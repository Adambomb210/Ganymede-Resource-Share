"""S3-compatible object storage for adapter weights (docs/02-architecture-v2.md 6.6).

Weights never travel through the coordinator's HTTP body -- workers PUT/GET
directly against object storage using URLs this module presigns. Storage is
self-hosted MinIO today and may become Cloudflare R2 later (6.6, "The
portability contract"); that swap must be a config change, not a code change.

To keep it that way, this module is restricted to the S3 subset every
S3-compatible implementation supports: ``put_object``, ``get_object``,
``head_object``, ``delete_object``, ``list_objects_v2``, ``create_bucket``, and
presigned URLs for PUT/GET. No lifecycle rules, no versioning, no tagging, no
multipart, no ``select_object_content`` -- R2 does not implement all of that,
and each one used is a migration blocker that goes unnoticed until the day you
try to move (6.6). Where a lifecycle rule would be tempting, the GC job (a
cron entry calling ``list_prefix`` + ``delete``) does the same thing portably.

The footgun this module exists to get right (6.6, "The presigned-URL host
footgun"): MinIO signs presigned URLs against whatever hostname it was told to
serve via ``MINIO_SERVER_URL``. Sign against the wrong hostname and every
presigned URL 403s with a signature mismatch that does not say why. Two rules
follow, both called out again at their point of use below:

1. Sign with the exact public endpoint workers will resolve -- byte-identical
   to the deployment's ``MINIO_SERVER_URL``. That's ``StorageConfig.endpoint_url``;
   never substitute a locally-reachable address such as a container-internal
   hostname or ``localhost``.
2. Use path-style addressing. Virtual-host style (``bucket.host/key``) needs
   wildcard DNS and a wildcard cert for what is one bucket; a self-hosted MinIO
   has neither (6.5).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from ganymede.coordinator.config import StorageConfig


class StoreError(Exception):
    """Wraps any botocore failure so callers never need to import botocore."""


class ObjectNotFound(StoreError):
    """Raised by ``get_bytes`` when the key does not exist."""


# --- Object key layout -------------------------------------------------------
#
# Round indices are zero-padded (%05d) so that lexicographic order -- which is
# what list_objects_v2 / list_prefix give you -- matches numeric order. The GC
# job (6.6, "Retention and garbage collection") walks rounds via list_prefix
# to find everything older than gc_keep_rounds; without zero-padding, "round
# 10" would sort before "round 2" and GC would delete the wrong things.


def adapter_key(run_id: str, round_idx: int, task_id: str) -> str:
    """Key for one worker's submitted adapter for a given round."""
    return f"runs/{run_id}/rounds/{round_idx:05d}/submissions/{task_id}.safetensors"


def base_adapter_key(run_id: str, round_idx: int) -> str:
    """Key for the round's base adapter (the aggregation result workers pull)."""
    return f"runs/{run_id}/rounds/{round_idx:05d}/base.safetensors"


def momentum_key(run_id: str) -> str:
    """Key for the outer-momentum state (DiLoCo combine mode, 5.2)."""
    return f"runs/{run_id}/momentum.safetensors"


class Store:
    """Thin wrapper over one long-lived boto3 S3 client."""

    def __init__(self, cfg: StorageConfig) -> None:
        self.cfg = cfg
        self._client = boto3.client(
            "s3",
            # Rule 1: sign with the exact public endpoint workers will use.
            # This must be byte-identical to the deployment's MINIO_SERVER_URL --
            # never a container-internal or localhost address, or every
            # presigned URL this client mints will 403 for real workers.
            endpoint_url=cfg.endpoint_url,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            config=Config(
                # Rule 2: path-style addressing. Virtual-host style needs
                # per-bucket wildcard DNS that a self-hosted MinIO won't have.
                s3={"addressing_style": "path"},
                signature_version="s3v4",
            ),
        )

    # -- bucket lifecycle -----------------------------------------------

    def ensure_bucket(self) -> None:
        """Create the bucket if absent. Called at startup; must be idempotent."""
        try:
            self._client.create_bucket(Bucket=self.cfg.bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            # Both are "the bucket already exists and is already mine" --
            # different S3 implementations report this differently, so both
            # count as success rather than failure.
            if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                return
            raise StoreError(f"ensure_bucket failed for {self.cfg.bucket!r}: {exc}") from exc

    # -- presigning -------------------------------------------------------

    def presign_put(self, key: str, expires_in: int | None = None) -> tuple[str, datetime]:
        return self._presign("put_object", key, expires_in)

    def presign_get(self, key: str, expires_in: int | None = None) -> tuple[str, datetime]:
        return self._presign("get_object", key, expires_in)

    def _presign(
        self, client_method: str, key: str, expires_in: int | None
    ) -> tuple[str, datetime]:
        ttl = expires_in if expires_in is not None else self.cfg.presign_expiry_sec
        try:
            url = self._client.generate_presigned_url(
                client_method,
                Params={"Bucket": self.cfg.bucket, "Key": key},
                ExpiresIn=ttl,
            )
        except ClientError as exc:
            raise StoreError(f"presign ({client_method}) failed for {key!r}: {exc}") from exc
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        return url, expires_at

    # -- direct object access ---------------------------------------------

    def put_bytes(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(Bucket=self.cfg.bucket, Key=key, Body=data)
        except ClientError as exc:
            raise StoreError(f"put_bytes failed for {key!r}: {exc}") from exc

    def get_bytes(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self.cfg.bucket, Key=key)
        except ClientError as exc:
            if _is_not_found(exc):
                raise ObjectNotFound(key) from exc
            raise StoreError(f"get_bytes failed for {key!r}: {exc}") from exc
        return resp["Body"].read()

    def head(self, key: str) -> dict[str, Any] | None:
        try:
            resp = self._client.head_object(Bucket=self.cfg.bucket, Key=key)
        except ClientError as exc:
            if _is_not_found(exc):
                return None
            raise StoreError(f"head failed for {key!r}: {exc}") from exc
        return {
            "size": resp["ContentLength"],
            "etag": resp["ETag"],
            "last_modified": resp["LastModified"],
        }

    def delete(self, key: str) -> None:
        # delete_object is itself idempotent in the S3 API (no error on a
        # missing key), so no existence check is needed here.
        try:
            self._client.delete_object(Bucket=self.cfg.bucket, Key=key)
        except ClientError as exc:
            raise StoreError(f"delete failed for {key!r}: {exc}") from exc

    def list_prefix(self, prefix: str) -> list[str]:
        keys: list[str] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            # list_objects_v2 truncates at 1000 keys per call; the paginator
            # follows ContinuationToken automatically. Round result history
            # alone can exceed 1000 objects over a long run, and GC depends
            # on seeing every key under a prefix, not just the first page.
            for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except ClientError as exc:
            raise StoreError(f"list_prefix failed for {prefix!r}: {exc}") from exc
        return keys


def _is_not_found(exc: ClientError) -> bool:
    """True for the various shapes a "no such object" error takes across
    S3-compatible backends (MinIO, AWS, R2 have all been observed to differ
    slightly here)."""
    error = exc.response.get("Error", {})
    code = error.get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("NoSuchKey", "404", "NotFound") or status == 404
