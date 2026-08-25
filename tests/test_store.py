"""Integration tests for ganymede.coordinator.store against a real MinIO container.

Why real MinIO and not moto/mocked S3: the entire point of this module is the
presigned-URL host-signing footgun (docs/02-architecture-v2.md 6.6). A mock
client can't reproduce "MinIO signed against the wrong hostname" -- only a
real MinIO process with MINIO_SERVER_URL set to a non-localhost name, hit by
a plain (non-boto3) HTTP client, actually exercises that failure mode.

The whole module is skipped -- not failed -- if Docker is unavailable, so this
suite doesn't hard-fail CI on a machine without Docker.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest

from ganymede.coordinator.config import StorageConfig
from ganymede.coordinator.store import (
    ObjectNotFound,
    Store,
    adapter_key,
    base_adapter_key,
    momentum_key,
)

CONTAINER_NAME = "ganymede-test-minio"
HOST_PORT = 9010  # deliberately not 9000, to avoid colliding with a real MinIO
PUBLIC_HOST = "storage-test.local"
PUBLIC_ENDPOINT = f"http://{PUBLIC_HOST}:{HOST_PORT}"
ROOT_USER = "ganymede-test"
ROOT_PASSWORD = "ganymede-test-secret"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=True
        )
    except Exception:
        return False
    return True


def _ensure_hosts_entry() -> None:
    """Make sure `storage-test.local` resolves, so the test genuinely exercises
    a non-localhost hostname rather than accidentally hitting loopback by name
    coincidence."""
    try:
        with open("/etc/hosts") as f:
            contents = f.read()
        if PUBLIC_HOST not in contents:
            with open("/etc/hosts", "a") as f:
                f.write(f"\n127.0.0.1 {PUBLIC_HOST}\n")
    except PermissionError:
        # Fall back to checking it already resolves (e.g. pre-provisioned).
        pass
    # Verify it actually resolves now, one way or another.
    socket.gethostbyname(PUBLIC_HOST)


def _wait_for_minio_ready(timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    url = f"{PUBLIC_ENDPOINT}/minio/health/live"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def minio_container() -> Iterator[None]:
    if not _docker_available():
        pytest.skip("Docker is not available -- skipping store integration tests")

    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False
    )

    try:
        _ensure_hosts_entry()
    except Exception as exc:
        pytest.skip(f"could not make {PUBLIC_HOST} resolve: {exc}")

    run_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{HOST_PORT}:9000",
        "-e",
        f"MINIO_ROOT_USER={ROOT_USER}",
        "-e",
        f"MINIO_ROOT_PASSWORD={ROOT_PASSWORD}",
        "-e",
        f"MINIO_SERVER_URL={PUBLIC_ENDPOINT}",
        "quay.io/minio/minio:latest",
        "server",
        "/data",
    ]
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"could not start MinIO container: {result.stderr}")

    try:
        if not _wait_for_minio_ready():
            logs = subprocess.run(
                ["docker", "logs", CONTAINER_NAME], capture_output=True, text=True
            )
            pytest.skip(f"MinIO did not become ready in time: {logs.stdout} {logs.stderr}")
        yield
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture()
def store(minio_container: None) -> Store:
    cfg = StorageConfig(
        endpoint_url=PUBLIC_ENDPOINT,
        bucket=f"ganymede-test-{uuid.uuid4().hex[:8]}",
        region="us-east-1",
        access_key=ROOT_USER,
        secret_key=ROOT_PASSWORD,
        presign_expiry_sec=900,
    )
    s = Store(cfg)
    s.ensure_bucket()
    return s


# --- basic CRUD --------------------------------------------------------------


def test_ensure_bucket_is_idempotent(store: Store) -> None:
    # Calling it again against the same (already-created) bucket must not raise.
    store.ensure_bucket()
    store.ensure_bucket()


def test_put_get_roundtrip_binary_with_null_bytes(store: Store) -> None:
    data = bytes(range(256)) * 4096  # includes 0x00, exercises binary safety
    store.put_bytes("blobs/binary.bin", data)
    assert store.get_bytes("blobs/binary.bin") == data


def test_get_bytes_missing_key_raises_object_not_found(store: Store) -> None:
    with pytest.raises(ObjectNotFound):
        store.get_bytes("blobs/does-not-exist.bin")


def test_head_existing_and_missing(store: Store) -> None:
    payload = b"x" * 1234
    store.put_bytes("blobs/headme.bin", payload)

    info = store.head("blobs/headme.bin")
    assert info is not None
    assert info["size"] == len(payload)
    assert "etag" in info
    assert "last_modified" in info

    assert store.head("blobs/nope.bin") is None


def test_delete_is_idempotent(store: Store) -> None:
    store.put_bytes("blobs/deleteme.bin", b"gone soon")
    store.delete("blobs/deleteme.bin")
    assert store.head("blobs/deleteme.bin") is None
    # Deleting an already-missing key must not raise.
    store.delete("blobs/deleteme.bin")


def test_list_prefix_scopes_to_prefix_and_empty_prefix_returns_empty(store: Store) -> None:
    store.put_bytes("scoped/a.bin", b"a")
    store.put_bytes("scoped/b.bin", b"b")
    store.put_bytes("other/c.bin", b"c")

    keys = store.list_prefix("scoped/")
    assert sorted(keys) == ["scoped/a.bin", "scoped/b.bin"]

    assert store.list_prefix("nothing-here/") == []


def test_list_prefix_paginates_past_1000(store: Store) -> None:
    # list_objects_v2 truncates at 1000 keys per page; 1005 objects forces the
    # paginator to actually follow ContinuationToken at least once.
    n = 1005
    for i in range(n):
        store.put_bytes(f"many/{i:05d}.bin", b"x")

    keys = store.list_prefix("many/")
    assert len(keys) == n
    assert len(set(keys)) == n  # no duplicates from mishandled pagination


# --- presigned URLs ----------------------------------------------------------


def test_presigned_roundtrip_over_public_hostname(store: Store) -> None:
    """The headline test.

    This is the check that catches a coordinator signing the wrong hostname
    (docs/02-architecture-v2.md 6.6, "The presigned-URL host footgun"): if
    Store signed against a locally-reachable address instead of
    StorageConfig.endpoint_url (the public MINIO_SERVER_URL), this request --
    made against PUBLIC_ENDPOINT by a client that is deliberately NOT boto3 --
    would fail with a signature mismatch. Using urllib here (rather than
    boto3) proves an arbitrary HTTP client, like a worker's uploader, can use
    the URL with no S3 SDK involved at all.
    """
    payload = b"lora adapter weights, not really, but binary-shaped \x00\x01\x02" * 1000
    key = "presign/roundtrip.safetensors"

    put_url, put_expires = store.presign_put(key)
    req = urllib.request.Request(put_url, data=payload, method="PUT")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status in (200, 204)

    get_url, get_expires = store.presign_get(key)
    with urllib.request.urlopen(get_url, timeout=10) as resp:
        fetched = resp.read()
        assert resp.status == 200

    assert fetched == payload
    assert put_expires > datetime.now(timezone.utc)
    assert get_expires > datetime.now(timezone.utc)


def test_presign_get_missing_key_404s_on_fetch_not_on_presign(store: Store) -> None:
    # Presigning never touches the object -- it's pure local signing -- so it
    # must succeed even for a key that doesn't exist. The 404 only shows up
    # when the URL is actually fetched. Document that by asserting the 404
    # rather than expecting presign_get itself to raise.
    url, _ = store.presign_get("presign/never-uploaded.bin")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(url, timeout=10)
    assert exc_info.value.code == 404


def test_expires_at_is_timezone_aware_and_future(store: Store) -> None:
    url, expires_at = store.presign_put("presign/expiry-check.bin", expires_in=60)
    assert url  # sanity
    assert expires_at.tzinfo is not None
    assert expires_at.tzinfo.utcoffset(expires_at) is not None
    assert expires_at > datetime.now(timezone.utc)
    assert expires_at <= datetime.now(timezone.utc) + timedelta(seconds=61)


# --- key builders --------------------------------------------------------------


def test_key_builders_zero_pad_and_sort_numerically() -> None:
    keys = [base_adapter_key("run-1", i) for i in (2, 10)]
    # Lexicographic sort of the raw list must already match numeric round
    # order -- that's the entire point of zero-padding, and it's what GC's
    # list_prefix-based walk relies on.
    assert sorted(keys) == keys
    assert "00002" in keys[0]
    assert "00010" in keys[1]

    a = adapter_key("run-1", 3, "task-abc")
    assert a == "runs/run-1/rounds/00003/submissions/task-abc.safetensors"

    m = momentum_key("run-1")
    assert m == "runs/run-1/momentum.safetensors"
