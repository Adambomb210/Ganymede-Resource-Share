"""Coordinator configuration.

Every deployment-specific value is an environment variable. No hostname appears
in the codebase (docs/02-architecture-v2.md 6.5), so moving from a laptop to a
server to R2 is a config change rather than a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"required environment variable {name} is not set")
    return val


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- Constants that 03-roadmap.md "Blocking check for M1" requires M1 to name. ---

# The third tier of the throughput ladder (2-architecture 3.5): first-ever worker of
# an unseen GPU class on an uncalibrated run. Deliberately low -- it costs one
# under-filled round and is replaced by a measurement immediately afterwards. The
# failure mode of guessing high (a truncated round, wasted work, every round) is
# much worse than the failure mode of guessing low (one short round).
COLD_START_STEPS_PER_MIN = 3.0

# The conservative arm of the 5.2 A/B: lr_outer = 1, beta = 0 reduces the outer step
# to a plain weighted mean. Both modes ship; a fresh run gets this one until M4 says
# the momentum variant is better on LoRA adapters specifically.
# Gate 4's tolerance and the aggregation dominance cap. Named here rather than
# written into the call sites so that GANYMEDE_NORM_REJECT_K and
# GANYMEDE_DOMINANCE_CAP are actually reachable from a deployment -- a tunable
# nobody reads is worse than no tunable, because the operator believes it works.
DEFAULT_NORM_REJECT_K = 5.0
DEFAULT_DOMINANCE_CAP = 2.0

# Base-model load plus adapter attach, in seconds. Sized from a measured M2
# round: 110 s for a 0.6B model with a warm HuggingFace cache. Larger models and
# cold caches are worse, so raise it per deployment -- a worker's submitted
# `setup_sec` metric is the number to raise it from.
DEFAULT_EST_SETUP_SEC = 120

DEFAULT_COMBINE_MODE = "mean"
DEFAULT_LR_OUTER = 1.0
DEFAULT_OUTER_MOMENTUM = 0.9  # only consulted when combine mode is "diloco"


@dataclass(frozen=True)
class StorageConfig:
    """S3-compatible object storage. Self-hosted MinIO now, R2 later, same code."""

    endpoint_url: str
    bucket: str
    region: str
    access_key: str
    secret_key: str
    # MinIO signs against whatever MINIO_SERVER_URL says. If the coordinator signs a
    # different hostname than the one workers resolve, every presigned URL 403s. The
    # value here must be byte-identical to the deployment's MINIO_SERVER_URL.
    presign_expiry_sec: int = 900

    @classmethod
    def from_env(cls) -> "StorageConfig":
        return cls(
            endpoint_url=_env("STORAGE_HOST"),
            bucket=_env("S3_BUCKET", "ganymede"),
            region=_env("S3_REGION", "us-east-1"),
            access_key=_env("S3_ACCESS_KEY"),
            secret_key=_env("S3_SECRET_KEY"),
            presign_expiry_sec=_env_int("S3_PRESIGN_EXPIRY_SEC", 900),
        )


@dataclass(frozen=True)
class Settings:
    db_path: str
    storage: StorageConfig
    coordinator_host: str
    require_tls: bool
    lease_duration_sec: int
    heartbeat_interval_sec: int
    # A worker claiming with less than this much usable time left in the round gets
    # 204 rather than work it cannot finish (2-architecture 3.2).
    min_usable_sec: int
    # Fixed overheads subtracted from remaining round time before sizing a budget.
    est_download_sec: int
    est_upload_sec: int
    # Loading the base model and attaching the adapter. A fixed per-task cost
    # like download and upload, not a rate -- see budget.usable_seconds for the
    # measurement that put it there.
    est_setup_sec: int
    safety_margin_sec: int
    # Gate 4: reject a tensor whose Frobenius norm exceeds k x the cohort median.
    norm_reject_k: float
    # No single worker may carry more than this multiple of the median contribution.
    dominance_cap: float
    # A worker that cannot reach this fraction of the median budget is not offered
    # work for this run (it stays eligible for others).
    throughput_floor_frac: float
    gc_keep_rounds: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=_env("GANYMEDE_DB", "/var/lib/ganymede/coordinator.db"),
            storage=StorageConfig.from_env(),
            coordinator_host=_env("COORDINATOR_HOST", "http://localhost:8000"),
            require_tls=_env_bool("GANYMEDE_REQUIRE_TLS", True),
            lease_duration_sec=_env_int("GANYMEDE_LEASE_SEC", 900),
            heartbeat_interval_sec=_env_int("GANYMEDE_HEARTBEAT_SEC", 60),
            min_usable_sec=_env_int("GANYMEDE_MIN_USABLE_SEC", 300),
            est_download_sec=_env_int("GANYMEDE_EST_DOWNLOAD_SEC", 60),
            est_upload_sec=_env_int("GANYMEDE_EST_UPLOAD_SEC", 30),
            est_setup_sec=_env_int("GANYMEDE_EST_SETUP_SEC", DEFAULT_EST_SETUP_SEC),
            safety_margin_sec=_env_int("GANYMEDE_SAFETY_MARGIN_SEC", 60),
            norm_reject_k=_env_float("GANYMEDE_NORM_REJECT_K", DEFAULT_NORM_REJECT_K),
            dominance_cap=_env_float("GANYMEDE_DOMINANCE_CAP", DEFAULT_DOMINANCE_CAP),
            throughput_floor_frac=_env_float("GANYMEDE_THROUGHPUT_FLOOR", 0.10),
            gc_keep_rounds=_env_int("GANYMEDE_GC_KEEP_ROUNDS", 3),
        )
