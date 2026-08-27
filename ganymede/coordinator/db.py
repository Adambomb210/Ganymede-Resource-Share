"""SQLite schema and connection handling (docs/02-architecture-v2.md 6.1).

WAL mode so readers never block the writer. Every state transition that two
workers could race -- claiming a task, closing a round -- runs inside
``BEGIN IMMEDIATE`` so SQLite takes the write lock at statement one rather than
at first write, which is what makes the claim path safe under concurrency.
"""

from __future__ import annotations

import sqlite3

from ganymede.coordinator import eligibility
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS contributors (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    enabled     INTEGER NOT NULL DEFAULT 1,
    clearance   TEXT NOT NULL DEFAULT 'open',   -- open | internal | restricted
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
    id                  TEXT PRIMARY KEY,
    contributor_id      TEXT NOT NULL REFERENCES contributors(id),
    compute_profile_json TEXT NOT NULL,
    image_tag           TEXT,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    rounds_joined       INTEGER NOT NULL DEFAULT 0,
    steps_total         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_workers_contributor ON workers(contributor_id);

CREATE TABLE IF NOT EXISTS runs (
    id                  TEXT PRIMARY KEY,
    status              TEXT NOT NULL,          -- draft | active | paused | done | failed
    base_model          TEXT NOT NULL,
    base_precision      TEXT NOT NULL,
    lora_cfg_json       TEXT NOT NULL,
    dataset_ref         TEXT NOT NULL,
    hyperparams_json    TEXT NOT NULL,
    current_round       INTEGER NOT NULL DEFAULT 0,
    target_rounds       INTEGER NOT NULL,
    combine_mode        TEXT NOT NULL DEFAULT 'mean',   -- mean | diloco
    lr_outer            REAL NOT NULL DEFAULT 1.0,
    outer_beta          REAL NOT NULL DEFAULT 0.0,
    outer_momentum_ref  TEXT,
    requires_json       TEXT NOT NULL DEFAULT '{}',
    -- Image tag a worker must be running to claim this run (8, 4.2 step 5).
    -- NULL means no requirement, which is the native-install case; a non-NULL
    -- value is also how 6.10's "restricted runs use the container path" is
    -- actually enforced, since a native worker has no image tag to match.
    required_image      TEXT,
    data_classification TEXT NOT NULL DEFAULT 'open',
    num_buckets         INTEGER NOT NULL DEFAULT 64,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rounds (
    run_id                TEXT NOT NULL REFERENCES runs(id),
    idx                   INTEGER NOT NULL,
    base_adapter_ref      TEXT NOT NULL,
    status                TEXT NOT NULL,        -- open | closing | closed
    target_steps          INTEGER NOT NULL,
    min_round_sec         INTEGER NOT NULL,
    max_round_sec         INTEGER NOT NULL,
    opened_at             TEXT NOT NULL,
    closed_at             TEXT,
    result_adapter_ref    TEXT,
    distinct_contributors INTEGER NOT NULL DEFAULT 0,
    eval_loss             REAL,
    adapter_divergence    REAL,
    PRIMARY KEY (run_id, idx)
);

CREATE TABLE IF NOT EXISTS tasks (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(id),
    round_idx         INTEGER NOT NULL,
    buckets_json      TEXT NOT NULL,
    local_steps       INTEGER NOT NULL,
    status            TEXT NOT NULL,            -- leased | submitted | abandoned | expired
    worker_id         TEXT REFERENCES workers(id),
    lease_expires_at  TEXT,
    attempts          INTEGER NOT NULL DEFAULT 1,
    -- Last progress figure the worker reported, for acceptance gate 5. Kept on
    -- the task rather than recovered from the audit log: the audit table grows
    -- with every heartbeat of every worker forever, so reading progress out of
    -- it would make each submit scan a table that never stops growing.
    last_heartbeat_steps INTEGER,
    -- Wall-clock ceiling handed to the worker as a safety net (8). Stored at
    -- claim rather than recomputed, so a worker resuming a held lease is told
    -- the same budget it was originally given; lease_expires_at is the harder
    -- bound and the worker honours whichever comes first.
    max_runtime_sec   INTEGER,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_round  ON tasks(run_id, round_idx, status);
CREATE INDEX IF NOT EXISTS idx_tasks_lease  ON tasks(status, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_tasks_worker ON tasks(worker_id, status);

CREATE TABLE IF NOT EXISTS submissions (
    task_id         TEXT PRIMARY KEY REFERENCES tasks(id),
    artifact_ref    TEXT NOT NULL,
    steps_completed INTEGER NOT NULL,
    tokens_seen     INTEGER NOT NULL DEFAULT 0,
    metrics_json    TEXT NOT NULL DEFAULT '{}',
    accepted        INTEGER,                    -- NULL until gated
    reject_reason   TEXT,
    received_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS buckets (
    run_id       TEXT NOT NULL REFERENCES runs(id),
    bucket_idx   INTEGER NOT NULL,
    times_trained INTEGER NOT NULL DEFAULT 0,
    last_round   INTEGER,
    PRIMARY KEY (run_id, bucket_idx)
);

CREATE TABLE IF NOT EXISTS throughput (
    run_id        TEXT NOT NULL REFERENCES runs(id),
    gpu_model     TEXT NOT NULL,
    steps_per_min REAL NOT NULL,
    samples       INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, gpu_model)
);

CREATE TABLE IF NOT EXISTS calibration (
    run_id           TEXT PRIMARY KEY REFERENCES runs(id),
    calibration_json TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

-- Rejections are logged per contributor: this is the raw material for the
-- Phase 2 reputation scoring, so it accumulates from M1 rather than being
-- instrumented later (2-architecture 5.1).
CREATE TABLE IF NOT EXISTS audit (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    at             TEXT NOT NULL,
    contributor_id TEXT,
    worker_id      TEXT,
    event          TEXT NOT NULL,
    detail_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_contributor ON audit(contributor_id, at);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection with the pragmas the concurrency story depends on."""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


# Columns added after a table's first release. ``CREATE TABLE IF NOT EXISTS``
# is a no-op against an existing table, so a new column in SCHEMA above reaches
# a fresh database and silently misses every existing one -- and the failure
# lands later, as a KeyError deep in the claim path, on the one deployment that
# has data worth keeping.
#
# Additive only, deliberately. Renames, drops and type changes need a real
# migration with a version number and a plan; adding a nullable column does
# not, and pretending otherwise here would mean writing a migration framework
# before there is a second thing to migrate.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "runs": {"required_image": "TEXT"},
    "tasks": {"max_runtime_sec": "INTEGER", "last_heartbeat_steps": "INTEGER"},
}


def _apply_additive_migrations(conn: sqlite3.Connection) -> list[str]:
    """Add any declared column the database does not already have."""
    applied = []
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table itself is new; CREATE TABLE already has the column
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                applied.append(f"{table}.{name}")
    if applied:
        conn.commit()
    return applied


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Lives in its own module because it is a diagnostic rather than
    # protocol state: nothing in the round lifecycle reads it, and
    # dropping the table would cost answers, not correctness.
    conn.executescript(eligibility.SCHEMA)
    _apply_additive_migrations(conn)


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block under an immediate write transaction.

    ``BEGIN IMMEDIATE`` acquires the write lock up front. With the default
    deferred behaviour SQLite takes it at the first write, which leaves a window
    where two claimants both read the same unleased task and both believe they
    won it. Every read-then-write state transition goes through here.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
