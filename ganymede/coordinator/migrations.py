"""Forward-only schema migrations (docs/05-data-model.md, "Prerequisite: a real
migration mechanism").

``db.py`` stayed additive-only for as long as a nullable column was the only
shape change anyone needed. The platform-expansion delta breaks that: it makes
``tasks.run_id`` / ``tasks.round_idx`` nullable (a SQLite table rebuild) and
changes what ``workers.id`` *means* (migration 004). Neither is expressible as
``ALTER TABLE ADD COLUMN``.

This module is the minimum that buys: a ``schema_version`` table holding one
row -- the cursor -- and a hand-ordered list of ``(version, name, fn)`` blocks.
``apply_pending`` runs every block past the cursor, in order, each inside the
existing ``db.immediate()`` write transaction, each bumping the cursor in the
same transaction so a crash leaves the database at a whole version or the one
before it, never between.

It is deliberately not a framework. There is no down-migration, no autogenerate,
no dependency graph -- a forward-only list of SQL is what a single-writer SQLite
coordinator needs and no more.

The split (docs/05):
  001  baseline -- anchors the cursor over today's ``db.SCHEMA`` +
       ``eligibility.SCHEMA``; no DDL of its own.
  002  the new tables -- ``images``, ``jobs``, ``submitters``, ``credit_events``,
       ``availability_ticks``, ``machine_weight``, ``enrollments``,
       ``machine_keys``.
  003  the additive column delta on ``contributors`` / ``workers`` / ``runs`` /
       ``tasks`` / ``worker_eligibility``, plus the ``tasks`` rebuild that makes
       ``run_id`` / ``round_idx`` nullable.
  004  the ``workers.id`` rework: ``sessions``, the backfill, and one synthesized
       consumed ``enrollments`` row per pre-existing machine (docs/08,
       "Migration 004").
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone

from ganymede.coordinator.db import immediate

# A migration is a callable handed the live connection. It does its DDL/DML with
# plain ``conn.execute`` calls inside a single ``with immediate(conn)`` block and
# records its own ``schema_version`` row in that same block. It must never call
# ``executescript`` -- that issues an implicit COMMIT and would end the
# transaction mid-migration.
Migration = Callable[[sqlite3.Connection], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _record(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, _now()),
    )


# --------------------------------------------------------------------------
# 001 -- baseline
# --------------------------------------------------------------------------


def _m001_baseline(conn: sqlite3.Connection) -> None:
    """Anchor the cursor. ``init_schema`` has already run ``executescript`` over
    ``db.SCHEMA`` and ``eligibility.SCHEMA``; version 1 simply declares "the
    database is now at the a1b4e36 shape" so 002+ have a floor to build on. A
    database that predates this module and already carries that schema reaches
    here with an empty ``schema_version`` and is brought to 1 with no DDL."""
    _record(conn, 1)


# --------------------------------------------------------------------------
# 002 -- the new tables
# --------------------------------------------------------------------------

# ``images`` is created before ``jobs`` because ``jobs.image_id`` references it.
_M002_TABLES = [
    # Uploaded payload containers (docs/05 "images"; Decision 18). ``digest`` is
    # the SHA-256 of the ``docker save`` archive -- the value a worker recomputes
    # from the bytes it pulls, not the OCI manifest digest. Images go to the
    # object store; only the handle lives here. ``finalized_at`` / ``scanned_at``
    # / ``scan_detail_json`` are the Stage 1 reconciliation adds from docs/11:
    # a row is worker-visible only once ``finalized_at`` is set, and immutable
    # thereafter -- a rebuild is a new row with a new digest.
    """
    CREATE TABLE IF NOT EXISTS images (
        id               TEXT PRIMARY KEY,
        submitter_id     TEXT NOT NULL REFERENCES contributors(id),
        digest           TEXT,
        size_bytes       INTEGER,
        object_ref       TEXT,
        uploaded_at      TEXT,
        scan_status      TEXT NOT NULL DEFAULT 'pending',  -- pending | clean | flagged
        finalized_at     TEXT,
        scanned_at       TEXT,
        scan_detail_json TEXT
    )
    """,
    # The generic parent (docs/05 "jobs"). A collab_lora_finetune job has a child
    # ``runs`` row; a batch_inference job has none. Nothing writes ``jobs`` rows
    # in this phase -- the scheduler workstream owns that -- so this is the
    # column shape only.
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id               TEXT PRIMARY KEY,
        owner_id         TEXT NOT NULL REFERENCES contributors(id),
        job_type         TEXT NOT NULL,
        spec_json        TEXT NOT NULL,
        image_id         TEXT REFERENCES images(id),
        status           TEXT NOT NULL,   -- draft|queued|running|paused|done|failed|cancelled
        priority_rank    INTEGER NOT NULL,
        constraints_json TEXT NOT NULL DEFAULT '{}',
        cancel_mode      TEXT,            -- soft | hard, set when moved to cancelled
        created_at       TEXT NOT NULL
    )
    """,
    # The vetted allowlist (docs/05 "submitters"; Decisions 3, 9). Only an
    # ``approved`` user may POST images or jobs.
    """
    CREATE TABLE IF NOT EXISTS submitters (
        user_id     TEXT PRIMARY KEY REFERENCES contributors(id),
        status      TEXT NOT NULL,   -- pending | approved | denied | revoked
        decided_by  TEXT REFERENCES contributors(id),
        decided_at  TEXT,
        note        TEXT
    )
    """,
    # Append-only ledger (docs/05 "credit_events"; Decisions 6, 7, 11). Never
    # updated, never deleted. Running total is
    # ``SUM(weighted_hours) WHERE kind = 'provisioned'`` -- no balance column,
    # no debit row. On a ``kind = 'work'`` row ``raw_seconds`` carries the
    # ``credit()`` WorkUnits scalar instead of availability seconds (Stage 1
    # reconciliation #6); every banked query filters ``kind = 'provisioned'`` so
    # the overload is safe.
    """
    CREATE TABLE IF NOT EXISTS credit_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id      TEXT REFERENCES workers(id),
        user_id         TEXT REFERENCES contributors(id),
        kind            TEXT NOT NULL,   -- provisioned | work
        weighted_hours  REAL NOT NULL DEFAULT 0.0,
        raw_seconds     INTEGER NOT NULL DEFAULT 0,
        system_weight   REAL NOT NULL DEFAULT 0.0,
        formula_version INTEGER NOT NULL DEFAULT 0,
        period_start    TEXT,
        period_end      TEXT,
        created_at      TEXT NOT NULL
    )
    """,
    # The integral input for provisioned accrual (docs/05 "availability_ticks").
    # Appended on every poll / heartbeat; GC-eligible once the accrual engine has
    # summed the window into a ``credit_events`` row.
    """
    CREATE TABLE IF NOT EXISTS availability_ticks (
        machine_id       TEXT NOT NULL REFERENCES workers(id),
        at               TEXT NOT NULL,
        leased           INTEGER NOT NULL DEFAULT 0,
        in_good_standing INTEGER NOT NULL DEFAULT 1
    )
    """,
    # The probe-derived per-machine multiplier (docs/05 "machine_weight";
    # Decision 12). ``formula_version`` is how a re-weighting rolls forward
    # without being retroactive.
    """
    CREATE TABLE IF NOT EXISTS machine_weight (
        machine_id      TEXT PRIMARY KEY REFERENCES workers(id),
        weight          REAL NOT NULL,
        components_json TEXT NOT NULL DEFAULT '{}',
        formula_version INTEGER NOT NULL DEFAULT 0,
        computed_at     TEXT NOT NULL
    )
    """,
    # Pending machine-enrollment tokens (docs/05 "enrollments"; docs/08). The
    # token is shown once, at issue; only its sha256 is stored. There is no
    # ``expires_at`` column on purpose -- docs/08 fixes expiry as
    # ``created_at + GANYMEDE_ENROLL_TTL_SEC`` evaluated at claim.
    """
    CREATE TABLE IF NOT EXISTS enrollments (
        id           TEXT PRIMARY KEY,
        user_id      TEXT NOT NULL REFERENCES contributors(id),
        token_hash   TEXT NOT NULL,
        display_name TEXT,
        created_at   TEXT NOT NULL,
        consumed_at  TEXT,
        machine_id   TEXT REFERENCES workers(id)
    )
    """,
    # Per-machine bearer credential (docs/05 "machine_keys"; docs/08). Replaces
    # the hand-issued contributor key for workers. Revocation is ``enabled = 0``,
    # never DELETE -- matching auth.py, so the audit trail survives.
    """
    CREATE TABLE IF NOT EXISTS machine_keys (
        machine_id TEXT NOT NULL REFERENCES workers(id),
        key_hash   TEXT NOT NULL UNIQUE,
        enabled    INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_enrollments_token ON enrollments(token_hash)",
    "CREATE INDEX IF NOT EXISTS idx_machine_keys_machine ON machine_keys(machine_id)",
    "CREATE INDEX IF NOT EXISTS idx_credit_events_machine ON credit_events(machine_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_availability_machine ON availability_ticks(machine_id, at)",
]


def _m002_new_tables(conn: sqlite3.Connection) -> None:
    with immediate(conn):
        for stmt in _M002_TABLES:
            conn.execute(stmt)
        _record(conn, 2)


# --------------------------------------------------------------------------
# 003 -- the additive column delta + the tasks rebuild
# --------------------------------------------------------------------------

# table -> {column: "<type + constraint>"} exactly as docs/05 "Existing tables --
# changes" lists them. Each is added only if absent, so 003 is safe against a
# database that already carries some of them.
_M003_ADDS: dict[str, dict[str, str]] = {
    "contributors": {
        "auth_provider": "TEXT NOT NULL DEFAULT 'local'",
        "auth_subject": "TEXT",
        "is_admin": "INTEGER NOT NULL DEFAULT 0",
        "email": "TEXT",
    },
    "workers": {
        "display_name": "TEXT",
        "enrolled_at": "TEXT",
        "hardware_fingerprint_json": "TEXT",
        "standing": "TEXT NOT NULL DEFAULT 'good'",
        "reputation": "REAL NOT NULL DEFAULT 0.25",
        "last_available_at": "TEXT",
    },
    "runs": {
        "job_id": "TEXT REFERENCES jobs(id)",
    },
    "worker_eligibility": {
        # docs/05: "add job_id; keep run_id for now". Nothing creates ``jobs``
        # rows yet, so the scheduler workstream rewires the writers -- 003 only
        # makes the column exist.
        "job_id": "TEXT REFERENCES jobs(id)",
    },
}

# The carry-over columns on ``tasks`` -- everything that predates 003. The
# rebuild copies exactly the intersection of this set with what the live table
# actually has, so an ancient database that never picked up ``max_runtime_sec``
# / ``last_heartbeat_steps`` migrates without a "no such column" on the SELECT.
_TASKS_CARRY = [
    "id", "run_id", "round_idx", "buckets_json", "local_steps", "status",
    "worker_id", "lease_expires_at", "attempts", "last_heartbeat_steps",
    "max_runtime_sec", "created_at",
]

# ``tasks`` after 003. ``run_id`` / ``round_idx`` lose ``NOT NULL`` (non-training
# jobs have neither); ``job_id`` / ``input_ref_json`` / ``attempt_group`` are the
# docs/05 adds. Column order is otherwise the original so a reader diffing the
# two sees only the intended change.
_TASKS_NEW_DDL = """
CREATE TABLE tasks_new (
    id                   TEXT PRIMARY KEY,
    run_id               TEXT REFERENCES runs(id),
    round_idx            INTEGER,
    job_id               TEXT REFERENCES jobs(id),
    buckets_json         TEXT NOT NULL,
    input_ref_json       TEXT,
    attempt_group        TEXT,
    local_steps          INTEGER NOT NULL,
    status               TEXT NOT NULL,
    worker_id            TEXT REFERENCES workers(id),
    lease_expires_at     TEXT,
    attempts             INTEGER NOT NULL DEFAULT 1,
    last_heartbeat_steps INTEGER,
    max_runtime_sec      INTEGER,
    created_at           TEXT NOT NULL
)
"""

_TASKS_INDEXES = [
    "CREATE INDEX idx_tasks_round  ON tasks(run_id, round_idx, status)",
    "CREATE INDEX idx_tasks_lease  ON tasks(status, lease_expires_at)",
    "CREATE INDEX idx_tasks_worker ON tasks(worker_id, status)",
]


def _m003_additive_delta(conn: sqlite3.Connection) -> None:
    # The ``tasks`` rebuild drops and recreates the table, so foreign keys must
    # be off while it runs or the DROP trips every child reference. The pragma
    # is a no-op inside a transaction, so it is toggled here, outside
    # ``immediate``; ``foreign_key_check`` after the commit is the assertion
    # that the copy kept every reference intact.
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        with immediate(conn):
            for table, cols in _M003_ADDS.items():
                have = _columns(conn, table)
                for name, decl in cols.items():
                    if name not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

            carry = [c for c in _TASKS_CARRY if c in _columns(conn, "tasks")]
            collist = ", ".join(carry)
            conn.execute(_TASKS_NEW_DDL)
            conn.execute(
                f"INSERT INTO tasks_new ({collist}) SELECT {collist} FROM tasks"
            )
            conn.execute("DROP TABLE tasks")
            conn.execute("ALTER TABLE tasks_new RENAME TO tasks")
            for stmt in _TASKS_INDEXES:
                conn.execute(stmt)
            _record(conn, 3)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(
                f"migration 003 left dangling references: {[tuple(v) for v in violations]}"
            )
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


# --------------------------------------------------------------------------
# 004 -- workers.id becomes enrollment-minted
# --------------------------------------------------------------------------

_M004_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    token_hash   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES contributors(id),
    created_at   TEXT,
    expires_at   TEXT NOT NULL,
    last_used_at TEXT
)
"""


def _m004_identity(conn: sqlite3.Connection) -> None:
    # Imported here rather than at module top: ``identity`` is the owner of the
    # fingerprint projection and the enrollment machinery, and it is not needed
    # until a database actually reaches version 4.
    from ganymede.coordinator.identity import fingerprint_from_profile

    with immediate(conn):
        conn.execute(_M004_SESSIONS_DDL)

        # Backfill every pre-existing machine to the enrollment-era record shape
        # (docs/08 "Backfill"). Existing ids are already unique primary keys --
        # derived, not wrong -- so 004 keeps them and rewrites no foreign key.
        rows = conn.execute(
            "SELECT id, contributor_id, compute_profile_json, image_tag, first_seen, "
            "display_name, enrolled_at, hardware_fingerprint_json FROM workers"
        ).fetchall()
        for row in rows:
            try:
                profile = json.loads(row["compute_profile_json"])
            except (TypeError, ValueError):
                profile = {}
            display_name = row["display_name"] or row["image_tag"] or (
                "machine-" + row["id"][:8]
            )
            conn.execute(
                """UPDATE workers
                     SET enrolled_at = COALESCE(enrolled_at, ?),
                         display_name = COALESCE(display_name, ?),
                         standing = COALESCE(standing, 'good'),
                         hardware_fingerprint_json =
                             COALESCE(hardware_fingerprint_json, ?)
                   WHERE id = ?""",
                (row["first_seen"], display_name,
                 fingerprint_from_profile(profile), row["id"]),
            )
            # One synthesized consumed enrollment per machine, so every row in
            # ``workers`` has a uniform provenance record. ``token_hash`` carries
            # a ':' which no real ``hash_key`` output (64 hex chars) can contain,
            # so the row can never be matched by a claim.
            already = conn.execute(
                "SELECT 1 FROM enrollments WHERE machine_id = ?", (row["id"],)
            ).fetchone()
            if already is None:
                conn.execute(
                    """INSERT INTO enrollments
                         (id, user_id, token_hash, display_name,
                          created_at, consumed_at, machine_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    ("migrated-" + row["id"], row["contributor_id"],
                     "migrated:" + row["id"], display_name,
                     row["first_seen"], row["first_seen"], row["id"]),
                )
        # No machine key is minted for a migrated row -- there is no channel to
        # deliver one. auth's transitional legacy-worker rule (docs/08 Spine
        # deviation 4) lets such a machine authenticate with its owner's
        # contributor key until it re-enrolls.
        _record(conn, 4)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

MIGRATIONS: list[tuple[int, str, Migration]] = [
    (1, "baseline", _m001_baseline),
    (2, "new_tables", _m002_new_tables),
    (3, "additive_delta", _m003_additive_delta),
    (4, "identity_machine_id", _m004_identity),
]


def _ensure_cursor(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "  version INTEGER PRIMARY KEY,"
        "  applied_at TEXT"
        ")"
    )


def current_version(conn: sqlite3.Connection) -> int:
    """The highest applied migration, or 0 on a database this runner has never
    touched."""
    _ensure_cursor(conn)
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return row["v"] or 0


def apply_pending(conn: sqlite3.Connection) -> list[int]:
    """Run every migration past the cursor, in order. Returns the versions
    applied this call (empty when already current). Called from
    ``db.init_schema`` on every connect -- reopening a deployment is the moment
    a pending migration has to run, because nobody invokes it by hand."""
    _ensure_cursor(conn)
    cursor = current_version(conn)
    applied: list[int] = []
    for version, _name, fn in MIGRATIONS:
        if version <= cursor:
            continue
        fn(conn)
        applied.append(version)
    return applied


LATEST_VERSION = MIGRATIONS[-1][0]
