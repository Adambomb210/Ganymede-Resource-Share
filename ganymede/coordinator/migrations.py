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
  005  the scheduler workstream (docs/07): adopt every pre-scheduler ``runs`` row
       under a generic ``jobs`` row and set ``runs.job_id``; rebuild
       ``worker_eligibility`` keyed by ``job_id`` instead of ``run_id``; index
       the queue walk.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
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
# 005 -- the scheduler workstream (docs/07)
# --------------------------------------------------------------------------

# Adopted / system-seeded jobs need an owner, and pre-scheduler ``runs`` have no
# owner concept. This synthetic ``contributors`` row is that owner; its
# ``key_hash`` carries a ':' which no real ``hash_key`` output (64 hex chars) can
# contain, so it can never authenticate (the same trick migration 004 uses for
# synthesized enrollment rows). ``coordinator.app`` and ``scripts.newrun`` reuse
# the id for the jobs they create around existing runs.
SYSTEM_OWNER_ID = "system"

_M005_WE_NEW_DDL = """
CREATE TABLE worker_eligibility_new (
    worker_id  TEXT NOT NULL REFERENCES workers(id),
    job_id     TEXT NOT NULL REFERENCES jobs(id),
    outcome    TEXT NOT NULL,
    reason     TEXT,
    checked_at TEXT NOT NULL,
    PRIMARY KEY (worker_id, job_id)
)
"""

# ``runs.status`` -> ``jobs.status``. A pre-scheduler ``active`` run is a job
# that has already fanned out at least one round, so it maps to ``running``, not
# ``queued`` -- the single-active-run case must walk identically to today.
_M005_STATUS_MAP = {
    "active": "running", "paused": "paused",
    "done": "done", "failed": "failed", "draft": "draft",
}


def _m005_scheduler(conn: sqlite3.Connection) -> None:
    # The ``worker_eligibility`` rebuild drops and recreates the table. Nothing
    # references it, but mirror 003's belt-and-braces: FKs off for the swap, a
    # ``foreign_key_check`` after the commit as the assertion the copy kept every
    # reference intact.
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        with immediate(conn):
            now = _now()
            conn.execute(
                """INSERT OR IGNORE INTO contributors
                     (id, name, key_hash, enabled, clearance, created_at)
                   VALUES (?, 'system', 'system:owner', 1, 'open', ?)""",
                (SYSTEM_OWNER_ID, now),
            )

            # Adopt every run that predates the scheduler under a generic jobs
            # row, oldest first, and point the run back at it. Sparse ranks
            # (10, 20, ...) leave room for ``reorder``'s before/after to insert
            # without a renumber.
            base_rank = conn.execute(
                "SELECT COALESCE(MAX(priority_rank), 0) AS m FROM jobs"
            ).fetchone()["m"]
            runs = conn.execute(
                "SELECT id, status, created_at FROM runs WHERE job_id IS NULL "
                "ORDER BY created_at, id"
            ).fetchall()
            for i, r in enumerate(runs, start=1):
                job_id = uuid.uuid4().hex
                conn.execute(
                    """INSERT INTO jobs
                         (id, owner_id, job_type, spec_json, image_id, status,
                          priority_rank, constraints_json, cancel_mode, created_at)
                       VALUES (?, ?, 'collab_lora_finetune', '{}', NULL, ?, ?,
                               '{}', NULL, ?)""",
                    (job_id, SYSTEM_OWNER_ID,
                     _M005_STATUS_MAP.get(r["status"], "running"),
                     base_rank + i * 10, r["created_at"] or now),
                )
                conn.execute(
                    "UPDATE runs SET job_id = ? WHERE id = ?", (job_id, r["id"])
                )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_queue "
                "ON jobs(status, priority_rank)"
            )

            # ``worker_eligibility``: run_id -> job_id (docs/05, docs/07 §3).
            # Table rebuild following 003's ``tasks`` pattern -- ``eligibility.
            # SCHEMA`` stays frozen at the historical (run_id) shape so
            # ``test_migrations``'s old-db fixture is still an honest a1b4e36
            # reproduction. A row whose ``run_id`` no longer maps to a job is
            # dropped: it is a stale diagnostic, GC-eligible like ``audit``.
            if "run_id" in _columns(conn, "worker_eligibility"):
                conn.execute(_M005_WE_NEW_DDL)
                conn.execute(
                    """INSERT OR IGNORE INTO worker_eligibility_new
                         (worker_id, job_id, outcome, reason, checked_at)
                       SELECT we.worker_id, r.job_id, we.outcome, we.reason,
                              we.checked_at
                         FROM worker_eligibility we
                         JOIN runs r ON r.id = we.run_id
                        WHERE r.job_id IS NOT NULL"""
                )
                conn.execute("DROP TABLE worker_eligibility")
                conn.execute(
                    "ALTER TABLE worker_eligibility_new RENAME TO worker_eligibility"
                )

            _record(conn, 5)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(
                f"migration 005 left dangling references: "
                f"{[tuple(v) for v in violations]}"
            )
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


# --------------------------------------------------------------------------
# 006 -- contributor agreement (docs/03 roadmap open question 2): contributors
# need to record acceptance of the data-handling terms before touching any
# non-`open` run. One nullable timestamp, append-only semantics upstream in
# ``budget.clearance_and_terms_permit``.
# --------------------------------------------------------------------------


def _m006_contributor_agreement(conn: sqlite3.Connection) -> None:
    with immediate(conn):
        if "agreed_at" not in _columns(conn, "contributors"):
            conn.execute("ALTER TABLE contributors ADD COLUMN agreed_at TEXT")
        _record(conn, 6)


# --------------------------------------------------------------------------
# 007 -- when a job ended (docs/11 1.2 retention). "Keep an image while any
# non-terminal job references it, and for image_keep_days after the last
# referencing job reaches a terminal status" needs that moment written down;
# ``created_at`` is the wrong end of the job and nothing else recorded it.
# Backfilled to NULL, which the GC reads as "ended at some unknown past time"
# and treats with the image's own finalize stamp instead -- an old job cannot
# pin an image forever just because this column did not exist when it ran.
# --------------------------------------------------------------------------


def _m007_job_terminal_at(conn: sqlite3.Connection) -> None:
    with immediate(conn):
        if "terminal_at" not in _columns(conn, "jobs"):
            conn.execute("ALTER TABLE jobs ADD COLUMN terminal_at TEXT")
        # The other half of retention: an image whose *bytes* have been
        # collected keeps its row, because terminal jobs still point at it and
        # that history is worth more than the row. ``collected_at`` plus a
        # nulled ``object_ref`` is how everything downstream can tell that the
        # archive is gone without inferring it from a 404.
        if "collected_at" not in _columns(conn, "images"):
            conn.execute("ALTER TABLE images ADD COLUMN collected_at TEXT")
        _record(conn, 7)


# --------------------------------------------------------------------------
# 008 -- Phase D's fairness substrate (docs/13). Four tables and two columns,
# every one of them inert on arrival: no ``submitter_quotas`` row means no cap,
# an empty ``share_accounting`` means every share is zero, and both new ``tasks``
# columns default to NULL, which is exactly "nothing has changed".
# --------------------------------------------------------------------------

_M008_TABLES = [
    # docs/13 §1.5. The share rollup, recomputed on the sweep like reputation.
    # ``decayed_seconds`` is as-of ``updated_at`` and the *reader* ages it
    # forward, so a sweep that stops fades fairness out rather than freezing
    # whoever happened to be ahead when it died.
    """
    CREATE TABLE IF NOT EXISTS share_accounting (
        owner_id        TEXT PRIMARY KEY REFERENCES contributors(id),
        decayed_seconds REAL NOT NULL,
        formula_version INTEGER NOT NULL,
        updated_at      TEXT NOT NULL
    )
    """,
    # docs/13 §3.1. Absent = uncapped, which is the default for everybody
    # including every submitter that already exists. NULL in one column is that
    # one dimension uncapped, so a concurrency cap with no budget is a row with
    # one column filled.
    """
    CREATE TABLE IF NOT EXISTS submitter_quotas (
        user_id              TEXT PRIMARY KEY REFERENCES contributors(id),
        max_concurrent_tasks INTEGER,
        monthly_task_hours   REAL,
        note                 TEXT,
        updated_by           TEXT REFERENCES contributors(id),
        updated_at           TEXT NOT NULL
    )
    """,
    # docs/13 §5.3. The *only* thing that distinguishes a probe from real work,
    # and it lives here rather than on ``tasks`` precisely so that nothing a
    # worker can see carries it (§5.1).
    """
    CREATE TABLE IF NOT EXISTS spot_check_issues (
        task_id        TEXT PRIMARY KEY REFERENCES tasks(id),
        source_task_id TEXT NOT NULL REFERENCES tasks(id),
        issued_at      TEXT NOT NULL,
        outcome        TEXT,
        decided_at     TEXT
    )
    """,
]

_M008_INDEXES = [
    # The share sweep's one scan: tasks leased inside the lookback window.
    "CREATE INDEX IF NOT EXISTS idx_tasks_leased_at ON tasks(leased_at)",
    # ``recompute_reputation`` reads a machine's spot-check history per sweep.
    "CREATE INDEX IF NOT EXISTS idx_spot_check_outcome "
    "ON spot_check_issues(outcome, decided_at)",
]


def _m008_fairness(conn: sqlite3.Connection) -> None:
    with immediate(conn):
        task_cols = _columns(conn, "tasks")
        if "leased_at" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN leased_at TEXT")
            # Backfill from ``created_at``. That is *exact*, not approximate,
            # for every row that can exist today: ``collab_lora_finetune``
            # inserts its task rows already ``leased``, so creation and lease
            # are the same instant. A static type's tasks are planned at
            # enqueue and leased later -- for those the two differ by an
            # unbounded margin, which is the whole reason this column exists --
            # but no static-type task has ever been leased on a live database,
            # so the backfill has nothing to be wrong about. A future reader
            # will otherwise read this as a guess.
            conn.execute(
                "UPDATE tasks SET leased_at = created_at "
                "WHERE leased_at IS NULL AND status <> 'planned'"
            )
        # docs/13 §4.2. NULL = not preempted, which is every row.
        if "preempt_mode" not in task_cols:
            conn.execute("ALTER TABLE tasks ADD COLUMN preempt_mode TEXT")
        for stmt in _M008_TABLES:
            conn.execute(stmt)
        for stmt in _M008_INDEXES:
            conn.execute(stmt)
        _record(conn, 8)


# --------------------------------------------------------------------------
# 009 -- multi-GPU hosts (docs/14). Schema only: this migration creates the
# device ledger and backfills it against today's data, but nothing in the
# codebase reads these tables yet -- the allocation logic, the claim-path
# changes, the worker changes and the API changes are later steps. Every new
# column defaults to 1 / uncapped, which is "nothing has changed" for a fleet
# of single-GPU hosts, following 008's own inertness discipline.
# --------------------------------------------------------------------------

_M009_TABLES = [
    # Per-device inventory (docs/14 §2, §3). Device enumeration -- populating
    # device_index > 0 for a real multi-GPU box -- is a later step; this table
    # exists now so ``task_devices`` and ``device_reservations`` have somewhere
    # to point, and so the worker backfill below has somewhere to land.
    # ``retired_at`` is NULL for a live device; nothing sets it yet.
    """
    CREATE TABLE IF NOT EXISTS worker_devices (
        worker_id          TEXT    NOT NULL REFERENCES workers(id),
        device_index       INTEGER NOT NULL,
        device_name        TEXT    NOT NULL,
        vram_mb            INTEGER NOT NULL,
        compute_capability TEXT,
        supports_json      TEXT    NOT NULL DEFAULT '[]',
        alloc_max_mb       INTEGER,
        bench_score        REAL,
        retired_at         TEXT,
        PRIMARY KEY (worker_id, device_index)
    )
    """,
    # The allocation ledger (docs/14 §1, §3-4). Append-only: a release stamps
    # ``released_at`` rather than deleting the row, so one table carries both
    # the invariant (via the partial unique index below) and the per-device
    # utilisation history. Nothing allocates through this table yet -- that is
    # the later allocation-logic step -- except the live-lease backfill below,
    # which has to run inside this migration: a migration that leaves an
    # in-flight lease unaccounted lets the very next claim double-book the card
    # that task is already holding.
    #
    # The key is a surrogate rowid, deliberately, and NOT ``(task_id,
    # device_index)``. Task ids are recycled: ``_claim_static_task`` re-leases
    # the *same* ``tasks`` row after an expiry, an abandon or a preemption
    # (``UPDATE tasks SET status='leased' ... WHERE id = ?``) rather than
    # minting a new id. A task that lands on the same device twice therefore
    # produces two legitimate rows for one (task, device) pair -- one released,
    # one live -- and a composite key over those columns would reject the
    # second allocation with an IntegrityError. History has no natural key
    # here; the *invariant* lives entirely in the partial unique index below,
    # which is the only uniqueness this table should enforce.
    """
    CREATE TABLE IF NOT EXISTS task_devices (
        id             INTEGER PRIMARY KEY,
        task_id        TEXT    NOT NULL REFERENCES tasks(id),
        worker_id      TEXT    NOT NULL REFERENCES workers(id),
        device_index   INTEGER NOT NULL,
        allocated_at   TEXT    NOT NULL,
        released_at    TEXT,
        release_reason TEXT
    )
    """,
    # docs/14 §6: a blocked wide job accumulates a reservation on devices as
    # they free, with a TTL so a dead job cannot hold cards forever. Nothing
    # writes here yet -- the reservation logic is a later step -- but the
    # primary key is the per-device half of §6's uniqueness requirement. The
    # other half ("at most one job may hold reservations on a given worker at
    # a time") is not a key constraint and is enforced in that later step's
    # application logic, not here.
    """
    CREATE TABLE IF NOT EXISTS device_reservations (
        worker_id    TEXT    NOT NULL REFERENCES workers(id),
        device_index INTEGER NOT NULL,
        job_id       TEXT    NOT NULL REFERENCES jobs(id),
        reserved_at  TEXT    NOT NULL,
        expires_at   TEXT    NOT NULL,
        PRIMARY KEY (worker_id, device_index)
    )
    """,
]

_M009_INDEXES = [
    # The load-bearing invariant of the whole feature (docs/14 §1, §3): at most
    # one *unreleased* row per (worker, device), enforced by SQLite rather than
    # by application logic, so two concurrent claims cannot double-book a card
    # even if the free-set computation that led to them raced. A released row
    # (``released_at IS NOT NULL``) drops out of the index and stops occupying
    # the slot while staying on the record -- many released rows for the same
    # device are fine and expected.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_task_devices_busy "
    "ON task_devices(worker_id, device_index) WHERE released_at IS NULL",
    # The operator / ledger read path (docs/14 §4 ``device_history``).
    "CREATE INDEX IF NOT EXISTS idx_task_devices_history "
    "ON task_devices(worker_id, allocated_at)",
    # ``devices.release`` looks a task's live rows up by task id on every
    # terminal path -- submit, abandon, expire, cancel, preempt -- and neither
    # index above can serve that. Partial on the same predicate as
    # ``idx_task_devices_busy``, so it stays the size of the *currently held*
    # set rather than growing with the append-only history behind it: the one
    # index here that would otherwise get slower every day the fleet runs
    # (§8.3 anticipates pruning that history, but nothing prunes it yet).
    "CREATE INDEX IF NOT EXISTS idx_task_devices_live "
    "ON task_devices(task_id) WHERE released_at IS NULL",
]


def _int_or(value: object, default: int) -> int:
    """Coerce a JSON-decoded value to ``int``, falling back rather than
    raising. A probe field that is present but the wrong shape (a string that
    is not a number, a bool, a list) must not abort a migration that runs on
    every coordinator startup (``db.init_schema`` -> ``apply_pending``)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _backfill_worker_devices(conn: sqlite3.Connection) -> None:
    """One synthesized ``worker_devices`` row at ``device_index = 0`` per
    existing worker (docs/14 §3), built from its flat ``compute_profile_json``
    -- the shape ``worker/probe.py``'s ``run_probe`` produces: ``device_name``,
    ``vram_mb``, ``compute_capability`` and ``supports`` at the top level,
    ``alloc_max_mb`` / ``bench_score`` nested under ``probe``.

    A worker whose profile is missing, not JSON, or not an object falls back
    to the exact pair ``run_probe`` itself uses when its own ``describe()``
    call raises (``probe.py``: ``"backend:unknown"``-shaped name, ``vram_mb``
    0) -- a name that reads as unknown rather than a fabricated one, and zero
    VRAM, which is fail-closed in the same sense docs/14 §2 invokes for a
    concurrent probe: it makes the worker ineligible for work rather than
    silently inventing capacity that may not exist.

    ``INSERT OR IGNORE`` against the ``(worker_id, device_index)`` primary key
    makes this safe to call more than once against the same database -- the
    runner's idempotency requirement (re-running ``apply_pending`` on an
    up-to-date database is a no-op) -- without needing this function to first
    check what ``_m009_multi_gpu_hosts`` already guarantees by construction.
    Mirrors the same defensiveness as the ``task_devices`` insert below.
    """
    for row in conn.execute(
        "SELECT id, compute_profile_json FROM workers"
    ).fetchall():
        try:
            profile = json.loads(row["compute_profile_json"])
            if not isinstance(profile, dict):
                profile = {}
        except (TypeError, ValueError):
            profile = {}
        probe = profile.get("probe")
        if not isinstance(probe, dict):
            probe = {}
        supports = profile.get("supports")
        if not isinstance(supports, list):
            supports = []
        device_name = profile.get("device_name")
        if not isinstance(device_name, str) or not device_name:
            device_name = "unknown"
        conn.execute(
            """INSERT OR IGNORE INTO worker_devices
                 (worker_id, device_index, device_name, vram_mb,
                  compute_capability, supports_json, alloc_max_mb, bench_score)
               VALUES (?, 0, ?, ?, ?, ?, ?, ?)""",
            (row["id"], device_name, _int_or(profile.get("vram_mb"), 0),
             profile.get("compute_capability"), json.dumps(supports),
             probe.get("alloc_max_mb"), probe.get("bench_score")),
        )


def _backfill_task_devices(conn: sqlite3.Connection) -> None:
    """One ``task_devices`` row at ``device_index = 0`` per currently-``leased``
    task (docs/14 §3) -- required for correctness, not tidiness: a migration
    that leaves a live lease unaccounted lets the very next claim double-book
    the card that task is already holding.

    ``allocated_at`` is taken from ``tasks.leased_at``, not this migration's
    own clock, so the history this table starts is honest about when the
    lease actually began. ``leased_at`` can be NULL -- migration 008
    backfills it from ``created_at`` for every non-``planned`` row at the
    moment 008 runs, but that is a point-in-time backfill, not a constraint,
    so a row that reaches ``leased`` status afterward without ever setting
    ``leased_at`` is possible in principle. ``COALESCE(leased_at, created_at)``
    is 008's own fallback for exactly this gap, and ``tasks.created_at`` is
    ``NOT NULL``, so the COALESCE can never itself be NULL going into
    ``task_devices.allocated_at TEXT NOT NULL``.

    Only ``status = 'leased'`` rows hold a device. A preempted task (docs/13
    §4: ``preempt_mode`` set, or ``status = 'preempted'`` after
    ``expire_leases``) has already given its card back and is waiting to be
    re-claimed, not holding one.

    Decision 4 (docs/07 §1, superseded by docs/14 §1) -- "a machine holds at
    most one leased task" -- was an application-level invariant, not a
    database one (see ``invariants.py``, which exists because it can be
    violated), so a pre-existing double-book on one worker is possible in
    principle. Two such rows would both target ``(worker_id, 0)``; ordering by
    the same ``COALESCE(leased_at, created_at)`` used for ``allocated_at``,
    then ``id``, makes the earlier lease win deterministically, and
    ``INSERT OR IGNORE`` -- backed by ``idx_task_devices_busy`` -- silently
    drops the second rather than raising. Refusing to open the database over a
    pre-existing anomaly would be worse than surfacing it as a dark second
    card, which is exactly what docs/14 §4's "the row that never got its
    released_at names the task that failed to release" is for once someone
    reads the ledger.

    ``worker_id IS NOT NULL`` is required because ``tasks.worker_id`` is
    nullable but ``task_devices.worker_id`` is not -- a ``leased`` row with no
    worker is already a data anomaly this migration should not be the one to
    raise on.
    """
    rows = conn.execute(
        """SELECT id, worker_id, leased_at, created_at FROM tasks
            WHERE status = 'leased' AND worker_id IS NOT NULL
            ORDER BY COALESCE(leased_at, created_at), id"""
    ).fetchall()
    for row in rows:
        conn.execute(
            """INSERT OR IGNORE INTO task_devices
                 (task_id, worker_id, device_index, allocated_at)
               VALUES (?, ?, 0, ?)""",
            (row["id"], row["worker_id"], row["leased_at"] or row["created_at"]),
        )


def _m009_multi_gpu_hosts(conn: sqlite3.Connection) -> None:
    with immediate(conn):
        for stmt in _M009_TABLES:
            conn.execute(stmt)
        for stmt in _M009_INDEXES:
            conn.execute(stmt)

        # docs/14 §3. A job/task defaults to one GPU, and a submitter defaults
        # to uncapped -- both "nothing has changed" for every job and every
        # submitter that already exists, matching 008's own inertness
        # discipline for its additive columns.
        if "gpu_count" not in _columns(conn, "jobs"):
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1"
            )
        if "gpu_count" not in _columns(conn, "tasks"):
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1"
            )
        if "max_concurrent_gpus" not in _columns(conn, "submitter_quotas"):
            conn.execute(
                "ALTER TABLE submitter_quotas "
                "ADD COLUMN max_concurrent_gpus INTEGER"
            )

        _backfill_worker_devices(conn)
        _backfill_task_devices(conn)

        _record(conn, 9)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

MIGRATIONS: list[tuple[int, str, Migration]] = [
    (1, "baseline", _m001_baseline),
    (2, "new_tables", _m002_new_tables),
    (3, "additive_delta", _m003_additive_delta),
    (4, "identity_machine_id", _m004_identity),
    (5, "scheduler", _m005_scheduler),
    (6, "contributor_agreement", _m006_contributor_agreement),
    (7, "job_terminal_at", _m007_job_terminal_at),
    (8, "fairness", _m008_fairness),
    (9, "multi_gpu_hosts", _m009_multi_gpu_hosts),
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
