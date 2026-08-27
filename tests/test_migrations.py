"""The forward-only migration runner (docs/05 "Prerequisite: a real migration
mechanism").

Two shapes matter and both are exercised here: a **fresh** database, where
``init_schema`` creates today's tables and then every migration runs over them;
and an **old** database, created from only the pre-delta ``db.SCHEMA`` +
``eligibility.SCHEMA`` with rows already in it, where ``init_schema`` has to
migrate forward without disturbing a single existing row or id.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from ganymede.coordinator import eligibility, migrations
from ganymede.coordinator.auth import hash_key
from ganymede.coordinator.db import SCHEMA, connect, init_schema


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _notnull(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    return {r[1]: r[3] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def fresh(tmp_path):
    conn = connect(str(tmp_path / "fresh.db"))
    init_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def old_db(tmp_path):
    """A database at the a1b4e36 shape: original schema, two real rows, a task."""
    path = str(tmp_path / "old.db")
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.executescript(eligibility.SCHEMA)
    cid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at) "
        "VALUES (?, 'alice', ?, 1, 'internal', '2024-01-01T00:00:00+00:00')",
        (cid, hash_key("alice-key")),
    )
    wid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO workers (id, contributor_id, compute_profile_json, image_tag, "
        "first_seen, last_seen, rounds_joined, steps_total) "
        "VALUES (?, ?, ?, 'ganymede/worker:v1', '2024-02-02T00:00:00+00:00', "
        "'2024-02-03T00:00:00+00:00', 7, 999)",
        (wid, cid, json.dumps({"backend": "cuda", "device_name": "RTX 3060",
                               "vram_mb": 12288})),
    )
    conn.execute(
        "INSERT INTO runs (id, status, base_model, base_precision, lora_cfg_json, "
        "dataset_ref, hyperparams_json, target_rounds, created_at) "
        "VALUES ('r1', 'active', 'm', 'bf16', '{}', 'd', '{}', 3, 'now')"
    )
    conn.execute(
        "INSERT INTO tasks (id, run_id, round_idx, buckets_json, local_steps, status, "
        "worker_id, attempts, created_at) "
        "VALUES ('t1', 'r1', 0, '[1,2,3]', 42, 'leased', ?, 1, 'now')",
        (wid,),
    )
    conn.commit()
    conn.close()
    conn = connect(path)
    yield conn, cid, wid
    conn.close()


# --------------------------------------------------------------------------
# Fresh database
# --------------------------------------------------------------------------


def test_fresh_reaches_latest_version(fresh):
    assert migrations.current_version(fresh) == migrations.LATEST_VERSION == 4
    rows = fresh.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    assert [r["version"] for r in rows] == [1, 2, 3, 4]


@pytest.mark.parametrize("table", [
    "jobs", "submitters", "images", "credit_events", "availability_ticks",
    "machine_weight", "enrollments", "machine_keys", "sessions", "schema_version",
])
def test_fresh_has_every_new_table(fresh, table):
    n = fresh.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()[0]
    assert n == 1


def test_fresh_has_every_new_column(fresh):
    assert {"auth_provider", "auth_subject", "is_admin", "email"} <= _cols(fresh, "contributors")
    assert {"display_name", "enrolled_at", "hardware_fingerprint_json", "standing",
            "reputation", "last_available_at"} <= _cols(fresh, "workers")
    assert "job_id" in _cols(fresh, "runs")
    assert "job_id" in _cols(fresh, "worker_eligibility")
    assert {"job_id", "input_ref_json", "attempt_group"} <= _cols(fresh, "tasks")
    assert {"finalized_at", "scanned_at", "scan_detail_json"} <= _cols(fresh, "images")


def test_tasks_run_id_and_round_idx_are_now_nullable(fresh):
    nn = _notnull(fresh, "tasks")
    assert nn["run_id"] == 0 and nn["round_idx"] == 0
    # The columns that were NOT NULL before still are.
    assert nn["buckets_json"] == 1 and nn["local_steps"] == 1 and nn["status"] == 1


def test_tasks_indexes_survive_the_rebuild(fresh):
    idx = {r[1] for r in fresh.execute("PRAGMA index_list(tasks)")}
    assert {"idx_tasks_round", "idx_tasks_lease", "idx_tasks_worker"} <= idx


def test_column_defaults_match_the_spec(fresh):
    cid = uuid.uuid4().hex
    fresh.execute(
        "INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at) "
        "VALUES (?, 'x', ?, 1, 'open', 'now')", (cid, hash_key("k")),
    )
    row = fresh.execute(
        "SELECT auth_provider, is_admin, auth_subject, email FROM contributors WHERE id=?",
        (cid,),
    ).fetchone()
    assert row["auth_provider"] == "local" and row["is_admin"] == 0
    assert row["auth_subject"] is None and row["email"] is None


def test_foreign_keys_pragma_is_left_on_after_the_rebuild(fresh):
    assert fresh.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert fresh.execute("PRAGMA foreign_key_check").fetchall() == []


def test_runner_is_idempotent(fresh):
    assert migrations.apply_pending(fresh) == []
    init_schema(fresh)  # the real re-entry path -- a coordinator restart
    assert migrations.current_version(fresh) == 4
    rows = fresh.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    assert rows == 4  # no duplicate cursor rows


# --------------------------------------------------------------------------
# Old database
# --------------------------------------------------------------------------


def test_old_db_migrates_forward(old_db):
    conn, _cid, _wid = old_db
    assert migrations.current_version(conn) == 0
    init_schema(conn)
    assert migrations.current_version(conn) == 4


def test_old_rows_survive_with_ids_and_values_intact(old_db):
    conn, cid, wid = old_db
    init_schema(conn)

    c = conn.execute("SELECT * FROM contributors WHERE id=?", (cid,)).fetchone()
    assert c["id"] == cid and c["name"] == "alice" and c["clearance"] == "internal"
    assert c["auth_provider"] == "local"  # backfilled default

    w = conn.execute("SELECT * FROM workers WHERE id=?", (wid,)).fetchone()
    assert w["id"] == wid  # migration 004 keeps every existing id
    assert w["rounds_joined"] == 7 and w["steps_total"] == 999
    assert w["enrolled_at"] == "2024-02-02T00:00:00+00:00"  # = first_seen
    assert w["standing"] == "good" and w["reputation"] == 0.25
    assert w["display_name"] == "ganymede/worker:v1"  # COALESCE(image_tag, ...)
    assert json.loads(w["hardware_fingerprint_json"])["gpu_model"] == "RTX 3060"

    t = conn.execute("SELECT * FROM tasks WHERE id='t1'").fetchone()
    assert t["run_id"] == "r1" and t["round_idx"] == 0 and t["worker_id"] == wid
    assert t["local_steps"] == 42 and t["job_id"] is None


def test_old_db_gets_one_synthesized_consumed_enrollment_per_machine(old_db):
    conn, cid, wid = old_db
    init_schema(conn)
    e = conn.execute("SELECT * FROM enrollments WHERE machine_id=?", (wid,)).fetchone()
    assert e is not None
    assert e["user_id"] == cid
    assert e["token_hash"] == "migrated:" + wid   # ':' -> un-matchable by any claim
    assert e["consumed_at"] == "2024-02-02T00:00:00+00:00"


def test_old_db_mints_no_machine_key_for_a_migrated_row(old_db):
    conn, _cid, wid = old_db
    init_schema(conn)
    n = conn.execute(
        "SELECT COUNT(*) FROM machine_keys WHERE machine_id=?", (wid,)
    ).fetchone()[0]
    assert n == 0


def test_old_db_rebuild_preserves_referential_integrity(old_db):
    conn, _cid, _wid = old_db
    init_schema(conn)
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_partial_upgrade_from_v2_runs_only_whats_pending(old_db):
    """A database left at an intermediate version picks up exactly the rest."""
    conn, _cid, _wid = old_db
    migrations._ensure_cursor(conn)
    migrations._m001_baseline(conn)
    migrations._m002_new_tables(conn)
    assert migrations.current_version(conn) == 2

    applied = migrations.apply_pending(conn)
    assert applied == [3, 4]
    assert migrations.current_version(conn) == 4
