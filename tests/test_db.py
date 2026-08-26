"""Schema concerns: the additive migration, and the pragmas the rest relies on."""

from __future__ import annotations

import pytest

from ganymede.coordinator.db import _apply_additive_migrations, connect, init_schema


@pytest.fixture
def db(tmp_path):
    conn = connect(str(tmp_path / "g.db"))
    init_schema(conn)
    return conn


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_a_new_column_reaches_a_database_that_already_exists(db):
    """``CREATE TABLE IF NOT EXISTS`` is a no-op against an existing table.

    So a column added to SCHEMA reaches every fresh database and silently misses
    every existing one -- and the miss surfaces much later, as an error deep in
    the claim path, on the one deployment whose data is worth keeping.
    """
    db.execute("ALTER TABLE runs DROP COLUMN required_image")
    db.execute("ALTER TABLE tasks DROP COLUMN max_runtime_sec")
    db.commit()
    assert "required_image" not in _columns(db, "runs")

    applied = _apply_additive_migrations(db)

    assert set(applied) == {"runs.required_image", "tasks.max_runtime_sec"}
    assert "required_image" in _columns(db, "runs")
    assert "max_runtime_sec" in _columns(db, "tasks")


def test_migration_is_idempotent(db):
    assert _apply_additive_migrations(db) == []
    assert _apply_additive_migrations(db) == []


def test_migration_preserves_existing_rows(db):
    """An ALTER that dropped data would be worse than the problem it solves."""
    db.execute("ALTER TABLE runs DROP COLUMN required_image")
    db.execute(
        """INSERT INTO runs (id, status, base_model, base_precision, lora_cfg_json,
                             dataset_ref, hyperparams_json, target_rounds, created_at)
           VALUES ('r', 'active', 'm', 'bf16', '{}', 'd', '{}', 3, 'now')"""
    )
    db.commit()

    _apply_additive_migrations(db)

    row = db.execute("SELECT id, base_model, required_image FROM runs").fetchone()
    assert row["id"] == "r" and row["base_model"] == "m"
    assert row["required_image"] is None  # added columns start null, by design


def test_init_schema_migrates_as_well_as_creates(tmp_path):
    path = str(tmp_path / "g.db")
    conn = connect(path)
    init_schema(conn)
    conn.execute("ALTER TABLE runs DROP COLUMN required_image")
    conn.commit()
    conn.close()

    # Reopening a deployment is the moment the migration has to run: nobody
    # invokes it by hand.
    conn = connect(path)
    init_schema(conn)
    assert "required_image" in _columns(conn, "runs")


def test_foreign_keys_and_wal_are_on(db):
    """Both are per-connection pragmas, not schema properties.

    Foreign keys default to *off* in SQLite, so a connection that forgot them
    would let orphaned tasks and submissions accumulate silently.
    """
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
