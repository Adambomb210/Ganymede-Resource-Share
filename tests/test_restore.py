"""Rebuilding a coordinator from an off-box backup (docs/02-architecture-v2.md 6.6).

The drill itself — a real MinIO, a real backup, the primary bucket and database
destroyed, a real worker claiming against the rebuilt coordinator — is recorded
in the roadmap's M5 status. These are the parts of it that should keep working
without one, and they concentrate on the half that is easy to get wrong.

Restoring the *database* is a file copy and hard to break. Restoring a
coordinator to a state a fleet can actually resume from is not: the snapshot
describes a moment that no longer exists, holding leases nobody will report
against and submissions whose artifacts were deliberately never backed up.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ganymede.coordinator.store import base_adapter_key
from scripts import restore as restore_mod


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def backup_store(store):
    """A second in-memory store standing in for the off-box bucket.

    `conftest`'s FakeStore matches Store's public surface exactly, so a
    divergence shows up as an AttributeError rather than as different
    behaviour — which is what lets the off-box copy be exercised without a
    second real endpoint.
    """
    return type(store)()


def _in_flight(conn, run_id: str, round_idx: int = 0):
    """A worker mid-round when the disk died: one lease, one submission."""
    now = _iso(datetime.now(timezone.utc))
    conn.execute(
        """INSERT INTO workers (id, contributor_id, compute_profile_json, image_tag,
                                first_seen, last_seen)
           VALUES ('w1', 'c-restore', '{}', NULL, ?, ?)""", (now, now))
    for tid, status in (("t-leased", "leased"), ("t-submitted", "submitted")):
        conn.execute(
            """INSERT INTO tasks (id, run_id, round_idx, buckets_json, local_steps, status,
                                  worker_id, lease_expires_at, attempts, max_runtime_sec, created_at)
               VALUES (?, ?, ?, '[0,1]', 50, ?, 'w1', ?, 1, 600, ?)""",
            (tid, run_id, round_idx, status, now, now))
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed, tokens_seen,
                                    metrics_json, received_at)
           VALUES ('t-submitted', 'runs/x/sub.safetensors', 50, 0, '{}', ?)""", (now,))
    conn.commit()


@pytest.fixture
def contributor_row(conn):
    now = _iso(datetime.now(timezone.utc))
    conn.execute(
        """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
           VALUES ('c-restore', 'restore', 'h-restore', 1, 'open', ?)""", (now,))
    conn.commit()


# --------------------------------------------------------------------------
# Reconcile: the half that decides whether the run can resume
# --------------------------------------------------------------------------


def test_leases_nobody_will_report_against_are_released(conn, seeded_run, contributor_row):
    """Those workers are talking to a machine that no longer exists. Left
    leased, their shards look spoken for until every lease expires."""
    run_id = seeded_run()
    _in_flight(conn, run_id)

    report = restore_mod.RestoreReport()
    restore_mod.reconcile(conn, report)

    assert report.leases_released == 2
    leased = conn.execute(
        "SELECT COUNT(*) c FROM tasks WHERE status IN ('leased','submitted')"
    ).fetchone()["c"]
    assert leased == 0


def test_submissions_whose_artifacts_were_never_backed_up_are_dropped(
    conn, seeded_run, contributor_row
):
    """6.6 deliberately does not back up individual submissions: they are
    already folded into the round result that *is* backed up, and they are the
    bulk of the bytes. So a restored database references objects that exist
    nowhere, and a round holding them can never close."""
    run_id = seeded_run()
    _in_flight(conn, run_id)

    report = restore_mod.RestoreReport()
    restore_mod.reconcile(conn, report)

    assert report.submissions_dropped == 1
    assert conn.execute("SELECT COUNT(*) c FROM submissions").fetchone()["c"] == 0


def test_a_round_caught_mid_close_is_reopened_rather_than_left_wedged(
    conn, seeded_run, contributor_row
):
    """`closing` is claimed by exactly one caller and released only when that
    caller finishes. A restore that left it there would wedge the run
    permanently — the same `stuck_close` invariants.py reports."""
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET status = 'closing' WHERE run_id = ?", (run_id,))
    conn.commit()

    restore_mod.reconcile(conn, restore_mod.RestoreReport())

    assert conn.execute(
        "SELECT status FROM rounds WHERE run_id = ?", (run_id,)
    ).fetchone()["status"] == "open"


def test_the_rewound_round_gets_a_fresh_deadline(conn, seeded_run, contributor_row):
    """Its original `opened_at` is however long ago the disk died. Keeping it
    would make the round instantly overdue and close on nothing."""
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET opened_at = '2020-01-01T00:00:00+00:00' WHERE run_id = ?",
                 (run_id,))
    conn.commit()

    now = datetime.now(timezone.utc)
    restore_mod.reconcile(conn, restore_mod.RestoreReport(), now=now)

    assert conn.execute(
        "SELECT opened_at FROM rounds WHERE run_id = ?", (run_id,)
    ).fetchone()["opened_at"] == now.isoformat()


def test_closed_rounds_are_left_alone(conn, seeded_run, contributor_row):
    """Only the round in flight is rewound. Rewinding history would discard
    every result the checkpoint chain is built on."""
    run_id = seeded_run()
    conn.execute("UPDATE runs SET current_round = 5 WHERE id = ?", (run_id,))
    conn.commit()

    report = restore_mod.RestoreReport()
    restore_mod.reconcile(conn, report)

    assert report.rounds_rewound == []
    assert conn.execute(
        "SELECT status FROM rounds WHERE run_id = ? AND idx = 0", (run_id,)
    ).fetchone()["status"] == "open"


# --------------------------------------------------------------------------
# Which objects have to come back
# --------------------------------------------------------------------------


def test_the_current_rounds_base_adapter_is_what_must_be_restored(conn, seeded_run):
    """Not the newest *result*: the base of the open round is the object the
    next worker is handed, and the one whose absence stops the run dead. For
    any round after the first they are the same object, which is why backing up
    the latest result is sufficient."""
    run_id = seeded_run()
    refs = restore_mod.referenced_adapters(conn)
    assert base_adapter_key(run_id, 0) in refs


def test_an_object_missing_from_both_stores_is_reported_not_swallowed(
    conn, store, backup_store, seeded_run, settings
):
    run_id = seeded_run()
    store.objects.clear()
    report = restore_mod.RestoreReport()
    restore_mod._plan_adapters(conn, backup_store, store, report, copy=True)

    assert report.problems
    assert base_adapter_key(run_id, 0) in report.problems[0]
    assert not report.ok


def test_an_object_already_in_the_primary_store_is_not_recopied(
    conn, store, backup_store, seeded_run
):
    """A restore onto a machine whose object store survived is a database-only
    restore, and should not re-upload gigabytes to prove it."""
    seeded_run()
    report = restore_mod.RestoreReport()
    restore_mod._plan_adapters(conn, backup_store, store, report, copy=True)

    assert report.adapters_copied == []
    assert report.adapters_already_present


# --------------------------------------------------------------------------
# Verification: the claim the drill exists to make
# --------------------------------------------------------------------------


def test_verify_passes_on_a_resumable_run(conn, store, seeded_run):
    seeded_run()
    report = restore_mod.RestoreReport()
    restore_mod._verify(conn, store, report)
    assert report.ok, report.problems


def test_verify_fails_when_the_base_adapter_is_gone(conn, store, seeded_run):
    """Reported rather than assumed: a restore that leaves a run unresumable is
    worse than one that fails loudly, because the operator would believe they
    were covered and find out at the next disaster."""
    seeded_run()
    store.objects.clear()
    report = restore_mod.RestoreReport()
    restore_mod._verify(conn, store, report)

    assert not report.ok
    assert "not in the primary store" in report.problems[0]


def test_verify_fails_when_the_current_round_cannot_be_claimed(conn, store, seeded_run):
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET status = 'closing' WHERE run_id = ?", (run_id,))
    conn.commit()
    report = restore_mod.RestoreReport()
    restore_mod._verify(conn, store, report)

    assert not report.ok
    assert "no worker can claim" in report.problems[0]


# --------------------------------------------------------------------------
# Picking a backup
# --------------------------------------------------------------------------


def test_the_newest_snapshot_wins(backup_store):
    """`backup.py` writes a UTC timestamp that sorts lexicographically in time
    order, which is why it is written that way rather than as anything
    friendlier."""
    for stamp in ("20260101T000000Z", "20260827T011617Z", "20260501T120000Z"):
        backup_store.put_bytes(f"backups/{stamp}/coordinator.db", b"x")

    assert restore_mod.latest_backup_key(backup_store) == \
        "backups/20260827T011617Z/coordinator.db"


def test_an_empty_backup_store_is_reported_rather_than_crashing(backup_store, store, tmp_path):
    report = restore_mod.restore(
        backup_store=backup_store, primary_store=store,
        db_path=str(tmp_path / "restored.db"),
    )
    assert not report.ok
    assert "no backups/" in report.problems[0]
    assert not (tmp_path / "restored.db").exists()
