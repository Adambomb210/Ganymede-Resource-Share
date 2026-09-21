"""The device ledger wired into the coordinator's claim path (docs/14 §5).

``test_devices.py`` covers the free-set arithmetic in isolation (docs/14's
step 3); this module covers the *wiring* step 4 adds on top of it: the
multi-lease reconcile, the capacity gate, allocation inside the lease
transaction, ``devices: [int]`` on the payload, release on every terminal
path, and the register-time reconcile. Every scenario here goes through the
real HTTP surface (``client`` / ``FakeWorker``) wherever the thing under test
is observable from there, the same discipline ``test_scheduler.py`` and
``test_cancel.py`` follow -- a wiring bug that only shows up through the
walk should be caught by a test that goes through the walk.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from ganymede.coordinator import devices, eligibility, rounds
from ganymede.coordinator.db import immediate
from tests.fake_worker import FakeWorker


def _job_id(conn, run_id: str) -> str:
    return conn.execute(
        "SELECT job_id FROM runs WHERE id = ?", (run_id,)
    ).fetchone()["job_id"]


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _live_row(conn, task_id: str):
    return conn.execute(
        "SELECT * FROM task_devices WHERE task_id = ? AND released_at IS NULL",
        (task_id,),
    ).fetchone()


@pytest.fixture
def admin(conn):
    """An admin contributor -- ``POST /jobs/{id}/cancel`` needs one, the same
    fixture ``test_cancel.py`` uses."""
    from ganymede.coordinator.auth import generate_key, hash_key

    cid, key = uuid.uuid4().hex, generate_key()
    conn.execute(
        """INSERT INTO contributors
             (id, name, key_hash, enabled, clearance, is_admin, created_at)
           VALUES (?, ?, ?, 1, 'open', 1, ?)""",
        (cid, "admin", hash_key(key), rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return cid, key


# ==========================================================================
# A v1 worker's behaviour is reproduced exactly (docs/14 §5.1)
# ==========================================================================


def test_v1_claim_with_no_active_task_ids_resumes_the_held_lease(
    client, store, conn, make_contributor, seeded_run
):
    """A v1 worker never sends ``active_task_ids``; ``ClaimRequest`` defaults
    it to ``[]``. Every held lease then looks unknown, so a second poll on a
    one-device worker must resume the same task -- never mint a second one --
    exactly what the old global held-lease check did."""
    seeded_run(run_id="r1")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)

    first = fw.claim()
    assert first is not None
    assert first["devices"] == [0]

    second = fw.claim()
    assert second is not None
    assert second["task_id"] == first["task_id"]

    # Still exactly one live allocation for this task -- resuming must not
    # allocate a second time.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_devices WHERE task_id = ? "
        "AND released_at IS NULL", (first["task_id"],),
    ).fetchone()["n"] == 1


# ==========================================================================
# The capacity gate (docs/14 §5.2)
# ==========================================================================


def test_capacity_gate_refuses_and_continues_the_walk_to_every_job(
    client, store, conn, make_contributor, seeded_run
):
    """A one-device worker whose device is already busy must be refused
    ``insufficient_free_devices`` on *every* remaining job in the walk, not
    just the first -- proving the gate `continue`s rather than `break`s
    (docs/14 §5.2's discipline, the same as the constraint gate's)."""
    seeded_run(run_id="hi")   # rank 10
    seeded_run(run_id="lo")   # rank 20
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)

    held = fw.claim()
    assert held is not None and held["run_id"] == "hi"

    # Tell the coordinator the worker already knows about its held lease, so
    # the outer reconcile falls through to the walk instead of re-serving it.
    resp = client.post(
        "/v1/tasks/claim", headers=fw.headers,
        json={"worker_id": fw.worker_id, "active_task_ids": [held["task_id"]]},
    )
    assert resp.status_code == 204

    verdicts = {v.job_id: v for v in eligibility.explain(conn, fw.worker_id).verdicts}
    assert verdicts[_job_id(conn, "hi")].outcome == eligibility.REFUSED
    assert verdicts[_job_id(conn, "hi")].reason == "insufficient_free_devices"
    assert verdicts[_job_id(conn, "lo")].outcome == eligibility.REFUSED
    assert verdicts[_job_id(conn, "lo")].reason == "insufficient_free_devices"


# ==========================================================================
# Release on every terminal path (docs/14 §5.5) -- zero unreleased rows and
# an immediately re-claimable card, for each of the five.
# ==========================================================================


def test_submit_releases_the_device(client, store, conn, make_contributor, seeded_run):
    seeded_run(run_id="r1")
    seeded_run(run_id="r2")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    assert task is not None

    result = fw.submit(task["local_steps"])
    assert result.status_code == 200

    row = conn.execute(
        "SELECT released_at, release_reason FROM task_devices WHERE task_id = ?",
        (task["task_id"],),
    ).fetchone()
    assert row["released_at"] is not None
    assert row["release_reason"] == "submitted"
    assert _live_row(conn, task["task_id"]) is None

    # Immediately re-claimable: the freed device backs a fresh lease.
    again = fw.claim()
    assert again is not None


def test_abandon_releases_the_device(client, store, conn, make_contributor, seeded_run):
    seeded_run(run_id="r1")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    assert task is not None

    result = fw.abandon()
    assert result.status_code == 200

    row = conn.execute(
        "SELECT released_at, release_reason FROM task_devices WHERE task_id = ?",
        (task["task_id"],),
    ).fetchone()
    assert row["released_at"] is not None
    assert row["release_reason"] == "abandoned"
    assert _live_row(conn, task["task_id"]) is None

    again = fw.claim()
    assert again is not None


def test_expired_lease_releases_the_device(client, store, conn, make_contributor, seeded_run):
    """``rounds.expire_leases``' first outcome -- a lease nobody heartbeated
    past its expiry (docs/14 §5.5)."""
    seeded_run(run_id="r1")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    assert task is not None

    conn.execute(
        "UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
        (rounds._iso(rounds.utcnow() - timedelta(hours=1)),
         task["task_id"]),
    )
    conn.commit()

    n = rounds.expire_leases(conn)
    assert n == 1
    row = conn.execute(
        "SELECT status, released_at, release_reason FROM task_devices "
        "JOIN tasks ON tasks.id = task_devices.task_id "
        "WHERE task_devices.task_id = ?", (task["task_id"],),
    ).fetchone()
    assert row["status"] == "expired"
    assert row["released_at"] is not None
    assert row["release_reason"] == "expired"
    assert _live_row(conn, task["task_id"]) is None

    again = fw.claim()
    assert again is not None


def test_a_cancelled_jobs_expired_lease_releases_the_device(
    client, store, conn, make_contributor, seeded_run, admin
):
    """``rounds.expire_leases``' second outcome -- a lease on a job an
    operator cancelled, reclaimed once its lease lapses (docs/11 §3, docs/14
    §5.5)."""
    seeded_run(run_id="r1")
    seeded_run(run_id="r2")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    assert task is not None

    _, admin_key = admin
    r = client.post(f"/v1/jobs/{_job_id(conn, 'r1')}/cancel", headers=_hdr(admin_key),
                    json={"mode": "hard"})
    assert r.status_code == 200, r.text

    conn.execute(
        "UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
        (rounds._iso(rounds.utcnow() - timedelta(hours=1)),
         task["task_id"]),
    )
    conn.commit()

    n = rounds.expire_leases(conn)
    assert n == 1
    row = conn.execute(
        "SELECT release_reason FROM task_devices WHERE task_id = ? "
        "AND released_at IS NOT NULL", (task["task_id"],),
    ).fetchone()
    assert row["release_reason"] == "cancelled"
    assert _live_row(conn, task["task_id"]) is None

    again = fw.claim()
    assert again is not None


def test_a_preempted_lease_releases_the_device(client, store, conn, make_contributor, seeded_run):
    """``rounds.expire_leases``' third outcome -- a hard preempt nobody
    acknowledged before its lease lapsed (docs/13 §4.3, docs/14 §5.5)."""
    seeded_run(run_id="r1")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()
    assert task is not None

    conn.execute(
        "UPDATE tasks SET preempt_mode = 'hard', lease_expires_at = ? WHERE id = ?",
        (rounds._iso(rounds.utcnow() - timedelta(hours=1)),
         task["task_id"]),
    )
    conn.commit()

    n = rounds.expire_leases(conn)
    assert n == 1
    row = conn.execute(
        "SELECT release_reason FROM task_devices WHERE task_id = ? "
        "AND released_at IS NOT NULL", (task["task_id"],),
    ).fetchone()
    assert row["release_reason"] == "preempted"
    assert _live_row(conn, task["task_id"]) is None

    again = fw.claim()
    assert again is not None


def test_a_straggler_left_behind_by_an_early_round_close_releases_its_device(
    client, store, conn, make_contributor, seeded_run
):
    """A round can close as soon as one worker's submission reaches
    ``target_steps`` (``plan.should_close``), while a second worker is still
    holding a lease on the same round. ``reduce.py`` force-expires that
    straggler the instant the round closes -- a fourth terminal path outside
    ``rounds.expire_leases``'s three, found while wiring this step (see the
    step 4 report), and it needs the same device release or that straggler's
    card stays dark until the next lease TTL happens to sweep it.
    ``release_reason`` is ``round_closed`` rather than ``expired`` (unlike
    the task's own ``status`` column, unchanged) -- distinct on purpose, so
    this assertion can only pass if ``reduce.py``'s own release ran; B's
    lease is nowhere near its TTL and ``expire_leases`` is never called
    anywhere in this test, so an ``expired`` reason here would not have
    proven anything."""
    seeded_run(run_id="r1")  # target_steps=100, min_round_sec=0 (conftest defaults)
    _, key_a = make_contributor(name="a")
    _, key_b = make_contributor(name="b")
    a = FakeWorker(client, store, key_a)
    b = FakeWorker(client, store, key_b)

    task_a = a.claim()
    task_b = b.claim()
    assert task_a is not None and task_b is not None

    result = a.submit(100)  # >= target_steps -- closes the round on this submit
    assert result.status_code == 200
    assert result["round_closed"] is True

    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_b["task_id"],)
    ).fetchone()
    assert row["status"] == "expired"
    b_row = conn.execute(
        "SELECT released_at, release_reason FROM task_devices WHERE task_id = ?",
        (task_b["task_id"],),
    ).fetchone()
    assert b_row["released_at"] is not None
    assert b_row["release_reason"] == "round_closed"
    assert _live_row(conn, task_b["task_id"]) is None

    # b's device is immediately free -- whatever the next round or job needs
    # it can have it right away, without waiting for a lease TTL to lapse.
    assert devices.free_devices(conn, b.worker_id, _job_id(conn, "r1")) == [0]


# ==========================================================================
# Register-time reconcile (docs/14 §5.6)
# ==========================================================================


def test_register_time_reconcile_frees_a_vanished_devices_allocation(
    client, store, conn, make_contributor
):
    """A worker that comes back reporting fewer devices than the ledger has
    on record has the vanished device retired and its live allocation
    released with reason ``reconciled`` (docs/14 §5.6). Worker-side
    per-device reporting is a later step, so this seeds the "before" state
    (two devices, one busy) directly, the way ``test_devices.py`` seeds
    ``worker_devices`` rows -- the coordinator side of the reconcile is what
    is under test, not a real multi-GPU worker."""
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    worker_id = fw.register()

    # A second device this worker apparently had, with a live allocation on
    # it -- the "before" picture a future multi-device worker's first
    # register call would already have produced.
    conn.execute(
        """INSERT INTO worker_devices
             (worker_id, device_index, device_name, vram_mb, retired_at)
           VALUES (?, 1, 'RTX 3060', 12288, NULL)""",
        (worker_id,),
    )
    task_id = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO tasks (id, buckets_json, local_steps, status, worker_id,
                              attempts, created_at)
           VALUES (?, '[]', 1, 'leased', ?, 1, ?)""",
        (task_id, worker_id, rounds._iso(rounds.utcnow())),
    )
    with immediate(conn):
        assert devices.allocate(conn, worker_id, task_id, 1, [1]) == [1]
    conn.commit()

    assert len(devices.inventory(conn, worker_id)) == 2

    # Re-register with the same (unchanged) flat profile -- no ``devices``
    # list, so the reconcile synthesizes device 0 only, exactly as it did on
    # first register. Device 1 has vanished from this report.
    again = fw.register()
    assert again == worker_id

    live = devices.inventory(conn, worker_id)
    assert [r["device_index"] for r in live] == [0]

    row = conn.execute(
        "SELECT released_at, release_reason FROM task_devices WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    assert row["released_at"] is not None
    assert row["release_reason"] == "reconciled"
