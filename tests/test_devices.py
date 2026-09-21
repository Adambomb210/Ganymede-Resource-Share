"""The per-device allocation ledger (docs/14 §4, §6).

This module (``coordinator/devices.py``) is not wired into the claim path yet
(docs/14's step 3 of 6) -- these tests exercise it directly, the way
``test_constraints.py`` exercises ``check_constraints`` without going through
``app.py``. The one exception is the concurrency test, which deliberately
opens a second real connection to the same on-disk database and races two
threads through the actual partial unique index -- the invariant this whole
module exists to protect is a database-level one, and a mock would not prove
anything about it.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import timedelta

import pytest

from ganymede.coordinator import devices, rounds
from ganymede.coordinator.db import connect, immediate

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _iso_ago(**kw) -> str:
    return rounds._iso(rounds.utcnow() - timedelta(**kw))


def _iso_from_now(**kw) -> str:
    return rounds._iso(rounds.utcnow() + timedelta(**kw))


@pytest.fixture
def worker(conn, make_contributor):
    """A real ``workers`` row. ``worker_devices``/``task_devices`` FK into it."""
    def _make(name: str = "w"):
        cid, _ = make_contributor(name=f"c-{name}-{uuid.uuid4().hex[:6]}")
        wid = uuid.uuid4().hex
        now = rounds._iso(rounds.utcnow())
        conn.execute(
            """INSERT INTO workers (id, contributor_id, compute_profile_json,
                                    first_seen, last_seen)
               VALUES (?, ?, '{"gpu_model": "CPU", "vram_mb": 1024}', ?, ?)""",
            (wid, cid, now, now),
        )
        conn.commit()
        return wid
    return _make


@pytest.fixture
def job(conn, make_contributor):
    """A real ``jobs`` row. ``device_reservations.job_id`` FKs into it."""
    def _make(name: str = "j", rank: int = 10):
        oid, _ = make_contributor(name=f"o-{name}-{uuid.uuid4().hex[:6]}")
        jid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO jobs
                 (id, owner_id, job_type, spec_json, status, priority_rank,
                  constraints_json, created_at)
               VALUES (?, ?, 'batch_inference', '{}', 'queued', ?, '{}', ?)""",
            (jid, oid, rank, rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return jid
    return _make


@pytest.fixture
def device(conn):
    """A ``worker_devices`` row."""
    def _make(worker_id: str, index: int, *, vram_mb: int = 8192,
             retired_at: str | None = None):
        conn.execute(
            """INSERT INTO worker_devices
                 (worker_id, device_index, device_name, vram_mb, retired_at)
               VALUES (?, ?, 'RTX 3060', ?, ?)""",
            (worker_id, index, vram_mb, retired_at),
        )
        conn.commit()
    return _make


@pytest.fixture
def task(conn):
    """A real ``tasks`` row -- ``task_devices.task_id`` FKs into it."""
    def _make(worker_id: str | None = None, status: str = "leased",
             task_id: str | None = None) -> str:
        tid = task_id or uuid.uuid4().hex
        conn.execute(
            """INSERT INTO tasks
                 (id, buckets_json, local_steps, status, worker_id, attempts,
                  created_at)
               VALUES (?, '[]', 1, ?, ?, 1, ?)""",
            (tid, status, worker_id, rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return tid
    return _make


# ==========================================================================
# inventory
# ==========================================================================


def test_inventory_excludes_retired_devices(conn, worker, device):
    wid = worker()
    device(wid, 0)
    device(wid, 1, retired_at=rounds._iso(rounds.utcnow()))
    rows = devices.inventory(conn, wid)
    assert [r["device_index"] for r in rows] == [0]


def test_inventory_is_index_ordered(conn, worker, device):
    wid = worker()
    for idx in (2, 0, 1):
        device(wid, idx)
    rows = devices.inventory(conn, wid)
    assert [r["device_index"] for r in rows] == [0, 1, 2]


# ==========================================================================
# max_inventory_width (docs/14 §9 -- gpu_count at submission)
# ==========================================================================


def test_max_inventory_width_is_zero_on_a_fresh_fleet(conn):
    """No worker has ever reconciled an inventory -- the state of a fresh
    coordinator, and the case app.create_job's own check must read as
    "unknown," never as "reject every job." """
    assert devices.max_inventory_width(conn) == 0


def test_max_inventory_width_is_the_widest_live_inventory(conn, worker, device):
    a = worker("a")
    device(a, 0)
    b = worker("b")
    device(b, 0)
    device(b, 1)
    device(b, 2)
    assert devices.max_inventory_width(conn) == 3


def test_max_inventory_width_excludes_retired_devices(conn, worker, device):
    """A box that lost a card must not go on advertising the old width --
    the same ``retired_at IS NULL`` filter ``inventory`` itself applies."""
    a = worker("a")
    device(a, 0)
    device(a, 1, retired_at=rounds._iso(rounds.utcnow()))
    device(a, 2, retired_at=rounds._iso(rounds.utcnow()))
    assert devices.max_inventory_width(conn) == 1


# ==========================================================================
# free_devices (docs/14 §4)
# ==========================================================================


def test_free_devices_excludes_a_live_lease(conn, worker, device, job, task):
    wid = worker()
    device(wid, 0)
    device(wid, 1)
    j = job()
    t = task(worker_id=wid)
    conn.execute(
        "INSERT INTO task_devices (task_id, worker_id, device_index, allocated_at) "
        "VALUES (?, ?, 0, ?)", (t, wid, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    assert devices.free_devices(conn, wid, j) == [1]


def test_free_devices_is_not_reduced_by_a_released_lease(conn, worker, device, job, task):
    """A released row drops out of the busy set -- the partial unique index's
    whole point, and the thing docs/14 §4 calls out by name as the most
    likely way to dark-card a machine if a query forgets it."""
    wid = worker()
    device(wid, 0)
    j = job()
    t = task(worker_id=wid)
    conn.execute(
        "INSERT INTO task_devices (task_id, worker_id, device_index, "
        "allocated_at, released_at, release_reason) VALUES (?, ?, 0, ?, ?, 'submitted')",
        (t, wid, _iso_ago(hours=1), _iso_ago(minutes=50)),
    )
    conn.commit()
    assert devices.free_devices(conn, wid, j) == [0]


def test_free_devices_excludes_a_reservation_held_by_another_job(conn, worker, device, job):
    wid = worker()
    device(wid, 0)
    device(wid, 1)
    mine = job(name="mine")
    other = job(name="other")
    conn.execute(
        "INSERT INTO device_reservations (worker_id, device_index, job_id, "
        "reserved_at, expires_at) VALUES (?, 1, ?, ?, ?)",
        (wid, other, rounds._iso(rounds.utcnow()), _iso_from_now(minutes=5)),
    )
    conn.commit()
    assert devices.free_devices(conn, wid, mine) == [0]


def test_free_devices_sees_through_its_own_reservation(conn, worker, device, job):
    """The requesting job's own reservation must not subtract from its own
    free set (docs/14 §4, §6) -- otherwise a job that successfully reserved a
    card could never actually claim it."""
    wid = worker()
    device(wid, 0)
    j = job()
    conn.execute(
        "INSERT INTO device_reservations (worker_id, device_index, job_id, "
        "reserved_at, expires_at) VALUES (?, 0, ?, ?, ?)",
        (wid, j, rounds._iso(rounds.utcnow()), _iso_from_now(minutes=5)),
    )
    conn.commit()
    assert devices.free_devices(conn, wid, j) == [0]


def test_free_devices_excludes_a_retired_device(conn, worker, device, job):
    wid = worker()
    device(wid, 0)
    device(wid, 1, retired_at=rounds._iso(rounds.utcnow()))
    j = job()
    assert devices.free_devices(conn, wid, j) == [0]


def test_free_devices_ignores_an_expired_unswept_reservation(conn, worker, device, job):
    """A dead job's reservation past its own TTL must not make this job wait
    behind it, even if ``expire_reservations`` has not run yet -- this is the
    deviation from docs/14 §4's literal wording; see the step 3 report."""
    wid = worker()
    device(wid, 0)
    mine = job(name="mine")
    dead = job(name="dead")
    conn.execute(
        "INSERT INTO device_reservations (worker_id, device_index, job_id, "
        "reserved_at, expires_at) VALUES (?, 0, ?, ?, ?)",
        (wid, dead, _iso_ago(minutes=10), _iso_ago(minutes=5)),
    )
    conn.commit()
    assert devices.free_devices(conn, wid, mine) == [0]


# ==========================================================================
# allocate (docs/14 §4-§5)
# ==========================================================================


def test_allocate_returns_the_requested_indices(conn, worker, device, task):
    wid = worker()
    device(wid, 0)
    device(wid, 1)
    t = task(worker_id=wid)
    with immediate(conn):
        result = devices.allocate(conn, wid, t, 2, [0, 1])
    assert result == [0, 1]
    live = conn.execute(
        "SELECT device_index FROM task_devices WHERE task_id = ? AND "
        "released_at IS NULL ORDER BY device_index", (t,),
    ).fetchall()
    assert [r["device_index"] for r in live] == [0, 1]


def test_allocate_returns_none_when_free_set_is_too_small(conn, worker, device, task):
    wid = worker()
    device(wid, 0)
    t = task(worker_id=wid)
    with immediate(conn):
        assert devices.allocate(conn, wid, t, 2, [0]) is None
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_devices WHERE task_id = ?", (t,)
    ).fetchone()["n"] == 0


def test_allocate_raises_for_nonpositive_count(conn, worker, device, task):
    wid = worker()
    device(wid, 0)
    t = task(worker_id=wid)
    with immediate(conn):
        with pytest.raises(ValueError):
            devices.allocate(conn, wid, t, 0, [0])


def test_allocate_returns_none_rather_than_raising_when_device_already_taken(
    conn, worker, device, task
):
    """The device is in ``free`` (stale free-set computation) but another task
    already holds it live -- the partial unique index rejects the insert, and
    that must come back as ``None``, not an exception."""
    wid = worker()
    device(wid, 0)
    holder = task(worker_id=wid)
    with immediate(conn):
        assert devices.allocate(conn, wid, holder, 1, [0]) == [0]

    contender = task(worker_id=wid)
    with immediate(conn):
        result = devices.allocate(conn, wid, contender, 1, [0])
    assert result is None
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_devices WHERE task_id = ?", (contender,)
    ).fetchone()["n"] == 0


def test_allocate_partial_failure_leaves_no_orphan_rows(conn, worker, device, task):
    """count=2, device 0 free and device 1 already taken: the insert for
    device 0 must succeed and then be undone when device 1's insert collides
    -- a caller that sees ``None`` must find nothing left behind, or the
    device-0 row would be a live allocation for a task that was never leased
    and nothing would ever release it."""
    wid = worker()
    device(wid, 0)
    device(wid, 1)
    holder = task(worker_id=wid)
    with immediate(conn):
        assert devices.allocate(conn, wid, holder, 1, [1]) == [1]

    contender = task(worker_id=wid)
    with immediate(conn):
        result = devices.allocate(conn, wid, contender, 2, [0, 1])
    assert result is None
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_devices WHERE task_id = ?", (contender,)
    ).fetchone()["n"] == 0
    # device 0 was never touched by the failed call either.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM task_devices WHERE worker_id = ? AND device_index = 0",
        (wid,),
    ).fetchone()["n"] == 0


def test_concurrent_allocations_for_the_same_device_exactly_one_wins(
    settings, conn, worker, device, job, task
):
    """Two threads, two real connections to the same on-disk database, racing
    through the actual ``idx_task_devices_busy`` partial unique index -- not a
    mock. Both read the same stale free set before either enters its write
    transaction, so the loser's ``None`` genuinely comes from the index
    rejecting its insert, not from a free-set recomputation that already saw
    the winner's row."""
    wid = worker()
    device(wid, 0)
    j = job()
    t_a = task(worker_id=wid)
    t_b = task(worker_id=wid)
    conn.commit()

    barrier = threading.Barrier(2)
    outcomes: dict[str, object] = {}

    def attempt(name: str, task_id: str) -> None:
        c = connect(settings.db_path)
        try:
            free = devices.free_devices(c, wid, j)
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
            with immediate(c):
                outcomes[name] = devices.allocate(c, wid, task_id, 1, free)
        finally:
            c.close()

    t1 = threading.Thread(target=attempt, args=("a", t_a))
    t2 = threading.Thread(target=attempt, args=("b", t_b))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    results = list(outcomes.values())
    assert len(results) == 2
    assert sum(r is None for r in results) == 1, outcomes
    assert sum(r == [0] for r in results) == 1, outcomes

    rows = conn.execute(
        "SELECT task_id, released_at FROM task_devices WHERE worker_id = ?", (wid,)
    ).fetchall()
    assert len(rows) == 1, "the loser must leave no orphan row"
    assert rows[0]["released_at"] is None


# ==========================================================================
# release
# ==========================================================================


def test_release_stamps_released_at_and_reason(conn, worker, device, task):
    wid = worker()
    device(wid, 0)
    t = task(worker_id=wid)
    with immediate(conn):
        devices.allocate(conn, wid, t, 1, [0])

    n = devices.release(conn, t, "submitted")
    assert n == 1
    row = conn.execute(
        "SELECT released_at, release_reason FROM task_devices WHERE task_id = ?", (t,)
    ).fetchone()
    assert row["released_at"] is not None
    assert row["release_reason"] == "submitted"


def test_release_is_idempotent(conn, worker, device, task):
    wid = worker()
    device(wid, 0)
    t = task(worker_id=wid)
    with immediate(conn):
        devices.allocate(conn, wid, t, 1, [0])

    first = devices.release(conn, t, "submitted")
    stamped_at = conn.execute(
        "SELECT released_at FROM task_devices WHERE task_id = ?", (t,)
    ).fetchone()["released_at"]

    second = devices.release(conn, t, "abandoned")
    row = conn.execute(
        "SELECT released_at, release_reason FROM task_devices WHERE task_id = ?", (t,)
    ).fetchone()

    assert first == 1
    assert second == 0
    # Untouched by the second call: same stamp, original reason.
    assert row["released_at"] == stamped_at
    assert row["release_reason"] == "submitted"


def test_release_on_a_recycled_task_stamps_only_the_live_row(conn, worker, device, task):
    """docs/14 §3: task ids are recycled by ``_claim_static_task``, so one
    task id can own more than one ``task_devices`` row over its lifetime --
    an earlier released one and a current live one. ``release`` must stamp
    only the live row and must not touch the earlier, already-released one."""
    wid = worker()
    device(wid, 0)
    t = task(worker_id=wid, task_id="recycled")

    # First lease of this task on device 0, already released.
    conn.execute(
        "INSERT INTO task_devices (task_id, worker_id, device_index, "
        "allocated_at, released_at, release_reason) "
        "VALUES (?, ?, 0, ?, ?, 'expired')",
        (t, wid, _iso_ago(hours=2), _iso_ago(hours=1)),
    )
    conn.commit()
    # Re-claimed onto the same device -- the live occupant.
    with immediate(conn):
        assert devices.allocate(conn, wid, t, 1, [0]) == [0]

    n = devices.release(conn, t, "submitted")
    assert n == 1

    rows = conn.execute(
        "SELECT allocated_at, released_at, release_reason FROM task_devices "
        "WHERE task_id = ? ORDER BY allocated_at", (t,),
    ).fetchall()
    assert len(rows) == 2
    # The earlier, already-released row is untouched.
    assert rows[0]["release_reason"] == "expired"
    # Only the second (live) row picked up this call's reason.
    assert rows[1]["release_reason"] == "submitted"
    assert rows[1]["released_at"] is not None


# ==========================================================================
# device_history
# ==========================================================================


def test_device_history_filters_by_since_and_orders_oldest_first(conn, worker, device, task):
    wid = worker()
    device(wid, 0)
    t1 = task(worker_id=wid)
    t2 = task(worker_id=wid)
    # t1's first (and only) lease on device 0, long since released.
    conn.execute(
        "INSERT INTO task_devices (task_id, worker_id, device_index, allocated_at, "
        "released_at) VALUES (?, ?, 0, ?, ?)",
        (t1, wid, _iso_ago(days=2), _iso_ago(days=1, hours=23)),
    )
    # t2's current, live lease on the same device -- recent.
    conn.execute(
        "INSERT INTO task_devices (task_id, worker_id, device_index, allocated_at) "
        "VALUES (?, ?, 0, ?)", (t2, wid, _iso_ago(hours=1)),
    )
    conn.commit()

    rows = devices.device_history(conn, wid, rounds.utcnow() - timedelta(days=1))
    assert len(rows) == 1
    assert rows[0]["task_id"] == t2

    all_rows = devices.device_history(conn, wid, rounds.utcnow() - timedelta(days=3))
    assert len(all_rows) == 2
    assert [r["task_id"] for r in all_rows] == [t1, t2]


# ==========================================================================
# reserve / expire_reservations (docs/14 §6)
# ==========================================================================


def test_reserve_grants_and_expire_reservations_removes_it(conn, worker, job):
    wid = worker()
    j = job()
    assert devices.reserve(conn, wid, j, [0, 1], ttl=0.01) is True
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM device_reservations WHERE worker_id = ?", (wid,)
    ).fetchone()["n"] == 2

    import time
    time.sleep(0.05)
    removed = devices.expire_reservations(conn)
    assert removed == 2
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM device_reservations WHERE worker_id = ?", (wid,)
    ).fetchone()["n"] == 0


def test_reserve_refuses_a_second_job_on_the_same_worker(conn, worker, job):
    wid = worker()
    first = job(name="first")
    second = job(name="second")
    assert devices.reserve(conn, wid, first, [0], ttl=300) is True
    assert devices.reserve(conn, wid, second, [1], ttl=300) is False
    # The refused job's device must not appear -- nothing was inserted for it.
    rows = conn.execute(
        "SELECT job_id, device_index FROM device_reservations WHERE worker_id = ?",
        (wid,),
    ).fetchall()
    assert [(r["job_id"], r["device_index"]) for r in rows] == [(first, 0)]


def test_reserve_lets_the_same_job_widen_its_reservation(conn, worker, job):
    wid = worker()
    j = job()
    assert devices.reserve(conn, wid, j, [0], ttl=300) is True
    assert devices.reserve(conn, wid, j, [0, 1], ttl=300) is True
    rows = conn.execute(
        "SELECT device_index FROM device_reservations WHERE worker_id = ? "
        "ORDER BY device_index", (wid,),
    ).fetchall()
    assert [r["device_index"] for r in rows] == [0, 1]


def test_reserving_job_still_sees_its_own_reserved_devices_as_free(conn, worker, device, job):
    """The reservation mechanism only makes sense if the job that holds it can
    still be handed the device (docs/14 §4, §6) -- an end-to-end check that
    ``reserve`` and ``free_devices`` agree with each other, not just each
    against the raw table."""
    wid = worker()
    device(wid, 0)
    j = job()
    assert devices.reserve(conn, wid, j, [0], ttl=300) is True
    assert devices.free_devices(conn, wid, j) == [0]


def test_reserve_purges_a_dead_jobs_expired_reservation_before_the_holder_check(
    conn, worker, job
):
    """A stale, already-expired reservation from a job that never came back
    must not permanently lock a worker -- ``reserve`` must not depend on
    ``expire_reservations`` having run first."""
    wid = worker()
    dead = job(name="dead")
    conn.execute(
        "INSERT INTO device_reservations (worker_id, device_index, job_id, "
        "reserved_at, expires_at) VALUES (?, 0, ?, ?, ?)",
        (wid, dead, _iso_ago(minutes=10), _iso_ago(minutes=5)),
    )
    conn.commit()

    fresh = job(name="fresh")
    assert devices.reserve(conn, wid, fresh, [0], ttl=300) is True
    row = conn.execute(
        "SELECT job_id FROM device_reservations WHERE worker_id = ? AND device_index = 0",
        (wid,),
    ).fetchone()
    assert row["job_id"] == fresh
