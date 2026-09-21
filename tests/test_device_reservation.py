"""Reservation with backfill (docs/14 §6), wired into the claim walk -- step 5.

``test_devices.py`` already proves ``reserve`` / ``expire_reservations`` in
isolation (step 3); this module covers what step 5 adds on top: the sweep
beside ``rounds.expire_leases``, the walk-order policy for *which* refused job
may call ``reserve``, the backfill peek that lets a smaller job borrow a
reserved device it can return in time, and the inertness guarantee that none
of this stirs while every job in the fleet is ``gpu_count == 1``. Every
scenario goes through the real HTTP surface, the same discipline
``test_device_wiring.py`` follows -- a wiring bug that only shows up through
the walk should be caught by a test that goes through the walk.

Job type: ``batch_inference`` (static, no ``shape_claim``, no image
requirement) -- the simplest static type with a real ``max_runtime_sec``
column to backfill against. ``gpu_count`` and ``max_runtime_sec`` are not yet
settable through the job-creation API (steps 6-9), so tests reach past the
API and set them directly on the ``jobs`` / ``tasks`` rows after a normal
``POST /v1/jobs`` + ``/enqueue`` has planned them -- the same liberty
``test_device_wiring.py``'s register-reconcile test takes with ``tasks`` /
``worker_devices`` rows.
"""

from __future__ import annotations

import uuid

import pytest

from ganymede.coordinator import devices, eligibility, rounds
from ganymede.coordinator.db import immediate


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _spec(shards, *, prefix: str = "out/j") -> dict:
    return {
        "model_ref": "hf://test-model",
        "shards": shards,
        "output_prefix": prefix,
        "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
    }


def _enqueue(client, skey: str, spec: dict) -> str:
    jid = client.post("/v1/jobs", headers=_hdr(skey),
                      json={"job_type": "batch_inference", "spec": spec}).json()["job_id"]
    r = client.post(f"/v1/jobs/{jid}/enqueue", headers=_hdr(skey))
    assert r.status_code == 200, r.text
    return jid


def _claim(client, wkey: str, worker_id: str):
    return client.post("/v1/tasks/claim", headers=_hdr(wkey),
                       json={"worker_id": worker_id})


@pytest.fixture
def make_submitter(conn, make_contributor):
    def _make(name: str = "submitter"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, "approved", rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


def _register(client, wkey: str, *, n_devices: int = 1) -> str:
    """Register a worker and pad it out to ``n_devices`` (docs/14 §2's flat
    fields synthesize exactly one, device 0 -- worker-side multi-device
    reporting is a later step, so extra devices are seeded directly, the way
    ``test_device_wiring.py``'s register-reconcile test does)."""
    resp = client.post(
        "/v1/workers/register", headers=_hdr(wkey),
        json={"compute_profile": {
            "backend": "cuda", "device_name": "RTX 3060", "vram_mb": 12288,
            "supports": ["bf16", "fp16", "nf4"],
            "probe": {"alloc_max_mb": 11288, "bench_score": 40.0},
        }},
    )
    assert resp.status_code == 200, resp.text
    worker_id = resp.json()["worker_id"]
    return worker_id


def _add_devices(conn, worker_id: str, indices) -> None:
    for idx in indices:
        conn.execute(
            """INSERT INTO worker_devices
                 (worker_id, device_index, device_name, vram_mb, retired_at)
               VALUES (?, ?, 'RTX 3060', 12288, NULL)""",
            (worker_id, idx),
        )
    conn.commit()


def _occupy(conn, worker_id: str, idx: int) -> str:
    """A dummy leased task holding ``idx`` -- a stand-in for "some other job
    is already running here" without needing a second real job and a second
    real claim to establish it."""
    task_id = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO tasks (id, buckets_json, local_steps, status, worker_id,
                              attempts, created_at)
           VALUES (?, '[]', 1, 'leased', ?, 1, ?)""",
        (task_id, worker_id, rounds._iso(rounds.utcnow())),
    )
    with immediate(conn):
        allocated = devices.allocate(conn, worker_id, task_id, 1, [idx])
    assert allocated == [idx]
    conn.commit()
    return task_id


def _release(conn, task_id: str) -> None:
    devices.release(conn, task_id, "test")
    conn.commit()


def _reservation_devices(conn, job_id: str) -> list[int]:
    rows = conn.execute(
        "SELECT device_index FROM device_reservations WHERE job_id = ? "
        "ORDER BY device_index",
        (job_id,),
    ).fetchall()
    return [r["device_index"] for r in rows]


# ==========================================================================
# A wide job accumulates a reservation as devices free (docs/14 §6)
# ==========================================================================


def test_a_four_card_job_accumulates_devices_as_they_free(
    client, store, conn, make_contributor, make_submitter
):
    """A 4-card job on a fully busy 4-card box builds its reservation one
    device at a time as each occupant releases, and only actually claims once
    all four are simultaneously free -- ``allocate`` is all-or-nothing, so
    there is no partial lease along the way, only a growing reservation."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1, 2, 3])

    jid = _enqueue(client, skey, _spec([{"ref": "s0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 4 WHERE id = ?", (jid,))
    conn.commit()

    occupants = [_occupy(conn, worker_id, i) for i in range(4)]

    # Fully busy: refused, nothing free to reserve yet.
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204
    assert _reservation_devices(conn, jid) == []

    for i, occ in enumerate(occupants):
        _release(conn, occ)
        resp = _claim(client, wkey, worker_id)
        if i < 3:
            assert resp.status_code == 204, resp.text
            assert _reservation_devices(conn, jid) == list(range(i + 1))
        else:
            assert resp.status_code == 200, resp.text
            assert sorted(resp.json()["devices"]) == [0, 1, 2, 3]


def test_a_reservation_does_not_outlive_the_claim_it_was_accumulating_for(
    client, store, conn, make_contributor, make_submitter
):
    """docs/14 §6: accumulation ends the moment the job fits.

    Left behind, the reservation rows would keep excluding every *other* job
    for the rest of the TTL after the winning task released its cards -- while
    the reserver, which sees through its own reservation, went unaffected. The
    devices the job won are recorded as ``task_devices`` rows, and those
    exclude everyone on their own; the reservation has nothing left to do.
    """
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1, 2, 3])

    jid = _enqueue(client, skey, _spec([{"ref": "s0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 4 WHERE id = ?", (jid,))
    conn.commit()

    occupants = [_occupy(conn, worker_id, i) for i in range(4)]

    # Accumulate a real reservation across three of the four cards.
    for occ in occupants[:3]:
        _release(conn, occ)
        assert _claim(client, wkey, worker_id).status_code == 204
    assert _reservation_devices(conn, jid) == [0, 1, 2]

    # The fourth frees, the job fits, and the claim succeeds.
    _release(conn, occupants[3])
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 200, resp.text

    assert _reservation_devices(conn, jid) == [], (
        "the reservation must be dropped the moment the job stops being blocked"
    )


# ==========================================================================
# Backfill: a smaller job that returns a reserved device in time may take it
# ==========================================================================


def test_a_short_task_backfills_onto_a_device_reserved_by_a_wider_job(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)  # one device, device 0

    wide_jid = _enqueue(client, skey, _spec([{"ref": "w0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 2 WHERE id = ?", (wide_jid,))
    conn.commit()

    # First poll: the wide job is the only selectable job, needs 2 devices,
    # this box has exactly 1 -- refused every time, but eligible (gpu_count >
    # 1), so it reserves the one device it can see.
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204
    assert _reservation_devices(conn, wide_jid) == [0]

    short_jid = _enqueue(client, skey, _spec([{"ref": "s0", "rows": 1}]))
    conn.execute(
        "UPDATE tasks SET max_runtime_sec = 30 WHERE job_id = ?", (short_jid,)
    )
    conn.commit()

    # Second poll: the wide job is refused and re-reserves again (it is still
    # first in walk order), then the walk continues to the short job. Its
    # plain free set is empty (device 0 is reserved by the wide job), but its
    # one task finishes in 30s -- nowhere near the wide job's still-fresh
    # reservation window -- so the backfill peek widens its free set and it
    # claims device 0 out from under the reservation without disturbing it.
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"] == short_jid
    assert body["devices"] == [0]

    # The reservation itself is untouched -- backfill lends the device, it
    # does not evict the reservation.
    assert _reservation_devices(conn, wide_jid) == [0]


def test_a_long_task_does_not_backfill_onto_a_reserved_device(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)

    wide_jid = _enqueue(client, skey, _spec([{"ref": "w0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 2 WHERE id = ?", (wide_jid,))
    conn.commit()
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204
    assert _reservation_devices(conn, wide_jid) == [0]

    long_jid = _enqueue(client, skey, _spec([{"ref": "s0", "rows": 1}]))
    # Longer than the reservation's TTL (default 300s) -- this task would
    # still be running well after the wide job's own reservation window says
    # it may come to collect the device, so backfill must refuse it.
    conn.execute(
        "UPDATE tasks SET max_runtime_sec = 100000 WHERE job_id = ?", (long_jid,)
    )
    conn.commit()

    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204

    verdicts = {v.job_id: v for v in eligibility.explain(conn, worker_id).verdicts}
    assert verdicts[long_jid].outcome == eligibility.REFUSED
    assert verdicts[long_jid].reason == "insufficient_free_devices"
    # gpu_count == 1: refused by ordinary contention, not eligible to reserve
    # (docs/14 §6) -- it must not have opened a reservation of its own.
    assert _reservation_devices(conn, long_jid) == []


def test_a_task_with_null_max_runtime_sec_does_not_backfill(
    client, store, conn, make_contributor, make_submitter
):
    """``batch_inference`` never sets ``max_runtime_sec`` on plan -- this is
    the default shape, not a constructed one. NULL cannot be shown to finish
    in time, so it must fail closed exactly like a declared-too-long task."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)

    wide_jid = _enqueue(client, skey, _spec([{"ref": "w0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 2 WHERE id = ?", (wide_jid,))
    conn.commit()
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204
    assert _reservation_devices(conn, wide_jid) == [0]

    null_jid = _enqueue(client, skey, _spec([{"ref": "s0", "rows": 1}]))
    assert conn.execute(
        "SELECT max_runtime_sec FROM tasks WHERE job_id = ?", (null_jid,)
    ).fetchone()["max_runtime_sec"] is None

    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204
    verdicts = {v.job_id: v for v in eligibility.explain(conn, worker_id).verdicts}
    assert verdicts[null_jid].outcome == eligibility.REFUSED
    assert verdicts[null_jid].reason == "insufficient_free_devices"


# ==========================================================================
# Walk-order policy: only one job may hold a reservation on a worker
# ==========================================================================


def test_a_second_wide_job_cannot_open_a_competing_reservation(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1, 2])  # 3 devices total
    _occupy(conn, worker_id, 2)  # 2 free (0, 1), 1 busy

    first_jid = _enqueue(client, skey, _spec([{"ref": "a0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 3 WHERE id = ?", (first_jid,))
    second_jid = _enqueue(client, skey, _spec([{"ref": "b0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 3 WHERE id = ?", (second_jid,))
    conn.commit()

    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204

    # Only the first job in walk order (rank order == enqueue order here)
    # ends up holding anything; the second is refused too, but never opens a
    # competing reservation -- neither via the walk's own gate (it never
    # calls ``reserve`` a second time this poll) nor, as a backstop,
    # ``reserve``'s own per-worker holder check.
    assert _reservation_devices(conn, first_jid) == [0, 1]
    assert _reservation_devices(conn, second_jid) == []

    verdicts = {v.job_id: v for v in eligibility.explain(conn, worker_id).verdicts}
    assert verdicts[second_jid].outcome == eligibility.REFUSED
    assert verdicts[second_jid].reason == "insufficient_free_devices"


# ==========================================================================
# TTL: a dead job's reservation stops blocking once it lapses
# ==========================================================================


def test_a_dead_jobs_reservation_stops_blocking_after_its_ttl(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)  # one device

    dead_jid = _enqueue(client, skey, _spec([{"ref": "w0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 2 WHERE id = ?", (dead_jid,))
    conn.commit()

    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204
    assert _reservation_devices(conn, dead_jid) == [0]

    # The reserving job dies -- cancelled, never to be walked again -- but
    # its reservation is not itself released by cancellation (docs/14 §6 does
    # not wire that, and this test is exactly why the TTL exists as the
    # backstop rather than relying on every terminal path to remember).
    r = client.post(f"/v1/jobs/{dead_jid}/cancel", headers=_hdr(skey),
                    json={"mode": "hard"})
    assert r.status_code == 200, r.text

    # Force the TTL to have already lapsed, the same way
    # ``test_expired_lease_releases_the_device`` forces a lease's expiry
    # rather than waiting on a real clock.
    conn.execute(
        "UPDATE device_reservations SET expires_at = ? WHERE job_id = ?",
        (rounds._iso(rounds.utcnow() - rounds.timedelta(hours=1)), dead_jid),
    )
    conn.commit()

    other_jid = _enqueue(client, skey, _spec([{"ref": "s0", "rows": 1}]))

    # The next poll's sweep (``devices.expire_reservations``, run beside
    # ``rounds.expire_leases`` at the top of ``claim``) clears the stale row
    # before the walk ever asks ``free_devices`` a question.
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == other_jid
    assert resp.json()["devices"] == [0]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM device_reservations WHERE job_id = ?", (dead_jid,)
    ).fetchone()["n"] == 0


# ==========================================================================
# Inertness: gpu_count == 1 everywhere means none of this ever activates
# ==========================================================================


def test_inertness_no_reservation_or_backfill_when_every_job_is_gpu_count_1(
    client, store, conn, make_contributor, make_submitter, monkeypatch
):
    """With every job at the fleet's actual default (``gpu_count == 1``,
    nothing sets it higher until step 9), device_reservations must never
    gain a row, and the backfill machinery this step adds must never even be
    consulted -- not just "consulted and found nothing," but never asked."""
    from ganymede.coordinator import devices as devices_module

    backfill_calls: list[tuple[str, int]] = []
    real_free_devices = devices_module.free_devices

    def _spy_free_devices(conn_, worker_id, job_id, now=None, max_runtime_sec=None):
        if max_runtime_sec is not None:
            backfill_calls.append((job_id, max_runtime_sec))
        return real_free_devices(conn_, worker_id, job_id, now=now,
                                 max_runtime_sec=max_runtime_sec)

    monkeypatch.setattr(devices_module, "free_devices", _spy_free_devices)

    # The guard that actually delivers inertness (the gate's cheap pre-check
    # before it would ever peek) is ``has_other_reservations``, not merely
    # the absence of a widened ``free_devices`` call downstream of it -- pin
    # that directly, so a future reordering of the gate's conditions that
    # let the peek slip past it fails here rather than only in production.
    reservation_checks: list[bool] = []
    real_has_other_reservations = devices_module.has_other_reservations

    def _spy_has_other_reservations(conn_, worker_id, job_id, now=None):
        result = real_has_other_reservations(conn_, worker_id, job_id, now=now)
        reservation_checks.append(result)
        return result

    monkeypatch.setattr(devices_module, "has_other_reservations",
                        _spy_has_other_reservations)

    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)  # one device, busy below

    occupant = _occupy(conn, worker_id, 0)
    hi_jid = _enqueue(client, skey, _spec([{"ref": "hi0", "rows": 1}]))
    lo_jid = _enqueue(client, skey, _spec([{"ref": "lo0", "rows": 1}]))

    # Several contended polls -- the exact scenario a wide job would have
    # started reserving devices in, except nothing here is wide.
    for _ in range(3):
        resp = _claim(client, wkey, worker_id)
        assert resp.status_code == 204

    verdicts = {v.job_id: v for v in eligibility.explain(conn, worker_id).verdicts}
    assert verdicts[hi_jid].outcome == eligibility.REFUSED
    assert verdicts[hi_jid].reason == "insufficient_free_devices"
    assert verdicts[lo_jid].outcome == eligibility.REFUSED
    assert verdicts[lo_jid].reason == "insufficient_free_devices"

    assert conn.execute(
        "SELECT COUNT(*) AS n FROM device_reservations"
    ).fetchone()["n"] == 0
    assert backfill_calls == []
    assert reservation_checks and all(r is False for r in reservation_checks)

    # And the device is still ordinarily claimable the instant it frees --
    # the fleet's existing behaviour, untouched.
    _release(conn, occupant)
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == hi_jid
    assert resp.json()["devices"] == [0]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM device_reservations"
    ).fetchone()["n"] == 0
