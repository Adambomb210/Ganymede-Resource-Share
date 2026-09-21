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


def _wide_job_holding_a_reservation(client, conn, skey, wkey):
    """A 2-device worker with device 1 already busy, and a ``gpu_count = 2``
    job that is therefore refused and reserves the one card it can see.

    The worker is sized so the wide job *could* run here once device 1 frees.
    That matters: docs/14 §6's third eligibility condition is that a job may
    only reserve on a worker whose own live inventory could ever satisfy it,
    so a 1-device box paired with a ``gpu_count = 2`` job -- the shape these
    tests originally used to manufacture a reservation cheaply -- no longer
    produces one at all. This is also the configuration §6's own narrative
    describes ("a blocked wide job accumulates a reservation on devices as
    they free"), rather than one where the job could never fit regardless.
    """
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1])
    _occupy(conn, worker_id, 1)

    wide_jid = _enqueue(client, skey, _spec([{"ref": "w0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 2 WHERE id = ?", (wide_jid,))
    conn.commit()
    return worker_id, wide_jid


def test_a_short_task_backfills_onto_a_device_reserved_by_a_wider_job(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id, wide_jid = _wide_job_holding_a_reservation(client, conn, skey, wkey)

    # First poll: the wide job needs 2 devices and only device 0 is free --
    # refused, but eligible (gpu_count > 1, and this box has 2 cards, so it
    # could run here once device 1 frees), so it reserves what it can see.
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
    worker_id, wide_jid = _wide_job_holding_a_reservation(client, conn, skey, wkey)
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
    worker_id, wide_jid = _wide_job_holding_a_reservation(client, conn, skey, wkey)
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
    worker_id, dead_jid = _wide_job_holding_a_reservation(client, conn, skey, wkey)

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


# ==========================================================================
# A job that can never fit this worker must not hold its reservation slot
# ==========================================================================


def test_an_over_wide_job_does_not_take_the_reservation_slot_of_one_that_fits(
    client, store, conn, make_contributor, make_submitter
):
    """docs/14 §6's third eligibility condition.

    ``create_job`` only checks a new job against the *fleet's* widest
    inventory, so a ``gpu_count = 4`` job is accepted while any 4-card host
    exists -- and then meets narrower hosts for the rest of its life, refused
    ``insufficient_free_devices`` on every one of them, structurally, forever.

    Reservation admits one holder per worker and one reserver per poll, so
    without the "could this job ever fit *here*" guard the over-wide job wins
    that slot on every single poll of this 3-card box, and the ``gpu_count =
    3`` job behind it in walk order -- which fits exactly -- never reserves
    anything and starves. That is precisely the starvation §6 exists to
    prevent, aimed at the wrong job.
    """
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1, 2])       # a 3-card box
    busy = _occupy(conn, worker_id, 2)           # one card already working

    # Ranked ahead: needs 4 cards, can never run on this 3-card box.
    over_jid = _enqueue(client, skey, _spec([{"ref": "o0", "rows": 4}]))
    # Ranked behind: needs 3, fits this box exactly once device 2 frees.
    fits_jid = _enqueue(client, skey, _spec([{"ref": "f0", "rows": 4}]))
    conn.execute("UPDATE jobs SET gpu_count = 4, priority_rank = 1 WHERE id = ?",
                 (over_jid,))
    conn.execute("UPDATE jobs SET gpu_count = 3, priority_rank = 2 WHERE id = ?",
                 (fits_jid,))
    conn.commit()

    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 204

    # Both are refused -- neither can run right now -- but only the one that
    # could eventually fit here holds the reservation.
    verdicts = {v.job_id: v for v in eligibility.explain(conn, worker_id).verdicts}
    assert verdicts[over_jid].reason == "insufficient_free_devices"
    assert verdicts[fits_jid].reason == "insufficient_free_devices"

    assert _reservation_devices(conn, over_jid) == [], (
        "a job wider than this whole worker must never reserve here"
    )
    assert _reservation_devices(conn, fits_jid) == [0, 1], (
        "the widest job that could actually run here should be accumulating"
    )

    # And the accumulation completes: when the busy card frees, the fitting
    # job claims all three rather than having been starved out of them.
    _release(conn, busy)
    resp = _claim(client, wkey, worker_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == fits_jid
    assert resp.json()["devices"] == [0, 1, 2]


# ==========================================================================
# The submission gate: gpu_count validated at POST (docs/14 §5, §8.1, §9)
# ==========================================================================


def _post_job(client, skey, *, job_type="batch_inference", spec=None, **extra):
    body = {"job_type": job_type,
            "spec": spec if spec is not None else _spec([{"ref": "a", "rows": 1}])}
    body.update(extra)
    return client.post("/v1/jobs", headers=_hdr(skey), json=body)


def test_a_non_positive_gpu_count_is_refused_without_consulting_the_fleet(
    client, conn, make_submitter
):
    """A caller bug regardless of what hardware exists -- no worker has
    registered here at all."""
    _, skey = make_submitter()
    r = _post_job(client, skey, gpu_count=0)
    assert r.status_code == 422
    assert "at least 1" in r.text


def test_a_default_job_is_accepted_on_a_fleet_that_has_never_registered(
    client, conn, make_submitter
):
    """docs/14 §5's "CAREFUL": ``max_inventory_width`` reads ``0`` on a fresh
    coordinator, and an unconditional width check there would 422 *every*
    job -- including the default ``gpu_count = 1`` -- on day one."""
    _, skey = make_submitter()
    assert _post_job(client, skey).status_code == 200


def test_a_job_wider_than_the_whole_fleet_is_refused_at_submission(
    client, conn, make_contributor, make_submitter
):
    """Refused now, rather than left as a row that polls 204 forever and
    gives the submitter nothing to go on."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    _register(client, wkey)          # one device -> widest inventory is 1

    r = _post_job(client, skey, gpu_count=2)
    assert r.status_code == 422
    assert "widest inventory" in r.text


def test_collab_lora_finetune_may_not_ask_for_more_than_one_device(
    client, conn, make_contributor, make_submitter
):
    """docs/14 §8.1 defers in-process multi-device training *by decision*,
    and the trainer picks a single device -- so a wide collab job would
    allocate cards its task body cannot use and strand them, allocated and
    idle, for the whole lease while the ledger correctly reports them busy.
    §7's supported way to use a wide box here is several one-card leases.
    """
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1, 2, 3])   # a real 4-card box exists

    # The fleet is wide enough, so only the job type's own limit refuses it.
    r = _post_job(client, skey, job_type="collab_lora_finetune", spec={},
                  gpu_count=2)
    assert r.status_code == 422
    assert "cannot use" in r.text

    # One device is still fine.
    assert _post_job(client, skey, job_type="collab_lora_finetune", spec={},
                     gpu_count=1).status_code == 200


def test_a_static_job_type_declares_no_device_limit_of_its_own(
    client, conn, make_contributor, make_submitter
):
    """The clamp is per job type, not a blanket rule -- ``batch_inference``
    sets no ``max_gpu_count``, so it is bounded only by the fleet."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="owner")
    worker_id = _register(client, wkey)
    _add_devices(conn, worker_id, [1, 2, 3])

    assert _post_job(client, skey, gpu_count=4).status_code == 200
