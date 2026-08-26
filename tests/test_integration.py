"""Fake-worker integration suite for the coordinator (docs/03-roadmap.md "M1 --
Coordinator", Exit criteria).

Every test drives the real FastAPI app (``create_app``) through a real
``TestClient`` against a real, file-backed SQLite database (``conftest.py``'s
``settings`` fixture points ``db_path`` at ``tmp_path``, never ``:memory:`` --
that matters for the concurrency tests, which need genuine cross-connection
locking behaviour). The only double is ``FakeStore`` (object storage), which
``conftest.py`` explains is deliberately covered by ``test_store.py`` against
real MinIO instead.

Round timing is driven by passing an explicit ``now`` to ``closer.maybe_close``
or by back-dating ``rounds.opened_at`` directly in the database -- never by
``time.sleep``. A suite that sleeps for round timing is a suite nobody runs.

Section order follows the M1 exit criteria as given:
  1. Round advance and work-based close
  2. Zero workers and the backstop
  3. Concurrency -- the BEGIN IMMEDIATE path
  4. Acceptance gates end to end
  5. Leases and failure
  6. Eligibility
  7. Budgets and heterogeneity
  8. Auth
  9. Bonus -- real coordinator bugs found while writing the above (xfail)
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import threading
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch
from fastapi.testclient import TestClient

from ganymede.coordinator import closer, rounds
from ganymede.coordinator.aggregate import (
    REJECT_DTYPE_MISMATCH,
    REJECT_KEY_MISMATCH,
    REJECT_NON_FINITE,
    REJECT_NORM_OUTLIER,
    REJECT_NOT_SAFETENSORS,
    REJECT_SHAPE_MISMATCH,
    load_adapter,
    save_adapter,
)
from ganymede.coordinator.app import create_app
from ganymede.coordinator.db import connect

from tests.conftest import make_adapter as build_adapter
from tests.fake_worker import FakeWorker

# A fixed run configuration reused verbatim across the 1/2/8-worker
# parametrize cases below -- the point of that test is that NOTHING here
# changes with fleet size.
QUORUM_RUN_KWARGS = dict(target_steps=1000, min_round_sec=0, max_round_sec=3600)


# ==========================================================================
# 1. Round advance and work-based close
# ==========================================================================


def test_single_worker_closes_round_and_advances_base(client, store, conn, make_contributor, seeded_run):
    """One worker completes a round; round 0 closes, result_adapter_ref is
    set, and round 1 opens with that result as its base -- and the base
    really is readable out of the store, not just referenced in the DB."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)

    r = fw.run_task()
    assert r.status_code == 200
    assert r["accepted"] is True
    assert r["round_closed"] is True

    round0 = conn.execute(
        "SELECT status, result_adapter_ref FROM rounds WHERE run_id=? AND idx=0", (run_id,)
    ).fetchone()
    assert round0["status"] == "closed"
    assert round0["result_adapter_ref"] is not None

    round1 = conn.execute(
        "SELECT status, base_adapter_ref FROM rounds WHERE run_id=? AND idx=1", (run_id,)
    ).fetchone()
    assert round1 is not None
    assert round1["status"] == "open"
    assert round1["base_adapter_ref"] == round0["result_adapter_ref"]

    adapter = load_adapter(store.get_bytes(round1["base_adapter_ref"]))
    assert len(adapter) > 0


@pytest.mark.parametrize("worker_count", [1, 2, 8])
def test_round_closes_with_1_2_8_workers_same_config(worker_count, client, store, conn, make_contributor, seeded_run):
    """No quorum retuning (3.2): the identical run config closes correctly
    whether 1, 2, or 8 fake workers show up -- a one-machine round is a
    legitimate round, not a degenerate case (rounds.py's module docstring).
    ``QUORUM_RUN_KWARGS`` is passed unmodified in every parametrize case."""
    run_id = seeded_run(run_id="quorum", **QUORUM_RUN_KWARGS)
    workers = []
    for i in range(worker_count):
        _, key = make_contributor(name=f"w{i}")
        fw = FakeWorker(client, store, key, device=f"gpu{i}")
        task = fw.claim(run_id)
        assert task is not None
        workers.append(fw)

    for fw in workers:
        # Any status is fine here -- once the first worker's local_steps
        # budget alone clears target_steps, the rest legitimately get 409.
        fw.submit(fw.task["local_steps"])

    round0 = conn.execute(
        "SELECT status, result_adapter_ref FROM rounds WHERE run_id=? AND idx=0", (run_id,)
    ).fetchone()
    assert round0["status"] == "closed"
    assert round0["result_adapter_ref"] is not None

    round1 = conn.execute("SELECT status FROM rounds WHERE run_id=? AND idx=1", (run_id,)).fetchone()
    assert round1 is not None
    assert round1["status"] == "open"


def test_aggregation_weighted_by_actual_steps(client, store, conn, make_contributor, seeded_run):
    """5.2: aggregation weights follow steps actually completed, not an
    equal per-worker share. mode='mean', lr_outer=1.0, beta=0.0 makes the
    outer step reduce to exactly the weighted mean (aggregate.combine's
    docstring), so this is checkable numerically against a directly computed
    expectation -- not just 'closer to the bigger worker'."""
    run_id = seeded_run(target_steps=350, min_round_sec=0, max_round_sec=3600,
                         combine_mode="mean", lr_outer=1.0, outer_beta=0.0)

    _, key1 = make_contributor(name="big")
    fw1 = FakeWorker(client, store, key1)
    fw1.claim(run_id)
    fw1.heartbeat(300)
    r1 = fw1.submit(300, scale=0.02, seed=101)
    assert r1.status_code == 200 and r1["accepted"] is True
    assert r1["round_closed"] is False  # 300 < 350, round must still be open

    _, key2 = make_contributor(name="small")
    fw2 = FakeWorker(client, store, key2)
    fw2.claim(run_id)
    fw2.heartbeat(100)
    r2 = fw2.submit(100, scale=0.02, seed=202)
    assert r2.status_code == 200 and r2["accepted"] is True
    assert r2["round_closed"] is True  # 300 + 100 = 400 >= 350

    a1 = build_adapter(scale=0.02, seed=101)
    a2 = build_adapter(scale=0.02, seed=202)
    w1, w2 = 300 / 400, 100 / 400
    expected = {k: w1 * a1[k] + w2 * a2[k] for k in a1}

    round1 = conn.execute("SELECT base_adapter_ref FROM rounds WHERE run_id=? AND idx=1", (run_id,)).fetchone()
    actual = load_adapter(store.get_bytes(round1["base_adapter_ref"]))
    for k in expected:
        torch.testing.assert_close(actual[k], expected[k], atol=1e-5, rtol=1e-5)


def test_run_reaches_target_rounds_marks_done_and_stops(client, store, conn, make_contributor, seeded_run):
    """A run that reaches target_rounds ends with runs.status == 'done' and
    does not open another round."""
    run_id = seeded_run(target_rounds=2, target_steps=10, min_round_sec=0, max_round_sec=3600)

    _, key1 = make_contributor(name="w1")
    r1 = FakeWorker(client, store, key1).run_task()
    assert r1.status_code == 200 and r1["round_closed"] is True

    _, key2 = make_contributor(name="w2")
    r2 = FakeWorker(client, store, key2).run_task()
    assert r2.status_code == 200 and r2["round_closed"] is True

    run_row = conn.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
    assert run_row["status"] == "done"
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM rounds WHERE run_id=? AND idx=2", (run_id,)
    ).fetchone()["n"] == 0

    # A run that's done offers no more work, to anyone.
    _, key3 = make_contributor(name="w3")
    assert FakeWorker(client, store, key3).claim(run_id) is None


# ==========================================================================
# 2. Zero workers and the backstop
# ==========================================================================


def test_zero_submission_round_reopens_quietly_at_backstop(conn, store, seeded_run):
    """A round that hits max_round_sec with zero submissions reopens
    quietly: opened_at moves forward, status stays 'open', the round index
    does not advance, and nothing lands in audit -- an unscheduled fleet
    being idle overnight is normal operation, not an incident
    (reopen_empty_round's docstring). Driven with an explicit `now`, no sleep."""
    run_id = seeded_run(max_round_sec=600, min_round_sec=0, target_steps=100)
    before = conn.execute(
        "SELECT opened_at FROM rounds WHERE run_id=? AND idx=0", (run_id,)
    ).fetchone()["opened_at"]

    future = rounds.utcnow() + timedelta(seconds=700)
    result = closer.maybe_close(conn, store, run_id, now=future)
    assert result is None

    row = conn.execute(
        "SELECT idx, status, opened_at FROM rounds WHERE run_id=? AND idx=0", (run_id,)
    ).fetchone()
    assert row["status"] == "open"
    assert row["idx"] == 0
    assert row["opened_at"] != before
    assert conn.execute("SELECT COUNT(*) AS n FROM audit").fetchone()["n"] == 0


def test_one_submission_round_closes_at_backstop(client, store, conn, make_contributor, seeded_run):
    """A round that hits max_round_sec with exactly one accepted submission
    closes normally, via the 'submissions >= 1' branch of should_close."""
    run_id = seeded_run(max_round_sec=600, min_round_sec=0, target_steps=10**9)
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    fw.heartbeat(10)
    r = fw.submit(10)
    assert r.status_code == 200 and r["accepted"] is True
    assert r["round_closed"] is False  # target unreachable; real elapsed time is ~0

    future = rounds.utcnow() + timedelta(seconds=700)
    result = closer.maybe_close(conn, store, run_id, now=future)
    assert result is not None
    assert result.reason == "max_round_sec"
    assert result.accepted == 1

    round1 = conn.execute("SELECT status FROM rounds WHERE run_id=? AND idx=1", (run_id,)).fetchone()
    assert round1 is not None and round1["status"] == "open"


def test_min_round_sec_prevents_early_close_on_fast_worker(client, store, conn, make_contributor, seeded_run):
    """min_round_sec is respected: a worker that blows past target_steps in
    the first few real seconds does not close the round early -- the floor
    exists so slower peers still downloading the base adapter aren't cut off
    (should_close's docstring)."""
    run_id = seeded_run(target_steps=10, min_round_sec=600, max_round_sec=3600)
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    fw.heartbeat(50)
    r = fw.submit(50)  # 50 >= target_steps(10), but ~0s of real time has elapsed
    assert r.status_code == 200
    assert r["round_closed"] is False

    row = conn.execute("SELECT status FROM rounds WHERE run_id=? AND idx=0", (run_id,)).fetchone()
    assert row["status"] == "open"


# ==========================================================================
# 3. Concurrency -- the BEGIN IMMEDIATE path
# ==========================================================================


def test_no_double_leasing_under_concurrent_claim(client, store, conn, make_contributor, seeded_run):
    """No double-leasing under concurrent claim: 8 fake workers claim
    simultaneously via a real ThreadPoolExecutor against the file-backed
    SQLite database -- every task_id returned is unique, and no worker ends
    up holding two leases. This is exactly what claim_task's BEGIN IMMEDIATE
    exists to guarantee."""
    run_id = seeded_run(target_steps=10**9, min_round_sec=0, max_round_sec=3600)
    n = 8
    workers = []
    for i in range(n):
        _, key = make_contributor(name=f"w{i}")
        workers.append(FakeWorker(client, store, key, device=f"gpu{i}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        tasks = list(ex.map(lambda fw: fw.claim(run_id), workers))

    assert all(t is not None for t in tasks)
    task_ids = [t["task_id"] for t in tasks]
    assert len(set(task_ids)) == n

    dupes = conn.execute(
        """SELECT worker_id, COUNT(*) AS n FROM tasks
           WHERE status='leased' GROUP BY worker_id HAVING n > 1"""
    ).fetchall()
    assert dupes == []


def test_claim_twice_returns_same_task_not_a_new_lease(client, store, conn, make_contributor, seeded_run):
    """The resume-after-network-blip path: a worker that claims twice (e.g.
    it never saw the first response) gets its existing lease back, not a
    second one forked off from the same shard."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    t1 = fw.claim(run_id)
    t2 = fw.claim(run_id)
    assert t1 is not None and t2 is not None
    assert t1["task_id"] == t2["task_id"]
    assert t1["buckets"] == t2["buckets"]


def test_concurrent_submits_all_land_round_closes_once(client, store, conn, make_contributor, seeded_run):
    """Concurrent submits from many workers all land: 8 workers submit
    simultaneously and every one is accepted and recorded. target_steps is
    set unreachably high so the concurrent phase itself can't trigger a
    close -- that isolates "do N simultaneous submits all land intact under
    BEGIN IMMEDIATE" from the round-*close* race, which is a separate,
    deeper bug covered on its own below
    (test_concurrent_round_close_race_crashes_the_loser, xfail): real
    concurrent submits that DO cross target_steps together hit that race
    unpredictably, which is exactly why this test doesn't rely on the close
    itself happening under concurrency. The round is then closed once,
    deliberately, and exactly one round-1 row must exist.
    """
    run_id = seeded_run(target_steps=10**9, min_round_sec=0, max_round_sec=3600)
    n = 8
    workers = []
    for i in range(n):
        _, key = make_contributor(name=f"w{i}")
        fw = FakeWorker(client, store, key, device=f"gpu{i}")
        assert fw.claim(run_id) is not None
        workers.append(fw)

    def do_submit(fw):
        fw.heartbeat(50)
        return fw.submit(50)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(do_submit, workers))

    assert [r.status_code for r in results] == [200] * n
    assert all(r["accepted"] is True for r in results)

    still_open = conn.execute("SELECT status FROM rounds WHERE run_id=? AND idx=0", (run_id,)).fetchone()
    assert still_open["status"] == "open"  # n*50 << target_steps: confirms no racy premature close

    result = closer.close_round(conn, store, run_id, 0, "test_forced")
    assert result is not None
    assert result.accepted == n
    assert result.total_steps == n * 50

    n_round1 = conn.execute(
        "SELECT COUNT(*) AS n FROM rounds WHERE run_id=? AND idx=1", (run_id,)
    ).fetchone()["n"]
    assert n_round1 == 1


# ==========================================================================
# 4. Acceptance gates end to end
# ==========================================================================

_GATE_CASES = [
    pytest.param("pickle", REJECT_NOT_SAFETENSORS, id="pickle"),
    pytest.param("missing_key", REJECT_KEY_MISMATCH, id="missing_key"),
    pytest.param("extra_key", REJECT_KEY_MISMATCH, id="extra_key"),
    pytest.param("wrong_shape", REJECT_SHAPE_MISMATCH, id="wrong_shape"),
    pytest.param("wrong_dtype", REJECT_DTYPE_MISMATCH, id="wrong_dtype"),
    pytest.param("nan", REJECT_NON_FINITE, id="nan"),
    pytest.param("inf", REJECT_NON_FINITE, id="inf"),
]


@pytest.mark.parametrize("corrupt, expected_reason", _GATE_CASES)
def test_structural_gate_rejects_bad_artifact_with_right_reason(
    corrupt, expected_reason, client, store, conn, make_contributor, seeded_run
):
    """Each of NaN, Inf, wrong shape, missing key, extra key, wrong dtype,
    and a pickle file (torch.save output) is rejected via the API with the
    correct reject_reason slug, and the submission row records that reason.
    A gate rejection is a normal 200, not an HTTP error -- the worker is
    told to reclaim, not raised at."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    task_id = fw.task["task_id"]

    r = fw.submit(10, corrupt=corrupt)
    assert r.status_code == 200
    assert r["accepted"] is False
    assert r["reject_reason"] == expected_reason
    assert r["next_action"] == "reclaim"

    row = conn.execute(
        "SELECT accepted, reject_reason FROM submissions WHERE task_id=?", (task_id,)
    ).fetchone()
    assert row["accepted"] == 0
    assert row["reject_reason"] == expected_reason


def test_rejected_submission_does_not_count_toward_round_target(client, store, conn, make_contributor, seeded_run):
    """A rejected submission must not contribute to the round's accumulated
    steps -- round_progress only sums accepted-or-ungated rows, so a
    structurally broken submission can't push the round over target_steps."""
    run_id = seeded_run(target_steps=10, min_round_sec=0, max_round_sec=3600)
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    r = fw.submit(500, corrupt="nan")  # reports far more than target_steps
    assert r.status_code == 200
    assert r["accepted"] is False
    assert r["round_closed"] is False

    row = conn.execute("SELECT status FROM rounds WHERE run_id=? AND idx=0", (run_id,)).fetchone()
    assert row["status"] == "open"


def test_rejected_submission_is_audited_against_right_contributor(client, store, conn, make_contributor, seeded_run):
    """A rejected submission is recorded in the audit table against the
    right contributor_id -- that log is the raw material for future
    reputation scoring (closer.gate_submission's docstring)."""
    run_id = seeded_run()
    cid, key = make_contributor(name="flaky")
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    r = fw.submit(10, corrupt="inf")
    assert r.status_code == 200 and r["accepted"] is False

    row = conn.execute(
        "SELECT contributor_id, detail_json FROM audit WHERE event='submission_rejected' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["contributor_id"] == cid
    assert json.loads(row["detail_json"])["reason"] == REJECT_NON_FINITE


def test_cohort_norm_gate_rejects_the_inflated_outlier(client, store, conn, make_contributor, seeded_run):
    """The cohort norm gate: 5 workers submit, one with a ~50x inflated
    tensor; the outlier is rejected at close (norm_outlier) and the other
    four are aggregated. Structural gates alone accept the outlier at submit
    time -- it's finite and correctly shaped -- so this only shows up once
    the round closes and gate 4 runs cohort-relative."""
    run_id = seeded_run(target_steps=1000, min_round_sec=0, max_round_sec=3600)

    normal_task_ids = []
    for i in range(4):
        _, key = make_contributor(name=f"normal{i}")
        fw = FakeWorker(client, store, key, device=f"gpu{i}")
        fw.claim(run_id)
        normal_task_ids.append(fw.task["task_id"])
        r = fw.submit(200, scale=0.02, seed=i)
        assert r.status_code == 200 and r["accepted"] is True
        assert r["round_closed"] is False

    _, key_out = make_contributor(name="outlier")
    fw_out = FakeWorker(client, store, key_out, device="gpu-outlier")
    fw_out.claim(run_id)
    outlier_task_id = fw_out.task["task_id"]
    r_out = fw_out.submit(200, scale=0.02, seed=99, corrupt="huge_norm")
    assert r_out.status_code == 200
    assert r_out["accepted"] is True  # structural gates pass it at submit time
    assert r_out["round_closed"] is True  # 5 * 200 = 1000 >= target_steps

    outlier_row = conn.execute(
        "SELECT accepted, reject_reason FROM submissions WHERE task_id=?", (outlier_task_id,)
    ).fetchone()
    assert outlier_row["accepted"] == 0
    assert outlier_row["reject_reason"] == REJECT_NORM_OUTLIER

    for tid in normal_task_ids:
        row = conn.execute(
            "SELECT accepted, reject_reason FROM submissions WHERE task_id=?", (tid,)
        ).fetchone()
        assert row["accepted"] == 1
        assert row["reject_reason"] is None

    round0 = conn.execute(
        "SELECT distinct_contributors FROM rounds WHERE run_id=? AND idx=0", (run_id,)
    ).fetchone()
    assert round0["distinct_contributors"] == 4


def test_artifact_key_must_match_derived_key(client, store, conn, make_contributor, seeded_run):
    """A worker cannot point a submission at someone else's object (or any
    key it didn't derive): artifact_key not matching the task's own derived
    key is rejected 422, before any bytes are even gated."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    r = fw.submit(10, artifact_key="runs/someone-elses-run/rounds/00000/submissions/x.safetensors")
    assert r.status_code == 422


# ==========================================================================
# 5. Leases and failure
# ==========================================================================


def test_expired_lease_is_reclaimed_by_another_worker(client, store, conn, make_contributor, seeded_run):
    """An expired lease is reclaimed: back-date it, then a different worker
    can claim the shard, and the original worker's heartbeat now gets 410."""
    run_id = seeded_run()
    _, key1 = make_contributor(name="stale")
    fw1 = FakeWorker(client, store, key1)
    t1 = fw1.claim(run_id)
    assert t1 is not None

    past = rounds._iso(rounds.utcnow() - timedelta(seconds=10))
    conn.execute("UPDATE tasks SET lease_expires_at=? WHERE id=?", (past, t1["task_id"]))

    _, key2 = make_contributor(name="fresh")
    fw2 = FakeWorker(client, store, key2)
    t2 = fw2.claim(run_id)  # claim() calls rounds.expire_leases(conn) first
    assert t2 is not None
    assert t2["task_id"] != t1["task_id"]

    hb = fw1.heartbeat(5)
    assert hb.status_code == 410


def test_heartbeat_after_round_closed_gets_409(client, store, conn, make_contributor, seeded_run):
    """Heartbeat on a task whose round already closed gets 409. This is
    exactly the real window between close_round flipping the round's status
    and it later expiring outstanding leases -- reproduced directly here by
    moving the round to 'closed' without touching the task, isolating
    heartbeat's own round-status check from that timing."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    conn.execute("UPDATE rounds SET status='closed' WHERE run_id=? AND idx=0", (run_id,))
    r = fw.heartbeat(5)
    assert r.status_code == 409


def test_submit_after_round_closed_gets_409(client, store, conn, make_contributor, seeded_run):
    """Submit after the round closed gets 409, via the same guard as
    heartbeat's."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    conn.execute("UPDATE rounds SET status='closed' WHERE run_id=? AND idx=0", (run_id,))
    r = fw.submit(10)
    assert r.status_code == 409


def test_abandon_releases_lease_for_reclaim(client, store, conn, make_contributor, seeded_run):
    """abandon() releases the lease and the same worker can claim again --
    the voluntary "host is going away" path gives the shard back cleanly."""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    t1 = fw.claim(run_id)
    assert t1 is not None
    r = fw.abandon()
    assert r.status_code == 200

    t2 = fw.claim(run_id)
    assert t2 is not None
    assert t2["task_id"] != t1["task_id"]


def test_durability_submission_survives_fresh_connection(client, store, conn, settings, make_contributor, seeded_run):
    """Killing the coordinator mid-round loses nothing already submitted:
    submit, close every DB connection, then build a fresh app + connection
    against the same db_path and confirm the submission and its accepted
    verdict are still there. (The object store's own durability is FakeStore
    by design here -- conftest.py's docstring explains that's covered
    against real MinIO in test_store.py instead.)"""
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    task_id = fw.task["task_id"]
    fw.heartbeat(10)
    r = fw.submit(10)
    assert r.status_code == 200 and r["accepted"] is True

    conn.close()  # simulate the coordinator process going away

    conn2 = connect(settings.db_path)
    row = conn2.execute(
        "SELECT accepted, steps_completed FROM submissions WHERE task_id=?", (task_id,)
    ).fetchone()
    assert row is not None
    assert row["accepted"] == 1
    assert row["steps_completed"] == 10
    conn2.close()

    # And a fresh app instance serves the same durable state.
    app2 = create_app(settings, store)
    client2 = TestClient(app2)
    status = client2.get("/status").json()
    run_status = next(r for r in status["runs"] if r["id"] == run_id)
    assert run_status["current_round"] == 0  # round 0 hasn't closed (target_steps not reached)


# ==========================================================================
# 6. Eligibility
# ==========================================================================


def test_requires_supports_excludes_worker_without_capability(client, store, conn, make_contributor, seeded_run):
    """A run requiring supports=['nf4'] is never offered to a worker whose
    profile lacks it, and is offered to one that has it."""
    run_id = seeded_run(requires={"supports": ["nf4"]})

    _, key_no = make_contributor(name="no-nf4")
    fw_no = FakeWorker(client, store, key_no, supports=["bf16", "fp16"])
    assert fw_no.claim(run_id) is None
    assert fw_no.last_response.status_code == 204

    _, key_yes = make_contributor(name="has-nf4")
    fw_yes = FakeWorker(client, store, key_yes, supports=["bf16", "fp16", "nf4"])
    assert fw_yes.claim(run_id) is not None


def test_requires_min_vram_excludes_undersized_card(client, store, conn, make_contributor, seeded_run):
    """A run requiring min_vram_mb=24000 excludes a 12 GB card."""
    run_id = seeded_run(requires={"min_vram_mb": 24000})
    _, key = make_contributor()
    fw = FakeWorker(client, store, key, vram_mb=12288)  # probe.alloc_max_mb ~ 11288
    assert fw.claim(run_id) is None
    assert fw.last_response.status_code == 204


def test_clearance_gates_claim_and_manifest_visibility(client, store, conn, make_contributor, seeded_run):
    """clearance: an open-clearance contributor gets 204 on an
    internal-classification run; an internal contributor gets a task. The
    manifest also hides the run from the first and shows it to the second."""
    run_id = seeded_run(classification="internal")

    _, key_open = make_contributor(clearance="open")
    fw_open = FakeWorker(client, store, key_open)
    assert fw_open.claim(run_id) is None
    assert fw_open.last_response.status_code == 204

    _, key_int = make_contributor(clearance="internal")
    fw_int = FakeWorker(client, store, key_int)
    assert fw_int.claim(run_id) is not None

    open_manifest = client.get("/v1/manifest", headers={"Authorization": f"Bearer {key_open}"}).json()
    assert run_id not in {r["run_id"] for r in open_manifest["runs"]}

    int_manifest = client.get("/v1/manifest", headers={"Authorization": f"Bearer {key_int}"}).json()
    assert run_id in {r["run_id"] for r in int_manifest["runs"]}


def test_worker_below_throughput_floor_refused_for_this_run(client, store, conn, make_contributor, seeded_run):
    """A worker whose budget falls below the throughput floor relative to
    its peers is refused for that run (204) -- with a fast peer already
    claimed, so a median exists to fall below."""
    run_id = seeded_run()
    conn.execute(
        "INSERT INTO throughput (run_id, gpu_model, steps_per_min, samples, updated_at) "
        "VALUES (?, 'A100', 400.0, 1, ?)",
        (run_id, rounds._iso(rounds.utcnow())),
    )
    _, key_fast = make_contributor(name="fast")
    fw_fast = FakeWorker(client, store, key_fast, device="A100")
    assert fw_fast.claim(run_id) is not None  # establishes the median

    _, key_slow = make_contributor(name="slow")
    fw_slow = FakeWorker(client, store, key_slow, device="ancient-gpu")  # cold_start=30, 30/400 < 10%
    assert fw_slow.claim(run_id) is None
    assert fw_slow.last_response.status_code == 204


# ==========================================================================
# 7. Budgets and heterogeneity
# ==========================================================================


def test_heterogeneous_throughput_scales_budget_and_buckets_3to1(client, store, conn, make_contributor, seeded_run):
    """The heterogeneity criterion: two workers whose measured throughput
    differs 3x both get budgets in ~3:1 ratio AND bucket counts in ~3:1
    ratio, from the same round."""
    run_id = seeded_run(max_round_sec=3600, min_round_sec=0, target_steps=10**9)
    now = rounds._iso(rounds.utcnow())
    conn.execute(
        "INSERT INTO throughput (run_id, gpu_model, steps_per_min, samples, updated_at) VALUES (?,?,?,1,?)",
        (run_id, "SlowGPU", 10.0, now),
    )
    conn.execute(
        "INSERT INTO throughput (run_id, gpu_model, steps_per_min, samples, updated_at) VALUES (?,?,?,1,?)",
        (run_id, "FastGPU", 30.0, now),
    )

    _, key_slow = make_contributor(name="slow")
    t_slow = FakeWorker(client, store, key_slow, device="SlowGPU").claim(run_id)
    _, key_fast = make_contributor(name="fast")
    t_fast = FakeWorker(client, store, key_fast, device="FastGPU").claim(run_id)

    assert t_slow is not None and t_fast is not None
    step_ratio = t_fast["local_steps"] / t_slow["local_steps"]
    bucket_ratio = len(t_fast["buckets"]) / len(t_slow["buckets"])
    assert step_ratio == pytest.approx(3.0, rel=0.1)
    assert bucket_ratio == pytest.approx(3.0, rel=0.15)


def test_claim_with_3_minutes_left_gets_204_with_retry_after(client, store, conn, make_contributor, seeded_run):
    """A worker claiming with ~3 minutes left before max_round_sec gets 204
    with a Retry-After header, not unfinishable work. Driven by opening a
    round with a small max_round_sec and back-dating opened_at."""
    run_id = seeded_run(max_round_sec=600, min_round_sec=0, target_steps=10**9)
    past = rounds.utcnow() - timedelta(seconds=600 - 180)  # ~180s ("3 min") remaining
    conn.execute(
        "UPDATE rounds SET opened_at=? WHERE run_id=? AND idx=0", (rounds._iso(past), run_id)
    )

    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    assert fw.claim(run_id) is None
    assert fw.last_response.status_code == 204
    assert fw.last_response.headers.get("Retry-After") is not None


def test_measured_throughput_folds_back_in_after_close(client, store, conn, make_contributor, seeded_run):
    """Measured throughput is folded back in after a round closes: submit
    with metrics.steps_per_min, then a throughput row appears for that GPU
    model, and a subsequent claim's budget reflects it rather than the
    cold-start default."""
    run_id = seeded_run(target_steps=10, min_round_sec=0, max_round_sec=3600)
    _, key = make_contributor()
    fw = FakeWorker(client, store, key, device="MeasureGPU")
    fw.claim(run_id)
    fw.heartbeat(50)
    r = fw.submit(50, steps_per_min=77.0)
    assert r.status_code == 200 and r["round_closed"] is True

    tp = conn.execute(
        "SELECT steps_per_min FROM throughput WHERE run_id=? AND gpu_model=?", (run_id, "MeasureGPU")
    ).fetchone()
    assert tp is not None
    assert tp["steps_per_min"] == pytest.approx(77.0)

    _, key_measured = make_contributor(name="measured")
    task_measured = FakeWorker(client, store, key_measured, device="MeasureGPU").claim(run_id)
    _, key_cold = make_contributor(name="cold")
    task_cold = FakeWorker(client, store, key_cold, device="NeverSeenGPU").claim(run_id)

    assert task_measured is not None and task_cold is not None
    assert task_measured["local_steps"] > task_cold["local_steps"]


# ==========================================================================
# 8. Auth
# ==========================================================================


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing"),
        pytest.param({"Authorization": "Basic dXNlcjpwYXNz"}, id="malformed_scheme"),
        pytest.param({"Authorization": "Bearer not-a-real-key"}, id="unknown_key"),
    ],
)
def test_auth_rejects_bad_credentials(client, headers):
    """No Authorization header, a malformed header, and an unknown key all
    fail closed with 401."""
    resp = client.get("/v1/manifest", headers=headers)
    assert resp.status_code == 401


def test_auth_rejects_revoked_contributor(client, make_contributor):
    """A revoked (enabled=0) contributor's key is rejected 401 even though
    the key itself is otherwise valid -- revocation is enabled=0, not row
    deletion (auth.py's docstring), so the row is found but must not pass."""
    _, key = make_contributor(enabled=False)
    resp = client.get("/v1/manifest", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 401


def test_require_tls_rejects_plain_http_accepts_forwarded_proto(settings, store, conn, make_contributor):
    """With settings.require_tls=True, a plain-HTTP request is refused 403,
    and one carrying X-Forwarded-Proto: https succeeds (the reverse-proxy
    deployment path, 6.5)."""
    tls_settings = dataclasses.replace(settings, require_tls=True)
    tls_client = TestClient(create_app(tls_settings, store))
    _, key = make_contributor()
    headers = {"Authorization": f"Bearer {key}"}

    resp = tls_client.get("/v1/manifest", headers=headers)
    assert resp.status_code == 403

    resp2 = tls_client.get("/v1/manifest", headers={**headers, "X-Forwarded-Proto": "https"})
    assert resp2.status_code == 200


def test_cross_contributor_task_access_is_404_not_403(client, store, conn, make_contributor, seeded_run):
    """Contributor A cannot heartbeat, submit to, or abandon contributor B's
    task -- each returns 404, not 403: a task belonging to someone else must
    be indistinguishable from one that does not exist
    (app.py's _worker_for_task docstring)."""
    run_id = seeded_run()
    _, key_a = make_contributor(name="a")
    _, key_b = make_contributor(name="b")
    fw_a = FakeWorker(client, store, key_a)
    task = fw_a.claim(run_id)
    assert task is not None
    task_id = task["task_id"]
    headers_b = {"Authorization": f"Bearer {key_b}"}

    r1 = client.post(f"/v1/tasks/{task_id}/heartbeat", headers=headers_b, json={"steps_completed": 1})
    assert r1.status_code == 404

    r2 = client.post(f"/v1/tasks/{task_id}/abandon", headers=headers_b)
    assert r2.status_code == 404

    up = client.post(f"/v1/tasks/{task_id}/upload-url", headers={"Authorization": f"Bearer {key_a}"})
    derived_key = up.json()["key"]
    r3 = client.post(
        f"/v1/tasks/{task_id}/submit", headers=headers_b,
        json={"artifact_key": derived_key, "steps_completed": 1, "tokens_seen": 0, "metrics": {}},
    )
    assert r3.status_code == 404


# ==========================================================================
# 9. Regression tests for three coordinator bugs this suite uncovered
#
# None of these are among the 30 exit-criteria tests above; each is a
# genuine defect this suite turned up while probing adjacent behaviour.
# Left as failing (not fixed) per instructions -- the coordinator is someone
# else's ownership boundary.
# ==========================================================================


def test_concurrent_round_close_race_crashes_the_loser(client, store, conn, settings, make_contributor, seeded_run, monkeypatch):
    run_id = seeded_run(target_steps=10**9, min_round_sec=0, max_round_sec=3600)
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim(run_id)
    r = fw.submit(10)
    assert r.status_code == 200 and r["accepted"] is True

    barrier = threading.Barrier(2)
    real_immediate = closer.immediate
    tl = threading.local()

    @contextmanager
    def synced_immediate(c):
        # Only the FIRST `with immediate(...)` per thread is barrier-gated --
        # that's the write that flips the round to 'closing', i.e. exactly
        # the boundary between the (unlocked) status check and the write.
        if not getattr(tl, "synced", False):
            tl.synced = True
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
        with real_immediate(c) as cc:
            yield cc

    monkeypatch.setattr(closer, "immediate", synced_immediate)

    outcomes: dict[str, tuple[str, object]] = {}

    def run_close(name):
        # The connection is opened *inside* the thread: SQLite forbids using a
        # connection from any thread but the one that created it, and the app
        # honours that by handing every request its own (app.get_conn).
        c = connect(settings.db_path)
        try:
            outcomes[name] = ("ok", closer.close_round(c, store, run_id, 0, "race"))
        except Exception as exc:
            outcomes[name] = ("err", exc)
        finally:
            c.close()

    t1 = threading.Thread(target=run_close, args=("a",))
    t2 = threading.Thread(target=run_close, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    kinds = [v[0] for v in outcomes.values()]
    # Neither caller may crash: both observed an open round and both were
    # entitled to try. Exactly one does the aggregation and returns a
    # CloseResult; the other is turned away with None, which is a clean no-op
    # rather than an exception, because losing this race is ordinary operation
    # and not something a worker did wrong.
    assert kinds.count("err") == 0, outcomes
    results = [v[1] for v in outcomes.values()]
    assert sum(r is not None for r in results) == 1, results
    assert sum(r is None for r in results) == 1, results


def test_norm_reject_k_setting_is_actually_honored_by_close(client, store, conn, settings, make_contributor, seeded_run):
    """Desired: a run's configured norm_reject_k should change gate 4's
    tolerance. A permissive k=1000 should accept a 50x outlier (50 << 1000);
    it doesn't, because the configured value never reaches check_norms."""
    permissive_settings = dataclasses.replace(settings, norm_reject_k=1000.0)
    permissive_client = TestClient(create_app(permissive_settings, store))

    run_id = seeded_run(target_steps=1000, min_round_sec=0, max_round_sec=3600)
    for i in range(4):
        _, key = make_contributor(name=f"n{i}")
        fw = FakeWorker(permissive_client, store, key, device=f"gpu{i}")
        fw.claim(run_id)
        fw.submit(200, scale=0.02, seed=i)

    _, key_out = make_contributor(name="outlier")
    fw_out = FakeWorker(permissive_client, store, key_out, device="gpu-out")
    fw_out.claim(run_id)
    outlier_task_id = fw_out.task["task_id"]
    fw_out.submit(200, scale=0.02, seed=99, corrupt="huge_norm")

    row = conn.execute(
        "SELECT accepted, reject_reason FROM submissions WHERE task_id=?", (outlier_task_id,)
    ).fetchone()
    assert row["accepted"] == 1
    assert row["reject_reason"] is None


def test_expected_manifest_cache_ignores_store_identity():
    from tests.conftest import FakeStore

    key = "runs/manifest-cache-collision-test/rounds/00000/base.safetensors"

    store_a = FakeStore()
    adapter_a = build_adapter(scale=0.01, seed=0)
    store_a.put_bytes(key, save_adapter(adapter_a))
    manifest_a = closer.expected_manifest(store_a, key)
    assert set(manifest_a) == set(adapter_a)

    store_b = FakeStore()
    adapter_b = dict(adapter_a)
    adapter_b["extra.lora_A.weight"] = torch.zeros(4, 8)  # a genuinely different manifest
    store_b.put_bytes(key, save_adapter(adapter_b))

    manifest_b = closer.expected_manifest(store_b, key)
    # Desired: reflects store_b's actual bytes, not store_a's stale cache entry.
    assert set(manifest_b) == set(adapter_b)
