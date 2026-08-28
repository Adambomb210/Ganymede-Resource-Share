"""The claim-path queue walk and the job / admin-queue endpoints (docs/07).

The single-active-job case is covered by ``test_integration.py`` (unchanged):
those tests still pass because one queued job walks exactly like today's one
active run. What is new here is the multi-job behaviour the walk exists for --
priority order, capability backfill past a refusal, one lease per machine across
*two* jobs -- plus the job submission / admin surface.
"""

from __future__ import annotations

import json
import uuid

import pytest

from ganymede.coordinator import eligibility, rounds
from tests.fake_worker import FakeWorker


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _job_id(conn, run_id: str) -> str:
    return conn.execute(
        "SELECT job_id FROM runs WHERE id = ?", (run_id,)
    ).fetchone()["job_id"]


def _set_constraints(conn, run_id: str, obj: dict) -> None:
    conn.execute(
        "UPDATE jobs SET constraints_json = ? WHERE id = ?",
        (json.dumps(obj), _job_id(conn, run_id)),
    )
    conn.commit()


def _set_rank(conn, run_id: str, rank: int) -> None:
    conn.execute(
        "UPDATE jobs SET priority_rank = ? WHERE id = ?", (rank, _job_id(conn, run_id))
    )
    conn.commit()


@pytest.fixture
def make_admin(conn):
    def _make(name: str = "admin"):
        from ganymede.coordinator.auth import generate_key, hash_key

        cid, key = uuid.uuid4().hex, generate_key()
        conn.execute(
            """INSERT INTO contributors
                 (id, name, key_hash, enabled, clearance, is_admin, created_at)
               VALUES (?, ?, ?, 1, 'open', 1, ?)""",
            (cid, name, hash_key(key), rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


@pytest.fixture
def make_submitter(conn, make_contributor):
    def _make(name: str = "submitter", status: str = "approved"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, status, rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# ==========================================================================
# The queue walk
# ==========================================================================


def test_lower_rank_job_is_walked_first(client, store, conn, make_contributor, seeded_run):
    """Two queued jobs; the one with the lower priority_rank hands out the
    task, whatever order they were created in."""
    seeded_run(run_id="early")     # rank 10
    seeded_run(run_id="late")      # rank 20
    _set_rank(conn, "late", 5)     # now 'late' sorts first

    _, key = make_contributor()
    task = FakeWorker(client, store, key).claim()
    assert task is not None
    assert task["run_id"] == "late"
    assert task["job_id"] == _job_id(conn, "late")


def test_capability_backfill_past_a_refusal(client, store, conn, make_contributor, seeded_run):
    """A machine no high-rank job accepts still reaches the first lower-rank
    job it fits (Decision 10). The walk `continue`s past the refusal -- no
    early break -- and the refusal is recorded against the skipped job."""
    seeded_run(run_id="pinned")    # rank 10 -- constrained away
    seeded_run(run_id="open")      # rank 20 -- fits
    _set_constraints(conn, "pinned", {"machine_ids": ["some-other-box"]})

    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    task = fw.claim()

    assert task is not None
    assert task["run_id"] == "open"

    verdicts = {v.job_id: v for v in eligibility.explain(conn, fw.worker_id).verdicts}
    assert verdicts[_job_id(conn, "pinned")].outcome == eligibility.REFUSED
    assert "pin list" in verdicts[_job_id(conn, "pinned")].reason
    assert verdicts[_job_id(conn, "open")].outcome == eligibility.LEASED


def test_predicate_constraint_refuses_then_backfills(client, store, conn, make_contributor, seeded_run):
    seeded_run(run_id="big-only")
    seeded_run(run_id="anything")
    _set_constraints(conn, "big-only", {"vram_gb": {">=": 999}})

    _, key = make_contributor()
    fw = FakeWorker(client, store, key, vram_mb=12288)
    task = fw.claim()
    assert task is not None and task["run_id"] == "anything"

    v = {x.job_id: x for x in eligibility.explain(conn, fw.worker_id).verdicts}
    assert v[_job_id(conn, "big-only")].outcome == eligibility.REFUSED
    assert "fails >=" in v[_job_id(conn, "big-only")].reason


def test_pin_does_not_waive_the_capability_gate(client, store, conn, make_contributor, seeded_run):
    """docs/07 §2: a pinned machine that cannot fit the model still refuses at
    the capability gate -- "only these machines", not "regardless of fit"."""
    run_id = seeded_run(requires={"min_vram_mb": 999_999})
    _, key = make_contributor()
    fw = FakeWorker(client, store, key, vram_mb=12288)
    fw.register()
    _set_constraints(conn, run_id, {"machine_ids": [fw.worker_id]})  # pin THIS machine

    assert fw.claim() is None
    assert fw.last_response.status_code == 204
    v = eligibility.explain(conn, fw.worker_id).verdicts[0]
    assert v.outcome == eligibility.REFUSED
    assert "vram_mb" in v.reason  # the is_eligible floor bit, not the pin


# ==========================================================================
# One lease per machine, global (Decision 4)
# ==========================================================================


def test_one_lease_per_machine_across_two_jobs(client, store, conn, make_contributor, seeded_run):
    """A machine holding a lease on job A does not also get a task from job B
    on its next poll -- it is re-served the task it already holds."""
    seeded_run(run_id="jobA")
    seeded_run(run_id="jobB")

    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    first = fw.claim()
    assert first is not None

    again = fw.claim()          # no pin -- walks the whole queue
    assert again is not None
    assert again["task_id"] == first["task_id"]
    assert again["run_id"] == first["run_id"]

    leased = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE worker_id = ? AND status = 'leased'",
        (fw.worker_id,),
    ).fetchone()["n"]
    assert leased == 1


def test_resumed_lease_gets_a_fresh_presign_same_task(client, store, conn, make_contributor, seeded_run):
    seeded_run(run_id="r")
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    a = fw.claim()
    b = fw.claim()
    assert a["task_id"] == b["task_id"]
    assert b["base_adapter_url"]  # a fresh presigned GET, not a replay


# ==========================================================================
# queued -> running flip
# ==========================================================================


def test_first_lease_flips_the_job_to_running(client, store, conn, make_contributor, seeded_run):
    run_id = seeded_run()
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (_job_id(conn, run_id),)
    ).fetchone()["status"] == "queued"

    _, key = make_contributor()
    assert FakeWorker(client, store, key).claim() is not None

    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (_job_id(conn, run_id),)
    ).fetchone()["status"] == "running"


def test_a_done_run_marks_its_job_done_and_leaves_the_walk(
    client, store, conn, make_contributor, seeded_run
):
    run_id = seeded_run(target_rounds=1, target_steps=10, min_round_sec=0, max_round_sec=3600)
    _, key = make_contributor()
    r = FakeWorker(client, store, key).run_task()
    assert r.status_code == 200 and r["round_closed"] is True

    assert conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()["status"] == "done"
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (_job_id(conn, run_id),)
    ).fetchone()["status"] == "done"

    # A done job is no longer walked, so explain() (non-terminal only) drops it.
    _, key2 = make_contributor(name="later")
    fw2 = FakeWorker(client, store, key2)
    assert fw2.claim() is None
    assert eligibility.explain(conn, fw2.worker_id).verdicts == []


# ==========================================================================
# Job submission & management endpoints
# ==========================================================================


def test_post_jobs_requires_an_approved_submitter(client, make_contributor, make_submitter):
    _, plain = make_contributor(name="nobody")
    r = client.post("/v1/jobs", headers=_hdr(plain),
                    json={"job_type": "collab_lora_finetune", "spec": {}})
    assert r.status_code == 404  # submitter surface not confirmed to exist

    _, skey = make_submitter()
    r = client.post("/v1/jobs", headers=_hdr(skey),
                    json={"job_type": "collab_lora_finetune", "spec": {}})
    assert r.status_code == 200
    assert r.json()["status"] == "draft"


def test_post_jobs_rejects_an_unknown_type_and_bad_constraints(client, make_submitter):
    _, skey = make_submitter()
    assert client.post("/v1/jobs", headers=_hdr(skey),
                       json={"job_type": "no_such_type", "spec": {}}).status_code == 422
    assert client.post("/v1/jobs", headers=_hdr(skey),
                       json={"job_type": "collab_lora_finetune", "spec": {},
                             "constraints": {"cores": {">=": 8}}}).status_code == 422
    assert client.post("/v1/jobs", headers=_hdr(skey),
                       json={"job_type": "collab_lora_finetune", "spec": {},
                             "constraints": {"vram_gb": {"~=": 8}}}).status_code == 422


def test_enqueue_moves_draft_to_queued_at_the_tail(client, conn, make_submitter):
    _, skey = make_submitter()
    jid = client.post("/v1/jobs", headers=_hdr(skey),
                      json={"job_type": "collab_lora_finetune", "spec": {},
                            "constraints": {"vram_gb": {">=": 16}}}).json()["job_id"]

    top = conn.execute("SELECT COALESCE(MAX(priority_rank),0) AS m FROM jobs").fetchone()["m"]
    r = client.post(f"/v1/jobs/{jid}/enqueue", headers=_hdr(skey))
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert r.json()["priority_rank"] == top + 1

    # A body priority_rank on POST /v1/jobs is ignored, not a 422 (docs/07 §5).
    r2 = client.post("/v1/jobs", headers=_hdr(skey),
                     json={"job_type": "collab_lora_finetune", "spec": {},
                           "priority_rank": -999})
    assert r2.status_code == 200


def test_jobs_get_is_404_not_403_across_owners(client, conn, make_submitter, make_contributor):
    _, s1 = make_submitter(name="s1")
    _, s2 = make_submitter(name="s2")
    jid = client.post("/v1/jobs", headers=_hdr(s1),
                      json={"job_type": "collab_lora_finetune", "spec": {}}).json()["job_id"]

    assert client.get(f"/v1/jobs/{jid}", headers=_hdr(s1)).status_code == 200
    assert client.get(f"/v1/jobs/{jid}", headers=_hdr(s2)).status_code == 404
    assert client.get(f"/v1/jobs/{uuid.uuid4().hex}", headers=_hdr(s1)).status_code == 404


def test_jobs_list_shows_own_to_submitter_all_to_admin(client, make_submitter, make_admin):
    _, s1 = make_submitter(name="s1")
    _, s2 = make_submitter(name="s2")
    _, akey = make_admin()
    j1 = client.post("/v1/jobs", headers=_hdr(s1),
                     json={"job_type": "collab_lora_finetune", "spec": {}}).json()["job_id"]
    client.post("/v1/jobs", headers=_hdr(s2),
                json={"job_type": "collab_lora_finetune", "spec": {}})

    mine = client.get("/v1/jobs", headers=_hdr(s1)).json()["jobs"]
    assert [j["job_id"] for j in mine] == [j1]
    all_jobs = client.get("/v1/jobs", headers=_hdr(akey)).json()["jobs"]
    assert len(all_jobs) >= 2


def test_cancel_sets_mode_and_state(client, conn, make_submitter):
    _, skey = make_submitter()
    jid = client.post("/v1/jobs", headers=_hdr(skey),
                      json={"job_type": "collab_lora_finetune", "spec": {}}).json()["job_id"]
    r = client.post(f"/v1/jobs/{jid}/cancel", headers=_hdr(skey), json={"mode": "hard"})
    assert r.status_code == 200
    row = conn.execute("SELECT status, cancel_mode FROM jobs WHERE id = ?", (jid,)).fetchone()
    assert row["status"] == "cancelled" and row["cancel_mode"] == "hard"

    assert client.post(f"/v1/jobs/{jid}/cancel", headers=_hdr(skey),
                       json={"mode": "sideways"}).status_code == 422


def test_a_cancelled_job_is_not_walked(client, store, conn, make_contributor, seeded_run, make_submitter):
    run_id = seeded_run()
    _, skey = make_submitter()
    # give the submitter ownership so cancel is allowed
    conn.execute("UPDATE jobs SET owner_id = (SELECT user_id FROM submitters LIMIT 1) "
                 "WHERE id = ?", (_job_id(conn, run_id),))
    conn.commit()
    client.post(f"/v1/jobs/{_job_id(conn, run_id)}/cancel", headers=_hdr(skey),
                json={"mode": "soft"})

    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    assert fw.claim() is None


# ==========================================================================
# Admin queue surface
# ==========================================================================


def test_admin_queue_lists_in_rank_order_with_lease_counts(
    client, store, conn, make_contributor, seeded_run, make_admin
):
    seeded_run(run_id="a")
    seeded_run(run_id="b")
    _set_rank(conn, "a", 30)
    _set_rank(conn, "b", 10)

    _, key = make_contributor()
    FakeWorker(client, store, key).claim()  # leases from 'b' (rank 10)

    _, akey = make_admin()
    q = client.get("/v1/admin/queue", headers=_hdr(akey))
    assert q.status_code == 200
    rows = q.json()["queue"]
    assert [r["id"] for r in rows] == [_job_id(conn, "b"), _job_id(conn, "a")]
    by_id = {r["id"]: r for r in rows}
    assert by_id[_job_id(conn, "b")]["leased_tasks"] == 1
    assert by_id[_job_id(conn, "a")]["leased_tasks"] == 0


def test_admin_queue_is_404_for_non_admin(client, make_contributor):
    _, key = make_contributor()
    assert client.get("/v1/admin/queue", headers=_hdr(key)).status_code == 404


def test_reorder_is_the_only_priority_writer_and_changes_the_walk(
    client, store, conn, make_contributor, seeded_run, make_admin
):
    seeded_run(run_id="first")     # rank 10
    seeded_run(run_id="second")    # rank 20
    _, akey = make_admin()

    # Move 'second' before 'first'.
    r = client.post("/v1/admin/queue/reorder", headers=_hdr(akey),
                    json={"job_id": _job_id(conn, "second"),
                          "before": _job_id(conn, "first")})
    assert r.status_code == 200
    assert r.json()["priority_rank"] == 10 - 1

    _, key = make_contributor()
    task = FakeWorker(client, store, key).claim()
    assert task["run_id"] == "second"

    # rank form
    r2 = client.post("/v1/admin/queue/reorder", headers=_hdr(akey),
                     json={"job_id": _job_id(conn, "first"), "rank": 3})
    assert r2.json()["priority_rank"] == 3

    # exactly one of before/after/rank
    assert client.post("/v1/admin/queue/reorder", headers=_hdr(akey),
                       json={"job_id": _job_id(conn, "first")}).status_code == 422
    assert client.post("/v1/admin/queue/reorder", headers=_hdr(akey),
                       json={"job_id": _job_id(conn, "first"), "rank": 1,
                             "after": _job_id(conn, "second")}).status_code == 422


def test_reorder_is_404_for_non_admin(client, conn, make_contributor, seeded_run):
    run_id = seeded_run()
    _, key = make_contributor()
    r = client.post("/v1/admin/queue/reorder", headers=_hdr(key),
                    json={"job_id": _job_id(conn, run_id), "rank": 1})
    assert r.status_code == 404


# ==========================================================================
# worker_eligibility keyed by job_id
# ==========================================================================


def test_eligibility_rows_are_keyed_by_job_id(client, store, conn, make_contributor, seeded_run):
    run_id = seeded_run(requires={"min_vram_mb": 999_999})
    _, key = make_contributor()
    fw = FakeWorker(client, store, key, vram_mb=12288)
    fw.claim()

    row = conn.execute(
        "SELECT job_id, outcome FROM worker_eligibility WHERE worker_id = ?",
        (fw.worker_id,),
    ).fetchone()
    assert row["job_id"] == _job_id(conn, run_id)
    assert row["outcome"] == eligibility.REFUSED


def test_explain_hides_terminal_jobs(client, store, conn, make_contributor, seeded_run):
    run_id = seeded_run()
    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    fw.claim()
    assert eligibility.explain(conn, fw.worker_id).verdicts  # visible while queued/running

    conn.execute("UPDATE jobs SET status = 'cancelled' WHERE id = ?", (_job_id(conn, run_id),))
    conn.commit()
    assert eligibility.explain(conn, fw.worker_id).verdicts == []
