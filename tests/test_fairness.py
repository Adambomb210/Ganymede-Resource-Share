"""Share accounting, fair-share ordering, quotas, and preemption (docs/13 §1-§4).

Four mechanisms that are all **off by default**, which makes the most important
tests in this file the boring ones: the assertions that with the defaults in
place, the queue orders exactly as it did before any of this existed and nothing
is ever refused. A scheduling feature that leaks when it is switched off is worse
than one that does not work, because the symptom is a fleet doing something
slightly wrong for a reason nobody thinks to look for.

The other half is the three-way task-status split. ``cancelled`` /
``preempted`` / ``abandoned`` look like bookkeeping and are not: two of them must
not touch the contributor's record, and exactly one of the three is terminal.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

import pytest

from ganymede.coordinator import fairness, ledger, rounds
from ganymede.coordinator.app import _selectable_jobs

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _iso_ago(**kw) -> str:
    return rounds._iso(rounds.utcnow() - timedelta(**kw))


@pytest.fixture
def owner(conn, make_contributor):
    def _make(name: str):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, "approved", rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


@pytest.fixture
def worker(conn, make_contributor):
    """A real ``workers`` row. ``tasks.worker_id`` is a foreign key, so a
    string literal will not do."""
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
def admin(conn):
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


def _job(conn, job_id: str, owner_id: str, rank: int, status: str = "queued",
         job_type: str = "batch_inference") -> str:
    conn.execute(
        """INSERT INTO jobs
             (id, owner_id, job_type, spec_json, status, priority_rank,
              constraints_json, created_at)
           VALUES (?, ?, ?, '{}', ?, ?, '{}', ?)""",
        (job_id, owner_id, job_type, status, rank, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return job_id


def _task(conn, task_id: str, job_id: str, status: str, *, leased_at=None,
          ended=None, worker_id=None, attempts: int = 1, preempt=None) -> str:
    conn.execute(
        """INSERT INTO tasks
             (id, job_id, buckets_json, local_steps, status, worker_id,
              lease_expires_at, leased_at, attempts, preempt_mode, created_at)
           VALUES (?, ?, '[]', 10, ?, ?, ?, ?, ?, ?, ?)""",
        (task_id, job_id, status, worker_id, ended, leased_at, attempts, preempt,
         rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return task_id


# ==========================================================================
# §1 Share accounting
# ==========================================================================


def test_a_lease_is_charged_for_the_reservation_not_the_work(conn, owner):
    """docs/13 §1.1. A machine held for fifteen minutes is fifteen minutes
    nobody else could have it, whether or not anything came back -- so an
    abandoned lease costs the same as a completed one."""
    a, _ = owner("a")
    b, _ = owner("b")
    _job(conn, "ja", a, 10)
    _job(conn, "jb", b, 10)
    # Both leased for 10 minutes an hour ago; a submitted, b abandoned.
    _task(conn, "ta", "ja", "submitted",
          leased_at=_iso_ago(minutes=70), ended=_iso_ago(minutes=60))
    _task(conn, "tb", "jb", "abandoned",
          leased_at=_iso_ago(minutes=70), ended=_iso_ago(minutes=60))

    totals = fairness.recompute_shares(conn)
    assert totals[a] == pytest.approx(totals[b], rel=1e-6)
    assert totals[a] == pytest.approx(600 * 0.5 ** (1 / 24), rel=1e-3)


def test_an_early_submission_ends_the_charge(conn, owner):
    """The machine was free from the moment the artifact landed, so a job that
    finishes in two minutes of a fifteen-minute lease is charged two."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10)
    conn.execute(
        """INSERT INTO tasks (id, job_id, buckets_json, local_steps, status,
                              lease_expires_at, leased_at, attempts, created_at)
           VALUES ('t1', 'ja', '[]', 10, 'submitted', ?, ?, 1, ?)""",
        (_iso_ago(minutes=45), _iso_ago(minutes=60), rounds._iso(rounds.utcnow())),
    )
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
                                    received_at)
           VALUES ('t1', 'a/b', 10, ?)""",
        (_iso_ago(minutes=58),),
    )
    conn.commit()
    totals = fairness.recompute_shares(conn)
    # Two minutes, not fifteen.
    assert totals[a] == pytest.approx(120 * 0.5 ** (58 / 60 / 24), rel=1e-2)


def test_an_in_flight_lease_is_charged_for_what_it_has_held_so_far(conn, owner):
    """A submitter's share rises *while* their work runs, not in a step when it
    finishes -- otherwise a long lease is free right up until it is not."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=5),
          ended=rounds._iso(rounds.utcnow() + timedelta(minutes=10)))
    totals = fairness.recompute_shares(conn)
    assert totals[a] == pytest.approx(300, rel=0.05)


def test_yesterdays_work_counts_half(conn, owner):
    """The 24-hour half-life, asserted as a number rather than trusted."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "submitted",
          leased_at=_iso_ago(hours=25), ended=_iso_ago(hours=24))
    totals = fairness.recompute_shares(conn)
    assert totals[a] == pytest.approx(3600 * 0.5, rel=1e-3)


def test_a_stale_rollup_is_aged_forward_not_served_stale(conn, owner):
    """docs/13 §1.5, and the property that makes a cached rollup safe on the
    claim path. A sweep that stopped a week ago must not still be telling the
    scheduler who was ahead when it died."""
    a, _ = owner("a")
    conn.execute(
        "INSERT INTO share_accounting (owner_id, decayed_seconds, formula_version,"
        " updated_at) VALUES (?, 1000.0, 0, ?)",
        (a, _iso_ago(hours=24)),
    )
    conn.commit()
    # One owner is always 100% of the total, so read the raw decay instead.
    row = conn.execute("SELECT * FROM share_accounting").fetchone()
    aged = fairness._decay(row["decayed_seconds"], 24.0)
    assert aged == pytest.approx(500.0)
    # And the fraction is still 1.0, because fractions are relative.
    assert fairness.share_fractions(conn)[a] == pytest.approx(1.0)


def test_nobody_has_used_anything_is_an_empty_map_not_a_crash(conn):
    """Every caller has to read ``{}`` as "no share is zero"."""
    assert fairness.share_fractions(conn) == {}


def test_a_pre_scheduler_task_with_no_job_is_not_charged_to_anyone(conn, owner):
    """A task from before migration 005 carries no ``job_id`` and so no owner.
    Those predate the scheduler entirely, in a world with one submitter, where a
    share is meaningless -- the JOIN drops them and the sweep must not trip."""
    conn.execute(
        """INSERT INTO tasks (id, buckets_json, local_steps, status, leased_at,
                              lease_expires_at, attempts, created_at)
           VALUES ('orphan', '[]', 10, 'submitted', ?, ?, 1, ?)""",
        (_iso_ago(hours=2), _iso_ago(hours=1), rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    assert fairness.recompute_shares(conn) == {}


def test_an_owner_who_drops_out_of_the_window_loses_their_row(conn, owner):
    """The sweep is delete-then-insert. An UPDATE-only pass would leave a stale
    share sitting there being decayed forward forever."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "submitted",
          leased_at=_iso_ago(hours=2), ended=_iso_ago(hours=1))
    assert fairness.recompute_shares(conn)
    conn.execute("DELETE FROM tasks")
    conn.commit()
    assert fairness.recompute_shares(conn) == {}
    assert conn.execute("SELECT COUNT(*) c FROM share_accounting").fetchone()["c"] == 0


# ==========================================================================
# §2 Fair-share ordering
# ==========================================================================


def test_the_default_spread_leaves_the_order_byte_identical(conn, owner, settings):
    """The single most important test in this file. ``spread = 0.0`` must not
    reorder anything -- not "usually", not "in practice"."""
    a, _ = owner("a")
    b, _ = owner("b")
    for i, (job, own, rank) in enumerate([
        ("j1", a, 30), ("j2", b, 10), ("j3", a, 20), ("j4", b, 10),
    ]):
        _job(conn, job, own, rank)
    # Give `a` the entire fleet's recent usage, so any leak shows.
    _task(conn, "t1", "j1", "submitted",
          leased_at=_iso_ago(minutes=30), ended=_iso_ago(minutes=10))
    fairness.recompute_shares(conn)

    baseline = [r["id"] for r in _selectable_jobs(conn, None, [], None)]
    with_settings = [r["id"] for r in _selectable_jobs(conn, None, [], settings)]
    assert settings.fairshare_spread == 0.0
    assert with_settings == baseline
    assert baseline == ["j2", "j4", "j3", "j1"]


def test_spread_demotes_the_submitter_who_has_had_the_fleet(conn, owner, settings):
    import dataclasses

    a, _ = owner("a")
    b, _ = owner("b")
    _job(conn, "hog", a, 10)
    _job(conn, "quiet", b, 15)
    _task(conn, "t1", "hog", "submitted",
          leased_at=_iso_ago(minutes=30), ended=_iso_ago(minutes=10))
    fairness.recompute_shares(conn)

    # Off: strict rank order, the hog first.
    assert [r["id"] for r in _selectable_jobs(conn, None, [], settings)] == \
        ["hog", "quiet"]
    # On, with a spread of one sparse slot: 10 + 10*1.0 = 20 > 15.
    hot = dataclasses.replace(settings, fairshare_spread=10.0)
    assert [r["id"] for r in _selectable_jobs(conn, None, [], hot)] == \
        ["quiet", "hog"]


def test_fair_share_only_ever_demotes(conn):
    """docs/13 §2.3. A formula that could *promote* would be a back door into
    the submitter-set priority docs/07 §5 forbids: starve yourself deliberately,
    get promoted. There is no such move."""
    for share in (0.0, 0.5, 1.0):
        assert fairness.effective_rank(10, 25.0, share) >= 10.0
    # And a nonsense negative share cannot buy a promotion either.
    assert fairness.effective_rank(10, 25.0, -5.0) == 10.0


def test_a_demoted_job_is_still_selectable(conn, owner, settings):
    """A sort term, never a filter (docs/13 §2.1). The hog must still be offered
    to a machine nothing else fits -- filtering it would idle the fleet in order
    to punish someone."""
    import dataclasses

    a, _ = owner("a")
    _job(conn, "hog", a, 10)
    _task(conn, "t1", "hog", "submitted",
          leased_at=_iso_ago(minutes=30), ended=_iso_ago(minutes=10))
    fairness.recompute_shares(conn)
    hot = dataclasses.replace(settings, fairshare_spread=1000.0)
    assert [r["id"] for r in _selectable_jobs(conn, None, [], hot)] == ["hog"]


# ==========================================================================
# §3 Quotas and budgets
# ==========================================================================


def test_no_row_is_no_cap(conn, owner):
    """The default for everybody, including every submitter that already
    exists. The feature is inert on arrival without needing a flag."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10)
    for i in range(20):
        _task(conn, f"t{i}", "ja", "leased", leased_at=_iso_ago(minutes=1))
    assert fairness.quota_refusal(conn, a) is None


def test_a_concurrency_cap_refuses_at_the_cap_not_past_it(conn, owner):
    a, _ = owner("a")
    conn.execute(
        "INSERT INTO submitter_quotas (user_id, max_concurrent_tasks, updated_at)"
        " VALUES (?, 2, ?)", (a, rounds._iso(rounds.utcnow())))
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1))
    conn.commit()
    assert fairness.quota_refusal(conn, a) is None
    _task(conn, "t2", "ja", "leased", leased_at=_iso_ago(minutes=1))
    assert fairness.quota_refusal(conn, a) == fairness.OVER_CONCURRENCY


def test_a_finished_task_frees_the_concurrency_slot(conn, owner):
    """A quota is self-clearing; that is what makes it different from a budget
    and why it is not checked at enqueue."""
    a, _ = owner("a")
    conn.execute(
        "INSERT INTO submitter_quotas (user_id, max_concurrent_tasks, updated_at)"
        " VALUES (?, 1, ?)", (a, rounds._iso(rounds.utcnow())))
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1))
    conn.commit()
    assert fairness.quota_refusal(conn, a) == fairness.OVER_CONCURRENCY
    conn.execute("UPDATE tasks SET status = 'submitted' WHERE id = 't1'")
    conn.commit()
    assert fairness.quota_refusal(conn, a) is None


def test_the_budget_is_undecayed_and_a_month_is_a_calendar_month(conn, owner):
    """docs/13 §3.3. Decay belongs to fairness, which is about recency; a budget
    has to match what a human gets adding the month up by hand."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10)
    # Two hours, eight days ago -- decayed almost to nothing, but still spent.
    _task(conn, "t1", "ja", "submitted",
          leased_at=_iso_ago(days=8, hours=2), ended=_iso_ago(days=8))
    hours = fairness.month_hours(conn, a)
    if rounds.utcnow().day > 8:  # both stamps inside this calendar month
        assert hours == pytest.approx(2.0, rel=1e-3)
        # And the *share* of the same lease has decayed by 2^-8.
        assert fairness.recompute_shares(conn)[a] < 7200 * 0.01


def test_a_spent_budget_refuses_the_enqueue_with_an_answer(client, conn, owner):
    """docs/13 §3.4. Refusing here rather than letting the job queue and never
    lease is the difference between an answer and a mystery."""
    a, key = owner("a")
    conn.execute(
        "INSERT INTO submitter_quotas (user_id, monthly_task_hours, updated_at)"
        " VALUES (?, 0.5, ?)", (a, rounds._iso(rounds.utcnow())))
    _job(conn, "spent", a, 10)
    _task(conn, "t1", "spent", "submitted",
          leased_at=_iso_ago(hours=2), ended=_iso_ago(hours=1))
    conn.commit()

    r = client.post("/v1/jobs", headers=_hdr(key), json={
        "job_type": "batch_inference", "spec": {
            "model_ref": "hf://m", "shards": [{"ref": "s0", "rows": 8}],
            "output_prefix": "out/x", "prompt_template": "{input}",
            "decode": {"mode": "greedy", "max_new_tokens": 4},
            "output_schema": {"id": "str", "output": "str"},
        }})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    r = client.post(f"/v1/jobs/{job_id}/enqueue", headers=_hdr(key))
    assert r.status_code == 409
    assert "monthly budget" in r.json()["detail"]


def test_the_concurrency_cap_does_not_block_an_enqueue(client, conn, owner):
    """Transient by nature. Refusing an enqueue because four tasks happen to be
    running right now would be nonsense."""
    a, key = owner("a")
    conn.execute(
        "INSERT INTO submitter_quotas (user_id, max_concurrent_tasks, updated_at)"
        " VALUES (?, 1, ?)", (a, rounds._iso(rounds.utcnow())))
    _job(conn, "busy", a, 10)
    _task(conn, "t1", "busy", "leased", leased_at=_iso_ago(minutes=1))
    conn.commit()
    r = client.post("/v1/jobs", headers=_hdr(key), json={
        "job_type": "batch_inference", "spec": {
            "model_ref": "hf://m", "shards": [{"ref": "s0", "rows": 8}],
            "output_prefix": "out/y", "prompt_template": "{input}",
            "decode": {"mode": "greedy", "max_new_tokens": 4},
            "output_schema": {"id": "str", "output": "str"},
        }})
    job_id = r.json()["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/enqueue", headers=_hdr(key)).status_code == 200


def test_only_an_admin_sets_a_quota(client, conn, owner, admin):
    """docs/07 §5's argument about ``priority_rank``, applied: a submitter who
    could raise their own ceiling does not have a ceiling."""
    a, akey = owner("a")
    _, adminkey = admin
    r = client.post(f"/v1/admin/submitters/{a}/quota", headers=_hdr(akey),
                    json={"max_concurrent_tasks": 100})
    # Not 403: docs/08 -- an authenticated non-admin gets 404 on the whole
    # /v1/admin/* tree, so the admin API is not confirmed to exist.
    assert r.status_code == 404
    r = client.post(f"/v1/admin/submitters/{a}/quota", headers=_hdr(adminkey),
                    json={"max_concurrent_tasks": 4, "monthly_task_hours": 12.5})
    assert r.status_code == 200, r.text
    assert r.json()["max_concurrent_tasks"] == 4
    row = fairness.quota_for(conn, a)
    assert row["max_concurrent_tasks"] == 4 and row["monthly_task_hours"] == 12.5


def test_a_quota_can_be_cleared_back_to_uncapped(client, conn, owner, admin):
    a, _ = owner("a")
    _, adminkey = admin
    client.post(f"/v1/admin/submitters/{a}/quota", headers=_hdr(adminkey),
                json={"max_concurrent_tasks": 1})
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1))
    assert fairness.quota_refusal(conn, a) == fairness.OVER_CONCURRENCY
    client.post(f"/v1/admin/submitters/{a}/quota", headers=_hdr(adminkey), json={})
    assert fairness.quota_refusal(conn, a) is None


# ==========================================================================
# §4 Preemption
# ==========================================================================


def test_a_preempt_rides_the_existing_heartbeat_field(conn, owner):
    """docs/13 §4.1-§4.2. No new transport: the same ``cancel`` the job-level
    path uses, on a job that is still perfectly alive."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="running")
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1), worker_id=None)
    assert rounds.cancel_outstanding(conn, "t1") is None
    conn.execute("UPDATE tasks SET preempt_mode = 'soft' WHERE id = 't1'")
    conn.commit()
    assert rounds.cancel_outstanding(conn, "t1") == "soft"
    # And the job is untouched -- this is the case the old job-keyed query,
    # which read `jobs.status = 'cancelled'`, could not have seen at all.
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = 'ja'").fetchone()["status"] == "running"


def test_a_cancelled_job_still_beats_a_preempt_marker(conn, owner):
    """A task that is both: the job going away outranks a scheduling decision
    about it, so this lands terminal rather than back in the pool."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="cancelled")
    conn.execute("UPDATE jobs SET cancel_mode = 'hard' WHERE id = 'ja'")
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1),
          worker_id=None, preempt="soft")
    conn.commit()
    assert rounds.abandon(conn, "t1", None) == "cancelled"


def test_a_preempted_task_is_not_an_infraction(conn, owner, worker):
    """The identical argument that split ``cancelled`` out in docs/11 §3. The
    machine was told to stop by the scheduler and did nothing wrong; only
    ``expired`` and ``abandoned`` count against it (docs/09 5.2)."""
    wid = worker()
    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="running")
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1),
          worker_id=wid, preempt="soft")
    conn.commit()

    assert rounds.abandon(conn, "t1", wid) == "preempted"
    assert ledger._infraction_since(
        conn, wid, rounds._iso(rounds.utcnow() - timedelta(days=1))) is False


def test_a_preempted_task_goes_back_in_the_pool_and_burns_no_attempt(
        conn, owner, worker):
    """docs/13 §4.3-§4.4. ``cancelled`` is terminal and this must not be: the
    work is still wanted. And the attempt budget bounds *the shard's* failures,
    so letting the scheduler spend it would fail a shard after five decisions
    nobody made about that shard."""
    from ganymede.coordinator.app import _claim_static_task

    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="running")
    _task(conn, "t1", "ja", "preempted", attempts=3)
    conn.execute("UPDATE tasks SET input_ref_json = '{}' WHERE id = 't1'")
    conn.commit()

    job = conn.execute("SELECT * FROM jobs WHERE id = 'ja'").fetchone()

    class _JT:
        def inputs_for(self, row, store):
            return {}

    class _S:
        lease_duration_sec = 900

    spec, _ = _claim_static_task(conn, _JT(), None, job, worker(), _S())
    assert spec is not None
    row = conn.execute("SELECT status, attempts FROM tasks WHERE id = 't1'").fetchone()
    assert row["status"] == "leased"
    assert row["attempts"] == 3  # not 4


def test_an_ordinary_reserve_still_burns_an_attempt(conn, owner, worker):
    """The other half of the CASE. A shard that expired *is* on its own budget."""
    from ganymede.coordinator.app import _claim_static_task

    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="running")
    _task(conn, "t1", "ja", "expired", attempts=3)
    conn.execute("UPDATE tasks SET input_ref_json = '{}' WHERE id = 't1'")
    conn.commit()
    job = conn.execute("SELECT * FROM jobs WHERE id = 'ja'").fetchone()

    class _JT:
        def inputs_for(self, row, store):
            return {}

    class _S:
        lease_duration_sec = 900

    _claim_static_task(conn, _JT(), None, job, worker(), _S())
    assert conn.execute(
        "SELECT attempts FROM tasks WHERE id = 't1'").fetchone()["attempts"] == 4


def test_a_wedged_preempted_worker_expires_onto_preempted(conn, owner, worker):
    """The worker never acknowledged: no abandon call, just a lease running out.
    ``expire_leases`` has to make the same three-way distinction."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="running")
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(hours=2),
          ended=_iso_ago(hours=1), worker_id=worker(), preempt="soft")
    assert rounds.expire_leases(conn) == 1
    row = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert row["status"] == "preempted"
    # The marker is cleared on the way out: leaving it set would preempt
    # whichever machine picks the shard up next, on a decision made about a
    # different machine.
    assert row["preempt_mode"] is None
    assert row["worker_id"] is None


def test_the_three_expiry_arms_partition(conn, owner, worker):
    """Three expired leases at once, one for each arm. A single-task test cannot
    tell "landed on preempted" from "counted twice" or "the arms overlapped",
    and the arms are three UPDATEs over the same predicate -- exactly the shape
    that double-counts if one of them forgets an exclusion."""
    a, _ = owner("a")
    _job(conn, "live", a, 10, status="running")
    _job(conn, "dead", a, 10, status="cancelled")
    conn.execute("UPDATE jobs SET cancel_mode = 'soft' WHERE id = 'dead'")
    stale = dict(leased_at=_iso_ago(hours=2), ended=_iso_ago(hours=1))
    _task(conn, "p", "live", "leased", worker_id=worker("p"), preempt="soft", **stale)
    _task(conn, "c", "dead", "leased", worker_id=worker("c"), **stale)
    _task(conn, "e", "live", "leased", worker_id=worker("e"), **stale)
    conn.commit()

    assert rounds.expire_leases(conn) == 3
    got = {r["id"]: r["status"] for r in conn.execute(
        "SELECT id, status FROM tasks").fetchall()}
    assert got == {"p": "preempted", "c": "cancelled", "e": "expired"}


def test_a_preempted_task_on_a_cancelled_job_expires_as_cancelled(conn, owner,
                                                                   worker):
    """Both markers at once, on the wedged-worker path. The job going away
    outranks a scheduling decision about it, so this must land terminal --
    the preempt arm has to exclude it, or a cancelled job keeps a live task."""
    a, _ = owner("a")
    _job(conn, "dead", a, 10, status="cancelled")
    conn.execute("UPDATE jobs SET cancel_mode = 'hard' WHERE id = 'dead'")
    _task(conn, "t1", "dead", "leased", leased_at=_iso_ago(hours=2),
          ended=_iso_ago(hours=1), worker_id=worker(), preempt="soft")
    conn.commit()
    assert rounds.expire_leases(conn) == 1
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = 't1'").fetchone()["status"] == "cancelled"


def test_preempting_a_task_that_is_not_leased_says_so(client, conn, owner, admin):
    a, _ = owner("a")
    _, adminkey = admin
    _job(conn, "ja", a, 10)
    _task(conn, "t1", "ja", "planned")
    r = client.post("/v1/admin/tasks/t1/preempt", headers=_hdr(adminkey),
                    json={"mode": "soft"})
    assert r.status_code == 409
    assert "not leased" in r.json()["detail"]


def test_every_preemption_is_audited(conn, owner, worker):
    """docs/13 §4.4 declines to cap preemptions per task, because a cap would
    silently turn "this shard keeps getting pushed around" into "this shard
    failed". The audit row is how the pushing-around stays visible instead."""
    a, _ = owner("a")
    _job(conn, "ja", a, 10, status="running")
    wid = worker()
    _task(conn, "t1", "ja", "leased", leased_at=_iso_ago(minutes=1), worker_id=wid)
    assert fairness.preempt(conn, "t1", "hard", cause="admin:root")
    row = conn.execute(
        "SELECT detail_json FROM audit WHERE event = 'task_preempted'").fetchone()
    detail = json.loads(row["detail_json"])
    assert detail["task_id"] == "t1" and detail["mode"] == "hard"
    assert detail["worker_id"] == wid and detail["cause"] == "admin:root"


# --- automatic preemption (docs/13 §4.6) -------------------------------------


def test_autopreempt_is_off_by_default(conn, owner, worker, settings):
    a, _ = owner("a")
    _job(conn, "starved", a, 10)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = 'starved'",
                 (_iso_ago(hours=5),))
    _job(conn, "fat", a, 90, status="running")
    _task(conn, "t1", "fat", "leased", leased_at=_iso_ago(hours=1),
          worker_id=worker())
    assert settings.autopreempt is False
    assert fairness.autopreempt(conn, settings) is None


def test_autopreempt_takes_the_longest_running_low_rank_lease(
        conn, owner, worker, settings):
    import dataclasses

    a, _ = owner("a")
    b, _ = owner("b")
    _job(conn, "starved", a, 10)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = 'starved'",
                 (_iso_ago(hours=5),))
    _job(conn, "fat", b, 90, status="running")
    _task(conn, "old", "fat", "leased", leased_at=_iso_ago(hours=3),
          worker_id=worker("old"))
    _task(conn, "new", "fat", "leased", leased_at=_iso_ago(minutes=5),
          worker_id=worker("new"))
    conn.commit()

    on = dataclasses.replace(settings, autopreempt=True)
    assert fairness.autopreempt(conn, on) == "old"
    assert conn.execute(
        "SELECT preempt_mode FROM tasks WHERE id = 'old'").fetchone()[0] == "soft"
    # One per sweep. The second call finds `old` already marked and moves to the
    # only remaining candidate rather than re-marking.
    assert fairness.autopreempt(conn, on) == "new"


def test_autopreempt_will_not_take_a_machine_that_refused_the_starved_job(
        conn, owner, worker, settings):
    """Step 3 of the policy, and the first thing to read ``worker_eligibility``
    as an *input* rather than a record. Absence of a refusal is weaker than a
    fresh evaluation, but it fails safe: no candidate, no preemption."""
    import dataclasses

    a, _ = owner("a")
    b, _ = owner("b")
    _job(conn, "starved", a, 10)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = 'starved'",
                 (_iso_ago(hours=5),))
    _job(conn, "fat", b, 90, status="running")
    wid = worker()
    _task(conn, "t1", "fat", "leased", leased_at=_iso_ago(hours=3), worker_id=wid)
    conn.execute(
        """INSERT INTO worker_eligibility (worker_id, job_id, outcome, reason,
                                           checked_at)
           VALUES (?, 'starved', 'refused', 'vram_mb 8000 < 24000', ?)""",
        (wid, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    on = dataclasses.replace(settings, autopreempt=True)
    assert fairness.autopreempt(conn, on) is None


def test_autopreempt_respects_the_rank_margin(conn, owner, worker, settings):
    """A job one rank away is not worth stopping work for."""
    import dataclasses

    a, _ = owner("a")
    b, _ = owner("b")
    _job(conn, "starved", a, 10)
    conn.execute("UPDATE jobs SET created_at = ? WHERE id = 'starved'",
                 (_iso_ago(hours=5),))
    _job(conn, "near", b, 15, status="running")
    _task(conn, "t1", "near", "leased", leased_at=_iso_ago(hours=3),
          worker_id=worker())
    conn.commit()
    on = dataclasses.replace(settings, autopreempt=True)
    assert fairness.autopreempt(conn, on) is None


def test_a_job_that_has_only_just_queued_is_not_starved(
        conn, owner, worker, settings):
    import dataclasses

    a, _ = owner("a")
    b, _ = owner("b")
    _job(conn, "fresh", a, 10)
    _job(conn, "fat", b, 90, status="running")
    _task(conn, "t1", "fat", "leased", leased_at=_iso_ago(hours=3),
          worker_id=worker())
    conn.commit()
    on = dataclasses.replace(settings, autopreempt=True)
    assert fairness.autopreempt(conn, on) is None
