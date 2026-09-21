"""Known-answer probes and reputation weighting (docs/13 §5, §6).

The anti-fraud pair, and the two halves want different kinds of test.

§5 is a mechanism whose whole value is that it cannot be detected, so the tests
are about *indistinguishability* and about the seams a probe must not leak
through: it must not decide whether a job is finished, must not be recycled as
ordinary work when it fails, and must not be checked against the machine that
produced the answer it is being checked against.

§6 is the one numerically live switch in Phase D. Two tests, and they check
different things: that a uniform reputation vector is *exactly* inert (the
formula is a reweighting, not a rescaling), and that the flag being off leaves
the default path alone. The interesting case -- mixed reputations across a real
cohort -- is not something a unit test can settle, and docs/13 §6.2 says so.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import timedelta

import pytest

from ganymede.coordinator import ledger, rounds, spotcheck
from ganymede.jobtypes.collab_lora_finetune import aggregate


def _iso_ago(**kw) -> str:
    return rounds._iso(rounds.utcnow() - timedelta(**kw))


class _AlwaysRoll:
    """An rng that always fires, so the rate is the only thing under test."""

    def random(self):
        return 0.0


class _NeverRoll:
    def random(self):
        return 1.0


@pytest.fixture
def owner(conn, make_contributor):
    cid, _ = make_contributor(name="submitter")
    conn.execute(
        "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
        (cid, "approved", rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return cid


@pytest.fixture
def worker(conn, make_contributor):
    def _make(name: str = "w"):
        cid, _ = make_contributor(name=f"c-{name}-{uuid.uuid4().hex[:6]}")
        wid = uuid.uuid4().hex
        now = rounds._iso(rounds.utcnow())
        conn.execute(
            """INSERT INTO workers (id, contributor_id, compute_profile_json,
                                    first_seen, last_seen)
               VALUES (?, ?, '{}', ?, ?)""",
            (wid, cid, now, now),
        )
        conn.commit()
        return wid
    return _make


def _spec() -> str:
    return json.dumps({
        "model_ref": "hf://m",
        "shards": [{"ref": "s0", "rows": 8}],
        "output_prefix": "out/x",
        "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
        "redundancy": {"agree_on": "output", "sample_rows": 4},
    })


@pytest.fixture
def job(conn, owner):
    conn.execute(
        """INSERT INTO jobs (id, owner_id, job_type, spec_json, status,
                             priority_rank, constraints_json, created_at)
           VALUES ('j1', ?, 'batch_inference', ?, 'running', 10, '{}', ?)""",
        (owner, _spec(), rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return conn.execute("SELECT * FROM jobs WHERE id = 'j1'").fetchone()


def _accepted_task(conn, store, task_id: str, worker_id: str, rows: list[dict],
                   *, accepted: int = 1) -> str:
    """A finished shard with its artifact in the store."""
    key = f"out/{task_id}.jsonl"
    store.put_bytes(key, "\n".join(json.dumps(r) for r in rows).encode())
    now = rounds._iso(rounds.utcnow())
    conn.execute(
        """INSERT INTO tasks (id, job_id, buckets_json, input_ref_json,
                              local_steps, status, worker_id, leased_at,
                              attempts, created_at)
           VALUES (?, 'j1', '[]', '{"ref": "s0"}', 8, 'submitted', ?, ?, 1, ?)""",
        (task_id, worker_id, now, now),
    )
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
                                    accepted, received_at)
           VALUES (?, ?, 8, ?, ?)""",
        (task_id, key, accepted, now),
    )
    conn.commit()
    return key


ROWS = [{"id": str(i), "output": f"answer-{i}"} for i in range(8)]
WRONG = [{"id": str(i), "output": f"garbage-{i}"} for i in range(8)]


class _Settings:
    spotcheck_rate = 1.0
    lease_duration_sec = 900


# ==========================================================================
# §5.1-§5.3 Issue
# ==========================================================================


def test_a_type_that_does_not_opt_in_is_never_probed(conn, store, owner, job, worker):
    """docs/13 §5.2, and the hole ``contained_batch`` opened.

    Probes are issued from the static-task reserve path, and until a third type
    existed ``batch_inference`` was the only type on it -- which made "static"
    an accurate proxy for "deterministic" *by accident*. ``contained_batch`` is
    static too and runs an image the coordinator did not build, so the proxy
    stopped holding: an honest machine on a job that samples, threads or stamps
    a timestamp would have been convicted, and ``judge`` would have reached that
    verdict through ``batch_inference``'s comparator regardless of the job's own
    type. docs/09 §5.1 rates a failed probe the largest single penalty there is.

    Gating issuance rather than making ``judge`` polymorphic is the smaller fix,
    and it makes that hardcoded comparator correct *by construction*: nothing
    but ``batch_inference`` is ever judged.
    """
    conn.execute(
        """INSERT INTO jobs (id, owner_id, job_type, spec_json, status,
                             priority_rank, constraints_json, created_at)
           VALUES ('j2', ?, 'contained_batch', '{}', 'running', 10, '{}', ?)""",
        (owner, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    contained = conn.execute("SELECT * FROM jobs WHERE id = 'j2'").fetchone()

    # A source exists and the rate is 1.0, so the *only* thing that can stop a
    # probe here is the type gate. (``job`` builds the j1 row ``_accepted_task``
    # hangs its task off; the update below moves it to the contained job.)
    _accepted_task(conn, store, "t-src", worker("src"), ROWS)
    conn.execute("UPDATE tasks SET job_id = 'j2' WHERE id = 't-src'")
    conn.commit()

    assert spotcheck.maybe_issue(conn, contained, worker("probe"), _Settings()) is None


def test_the_deterministic_type_still_is_probed(conn, store, job, worker):
    """The other half: the gate must not have turned the feature off. This is
    the same call as above against the type that opts in."""
    _accepted_task(conn, store, "t-src", worker("src"), ROWS)
    assert spotcheck.maybe_issue(conn, job, worker("probe"), _Settings()) is not None


def test_the_default_rate_never_issues_a_probe(conn, store, job, worker):
    """Off is off. ``spotcheck_rate`` defaults to 0.0 and the roll is skipped
    entirely rather than rolled against zero."""
    class _Off:
        spotcheck_rate = 0.0
        lease_duration_sec = 900

    _accepted_task(conn, store, "t1", worker("a"), ROWS)
    assert spotcheck.maybe_issue(conn, job, worker("b"), _Off(),
                                 rng=_AlwaysRoll()) is None
    assert conn.execute(
        "SELECT COUNT(*) c FROM spot_check_issues").fetchone()["c"] == 0


def test_a_probe_is_a_re_issue_of_an_accepted_shard(conn, store, job, worker):
    """docs/13 §5.1. The payload is the source's, verbatim -- that is what makes
    it indistinguishable, because there is nothing to distinguish."""
    a = worker("a")
    _accepted_task(conn, store, "t1", a, ROWS)
    probe_id = spotcheck.maybe_issue(conn, job, worker("b"), _Settings(),
                                     rng=_AlwaysRoll())
    assert probe_id is not None
    probe = conn.execute("SELECT * FROM tasks WHERE id = ?", (probe_id,)).fetchone()
    source = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert probe["input_ref_json"] == source["input_ref_json"]
    assert probe["buckets_json"] == source["buckets_json"]
    assert probe["local_steps"] == source["local_steps"]
    assert probe["status"] == "leased" and probe["attempts"] == 1


def test_a_machine_is_never_checked_against_itself(conn, store, job, worker):
    """Load-bearing, not a nicety: a machine compared with its own earlier
    answer agrees with itself, so the probe would measure determinism rather
    than honesty."""
    a = worker("a")
    _accepted_task(conn, store, "t1", a, ROWS)
    assert spotcheck.maybe_issue(conn, job, a, _Settings(),
                                 rng=_AlwaysRoll()) is None


def test_nothing_accepted_yet_means_nothing_to_check_against(conn, store, job,
                                                              worker):
    """Not an error. Early in a job there is no trusted answer yet."""
    _accepted_task(conn, store, "t1", worker("a"), ROWS, accepted=0)
    assert spotcheck.maybe_issue(conn, job, worker("b"), _Settings(),
                                 rng=_AlwaysRoll()) is None


def test_a_probe_is_never_the_source_of_another_probe(conn, store, job, worker):
    """Chaining would compare an unverified answer against another unverified
    answer and call the result a known-answer check."""
    a, b, c = worker("a"), worker("b"), worker("c")
    _accepted_task(conn, store, "t1", a, ROWS)
    probe_id = spotcheck.maybe_issue(conn, job, b, _Settings(), rng=_AlwaysRoll())
    # Accept the probe's own submission, so it looks like a candidate source.
    key = f"out/{probe_id}.jsonl"
    store.put_bytes(key, "\n".join(json.dumps(r) for r in ROWS).encode())
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
                                    accepted, received_at)
           VALUES (?, ?, 8, 1, ?)""",
        (probe_id, key, rounds._iso(rounds.utcnow())),
    )
    conn.execute("UPDATE tasks SET status = 'submitted' WHERE id = ?", (probe_id,))
    conn.commit()
    source = spotcheck._source_for(conn, "j1", c)
    assert source["id"] == "t1"


def test_the_rate_is_the_probability(conn, store, job, worker):
    """A boring test with a purpose: the roll is ``rng.random() < rate``, and
    getting that comparison backwards would make 0.1 mean 90%."""
    class _Tenth:
        spotcheck_rate = 0.1
        lease_duration_sec = 900

    _accepted_task(conn, store, "t1", worker("a"), ROWS)
    b = worker("b")
    rng = random.Random(7)
    hits = sum(
        1 for _ in range(400)
        if spotcheck.maybe_issue(conn, job, b, _Tenth(), rng=rng) is not None
    )
    assert 20 < hits < 70  # ~40 expected


# ==========================================================================
# §5.3 Judge
# ==========================================================================


def _issue_and_answer(conn, store, job, worker, rows) -> str:
    a = worker("a")
    _accepted_task(conn, store, "t1", a, ROWS)
    probe_id = spotcheck.maybe_issue(conn, job, worker("b"), _Settings(),
                                     rng=_AlwaysRoll())
    key = f"out/{probe_id}.jsonl"
    store.put_bytes(key, "\n".join(json.dumps(r) for r in rows).encode())
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
                                    accepted, received_at)
           VALUES (?, ?, 8, 1, ?)""",
        (probe_id, key, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return probe_id


def test_the_same_answer_passes(conn, store, job, worker):
    probe_id = _issue_and_answer(conn, store, job, worker, ROWS)
    assert spotcheck.judge(conn, store, probe_id, job) == spotcheck.PASSED


def test_a_different_answer_fails(conn, store, job, worker):
    probe_id = _issue_and_answer(conn, store, job, worker, WRONG)
    assert spotcheck.judge(conn, store, probe_id, job) == spotcheck.FAILED
    assert conn.execute(
        "SELECT 1 FROM audit WHERE event = 'spot_check_failed'").fetchone()


def test_an_unreadable_artifact_is_not_a_conviction(conn, store, job, worker):
    """"We could not look" is not "we looked and it was wrong", and this is the
    largest penalty in the system."""
    probe_id = _issue_and_answer(conn, store, job, worker, ROWS)
    conn.execute(
        "UPDATE submissions SET artifact_ref = 'gone/missing' WHERE task_id = ?",
        (probe_id,))
    conn.commit()
    assert spotcheck.judge(conn, store, probe_id, job) is None
    assert conn.execute(
        "SELECT outcome FROM spot_check_issues WHERE task_id = ?",
        (probe_id,)).fetchone()["outcome"] is None


def test_judging_an_ordinary_task_is_a_no_op(conn, store, job, worker):
    _accepted_task(conn, store, "t1", worker("a"), ROWS)
    assert spotcheck.judge(conn, store, "t1", job) is None


def test_a_probe_is_judged_once(conn, store, job, worker):
    probe_id = _issue_and_answer(conn, store, job, worker, ROWS)
    assert spotcheck.judge(conn, store, probe_id, job) == spotcheck.PASSED
    assert spotcheck.judge(conn, store, probe_id, job) is None


def test_a_probe_that_never_came_back_is_voided_not_left_hanging(conn, store,
                                                                  job, worker):
    """It says nothing about the machine -- the machine went away, and the
    ordinary abandoned/expired path already counts that. But it must not sit at
    NULL forever."""
    _accepted_task(conn, store, "t1", worker("a"), ROWS)
    probe_id = spotcheck.maybe_issue(conn, job, worker("b"), _Settings(),
                                     rng=_AlwaysRoll())
    conn.execute("UPDATE tasks SET status = 'expired' WHERE id = ?", (probe_id,))
    conn.commit()
    assert spotcheck.void_stale(conn) == 1
    assert conn.execute(
        "SELECT outcome FROM spot_check_issues WHERE task_id = ?",
        (probe_id,)).fetchone()["outcome"] == spotcheck.VOID
    # And a voided probe is not an outcome anything counts.
    assert spotcheck.outcomes_for(conn, "anyone", _iso_ago(days=1)) == (0, 0)


# ==========================================================================
# §5.4 A probe leaks into nothing
# ==========================================================================


def test_a_failed_probe_does_not_wedge_the_job(conn, store, job, worker):
    """The probe never passes, so a job whose completion gate counted it would
    never finish. docs/13 §5.4's exclusion clause, asserted end to end."""
    from ganymede.coordinator import close

    probe_id = _issue_and_answer(conn, store, job, worker, WRONG)
    conn.execute(
        "UPDATE submissions SET accepted = 0 WHERE task_id = ?", (probe_id,))
    conn.execute("UPDATE tasks SET status = 'submitted' WHERE id = ?", (probe_id,))
    conn.commit()

    close.advance_job(conn, store, job_id="j1", settings=None)
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = 'j1'").fetchone()["status"] == "done"


def test_a_failed_probe_is_not_recycled_as_ordinary_work(conn, store, job, worker):
    """A failed probe is a ``submitted`` row with ``accepted = 0``, which is
    exactly what the re-serve recycles. Its shard was accepted long ago, so
    handing it out again would burn attempts re-doing finished work because one
    machine got it wrong."""
    from ganymede.coordinator.app import _claim_static_task

    probe_id = _issue_and_answer(conn, store, job, worker, WRONG)
    conn.execute(
        "UPDATE submissions SET accepted = 0 WHERE task_id = ?", (probe_id,))
    conn.execute("UPDATE tasks SET status = 'submitted' WHERE id = ?", (probe_id,))
    conn.commit()

    class _JT:
        def inputs_for(self, row, store):
            return {}

    class _Off:
        spotcheck_rate = 0.0
        lease_duration_sec = 900

    spec, _ = _claim_static_task(conn, _JT(), store, job, worker("c"), _Off(),
                                 free_devices=[0])
    assert spec is None


# ==========================================================================
# §5.5 Into the reputation score
# ==========================================================================


def _probe_outcome(conn, worker_id: str, outcome: str, task_id: str) -> None:
    now = rounds._iso(rounds.utcnow())
    conn.execute(
        """INSERT INTO tasks (id, job_id, buckets_json, local_steps, status,
                              worker_id, leased_at, attempts, created_at)
           VALUES (?, 'j1', '[]', 8, 'submitted', ?, ?, 1, ?)""",
        (task_id, worker_id, now, now),
    )
    conn.execute(
        """INSERT INTO spot_check_issues (task_id, source_task_id, issued_at,
                                          outcome, decided_at)
           VALUES (?, ?, ?, ?, ?)""",
        (task_id, task_id, now, outcome, now),
    )
    conn.commit()


def test_one_failed_probe_forces_probation_on_a_spotless_machine(conn, job,
                                                                  worker):
    """docs/09 §5.3: *one* spot-check failure, regardless of score. A machine
    with months of clean work has enough headroom that the multiplier alone
    would leave it in good standing -- which is exactly the machine this rule
    exists for."""
    wid = worker("a")
    conn.execute("UPDATE workers SET reputation = 0.99, standing = 'good' "
                 "WHERE id = ?", (wid,))
    conn.commit()
    _probe_outcome(conn, wid, spotcheck.FAILED, "p1")
    ledger.recompute_reputation(conn, wid)
    row = conn.execute("SELECT reputation, standing FROM workers WHERE id = ?",
                       (wid,)).fetchone()
    assert row["standing"] == "probation"
    assert row["reputation"] == pytest.approx(0.99 * 0.5 - 0.20)


def test_a_second_failure_on_probation_revokes(conn, job, worker):
    wid = worker("a")
    conn.execute("UPDATE workers SET reputation = 0.80, standing = 'probation' "
                 "WHERE id = ?", (wid,))
    conn.commit()
    _probe_outcome(conn, wid, spotcheck.FAILED, "p1")
    ledger.recompute_reputation(conn, wid)
    assert conn.execute("SELECT standing FROM workers WHERE id = ?",
                        (wid,)).fetchone()["standing"] == "revoked"


def test_a_failed_probe_costs_more_than_a_minority_disagreement(conn, job,
                                                                 worker):
    """A probe is compared against an already-accepted answer, so there is no
    question which side is wrong. A minority is merely outvoted."""
    probe_w, minority_w = worker("p"), worker("m")
    for wid in (probe_w, minority_w):
        conn.execute("UPDATE workers SET reputation = 0.80 WHERE id = ?", (wid,))
    _probe_outcome(conn, probe_w, spotcheck.FAILED, "p1")
    conn.execute(
        "INSERT INTO audit (at, event, detail_json) VALUES (?, ?, ?)",
        (rounds._iso(rounds.utcnow()), "attempt_group_disagreement",
         json.dumps({"attempt_group": "g1",
                     "members": [{"task": "x", "worker_id": minority_w}]})),
    )
    conn.commit()
    ledger.recompute_reputation(conn, probe_w)
    ledger.recompute_reputation(conn, minority_w)
    probe_score = conn.execute("SELECT reputation FROM workers WHERE id = ?",
                               (probe_w,)).fetchone()["reputation"]
    minority_score = conn.execute("SELECT reputation FROM workers WHERE id = ?",
                                  (minority_w,)).fetchone()["reputation"]
    assert probe_score < minority_score
    assert minority_score < 0.80  # but still a hard hit


def test_recovery_from_probation_needs_a_passed_probe(conn, job, worker):
    """docs/09 §5.3. This is the clause that stops probation being waited out in
    silence: a machine that stops working stops being able to earn its way
    back."""
    wid = worker("a")
    conn.execute("UPDATE workers SET reputation = 0.95, standing = 'probation' "
                 "WHERE id = ?", (wid,))
    conn.commit()
    ledger.recompute_reputation(conn, wid)
    assert conn.execute("SELECT standing FROM workers WHERE id = ?",
                        (wid,)).fetchone()["standing"] == "probation"

    _probe_outcome(conn, wid, spotcheck.PASSED, "p1")
    ledger.recompute_reputation(conn, wid)
    assert conn.execute("SELECT standing FROM workers WHERE id = ?",
                        (wid,)).fetchone()["standing"] == "good"


def test_a_passed_probe_is_worth_twice_a_clean_submission(conn, job, worker):
    """Corroborated, not merely well-formed."""
    assert ledger.REP_PROBE_PASS_STEP == 2 * ledger.REP_CLEAN_STEP
    wid = worker("a")
    conn.execute("UPDATE workers SET reputation = 0.50 WHERE id = ?", (wid,))
    conn.commit()
    _probe_outcome(conn, wid, spotcheck.PASSED, "p1")
    ledger.recompute_reputation(conn, wid)
    assert conn.execute("SELECT reputation FROM workers WHERE id = ?",
                        (wid,)).fetchone()["reputation"] == pytest.approx(0.54)


# ==========================================================================
# §6 Reputation-weighted aggregation
# ==========================================================================


def test_a_uniform_reputation_vector_is_exactly_inert():
    """The test that says this is a *reweighting* and not a rescaling. Every
    machine enrols at REP_ENROLL, so a cohort that has all enrolled and none
    diverged has a constant vector -- and a constant factor cancels in the
    normalisation."""
    steps = [100, 250, 40, 700]
    keys = ["a", "b"]
    plain = aggregate.dense_weights(steps, keys)
    for c in (ledger.REP_ENROLL, 1.0, 0.37):
        weighted = aggregate.dense_weights(steps, keys,
                                           reputation=[c] * len(steps))
        for p, w in zip(plain, weighted):
            for k in keys:
                assert w[k] == pytest.approx(p[k], rel=1e-12)


def test_none_is_the_same_as_not_passing_it():
    steps = [10, 20, 30]
    assert aggregate.dense_weights(steps, ["a"]) == \
        aggregate.dense_weights(steps, ["a"], reputation=None)


def test_a_distrusted_machine_carries_less(conn):
    steps = [100, 100, 100]
    keys = ["a"]
    w = aggregate.dense_weights(steps, keys, reputation=[1.0, 1.0, 0.1])
    assert w[2]["a"] < w[0]["a"]
    assert sum(x["a"] for x in w) == pytest.approx(1.0)


def test_a_cohort_of_zeros_falls_back_rather_than_killing_the_round():
    """Reachable -- ``recompute_reputation`` floors at 0.0 -- and an aggregation
    nobody trusts still beats no aggregation. The machines are already excluded
    from accrual by their standing."""
    steps = [10, 20, 30]
    keys = ["a"]
    fell_back = aggregate.dense_weights(steps, keys, reputation=[0.0, 0.0, 0.0])
    assert fell_back == aggregate.dense_weights(steps, keys)


def test_a_majority_zero_cohort_does_not_wedge_the_round():
    """The gap between "every machine is at zero" (handled above) and "one is
    not" -- and the second one used to be a ``ZeroDivisionError``.

    With at least half the cohort at reputation 0.0 by sorted position, the
    median *share* is 0.0, so ``limit = cap * 0`` clamps every share to zero
    and the renormalisation divides by zero. That exception leaves
    ``reduce_close`` and reaches ``close_round``, which reopens the round and
    re-raises -- so the round then fails again on the next submit or claim, and
    every one after that. A wedged round, from a cohort that is merely
    lopsided rather than broken.

    Reachable for real: ``recompute_reputation`` floors at 0.0, so a mixed
    cohort of one honest machine and two floored ones is an ordinary state.
    """
    keys = ["a"]
    w = aggregate.dense_weights([100, 100, 100], keys, cap=2.0,
                                reputation=[0.0, 0.0, 1.0])
    assert sum(x["a"] for x in w) == pytest.approx(1.0)
    # The machines at zero carry nothing; the one that is trusted carries it all.
    assert w[0]["a"] == pytest.approx(0.0)
    assert w[1]["a"] == pytest.approx(0.0)
    assert w[2]["a"] == pytest.approx(1.0)


def test_a_zero_median_does_not_disable_the_cap_for_ordinary_cohorts():
    """The skip above is scoped to the zero-median case only -- a cohort whose
    median is positive must still be capped, or the fix would have quietly
    turned the dominance cap off."""
    keys = ["a"]
    w = aggregate.dense_weights([100, 100, 10_000], keys, cap=2.0,
                                reputation=[1.0, 1.0, 1.0])
    shares = [x["a"] for x in w]
    assert sum(shares) == pytest.approx(1.0)
    # Raw shares would be ~[0.0098, 0.0098, 0.98]; the cap holds it to 2x the
    # median instead of letting one machine carry the round.
    assert shares[2] == pytest.approx(0.5)
    assert max(shares) <= 2.0 * sorted(shares)[len(shares) // 2] + 1e-9


def test_an_even_cohort_with_half_at_zero_is_still_capped_normally():
    """An even split averages the two middle shares, so the median is positive
    and the ordinary cap path runs -- included so the boundary between the two
    behaviours is pinned rather than incidental."""
    keys = ["a"]
    w = aggregate.dense_weights([100, 100, 100, 100], keys, cap=2.0,
                                reputation=[0.0, 0.0, 1.0, 1.0])
    shares = [x["a"] for x in w]
    assert sum(shares) == pytest.approx(1.0)
    assert shares[0] == pytest.approx(0.0)
    assert shares[2] == pytest.approx(0.5)


def test_a_mismatched_reputation_vector_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError, match="2 entries for 3 workers"):
        aggregate.dense_weights([1, 2, 3], ["a"], reputation=[1.0, 1.0])


def test_the_dominance_cap_still_binds_on_the_weighted_shares():
    """The cap means "no worker carries more than cap x the median" -- it is
    just measuring a contribution that now accounts for trust."""
    steps = [10, 10, 10, 10, 1000]
    w = aggregate.dense_weights(steps, ["a"], cap=2.0,
                                reputation=[0.2, 0.2, 0.2, 0.2, 1.0])
    shares = sorted(x["a"] for x in w)
    median = shares[len(shares) // 2]
    assert max(shares) <= 2.0 * median + 1e-9
