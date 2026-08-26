"""Every invariant needs a case that actually trips it.

A checker whose violations have never been observed firing is decoration: the
first time one matters is the worst time to discover the query was wrong. So
each check here gets a database put deliberately into the broken state, and one
that is merely *unusual* -- the transients that must not be reported.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

from ganymede.coordinator import invariants, rounds


def _worker(conn, contributor_id, worker_id="w1"):
    now = rounds._iso(rounds.utcnow())
    conn.execute(
        """INSERT INTO workers (id, contributor_id, compute_profile_json, first_seen, last_seen)
           VALUES (?, ?, '{}', ?, ?)""",
        (worker_id, contributor_id, now, now),
    )
    return worker_id


def _task(conn, run_id, worker_id, buckets, status="leased", round_idx=0, expires_in=900):
    task_id = uuid.uuid4().hex
    expires = (
        None if expires_in is None
        else rounds._iso(rounds.utcnow() + timedelta(seconds=expires_in))
    )
    conn.execute(
        """INSERT INTO tasks (id, run_id, round_idx, buckets_json, local_steps,
                              status, worker_id, lease_expires_at, created_at)
           VALUES (?, ?, ?, ?, 10, ?, ?, ?, ?)""",
        (task_id, run_id, round_idx, json.dumps(buckets), status, worker_id,
         expires, rounds._iso(rounds.utcnow())),
    )
    return task_id


def _checks(conn, **kw) -> set[str]:
    return {v.check for v in invariants.check(conn, **kw)}


def test_a_healthy_run_reports_nothing(conn, seeded_run, make_contributor):
    run_id = seeded_run()
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid), [0, 1])

    assert invariants.check(conn) == []


# --------------------------------------------------------------------------
# double_lease
# --------------------------------------------------------------------------


def test_two_live_leases_for_one_worker_in_one_round_is_a_violation(
    conn, seeded_run, make_contributor
):
    run_id = seeded_run()
    cid, _ = make_contributor()
    worker = _worker(conn, cid)
    _task(conn, run_id, worker, [0])
    _task(conn, run_id, worker, [1])

    assert "double_lease" in _checks(conn)


def test_one_lease_per_round_across_two_rounds_is_fine(conn, seeded_run, make_contributor):
    """A worker that worked round 0 and now holds round 1 is the normal case --
    grouping on worker alone would report every healthy multi-round run."""
    run_id = seeded_run()
    cid, _ = make_contributor()
    worker = _worker(conn, cid)
    _task(conn, run_id, worker, [0], status="submitted", round_idx=0)
    _task(conn, run_id, worker, [1], round_idx=1)

    assert "double_lease" not in _checks(conn)


# --------------------------------------------------------------------------
# repeat_before_coverage
# --------------------------------------------------------------------------


def test_retraining_one_bucket_while_another_goes_untouched_is_a_violation(
    conn, seeded_run, make_contributor
):
    run_id = seeded_run(num_buckets=8)
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid, "w1"), [3, 4])
    _task(conn, run_id, _worker(conn, cid, "w2"), [4, 5])

    violation = next(v for v in invariants.check(conn) if v.check == "repeat_before_coverage")
    assert "1 bucket(s) more than once" in violation.detail


def test_overlap_with_nowhere_else_to_go_is_pigeonholes_not_a_bug(
    conn, seeded_run, make_contributor
):
    """A worker whose budget wants every bucket is assigned every bucket, and
    the next worker in that round has no unheld shard left to take. Reporting
    that would flag a design consequence as a fault."""
    run_id = seeded_run(num_buckets=4)
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid, "w1"), [0, 1, 2, 3])
    _task(conn, run_id, _worker(conn, cid, "w2"), [0, 1, 2, 3])

    assert "repeat_before_coverage" not in _checks(conn)


def test_a_bucket_reissued_after_an_expiry_is_not_a_repeat(
    conn, seeded_run, make_contributor
):
    """Expired and abandoned shards are *meant* to come back round. Counting
    them as held would make every recovered worker failure look like a bug."""
    run_id = seeded_run(num_buckets=8)
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid, "w1"), [7], status="expired")
    _task(conn, run_id, _worker(conn, cid, "w2"), [7], status="abandoned")
    _task(conn, run_id, _worker(conn, cid, "w3"), [7])

    assert "repeat_before_coverage" not in _checks(conn)


def test_the_same_bucket_in_two_different_rounds_is_not_a_repeat(
    conn, seeded_run, make_contributor
):
    """Buckets are meant to be revisited across rounds -- that is what
    times_trained counts. Only within one round is a repeat suspicious."""
    run_id = seeded_run(num_buckets=8)
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid, "w1"), [2], status="submitted", round_idx=0)
    _task(conn, run_id, _worker(conn, cid, "w2"), [2], round_idx=1)

    assert "repeat_before_coverage" not in _checks(conn)


# --------------------------------------------------------------------------
# stuck_lease
# --------------------------------------------------------------------------


def test_a_lease_long_past_expiry_is_stuck(conn, seeded_run, make_contributor):
    run_id = seeded_run()
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid), [0], expires_in=-3600)

    assert "stuck_lease" in _checks(conn)


def test_a_lease_just_past_expiry_is_a_transient_not_a_violation(
    conn, seeded_run, make_contributor
):
    """Leases are reclaimed lazily, on the next claim. On a fleet with nothing
    running they sit expired by design, and alerting on that would fire nightly."""
    run_id = seeded_run()
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid), [0], expires_in=-30)

    assert "stuck_lease" not in _checks(conn)


# --------------------------------------------------------------------------
# stuck_close
# --------------------------------------------------------------------------


def test_a_round_wedged_in_closing_is_a_violation(conn, seeded_run):
    """The failure with no symptom of its own: closing is claimed by exactly one
    caller and released only when that caller finishes, so a close that dies
    partway leaves every worker getting 204 forever and nothing logging why."""
    run_id = seeded_run()
    past = rounds._iso(rounds.utcnow() - timedelta(seconds=3600))
    conn.execute(
        "UPDATE rounds SET status = 'closing', opened_at = ? WHERE run_id = ?",
        (past, run_id),
    )

    assert "stuck_close" in _checks(conn)


def test_a_round_that_just_started_closing_is_not_reported(conn, seeded_run):
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET status = 'closing' WHERE run_id = ?", (run_id,))

    assert "stuck_close" not in _checks(conn)


# --------------------------------------------------------------------------
# the rest
# --------------------------------------------------------------------------


def test_a_closed_round_with_accepted_work_and_no_adapter_is_a_broken_chain(
    conn, seeded_run, make_contributor
):
    run_id = seeded_run()
    cid, _ = make_contributor()
    task_id = _task(conn, run_id, _worker(conn, cid), [0], status="submitted")
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed, accepted, received_at)
           VALUES (?, 'a.safetensors', 10, 1, ?)""",
        (task_id, rounds._iso(rounds.utcnow())),
    )
    conn.execute(
        "UPDATE rounds SET status = 'closed', result_adapter_ref = NULL WHERE run_id = ?",
        (run_id,),
    )

    assert "no_result_adapter" in _checks(conn)


def test_a_closed_round_that_accepted_nothing_published_nothing_and_that_is_fine(
    conn, seeded_run
):
    """An empty round is the expected overnight state, not an incident."""
    run_id = seeded_run()
    conn.execute(
        "UPDATE rounds SET status = 'closed', result_adapter_ref = NULL WHERE run_id = ?",
        (run_id,),
    )

    assert "no_result_adapter" not in _checks(conn)


def test_an_active_run_pointing_at_a_round_that_does_not_exist_is_a_violation(
    conn, seeded_run
):
    run_id = seeded_run()
    conn.execute("UPDATE runs SET current_round = 7 WHERE id = ?", (run_id,))

    assert "dangling_current_round" in _checks(conn)


def test_a_finished_run_pointing_past_its_last_round_is_not_a_violation(conn, seeded_run):
    run_id = seeded_run()
    conn.execute("UPDATE runs SET current_round = 7, status = 'done' WHERE id = ?", (run_id,))

    assert "dangling_current_round" not in _checks(conn)


def test_a_submission_against_an_abandoned_lease_is_a_violation(
    conn, seeded_run, make_contributor
):
    run_id = seeded_run()
    cid, _ = make_contributor()
    task_id = _task(conn, run_id, _worker(conn, cid), [0], status="abandoned")
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed, accepted, received_at)
           VALUES (?, 'a.safetensors', 10, 1, ?)""",
        (task_id, rounds._iso(rounds.utcnow())),
    )

    assert "orphan_submission" in _checks(conn)


def test_a_submission_on_a_lease_the_close_expired_is_not_an_orphan(
    conn, seeded_run, make_contributor
):
    """close_round expires every lease outstanding in the round it closed,
    including ones that had already submitted -- so 'submitted then expired' is
    the ordinary end state of a round, not evidence of a lost lease."""
    run_id = seeded_run()
    cid, _ = make_contributor()
    task_id = _task(conn, run_id, _worker(conn, cid), [0], status="expired")
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed, accepted, received_at)
           VALUES (?, 'a.safetensors', 10, 1, ?)""",
        (task_id, rounds._iso(rounds.utcnow())),
    )

    assert "orphan_submission" not in _checks(conn)


# --------------------------------------------------------------------------
# coverage and the loss curve
# --------------------------------------------------------------------------


def test_coverage_distinguishes_advancing_from_retraining(conn, seeded_run):
    run_id = seeded_run(num_buckets=8)
    for b in (0, 1, 2, 3):
        conn.execute(
            "UPDATE buckets SET times_trained = 1 WHERE run_id = ? AND bucket_idx = ?",
            (run_id, b),
        )
    advancing = invariants.coverage(conn, run_id)
    assert advancing == {"buckets": 8, "distinct_trained": 4, "min": 0, "max": 1,
                         "spread": 1, "total": 4, "whole_dataset_tasks": 0}

    # Same total work, all of it on one shard: the shape that says a run is
    # busy but learning nothing new.
    conn.execute("UPDATE buckets SET times_trained = 0 WHERE run_id = ?", (run_id,))
    conn.execute(
        "UPDATE buckets SET times_trained = 4 WHERE run_id = ? AND bucket_idx = 0", (run_id,)
    )
    retraining = invariants.coverage(conn, run_id)
    assert retraining["total"] == advancing["total"]
    assert retraining["distinct_trained"] == 1
    assert retraining["spread"] == 4


def test_the_loss_curve_skips_unevaluated_rounds_rather_than_zeroing_them(conn, seeded_run):
    run_id = seeded_run()
    rounds.open_round(conn, run_id, 1, "b", 100, 0, 3600)
    rounds.open_round(conn, run_id, 2, "c", 100, 0, 3600)
    conn.execute("UPDATE rounds SET eval_loss = 2.5 WHERE run_id = ? AND idx = 0", (run_id,))
    conn.execute("UPDATE rounds SET eval_loss = 2.1 WHERE run_id = ? AND idx = 2", (run_id,))

    assert invariants.loss_curve(conn, run_id) == [(0, 2.5), (2, 2.1)]


def test_the_cli_exits_nonzero_on_a_violation_and_zero_when_clean(
    conn, settings, seeded_run, capsys
):
    """The exit code is the whole point: this is meant to be a cron check."""
    run_id = seeded_run()
    assert invariants.main(["--db", settings.db_path, "--run-id", run_id]) == 0

    past = rounds._iso(rounds.utcnow() - timedelta(seconds=3600))
    conn.execute(
        "UPDATE rounds SET status = 'closing', opened_at = ? WHERE run_id = ?", (past, run_id)
    )
    conn.commit()
    assert invariants.main(["--db", settings.db_path]) == 1
    assert "stuck_close" in capsys.readouterr().err


def test_coverage_counts_the_workers_handed_the_entire_dataset(
    conn, seeded_run, make_contributor
):
    """The pathology `spread` cannot see. When every worker in a round is
    assigned every bucket, nobody is sharding anything -- but times_trained
    rises uniformly, so spread stays at zero and coverage looks perfect."""
    run_id = seeded_run(num_buckets=4)
    cid, _ = make_contributor()
    for i in range(3):
        _task(conn, run_id, _worker(conn, cid, f"w{i}"), [0, 1, 2, 3])

    assert invariants.coverage(conn, run_id)["whole_dataset_tasks"] == 3


def test_a_task_holding_a_proper_subset_is_not_counted_as_whole_dataset(
    conn, seeded_run, make_contributor
):
    run_id = seeded_run(num_buckets=4)
    cid, _ = make_contributor()
    _task(conn, run_id, _worker(conn, cid), [0, 1, 2])

    assert invariants.coverage(conn, run_id)["whole_dataset_tasks"] == 0
