"""The evaluator's selection and resilience, without loading a model.

What actually needs a model -- that the loss it computes is a real held-out
loss, and that it descends across a run's rounds -- is exercised in
``test_worker_concurrency``, against rounds produced by real workers. Here the
concern is narrower and cheaper: which rounds get picked up, in what order, and
what a failure on one costs the rest.
"""

from __future__ import annotations

import pytest

from ganymede.coordinator import rounds
from ganymede.jobtypes.collab_lora_finetune import plan
from scripts import evalround


@pytest.fixture
def closed_rounds(conn, seeded_run):
    """A run with three closed rounds, each having published an adapter."""
    run_id = seeded_run()
    for idx in (1, 2):
        plan.open_round(conn, run_id, idx, f"base-{idx}", 100, 0, 3600)
    for idx in (0, 1, 2):
        conn.execute(
            """UPDATE rounds SET status = 'closed', result_adapter_ref = ?
               WHERE run_id = ? AND idx = ?""",
            (f"result-{idx}", run_id, idx),
        )
    return run_id


class StubEvaluator:
    """Records what it was asked to evaluate, and can be told to fail on one."""

    def __init__(self, fail_on: int | None = None):
        self.seen: list[int] = []
        self.fail_on = fail_on

    def evaluate(self, conn, rnd):
        self.seen.append(int(rnd["idx"]))
        if self.fail_on == int(rnd["idx"]):
            raise RuntimeError("adapter went missing")
        conn.execute(
            "UPDATE rounds SET eval_loss = ? WHERE run_id = ? AND idx = ?",
            (1.0, rnd["run_id"], rnd["idx"]),
        )
        return 1.0


def test_only_closed_rounds_that_published_an_adapter_are_pending(conn, seeded_run):
    """An open round has nothing to evaluate yet, and a round that closed with
    no accepted work published nothing -- neither is a backlog item."""
    run_id = seeded_run()
    plan.open_round(conn, run_id, 1, "base-1", 100, 0, 3600)
    conn.execute(
        "UPDATE rounds SET status = 'closed', result_adapter_ref = NULL WHERE idx = 0"
    )

    assert evalround.pending_rounds(conn, run_id) == []


def test_an_already_evaluated_round_is_not_picked_up_again(conn, closed_rounds):
    """Idempotence is what makes re-running after a crash free."""
    conn.execute("UPDATE rounds SET eval_loss = 2.0 WHERE idx IN (0, 1)")

    assert [r["idx"] for r in evalround.pending_rounds(conn, closed_rounds)] == [2]


def test_the_backlog_is_worked_oldest_first(conn, closed_rounds):
    """Newest-first would leave a hole in the middle of the curve while the
    backlog drains, which is exactly when someone is watching it."""
    stub = StubEvaluator()
    assert evalround.run_once(conn, stub, closed_rounds) == 3
    assert stub.seen == [0, 1, 2]


def test_one_unreadable_adapter_does_not_stop_the_backlog(conn, closed_rounds):
    """The round stays pending rather than being marked evaluated, so a later
    pass retries it -- a transient storage failure costs a pass, not the curve."""
    stub = StubEvaluator(fail_on=1)

    assert evalround.run_once(conn, stub, closed_rounds) == 2
    assert stub.seen == [0, 1, 2]
    assert [r["idx"] for r in evalround.pending_rounds(conn, closed_rounds)] == [1]


def test_a_limit_caps_one_pass_without_losing_the_rest(conn, closed_rounds):
    stub = StubEvaluator()
    assert evalround.run_once(conn, stub, closed_rounds, limit=2) == 2
    assert [r["idx"] for r in evalround.pending_rounds(conn, closed_rounds)] == [2]


def test_run_id_scopes_the_backlog_to_one_run(conn, closed_rounds, seeded_run):
    other = seeded_run(run_id="other")
    conn.execute(
        "UPDATE rounds SET status = 'closed', result_adapter_ref = 'x' WHERE run_id = ?",
        (other,),
    )

    assert {r["run_id"] for r in evalround.pending_rounds(conn, closed_rounds)} == {closed_rounds}
    assert len(evalround.pending_rounds(conn)) == 4
