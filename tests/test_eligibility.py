"""Why a worker never gets work (roadmap M5, 6.9, 6.10).

The load-bearing test in this file is
``test_the_recorded_reason_is_the_one_the_claim_path_actually_produced``. Every
other assertion here is about presentation; that one is about whether the
feature is honest. A diagnostic that reimplements the decision it explains will
eventually disagree with it, and a contributor told they are eligible for a run
that keeps turning them away is worse off than one told nothing.
"""

from __future__ import annotations

import json

import pytest

from ganymede.coordinator import eligibility


def _claim(client, key, worker_id, **body):
    payload = {"worker_id": worker_id, **body}
    return client.post("/v1/tasks/claim", json=payload,
                       headers={"Authorization": f"Bearer {key}"})


# --------------------------------------------------------------------------
# The property the whole feature rests on
# --------------------------------------------------------------------------


def test_the_recorded_reason_is_the_one_the_claim_path_actually_produced(
    client, conn, make_contributor, seeded_run, register_worker
):
    """Not "a reason of the same kind" -- the same string, from the same call.

    This is why `record` is fed from inside the claim loop rather than being a
    second implementation that runs later. If this test can be made to pass by
    a reimplementation, the reimplementation is what will drift."""
    run_id = seeded_run(requires={"min_vram_mb": 999_999})
    _, key = make_contributor()
    worker_id = register_worker(key)

    assert _claim(client, key, worker_id).status_code == 204

    answer = eligibility.explain(conn, worker_id)
    verdict = next(v for v in answer.verdicts if v.run_id == run_id)
    assert verdict.outcome == eligibility.REFUSED
    # The exact text budget.is_eligible produced, numbers and all.
    assert "vram_mb" in verdict.reason
    assert "999999" in verdict.reason.replace(",", "")


def test_a_successful_claim_clears_a_previous_refusal(
    client, conn, make_contributor, seeded_run, register_worker
):
    """A stale "refused" is the answer a contributor would act on. Leaving one
    behind sends them looking for a fault in a machine that is now fine."""
    run_id = seeded_run(requires={"min_vram_mb": 999_999})
    _, key = make_contributor()
    worker_id = register_worker(key)

    _claim(client, key, worker_id)
    assert eligibility.explain(conn, worker_id).verdicts[0].outcome == eligibility.REFUSED

    # Drop the requirement and poll again.
    conn.execute("UPDATE runs SET requires_json = '{}' WHERE id = ?", (run_id,))
    conn.commit()
    assert _claim(client, key, worker_id).status_code == 200

    answer = eligibility.explain(conn, worker_id)
    verdict = next(v for v in answer.verdicts if v.run_id == run_id)
    assert verdict.outcome == eligibility.LEASED
    assert verdict.reason is None


# --------------------------------------------------------------------------
# The three outcomes stay distinct
# --------------------------------------------------------------------------


def test_eligible_but_nothing_to_do_is_not_a_refusal(
    client, conn, make_contributor, seeded_run, register_worker
):
    """A fleet that is uniformly idle is a run with no open round -- an
    operator problem. A fleet that is uniformly refused is a run nobody can
    meet -- a different operator problem. Collapsing them loses that."""
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET status = 'closed' WHERE run_id = ?", (run_id,))
    conn.commit()

    _, key = make_contributor()
    worker_id = register_worker(key)
    assert _claim(client, key, worker_id).status_code == 204

    verdict = eligibility.explain(conn, worker_id).verdicts[0]
    assert verdict.outcome == eligibility.IDLE
    assert verdict.eligible


def test_a_worker_that_has_never_polled_says_so_rather_than_looking_refused(
    client, conn, make_contributor, register_worker
):
    """The distinction is the useful half: no rows means the agent is not
    running or cannot reach the coordinator, which is upstream of eligibility
    entirely."""
    _, key = make_contributor()
    worker_id = register_worker(key)

    answer = eligibility.explain(conn, worker_id)
    assert answer.verdicts == []
    assert "no polls recorded" in str(answer)


def test_an_image_mismatch_is_reported_as_such(
    client, conn, make_contributor, seeded_run, register_worker
):
    run_id = seeded_run(required_image="ganymede/worker-llm:v9")
    _, key = make_contributor()
    worker_id = register_worker(key, image_tag="ganymede/worker-llm:v2")
    _claim(client, key, worker_id)

    verdict = next(v for v in eligibility.explain(conn, worker_id).verdicts
                   if v.run_id == run_id)
    assert verdict.outcome == eligibility.REFUSED
    assert "image" in verdict.reason


def test_clearance_refusal_is_recorded(
    client, conn, make_contributor, seeded_run, register_worker
):
    run_id = seeded_run(classification="restricted")
    _, key = make_contributor(clearance="open")
    worker_id = register_worker(key)
    _claim(client, key, worker_id)

    verdict = next(v for v in eligibility.explain(conn, worker_id).verdicts
                   if v.run_id == run_id)
    assert verdict.outcome == eligibility.REFUSED
    assert verdict.reason == "clearance 'open' < classification 'restricted'"


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def test_the_endpoint_answers_the_contributors_question(
    client, conn, make_contributor, seeded_run, register_worker
):
    seeded_run(requires={"min_vram_mb": 999_999})
    _, key = make_contributor()
    worker_id = register_worker(key)
    _claim(client, key, worker_id)

    r = client.get(f"/v1/workers/{worker_id}/eligibility",
                   headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["eligible_for_something"] is False
    assert body["runs"][0]["outcome"] == eligibility.REFUSED
    # Half of every refusal is a fact about this machine; a contributor
    # comparing the reason against what they think their card has needs to see
    # what the coordinator measured (6.9).
    assert "backend" in body["compute_profile"]


def test_another_contributors_worker_is_not_enumerable(
    client, make_contributor, register_worker
):
    """404, not 403: a worker id belonging to someone else must not be
    distinguishable from one that does not exist."""
    _, key_a = make_contributor()
    worker_a = register_worker(key_a)

    _, key_b = make_contributor()
    r = client.get(f"/v1/workers/{worker_a}/eligibility",
                   headers={"Authorization": f"Bearer {key_b}"})
    assert r.status_code == 404


# --------------------------------------------------------------------------
# The operator's view
# --------------------------------------------------------------------------


def test_the_fleet_view_groups_one_problem_into_one_line(
    client, conn, make_contributor, seeded_run, register_worker
):
    """One machine refused for memory is that machine's business. Every machine
    refused for memory is a run whose requires block is wrong -- and the two
    read identically one worker at a time."""
    seeded_run(requires={"min_vram_mb": 999_999})

    for i, vram in enumerate((4096, 6144, 8192)):
        _, key = make_contributor(name=f"c{i}")
        worker_id = register_worker(
            key, vram_mb=vram, probe={"alloc_max_mb": vram, "bench_score": 40.0}
        )
        _claim(client, key, worker_id)

    summary = eligibility.fleet_summary(conn)
    # Three different measured numbers, one problem, one line.
    assert len(summary) == 1
    assert len(next(iter(summary.values()))) == 3


def test_shape_strips_the_numbers_that_would_scatter_one_problem():
    assert eligibility._shape("vram_mb 4096 < 8000") == eligibility._shape("vram_mb 6144 < 8000")
    assert eligibility._shape("missing capability: nf4") != eligibility._shape("vram_mb 1 < 2")


# --------------------------------------------------------------------------
# It must not be able to break what it diagnoses
# --------------------------------------------------------------------------


def test_a_broken_diagnostic_table_does_not_cost_a_round(
    client, conn, make_contributor, seeded_run, register_worker
):
    """Recording is best-effort by construction. A failure here turning a
    successful claim into a 500 would be a strictly worse outcome than not
    knowing why a worker is idle."""
    seeded_run()
    _, key = make_contributor()
    worker_id = register_worker(key)

    conn.execute("DROP TABLE worker_eligibility")
    conn.commit()

    assert _claim(client, key, worker_id).status_code == 200


def test_record_with_no_verdicts_is_a_no_op(conn):
    eligibility.record(conn, "worker-x", [])
    assert eligibility.explain(conn, "worker-x").verdicts == []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_the_cli_reports_json(tmp_path, client, conn, make_contributor, seeded_run,
                              register_worker, settings, capsys):
    seeded_run(requires={"min_vram_mb": 999_999})
    _, key = make_contributor()
    worker_id = register_worker(key)
    _claim(client, key, worker_id)

    rc = eligibility.main(["--db", settings.db_path, "--worker-id", worker_id, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["worker_id"] == worker_id
    assert payload[0]["verdicts"][0]["outcome"] == eligibility.REFUSED


def test_the_cli_needs_a_subject(settings):
    with pytest.raises(SystemExit):
        eligibility.main(["--db", settings.db_path])
