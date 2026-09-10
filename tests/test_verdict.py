"""M4b's verdict harness (scripts/verdict.py).

The passing case is not invented: it is M4a's own numbers from
docs/03-roadmap.md -- loss 4.878 -> 4.472 over four rounds, divergence
0.657 -> 0.006 -- so "what a pass looks like" is a run that actually happened
rather than a shape chosen to make the assertions work.

The three-state exit code is the thing most worth testing. `0` and `1` are
ordinary; `2` is the one that matters, because a criterion nobody measured must
not be reported as one that passed, and must not be reported as one that failed
either -- an exit code that fires on missing data is an exit code operators
learn to ignore.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import uuid

import pytest

from ganymede.coordinator.db import connect, init_schema
from ganymede.coordinator.rounds import _iso, utcnow
from scripts import verdict as V

# M4a's measured run (docs/03-roadmap.md "The numbers").
M4A_LOSS = [4.878, 4.589, 4.491, 4.472]
M4A_DIVERGENCE = [0.657, 0.312, 0.030, 0.006]

_PROBE_SRC = """import sys
from scripts import verdict
heavy = sorted({m.split(".")[0] for m in sys.modules
                if m.split(".")[0] in
                ("torch", "transformers", "peft", "datasets", "boto3", "botocore")})
print(",".join(heavy))
"""


def _baseline(band: float = 4.5, timed: bool = True) -> dict:
    curve = [
        {"step": 0, "mean": 5.2, "stdev": 0.01, "min": 5.19, "max": 5.21},
        {"step": 400, "mean": 4.7, "stdev": 0.01, "min": 4.69, "max": 4.71},
        {"step": 800, "mean": 4.45, "stdev": 0.01, "min": 4.44, "max": 4.46},
    ]
    if timed:
        for point, sec in zip(curve, (0.0, 900.0, 1800.0)):
            point["train_sec_mean"] = sec
    return {"summary": {"curve": curve,
                        "tolerance": {"k": 2.0, "pass_if_final_loss_at_most": band}}}


@pytest.fixture
def run_db(tmp_path):
    """A four-round run with three workers, shaped like M4a's.

    Built here rather than driven through the coordinator: the harness reads
    finished state, and a fixture that produced that state by running the
    scheduler would be testing the scheduler.
    """
    path = str(tmp_path / "verdict.db")
    conn = connect(path)
    init_schema(conn)
    now = utcnow()
    run_id = "run-m4a"

    conn.execute(
        """INSERT INTO runs (id, status, base_model, base_precision, lora_cfg_json,
                    dataset_ref, hyperparams_json, current_round, target_rounds,
                    combine_mode, num_buckets, created_at)
           VALUES (?, 'done', 'Qwen/Qwen3-0.6B', 'bf16', '{}', 'd', '{}', 4, 4,
                   'mean', 32, ?)""",
        (run_id, _iso(now)),
    )
    workers = []
    for i in range(3):
        wid = uuid.uuid4().hex
        workers.append(wid)
        cid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
               VALUES (?, ?, ?, 1, 'open', ?)""",
            (cid, f"c{i}", uuid.uuid4().hex, _iso(now)),
        )
        conn.execute(
            """INSERT INTO workers (id, contributor_id, compute_profile_json,
                        first_seen, last_seen)
               VALUES (?, ?, '{}', ?, ?)""",
            (wid, cid, _iso(now), _iso(now)),
        )

    from datetime import timedelta

    for idx, (loss, div) in enumerate(zip(M4A_LOSS, M4A_DIVERGENCE)):
        opened = now + timedelta(seconds=idx * 300)
        closed = opened + timedelta(seconds=280)
        conn.execute(
            """INSERT INTO rounds (run_id, idx, base_adapter_ref, status, target_steps,
                        min_round_sec, max_round_sec, opened_at, closed_at,
                        result_adapter_ref, distinct_contributors, eval_loss,
                        adapter_divergence)
               VALUES (?, ?, ?, 'closed', 200, 0, 600, ?, ?, ?, 3, ?, ?)""",
            (run_id, idx, f"a/{idx}", _iso(opened), _iso(closed),
             f"a/{idx + 1}", loss, div),
        )
        for slot, (w, steps) in enumerate(zip(workers, (70, 68, 72))):
            task_id = f"t{idx}-{w[:6]}"
            # Disjoint buckets, advancing round on round. Handing every task the
            # same bucket would trip `repeat_before_coverage` -- correctly, and
            # the harness reporting that is the point of including
            # invariants.check at all.
            buckets = [idx * 3 + slot]
            conn.execute(
                """INSERT INTO tasks (id, run_id, round_idx, buckets_json, local_steps,
                            status, worker_id, max_runtime_sec, created_at)
                   VALUES (?, ?, ?, ?, 100, 'submitted', ?, 300, ?)""",
                (task_id, run_id, idx, json.dumps(buckets), w, _iso(opened)),
            )
            conn.execute(
                """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
                            accepted, received_at)
                   VALUES (?, 'x', ?, 1, ?)""",
                (task_id, steps, _iso(opened + timedelta(seconds=270))),
            )
    for b in range(32):
        conn.execute(
            "INSERT INTO buckets (run_id, bucket_idx, times_trained, last_round) "
            "VALUES (?, ?, ?, 3)", (run_id, b, 1 if b < 12 else 0),
        )
    conn.commit()
    conn.close()
    return path


def _open(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _collect(path, baseline=None, **over):
    conn = _open(path)
    try:
        run = V.pick_run(conn, None)
        kwargs = {"generations": False, "out_dir": None, "cap": 2.0,
                  "floor": 0.5, "ceiling": 1.05}
        kwargs.update(over)
        return {v.key: v for v in V.collect(conn, run, baseline, **kwargs)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The criteria
# ---------------------------------------------------------------------------


def test_a_run_matching_m4a_passes_every_criterion_it_can_evaluate(run_db):
    v = _collect(run_db, _baseline())
    assert v["loss_vs_steps"].state == V.PASS
    assert v["divergence"].state == V.PASS
    assert v["budgets"].state == V.PASS
    assert v["dominance"].state == V.PASS
    assert v["invariants"].state == V.PASS


def test_the_loss_band_is_the_baseline_s_own_tolerance(run_db):
    """4.472 against a band of 4.5 passes; the same run against 4.4 fails. The
    harness never invents a threshold -- ``pass_if_final_loss_at_most`` is
    computed by ganymede-baseline from measured seed spread."""
    assert _collect(run_db, _baseline(band=4.5))["loss_vs_steps"].state == V.PASS
    tight = _collect(run_db, _baseline(band=4.4))["loss_vs_steps"]
    assert tight.state == V.FAIL
    assert "4.47" in tight.detail and "4.4" in tight.detail


def test_cumulative_steps_counts_only_accepted_submissions(run_db):
    """A rejected submission was never aggregated, so counting its steps would
    overstate the training signal the loss is being credited to."""
    conn = _open(run_db)
    conn.execute("UPDATE submissions SET accepted = 0 WHERE task_id LIKE 't3-%'")
    conn.commit()
    conn.close()
    v = _collect(run_db, _baseline())["loss_vs_steps"]
    # Three rounds of 210 rather than four.
    assert v.data["cumulative_steps"] == 630


def test_growing_divergence_fails_and_names_local_steps(run_db):
    """5.2 records divergence because it measures what DiLoCo trades away, and
    *growing* drift is the signal that local_steps is too high. M4a's collapsed
    by two orders of magnitude; this is the inverse."""
    conn = _open(run_db)
    for idx, div in enumerate([0.006, 0.030, 0.312, 0.657]):
        conn.execute("UPDATE rounds SET adapter_divergence = ? WHERE idx = ?",
                     (div, idx))
    conn.commit()
    conn.close()
    v = _collect(run_db)["divergence"]
    assert v.state == V.FAIL
    assert "local_steps" in v.detail


def test_two_rounds_are_not_enough_to_call_a_trend(run_db):
    conn = _open(run_db)
    conn.execute("UPDATE rounds SET adapter_divergence = NULL WHERE idx >= 2")
    conn.commit()
    conn.close()
    v = _collect(run_db)["divergence"]
    assert v.state == V.UNKNOWN
    assert "at least 3" in v.detail


def test_a_worker_finishing_far_under_budget_fails_the_budget_criterion(run_db):
    """3.5's budgets are what M4b confirms, and under-filling fails in the
    direction people forget: a worker that finishes at 20% of its budget wasted
    80% of a round, and nothing errored."""
    conn = _open(run_db)
    for row in conn.execute(
            "SELECT id, created_at FROM tasks WHERE id LIKE 't0-%'").fetchall():
        from datetime import datetime, timedelta as td
        quick = datetime.fromisoformat(row["created_at"]) + td(seconds=30)
        conn.execute("UPDATE submissions SET received_at = ? WHERE task_id = ?",
                     (quick.isoformat(), row["id"]))
    conn.commit()
    conn.close()
    v = _collect(run_db)["budgets"]
    assert v.state == V.FAIL
    assert "under" in v.detail


def test_the_cap_binding_in_every_round_means_one_machine_carried_the_run(run_db):
    """Being bound is not itself a failure -- the cap exists to be applied.
    Being bound in every multi-worker round is the finding."""
    conn = _open(run_db)
    conn.execute("UPDATE submissions SET steps_completed = 1000 "
                 "WHERE task_id LIKE '%' || (SELECT worker_id FROM tasks "
                 "WHERE id = submissions.task_id) || '%' AND task_id LIKE 't%'")
    # Make exactly one worker dominant in every round.
    conn.execute("UPDATE submissions SET steps_completed = 10 "
                 "WHERE task_id IN (SELECT id FROM tasks WHERE worker_id != ("
                 "  SELECT worker_id FROM tasks ORDER BY id LIMIT 1))")
    conn.commit()
    conn.close()
    v = _collect(run_db)["dominance"]
    assert v.state == V.FAIL
    assert "carried the run" in v.detail


def test_a_solo_run_says_the_cap_had_nothing_to_bind(run_db):
    """`distinct_contributors = 1` is expected while you are the only
    contributor (roadmap, "A note on being the only contributor"). Reporting
    that as a pass would claim a fleet property from a one-machine run."""
    conn = _open(run_db)
    conn.execute("DELETE FROM submissions WHERE task_id NOT LIKE '%' || ("
                 "  SELECT substr(worker_id, 1, 6) FROM tasks ORDER BY id LIMIT 1) || '%'")
    conn.commit()
    conn.close()
    v = _collect(run_db)["dominance"]
    assert v.state == V.UNKNOWN
    assert "single contributor" in v.detail


# ---------------------------------------------------------------------------
# Wall-clock, and the missing-measurement path
# ---------------------------------------------------------------------------


def test_wallclock_compares_time_to_reach_the_same_loss(run_db):
    """Not "who was better at time T": the two runs share no step grid, and
    comparing at an arbitrary timestamp rewards whoever evaluated most
    recently."""
    v = _collect(run_db, _baseline())["loss_vs_wallclock"]
    assert v.state == V.PASS
    assert v.data["baseline_sec"] == 1800.0
    assert v.data["distributed_sec"] < 1800.0
    # The direction has to be unmissable in the text: this is the number the
    # milestone turns on, and a bare ratio reads either way round.
    assert "faster" in v.detail


def test_a_distributed_run_slower_than_one_gpu_fails_and_says_slower(run_db):
    """"Otherwise the system is an expensive way to train slower" is the whole
    point of the criterion, so the failing direction gets its own test."""
    conn = _open(run_db)
    from datetime import datetime, timedelta as td
    opened = datetime.fromisoformat(
        conn.execute("SELECT opened_at FROM rounds WHERE idx = 0").fetchone()[0])
    conn.execute("UPDATE rounds SET closed_at = ? WHERE idx = 3",
                 ((opened + td(seconds=5400)).isoformat(),))
    conn.commit()
    conn.close()
    v = _collect(run_db, _baseline())["loss_vs_wallclock"]
    assert v.state == V.FAIL
    assert "slower" in v.detail


def test_a_baseline_without_timing_is_unevaluated_not_failed(run_db):
    """`baseline.json` is a checked-in artifact of a GPU run, and the committed
    one predates per-point timing. A missing measurement is not a failed
    criterion -- and the message has to say what to run."""
    v = _collect(run_db, _baseline(timed=False))["loss_vs_wallclock"]
    assert v.state == V.UNKNOWN
    assert "ganymede-baseline" in v.detail


def test_an_unevaluated_run_says_to_run_evalround(run_db):
    conn = _open(run_db)
    conn.execute("UPDATE rounds SET eval_loss = NULL")
    conn.commit()
    conn.close()
    v = _collect(run_db, _baseline())
    assert v["loss_vs_steps"].state == V.UNKNOWN
    assert "evalround" in v["loss_vs_steps"].detail


def test_generations_are_unevaluated_unless_asked_for(run_db):
    """The one criterion needing transformers, a base model and S3 credentials.
    Off by default so the harness runs on the coordinator box."""
    v = _collect(run_db, _baseline())["generations"]
    assert v.state == V.UNKNOWN
    assert "--generations" in v.detail


# ---------------------------------------------------------------------------
# Exit codes -- three states
# ---------------------------------------------------------------------------


def test_exit_code_zero_only_when_everything_evaluated_passed():
    passed = [V.Verdict("a", "", V.PASS, "")]
    assert V.exit_code(passed) == 0


def test_a_failure_outranks_a_missing_measurement():
    mixed = [V.Verdict("a", "", V.FAIL, ""), V.Verdict("b", "", V.UNKNOWN, "")]
    assert V.exit_code(mixed) == 1


def test_a_missing_measurement_is_two_not_one_and_not_zero():
    """The distinction the whole report rests on: nobody measured it is neither
    'the milestone passed' nor 'the milestone failed'."""
    assert V.exit_code([V.Verdict("a", "", V.PASS, ""),
                        V.Verdict("b", "", V.UNKNOWN, "")]) == 2


def test_the_run_exits_two_and_names_what_was_not_evaluated(run_db, capsys):
    code = V.main(["--db", run_db, "--baseline", "does-not-exist.json"])
    out = capsys.readouterr().out
    assert code == 2
    assert "not evaluated:" in out
    assert "generations" in out


# ---------------------------------------------------------------------------
# Deployability
# ---------------------------------------------------------------------------


def test_the_harness_needs_no_storage_configuration(run_db, monkeypatch, capsys):
    """The machine you have after the rentals are destroyed is one with a copy
    of the database and no bucket. `Settings.from_env()` raises without
    STORAGE_HOST, so a harness that built a Store at startup would refuse to run
    exactly there."""
    for var in ("STORAGE_HOST", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY",
                "GANYMEDE_DB", "COORDINATOR_HOST"):
        monkeypatch.delenv(var, raising=False)
    assert V.main(["--db", run_db]) in (0, 1, 2)
    assert "run run-m4a" in capsys.readouterr().out


def test_it_imports_no_trainer_dependency_at_module_level():
    """`pyproject` keeps transformers/peft/datasets out of the coordinator, so
    importing one at module level would make this script undeployable on the
    box that holds the database."""
    import ast
    import pathlib

    src = pathlib.Path(V.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = []
    for node in top_level:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        else:
            names.append(node.module or "")
    banned = ("transformers", "peft", "datasets", "torch", "ganymede.trainer")
    offenders = [n for n in names if any(n.startswith(b) for b in banned)]
    assert offenders == [], f"module-level import of {offenders}"


def test_importing_it_pulls_in_no_heavy_dependency(tmp_path):
    """The AST check above states the intent; this measures the reality,
    transitively.

    In a **subprocess**, deliberately: `tests/conftest.py` imports torch for the
    coordinator fixtures, so an in-process version of this would find torch
    already in `sys.modules`, diff to nothing, and pass by being inert -- the
    same shape as a stand-in that cannot disagree with the code under test.
    """
    import os
    import subprocess
    import sys as _sys

    root = pathlib.Path(V.__file__).parents[1]
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE_SRC, encoding="utf-8")
    # PYTHONPATH, not just cwd: Python puts the *script's* directory on
    # sys.path, which here is tmp_path.
    out = subprocess.run([_sys.executable, str(probe)], capture_output=True,
                         text=True, cwd=str(root),
                         env={**os.environ, "PYTHONPATH": str(root)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        f"importing scripts.verdict pulls in {out.stdout.strip()} -- it has to "
        f"run on a box that has none of them")


def test_a_missing_database_is_exit_two_not_a_traceback(tmp_path, capsys):
    assert V.main(["--db", str(tmp_path / "nope.db")]) == 2
    assert "no such database" in capsys.readouterr().err


def test_the_json_report_carries_every_number_for_the_ab_diff(run_db, tmp_path):
    """Criterion 5 is a diff of two of these files, so each has to be complete
    on its own -- the numbers, not just the verdicts."""
    out = tmp_path / "verdict.json"
    V.main(["--db", run_db, "-o", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["run"]["combine_mode"] == "mean"
    keys = {c["criterion"] for c in report["criteria"]}
    assert keys == {"loss_vs_steps", "loss_vs_wallclock", "generations",
                    "divergence", "combine_mode", "budgets", "dominance",
                    "invariants"}
    divergence = next(c for c in report["criteria"] if c["criterion"] == "divergence")
    assert [p["divergence"] for p in divergence["per_round"]] == M4A_DIVERGENCE


# ---------------------------------------------------------------------------
# Generation checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("Paris is the capital of France.", None),
    ("", "empty"),
    ("   ", "empty"),
    ("Sure.\n\n### Instruction:\nWhat is...", "template leak"),
    ("the the the the the the the the the the the the the the", "repetition"),
    ("A short one.", None),
])
def test_the_mechanical_generation_checks(text, expected):
    """Conservative on purpose: each of these is wrong regardless of taste, and
    anything subtler belongs in the generations.json a person reads."""
    assert V._generation_problem(text) == expected


# ---------------------------------------------------------------------------
# Is this baseline about this run?
# ---------------------------------------------------------------------------


def _matching_baseline(**over) -> dict:
    """A baseline whose `run` block describes the fixture's run."""
    b = _baseline()
    b["run"] = {"base_model": "Qwen/Qwen3-0.6B", "base_precision": "bf16",
                "dataset_ref": "d", "lora_cfg": {}}
    b["run"].update(over)
    return b


def test_a_baseline_for_a_different_model_is_ignored_not_failed(run_db):
    """Found by running the harness against a real fleet database rather than
    this file's fixture.

    It compared a 107k-parameter model trained on synthetic rows against the
    committed Qwen3-1.7B/Dolly baseline and reported a confident FAIL. The
    fixture could never catch it: the same person wrote the baseline and the
    run, so they always agreed.

    The failing direction is merely wrong. The passing one is dangerous -- on a
    rented afternoon nobody questions a PASS -- and an unusable comparison is a
    criterion that was not evaluated, which is what the third exit code says.
    """
    v = _collect(run_db, _matching_baseline(base_model="Qwen/Qwen3-1.7B-Base"))
    assert v["loss_vs_steps"].state == V.UNKNOWN
    assert "not about this run" in v["loss_vs_steps"].detail
    assert "Qwen3-1.7B-Base" in v["loss_vs_steps"].detail
    # The same fact stops the wall-clock criterion too: both compare losses.
    assert v["loss_vs_wallclock"].state == V.UNKNOWN


def test_a_matching_baseline_is_used(run_db):
    """The guard has to not fire on the case it exists to protect."""
    v = _collect(run_db, _matching_baseline())
    assert v["loss_vs_steps"].state == V.PASS
    assert v["loss_vs_wallclock"].state == V.PASS


@pytest.mark.parametrize("field,value", [
    ("base_precision", "nf4"),
    ("dataset_ref", "hf://somewhere/else"),
])
def test_precision_and_dataset_also_make_a_baseline_unusable(run_db, field, value):
    """Each of these changes the curve on its own, so a mismatch does not make
    the comparison imprecise -- it makes it meaningless."""
    v = _collect(run_db, _matching_baseline(**{field: value}))
    assert v["loss_vs_steps"].state == V.UNKNOWN
    assert field in v["loss_vs_steps"].detail


def test_a_different_lora_rank_makes_a_baseline_unusable(run_db):
    conn = _open(run_db)
    conn.execute("""UPDATE runs SET lora_cfg_json = '{"rank": 16}' """)
    conn.commit()
    conn.close()
    v = _collect(run_db, _matching_baseline(lora_cfg={"rank": 8}))
    assert v["loss_vs_steps"].state == V.UNKNOWN
    assert "lora_cfg.rank" in v["loss_vs_steps"].detail


def test_the_mismatch_is_printed_where_it_cannot_be_missed(run_db, tmp_path, capsys):
    """Buried in one criterion's detail it reads as that criterion's problem.
    It is the whole report's problem."""
    import json as _json

    b = tmp_path / "baseline.json"
    b.write_text(_json.dumps(_matching_baseline(base_model="other/model")),
                 encoding="utf-8")
    V.main(["--db", run_db, "--baseline", str(b)])
    assert "BASELINE IGNORED" in capsys.readouterr().out


def test_a_missing_baseline_does_not_send_you_to_re_run_the_gpu_job(run_db):
    """The wall-clock criterion used to blame missing per-point timing for a
    baseline that was simply never passed -- sending someone off to spend GPU
    hours on the wrong problem."""
    v = _collect(run_db, None)["loss_vs_wallclock"]
    assert v.state == V.UNKNOWN
    assert "no baseline.json found" in v.detail
    assert "ganymede-baseline" not in v.detail
