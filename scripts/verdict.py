"""``ganymede-verdict`` -- did the distributed run work? (docs/03-roadmap.md M4b)

M4b rents three machines for an afternoon and asks seven questions of what comes
back. Answering them by reading a loss curve at the end of that afternoon is how
you talk yourself into a result. This script answers them from the run's own
tables, prints a verdict per criterion, and exits with a code that says whether
the milestone passed -- so the afternoon has a stopping condition rather than a
judgement call.

Deployability (the constraint that shaped this file)
----------------------------------------------------
Six of the seven criteria, plus ``invariants.check``, run against **a database
file and ``baseline.json``**. Nothing else: no torch, no transformers, no object
store, no S3 credentials, no ``STORAGE_HOST``. That is deliberate and it is the
main design decision here --

* ``pyproject`` keeps the coordinator small on purpose: it has no
  ``transformers``/``peft``/``datasets``, so anything importing them at module
  level is undeployable on the box that actually holds the database.
* ``Settings.from_env()`` raises when storage is unconfigured. Calling it at
  startup would make the whole script refuse to run on a machine with no bucket
  -- which is exactly the machine you have after the rentals are destroyed and
  all you kept was a copy of ``ganymede.db``.

So the store and the trainer stack are reached **only** inside `--generations`,
imported lazily, and their absence is reported as *not evaluated* rather than
crashing or quietly passing. `python -m scripts.verdict --db run.db` works on a
laptop with nothing installed but the coordinator's own dependencies, on any OS.

Three exit codes, not two
-------------------------
``0`` every evaluated criterion passed. ``1`` at least one failed. ``2`` a fact
needed to reach a verdict was missing -- nobody ran ``scripts/evalround.py``,
the baseline predates timing, the trainer extra is absent. Conflating the third
with the second is how an operator learns to ignore the exit code; conflating it
with the first is how a milestone gets declared on data nobody collected.
``status.py`` makes the same distinction for the same reason.

The A/B is a diff of two of these
---------------------------------
Criterion 5 (mean vs. diloco, 5.2's research risk) needs two completed runs,
which is two afternoons. Rather than ship a ``--compare-run`` path that cannot
be tested against real data until the day it matters, this emits every number
into ``verdict.json`` and the A/B is ``ganymede-verdict --db a.db -o a.json``
against the same for ``b``. The criterion reports which mode the run used and
what its final loss was, which is the half a single run can honestly answer.

Running it
----------
    python -m scripts.verdict --db ganymede.db --run-id live
    python -m scripts.verdict --db ganymede.db --generations -o verdict.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Deliberately nothing from ganymede.trainer at module level, and nothing that
# reaches Settings/Store. See the docstring.
from ganymede.coordinator import invariants

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"   # not evaluated: something needed was missing

_MARK = {PASS: "PASS", FAIL: "FAIL", UNKNOWN: "  ? "}


class Verdict:
    """One exit criterion's answer, its numbers, and why."""

    def __init__(self, key: str, title: str, state: str, detail: str,
                 data: dict[str, Any] | None = None) -> None:
        self.key = key
        self.title = title
        self.state = state
        self.detail = detail
        self.data = data or {}

    def as_dict(self) -> dict[str, Any]:
        return {"criterion": self.key, "title": self.title, "state": self.state,
                "detail": self.detail, **self.data}


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _elapsed(start: str | None, end: str | None) -> float | None:
    a, b = _parse(start), _parse(end)
    if a is None or b is None:
        return None
    # A database written by more than one tool can hold both naive and aware
    # timestamps; subtracting one from the other raises rather than returning a
    # wrong number, and an unreadable pair should read as "no measurement".
    if (a.tzinfo is None) != (b.tzinfo is None):
        return None
    return (b - a).total_seconds()


# ---------------------------------------------------------------------------
# Reading the run
# ---------------------------------------------------------------------------


def pick_run(conn: sqlite3.Connection, run_id: str | None) -> sqlite3.Row | None:
    if run_id:
        return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return rows[0] if rows else None


def round_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? ORDER BY idx", (run_id,)
    ).fetchall()


def accepted_steps_by_round(conn: sqlite3.Connection, run_id: str) -> dict[int, list[tuple[str, int]]]:
    """``round_idx -> [(worker_id, steps_completed), ...]`` for accepted work.

    Accepted only. A rejected submission's steps were never aggregated, so
    counting them would overstate both the training signal and every worker's
    share of it.
    """
    out: dict[int, list[tuple[str, int]]] = {}
    for r in conn.execute(
        """SELECT t.round_idx AS idx, t.worker_id AS worker_id,
                  s.steps_completed AS steps
             FROM submissions s JOIN tasks t ON t.id = s.task_id
            WHERE t.run_id = ? AND s.accepted = 1
            ORDER BY t.round_idx""",
        (run_id,),
    ):
        out.setdefault(int(r["idx"]), []).append(
            (r["worker_id"] or "?", int(r["steps"])))
    return out


# ---------------------------------------------------------------------------
# Is this baseline even about this run?
# ---------------------------------------------------------------------------

# What has to match before two loss numbers can be compared at all. Each of
# these changes the curve on its own, so a mismatch does not make the comparison
# imprecise -- it makes it meaningless.
_MUST_MATCH = ("base_model", "base_precision", "dataset_ref")


def baseline_mismatch(run: sqlite3.Row, baseline: dict | None) -> str | None:
    """Why this baseline cannot judge this run, or None if it can.

    Found by running the harness against a real fleet database: it compared a
    107k-parameter model trained on synthetic rows against the committed
    Qwen3-1.7B/Dolly baseline and reported a confident FAIL. The failing
    direction is merely wrong; the passing direction is the dangerous one, and
    on a rented afternoon nobody would question a PASS.

    Deliberately not a warning. An unusable comparison is a criterion that was
    not evaluated, which is what the third exit code exists to say.
    """
    if not baseline:
        return None
    ref = baseline.get("run") or {}
    for field in _MUST_MATCH:
        want, got = ref.get(field), (run[field] if field in run.keys() else None)
        if want and got and want != got:
            return f"{field}: baseline has {want!r}, this run has {got!r}"

    want_lora = ref.get("lora_cfg") or {}
    try:
        got_lora = json.loads(run["lora_cfg_json"] or "{}")
    except ValueError:
        got_lora = {}
    for key in ("rank", "alpha", "target_modules"):
        want, got = want_lora.get(key), got_lora.get(key)
        if want is not None and got is not None and want != got:
            return f"lora_cfg.{key}: baseline has {want!r}, this run has {got!r}"
    return None


# ---------------------------------------------------------------------------
# 1 -- loss against cumulative steps, inside the baseline's seed band
# ---------------------------------------------------------------------------


def criterion_loss_vs_steps(rounds: list[sqlite3.Row],
                            steps: dict[int, list[tuple[str, int]]],
                            baseline: dict | None,
                            mismatch: str | None = None) -> Verdict:
    title = "held-out loss vs cumulative steps is inside the baseline band"
    evaluated = [(int(r["idx"]), float(r["eval_loss"]))
                 for r in rounds if r["eval_loss"] is not None]
    if not evaluated:
        return Verdict("loss_vs_steps", title, UNKNOWN,
                       "no round has an eval_loss -- run `python -m scripts.evalround "
                       "--once` against this database first")
    if not baseline:
        return Verdict("loss_vs_steps", title, UNKNOWN,
                       "no baseline.json found (--baseline), so there is no band")
    if mismatch:
        return Verdict("loss_vs_steps", title, UNKNOWN,
                       f"this baseline is not about this run -- {mismatch}. "
                       "Comparing them would be a verdict on nothing.",
                       {"mismatch": mismatch})

    band = (baseline.get("summary") or {}).get("tolerance", {}).get(
        "pass_if_final_loss_at_most")
    if band is None:
        return Verdict("loss_vs_steps", title, UNKNOWN,
                       "baseline.json has no tolerance band")

    # Carried forward across every round, not just rounds that had accepted
    # work: a round whose submissions were all rejected still has a loss, and
    # `cumulative.get(that_round)` must be the steps trained so far rather than
    # a hole.
    cumulative, total = {}, 0
    for r in rounds:
        idx = int(r["idx"])
        total += sum(n for _, n in steps.get(idx, []))
        cumulative[idx] = total

    final_idx, final_loss = evaluated[-1]
    curve = [{"round": i, "steps": cumulative.get(i), "loss": l}
             for i, l in evaluated]
    ok = final_loss <= float(band)
    return Verdict(
        "loss_vs_steps", title, PASS if ok else FAIL,
        f"final loss {final_loss:.5f} after {cumulative.get(final_idx, 0)} accepted "
        f"steps; band is <= {float(band):.5f} "
        f"({'inside' if ok else 'outside'} by {abs(final_loss - float(band)):.5f})",
        {"final_loss": final_loss, "band": float(band),
         "cumulative_steps": cumulative.get(final_idx), "curve": curve},
    )


# ---------------------------------------------------------------------------
# 2 -- loss against wall-clock, against one GPU
# ---------------------------------------------------------------------------


def criterion_loss_vs_wallclock(rounds: list[sqlite3.Row],
                                baseline: dict | None,
                                mismatch: str | None = None) -> Verdict:
    title = "held-out loss vs wall-clock beats single-node"
    evaluated = [r for r in rounds if r["eval_loss"] is not None]
    if not evaluated:
        return Verdict("loss_vs_wallclock", title, UNKNOWN,
                       "no round has an eval_loss -- run scripts/evalround first")

    opened = _parse(rounds[0]["opened_at"])
    distributed: list[dict[str, Any]] = []
    for r in evaluated:
        closed = _parse(r["closed_at"])
        if opened is None or closed is None:
            continue
        distributed.append({"round": int(r["idx"]),
                            "sec": round((closed - opened).total_seconds(), 2),
                            "loss": float(r["eval_loss"])})
    if not distributed:
        return Verdict("loss_vs_wallclock", title, UNKNOWN,
                       "rounds carry no usable opened_at/closed_at")

    if not baseline:
        # Said separately from the timing case below, which used to swallow it:
        # telling someone to re-run ganymede-baseline when they simply did not
        # pass one sends them off to spend GPU hours on the wrong problem.
        return Verdict("loss_vs_wallclock", title, UNKNOWN,
                       "no baseline.json found (--baseline), so there is nothing "
                       "to compare against", {"distributed": distributed})
    if mismatch:
        return Verdict("loss_vs_wallclock", title, UNKNOWN,
                       f"this baseline is not about this run -- {mismatch}",
                       {"mismatch": mismatch, "distributed": distributed})

    curve = baseline.get("summary", {}).get("curve") or []
    timed = [p for p in curve if p.get("train_sec_mean") is not None]
    if not timed:
        return Verdict(
            "loss_vs_wallclock", title, UNKNOWN,
            "baseline.json has no per-point timing -- re-run `ganymede-baseline` "
            "to record it; a baseline written before that change cannot answer this",
            {"distributed": distributed})

    # The honest comparison is "who reached this loss first", not "who was
    # better at time T": the two runs do not share a step grid, and comparing
    # losses at an arbitrary shared timestamp would reward whichever happened to
    # have just evaluated.
    target = distributed[-1]["loss"]
    ours = distributed[-1]["sec"]
    theirs = next((p["train_sec_mean"] for p in timed if p["mean"] <= target), None)
    if theirs is None:
        return Verdict(
            "loss_vs_wallclock", title, PASS,
            f"the baseline never reached {target:.5f} (best {min(p['mean'] for p in timed):.5f}); "
            f"the run got there in {ours:.0f}s",
            {"target_loss": target, "distributed_sec": ours, "baseline_sec": None,
             "distributed": distributed})

    ok = ours < theirs
    # Spelled out rather than left as a ratio: this is the number the whole
    # milestone turns on, and "0.66x" reads as either direction depending on
    # what the reader assumes is being divided.
    ratio = (theirs / ours) if ours else 0.0
    how = f"{ratio:.2f}x faster" if ok else f"{(ours / theirs):.2f}x slower"
    return Verdict(
        "loss_vs_wallclock", title, PASS if ok else FAIL,
        f"loss {target:.5f} reached in {ours:.0f}s distributed vs {theirs:.0f}s "
        f"single-node -- {how}",
        {"target_loss": target, "distributed_sec": ours, "baseline_sec": theirs,
         "speedup": round(theirs / ours, 3) if ours else None,
         "distributed": distributed},
    )


# ---------------------------------------------------------------------------
# 4 -- divergence stable rather than growing
# ---------------------------------------------------------------------------


def criterion_divergence(rounds: list[sqlite3.Row]) -> Verdict:
    title = "inter-worker adapter divergence is stable, not growing"
    seen = [(int(r["idx"]), float(r["adapter_divergence"]))
            for r in rounds if r["adapter_divergence"] is not None]
    if len(seen) < 3:
        return Verdict("divergence", title, UNKNOWN,
                       f"only {len(seen)} round(s) recorded a divergence; "
                       "a trend needs at least 3")

    values = [v for _, v in seen]
    third = max(1, len(values) // 3)
    early = statistics.fmean(values[:third])
    late = statistics.fmean(values[-third:])
    # Growing drift is 5.2's signal that local_steps is too high. Falling or
    # flat is agreement. The threshold is deliberately loose: what matters is
    # the direction, and a small rise on a short run is noise.
    ok = late <= early * 1.5
    return Verdict(
        "divergence", title, PASS if ok else FAIL,
        f"first third mean {early:.4f} -> last third mean {late:.4f} "
        f"({'stable or falling' if ok else 'GROWING -- local_steps may be too high'})",
        {"early_mean": round(early, 6), "late_mean": round(late, 6),
         "per_round": [{"round": i, "divergence": v} for i, v in seen]},
    )


# ---------------------------------------------------------------------------
# 5 -- the combine-mode A/B, as much of it as one run can answer
# ---------------------------------------------------------------------------


def criterion_combine_mode(run: sqlite3.Row, rounds: list[sqlite3.Row]) -> Verdict:
    title = "both combine modes A/B'd (5.2's research risk)"
    mode = run["combine_mode"]
    evaluated = [float(r["eval_loss"]) for r in rounds if r["eval_loss"] is not None]
    final = evaluated[-1] if evaluated else None
    return Verdict(
        "combine_mode", title, UNKNOWN,
        f"this run used combine_mode={mode!r}"
        + (f", final loss {final:.5f}" if final is not None else "")
        + ". The A/B needs the other mode's run: diff this verdict.json against "
          "one from a run with the other mode. If diloco does not beat mean on "
          "LoRA adapters, ship the plain mean and record the negative result.",
        {"combine_mode": mode, "final_loss": final,
         "lr_outer": run["lr_outer"], "outer_beta": run["outer_beta"]},
    )


# ---------------------------------------------------------------------------
# 6 -- mixed-speed workers both land near the round deadline
# ---------------------------------------------------------------------------


def criterion_budgets(conn: sqlite3.Connection, run_id: str,
                      floor: float, ceiling: float) -> Verdict:
    title = "mixed-speed workers land near the round deadline (3.5's budgets)"
    rows = conn.execute(
        """SELECT t.round_idx AS idx, t.worker_id AS worker_id,
                  t.created_at AS claimed_at, t.max_runtime_sec AS budget,
                  s.received_at AS submitted_at
             FROM submissions s JOIN tasks t ON t.id = s.task_id
            WHERE t.run_id = ? AND s.accepted = 1
            ORDER BY t.round_idx""",
        (run_id,),
    ).fetchall()

    used: list[dict[str, Any]] = []
    for r in rows:
        took = _elapsed(r["claimed_at"], r["submitted_at"])
        budget = r["budget"]
        if took is None or not budget:
            continue
        used.append({"round": int(r["idx"]), "worker": r["worker_id"],
                     "took_sec": round(took, 1), "budget_sec": int(budget),
                     "fraction": round(took / float(budget), 3)})
    if not used:
        return Verdict("budgets", title, UNKNOWN,
                       "no accepted submission carries both a claim time and a "
                       "max_runtime_sec budget")

    fractions = [u["fraction"] for u in used]
    low = [u for u in used if u["fraction"] < floor]
    high = [u for u in used if u["fraction"] > ceiling]
    ok = not low and not high
    parts = []
    if low:
        parts.append(f"{len(low)} finished under {floor:.0%} of budget "
                     f"(round under-filled)")
    if high:
        parts.append(f"{len(high)} ran past {ceiling:.0%} of budget")
    return Verdict(
        "budgets", title, PASS if ok else FAIL,
        f"{len(used)} accepted tasks, budget used min {min(fractions):.0%} / "
        f"median {statistics.median(fractions):.0%} / max {max(fractions):.0%}"
        + ("; " + "; ".join(parts) if parts else ""),
        {"tasks": used, "floor": floor, "ceiling": ceiling},
    )


# ---------------------------------------------------------------------------
# 7 -- no single worker exceeds the dominance cap
# ---------------------------------------------------------------------------


def criterion_dominance(steps: dict[int, list[tuple[str, int]]], cap: float) -> Verdict:
    title = f"no worker exceeds the dominance cap ({cap}x median)"
    if not steps:
        return Verdict("dominance", title, UNKNOWN, "no accepted submissions")

    bound: list[dict[str, Any]] = []
    for idx in sorted(steps):
        contributions = steps[idx]
        if len(contributions) < 2:
            continue   # a solo round has no median to dominate
        counts = [n for _, n in contributions]
        median = statistics.median(counts)
        if median <= 0:
            continue
        for worker, n in contributions:
            if n > median * cap:
                bound.append({"round": idx, "worker": worker, "steps": n,
                              "median": median, "ratio": round(n / median, 3)})
    rounds_seen = sum(1 for idx in steps if len(steps[idx]) >= 2)
    if rounds_seen == 0:
        return Verdict("dominance", title, UNKNOWN,
                       "every round closed with a single contributor -- the cap "
                       "has nothing to bind and this says nothing about a fleet")
    # Being bound is not a failure: the cap exists to be applied. Being bound in
    # *every* multi-worker round means one machine carried the run, which is the
    # thing M4b is checking for.
    rounds_bound = len({b["round"] for b in bound})
    ok = rounds_bound < rounds_seen
    return Verdict(
        "dominance", title, PASS if ok else FAIL,
        f"the cap bound in {rounds_bound} of {rounds_seen} multi-worker rounds"
        + ("" if ok else " -- every one, so one machine carried the run"),
        {"capped": bound, "rounds_with_a_cohort": rounds_seen, "cap": cap},
    )


# ---------------------------------------------------------------------------
# 3 -- greedy generations (the only criterion needing torch and the store)
# ---------------------------------------------------------------------------


def criterion_generations(run: sqlite3.Row, rounds: list[sqlite3.Row],
                          out_dir: Path | None) -> Verdict:
    """Decode the fixed prompt set through the final aggregated adapter.

    Lazily imported and separately flagged: this is the one criterion that needs
    `transformers`/`peft`, a base-model download and S3 credentials, none of
    which the coordinator box has.
    """
    title = "greedy generations show no collapse or template corruption"
    closed = [r for r in rounds if r["result_adapter_ref"]]
    if not closed:
        return Verdict("generations", title, UNKNOWN,
                       "no closed round has a result adapter")
    ref = closed[-1]["result_adapter_ref"]

    try:
        from ganymede.coordinator.config import Settings
        from ganymede.coordinator.store import Store
        from ganymede.jobtypes.collab_lora_finetune import aggregate
        from ganymede.trainer import evaluate as eval_mod
        from ganymede.trainer import model as model_mod
    except ImportError as exc:
        return Verdict("generations", title, UNKNOWN,
                       f"needs the trainer extra (`pip install -e '.[trainer]'`): {exc}")

    try:
        store = Store(Settings.from_env().storage)
        adapter = aggregate.load_adapter(store.get_bytes(ref))
        base = model_mod.load_base(run["base_model"], run["base_precision"])
        tokenizer = model_mod.load_tokenizer(run["base_model"])
        peft_model = model_mod.attach_lora(
            base, json.loads(run["lora_cfg_json"]), init_from=adapter)
        completions = eval_mod.smoke_generate(
            peft_model, tokenizer,
            prompt_format=json.loads(run["hyperparams_json"]).get(
                "prompt_format", "dolly-v1"))
    except Exception as exc:  # noqa: BLE001 - any of storage/model/decode
        return Verdict("generations", title, UNKNOWN,
                       f"could not decode: {type(exc).__name__}: {exc}")

    flagged = [{"instruction": c["instruction"], "completion": c["completion"],
                "why": why}
               for c in completions
               for why in [_generation_problem(c["completion"])] if why]

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "generations.json").write_text(
            json.dumps(completions, indent=2), encoding="utf-8")

    ok = not flagged
    return Verdict(
        "generations", title, PASS if ok else FAIL,
        f"{len(completions)} prompts decoded, {len(flagged)} flagged"
        + ("" if ok else f": {', '.join(sorted({f['why'] for f in flagged}))}")
        + ". Mechanical checks only -- read generations.json before believing a pass.",
        {"flagged": flagged, "adapter_ref": ref, "prompts": len(completions)},
    )


# Dolly-v1's section markers. A completion containing one means the model is
# reproducing the training template instead of answering -- the specific failure
# `evaluate.py` calls out as barely moving the loss curve.
_TEMPLATE_MARKERS = ("### Instruction:", "### Response:", "### Input:")


def _generation_problem(text: str) -> str | None:
    """The mechanical half of "did it collapse". Deliberately conservative:
    every check here is a thing that is wrong regardless of taste."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    for marker in _TEMPLATE_MARKERS:
        if marker in text:
            return "template leak"
    words = stripped.split()
    if len(words) >= 12:
        # A degenerate loop repeats a short phrase until the token budget runs
        # out. Unique-word ratio catches that without a threshold on length.
        if len(set(words)) / len(words) < 0.35:
            return "repetition"
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def collect(conn: sqlite3.Connection, run: sqlite3.Row, baseline: dict | None,
            *, generations: bool, out_dir: Path | None,
            cap: float, floor: float, ceiling: float) -> list[Verdict]:
    run_id = run["id"]
    rounds = round_rows(conn, run_id)
    steps = accepted_steps_by_round(conn, run_id)
    mismatch = baseline_mismatch(run, baseline)
    verdicts = [
        criterion_loss_vs_steps(rounds, steps, baseline, mismatch),
        criterion_loss_vs_wallclock(rounds, baseline, mismatch),
        criterion_generations(run, rounds, out_dir) if generations else Verdict(
            "generations",
            "greedy generations show no collapse or template corruption",
            UNKNOWN, "not requested (pass --generations; needs the trainer extra "
                     "and storage credentials)"),
        criterion_divergence(rounds),
        criterion_combine_mode(run, rounds),
        criterion_budgets(conn, run_id, floor, ceiling),
        criterion_dominance(steps, cap),
    ]
    violations = invariants.check(conn)
    verdicts.append(Verdict(
        "invariants", "the database holds no structural violations",
        PASS if not violations else FAIL,
        "clean" if not violations
        else f"{len(violations)} violation(s): " + "; ".join(str(v) for v in violations[:5]),
        {"violations": [str(v) for v in violations]},
    ))
    return verdicts


def render(run: sqlite3.Row, verdicts: list[Verdict],
           coverage: dict, stream=None) -> None:
    # Resolved here, not as a default argument: a default binds ``sys.stdout``
    # once at import, so anything that replaces it afterwards -- a test's
    # capture, a caller redirecting output to a file -- is written past.
    stream = stream if stream is not None else sys.stdout
    print(f"run {run['id']}  status={run['status']}  "
          f"combine_mode={run['combine_mode']}  "
          f"round {run['current_round']}/{run['target_rounds']}", file=stream)
    print(f"coverage: {coverage['distinct_trained']}/{coverage['buckets']} buckets "
          f"trained, spread {coverage['spread']}", file=stream)
    mismatch = next((v.data.get("mismatch") for v in verdicts
                     if v.data.get("mismatch")), None)
    if mismatch:
        print(f"BASELINE IGNORED -- {mismatch}", file=stream)
    print("", file=stream)
    for v in verdicts:
        print(f"[{_MARK[v.state]}] {v.title}", file=stream)
        for line in _wrap(v.detail):
            print(f"        {line}", file=stream)
    print("", file=stream)

    failed = [v for v in verdicts if v.state == FAIL]
    unknown = [v for v in verdicts if v.state == UNKNOWN]
    print(f"{len(verdicts) - len(failed) - len(unknown)} passed, "
          f"{len(failed)} failed, {len(unknown)} not evaluated", file=stream)
    if unknown:
        # Loudly, because a criterion nobody measured is the one that gets
        # remembered as having passed.
        print("not evaluated: " + ", ".join(v.key for v in unknown), file=stream)


def _wrap(text: str, width: int = 88) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def exit_code(verdicts: list[Verdict]) -> int:
    if any(v.state == FAIL for v in verdicts):
        return 1
    if any(v.state == UNKNOWN for v in verdicts):
        return 2
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ganymede-verdict",
        description="Answer M4b's exit criteria from a run's own tables.")
    p.add_argument("--db", required=True,
                   help="path to the coordinator database (a copy is fine)")
    p.add_argument("--run-id", default=None,
                   help="which run (default: the most recently created)")
    p.add_argument("--baseline", default="baseline.json",
                   help="single-node reference produced by ganymede-baseline")
    p.add_argument("--generations", action="store_true",
                   help="also decode the fixed prompt set through the final "
                        "adapter; needs the trainer extra and storage credentials")
    p.add_argument("-o", "--out", default=None,
                   help="write the full verdict as JSON here (the A/B is a diff "
                        "of two of these)")
    p.add_argument("--dominance-cap", type=float, default=2.0,
                   help="multiple of the median a worker may contribute (3.5)")
    p.add_argument("--budget-floor", type=float, default=0.5,
                   help="a worker finishing under this fraction of its budget "
                        "means the round was under-filled")
    p.add_argument("--budget-ceiling", type=float, default=1.05,
                   help="a worker over this fraction overran its budget")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    db = Path(args.db)
    if not db.exists():
        print(f"no such database: {db}", file=sys.stderr)
        return 2
    # Read-only intent, and no db.connect(): that would create the file and run
    # migrations against someone's copied artifact.
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        run = pick_run(conn, args.run_id)
        if run is None:
            print("no runs in this database", file=sys.stderr)
            return 2

        baseline = None
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        out = Path(args.out) if args.out else None
        verdicts = collect(
            conn, run, baseline, generations=args.generations,
            out_dir=out.parent if out else None,
            cap=args.dominance_cap, floor=args.budget_floor,
            ceiling=args.budget_ceiling)
        cov = invariants.coverage(conn, run["id"])
        render(run, verdicts, cov)

        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run": {k: run[k] for k in run.keys()},
                "coverage": cov,
                "criteria": [v.as_dict() for v in verdicts],
            }, indent=2, default=str), encoding="utf-8")
            print(f"wrote {out}", file=sys.stderr)
        return exit_code(verdicts)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
