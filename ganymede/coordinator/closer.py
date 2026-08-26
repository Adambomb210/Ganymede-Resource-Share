"""Closing a round: cohort gate, combine, publish, advance.

docs/02-architecture-v2.md sections 5.1, 5.2 and 3.1.

The gates are split across two moments on purpose. The structural gates (1-3, 5
in 5.1) run at **submit** time, because they are per-submission and a worker can
be told immediately that its adapter was malformed -- feedback at the moment the
bug happened rather than forty minutes later. Gate 4 is cohort-relative: an
adapter is an outlier only with respect to the other adapters in the round, so
it cannot run until the round is done.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import weakref
from dataclasses import dataclass
from datetime import datetime

from ganymede.coordinator import aggregate, rounds
from ganymede.coordinator.config import DEFAULT_DOMINANCE_CAP, DEFAULT_NORM_REJECT_K
from ganymede.coordinator.db import immediate
from ganymede.coordinator.store import Store, base_adapter_key, momentum_key

log = logging.getLogger("ganymede.coordinator.closer")


@dataclass(frozen=True)
class CloseResult:
    run_id: str
    round_idx: int
    reason: str
    accepted: int
    rejected: int
    total_steps: int
    distinct_contributors: int
    result_adapter_ref: str | None
    divergence: float | None
    next_round_opened: bool


# A round's base adapter never changes once written, so its manifest is safe to
# cache by key. Without this every submit re-downloads and re-parses ~25 MB just
# to learn a key set the coordinator already knew -- eight workers a round would
# pay for that eight times over an artifact none of them changed.
# Keyed by the Store the bytes actually live in, not by the ref string alone. A
# key path is only unique *within* a bucket: the same run id recreated against
# fresh storage, or a process holding two stores, would otherwise be served a
# manifest describing an adapter it never read. Weak keys so a discarded Store
# takes its cache with it.
_MANIFEST_CACHE: "weakref.WeakKeyDictionary[object, dict[str, dict]]" = (
    weakref.WeakKeyDictionary()
)


def expected_manifest(store: Store, base_adapter_ref: str) -> dict:
    """Key set and shapes a submission must match, from the round's base adapter."""
    per_store = _MANIFEST_CACHE.setdefault(store, {})
    cached = per_store.get(base_adapter_ref)
    if cached is None:
        cached = aggregate.manifest_of(
            aggregate.load_adapter(store.get_bytes(base_adapter_ref))
        )
        # One entry per round; a long run would otherwise accumulate one dict
        # per round for the lifetime of the process.
        if len(per_store) > 4:
            per_store.clear()
        per_store[base_adapter_ref] = cached
    return cached


def gate_submission(
    conn: sqlite3.Connection,
    store: Store,
    task_id: str,
    expected: dict,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Run the structural gates against a just-received artifact.

    Records the verdict on the submission row and, on rejection, in the audit
    log keyed by contributor -- that log is the raw material for Phase 2
    reputation scoring, so it accumulates from day one rather than being
    retrofitted.
    """
    now = now or rounds.utcnow()
    sub = conn.execute("SELECT * FROM submissions WHERE task_id = ?", (task_id,)).fetchone()
    if sub is None:
        return False, "no_submission"

    task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    hb = rounds.last_heartbeat_steps(conn, task_id)

    try:
        raw = store.get_bytes(sub["artifact_ref"])
    except Exception:
        verdict, _ = aggregate.GateResult(False, "missing_artifact"), None
        _record_verdict(conn, task_id, task, verdict, now)
        return False, "missing_artifact"

    verdict, _adapter = aggregate.check_structural(
        raw, expected, sub["steps_completed"], heartbeat_steps=hb
    )
    _record_verdict(conn, task_id, task, verdict, now)
    return verdict.accepted, verdict.reason


def _record_verdict(conn, task_id, task, verdict, now) -> None:
    with immediate(conn):
        conn.execute(
            "UPDATE submissions SET accepted = ?, reject_reason = ? WHERE task_id = ?",
            (1 if verdict.accepted else 0, verdict.reason, task_id),
        )
        if not verdict.accepted and task is not None:
            row = conn.execute(
                "SELECT contributor_id FROM workers WHERE id = ?", (task["worker_id"],)
            ).fetchone()
            conn.execute(
                """INSERT INTO audit (at, contributor_id, worker_id, event, detail_json)
                   VALUES (?, ?, ?, 'submission_rejected', ?)""",
                (
                    rounds._iso(now),
                    row["contributor_id"] if row else None,
                    task["worker_id"],
                    json.dumps({"task": task_id, "reason": verdict.reason,
                                "detail": verdict.detail}),
                ),
            )


def close_round(
    conn: sqlite3.Connection,
    store: Store,
    run_id: str,
    round_idx: int,
    reason: str,
    now: datetime | None = None,
    settings=None,
) -> CloseResult | None:
    """Aggregate a round's accepted submissions and advance the run.

    Returns None when another caller is already closing this round. Closing is
    driven opportunistically from the request path, so two workers whose
    submissions land together will both evaluate the close rule and both find
    it satisfied -- that is the normal case, not an error.
    """
    now = now or rounds.utcnow()
    norm_k = getattr(settings, "norm_reject_k", DEFAULT_NORM_REJECT_K)
    cap = getattr(settings, "dominance_cap", DEFAULT_DOMINANCE_CAP)

    # Claim the close atomically. Reading the status and then writing it in a
    # separate statement leaves a window where two callers both see 'open' and
    # both run the whole aggregate-and-advance path; the loser then fails on the
    # rounds-table UNIQUE constraint, which reaches a blameless worker as a 500.
    # The conditional UPDATE makes exactly one caller the winner, and it is the
    # same write that fences off late submissions with a clean 409.
    with immediate(conn):
        claimed = conn.execute(
            """UPDATE rounds SET status = 'closing'
               WHERE run_id = ? AND idx = ? AND status = 'open'""",
            (run_id, round_idx),
        ).rowcount
    if not claimed:
        return None

    try:
        return _close_claimed_round(conn, store, run_id, round_idx, reason, now,
                                    norm_k, cap)
    except Exception:
        # Give the round back. Everything between claiming 'closing' and the
        # status update below is storage I/O and tensor arithmetic, any of which
        # can fail -- and 'closing' is claimed by exactly one caller and
        # released only by that caller finishing. Leaving it set would wedge the
        # round permanently: no worker could claim it, no submission could
        # reopen it, and nothing would report an error. Every worker would
        # simply get 204 forever.
        #
        # Reopening is safe to retry. The result adapter is written under a key
        # derived from (run, round), so a second attempt overwrites the same
        # object rather than accumulating; the submissions it aggregates are
        # unchanged; and the close rule that fired once will fire again on the
        # next submit or claim.
        conn.execute(
            """UPDATE rounds SET status = 'open'
               WHERE run_id = ? AND idx = ? AND status = 'closing'""",
            (run_id, round_idx),
        )
        log.exception("closing round %s#%s failed; reopened for retry", run_id, round_idx)
        raise


def _close_claimed_round(
    conn: sqlite3.Connection,
    store: Store,
    run_id: str,
    round_idx: int,
    reason: str,
    now: datetime,
    norm_k: float,
    cap: float,
) -> CloseResult | None:
    """The body of ``close_round``, after the close has been claimed."""
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    rnd = conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? AND idx = ?", (run_id, round_idx)
    ).fetchone()

    subs = conn.execute(
        """SELECT s.*, t.worker_id FROM submissions s
           JOIN tasks t ON t.id = s.task_id
           WHERE t.run_id = ? AND t.round_idx = ? AND s.accepted = 1
           ORDER BY s.received_at""",
        (run_id, round_idx),
    ).fetchall()

    base = aggregate.load_adapter(store.get_bytes(rnd["base_adapter_ref"]))
    keys = sorted(base.keys())

    adapters: list[dict] = []
    kept: list[sqlite3.Row] = []
    for s in subs:
        try:
            adapters.append(aggregate.load_adapter(store.get_bytes(s["artifact_ref"])))
            kept.append(s)
        except Exception:
            # Already passed the structural gates at submit; a failure here means
            # the object went missing between then and now. Drop it and carry on
            # rather than failing the whole round for one absent artifact.
            _record_verdict(conn, s["task_id"], None,
                            aggregate.GateResult(False, "missing_artifact"), now)

    rejected = len(subs) - len(kept)

    # Gate 4: cohort-relative outlier rejection, only meaningful now.
    if adapters:
        norm_results = aggregate.check_norms(adapters, k=norm_k)
        surviving, surviving_subs = [], []
        for adapter, sub, res in zip(adapters, kept, norm_results, strict=True):
            if res.accepted:
                surviving.append(adapter)
                surviving_subs.append(sub)
            else:
                rejected += 1
                task = conn.execute(
                    "SELECT * FROM tasks WHERE id = ?", (sub["task_id"],)
                ).fetchone()
                _record_verdict(conn, sub["task_id"], task, res, now)
        adapters, kept = surviving, surviving_subs

    total_steps = sum(int(s["steps_completed"]) for s in kept)
    contributors = conn.execute(
        """SELECT COUNT(DISTINCT w.contributor_id) AS n FROM submissions s
           JOIN tasks t ON t.id = s.task_id
           JOIN workers w ON w.id = t.worker_id
           WHERE t.run_id = ? AND t.round_idx = ? AND s.accepted = 1""",
        (run_id, round_idx),
    ).fetchone()["n"]

    result_ref: str | None = None
    divergence: float | None = None

    if adapters:
        weights = aggregate.dense_weights(
            [int(s["steps_completed"]) for s in kept], keys, cap=cap
        )
        momentum = None
        if run["outer_momentum_ref"]:
            try:
                momentum = aggregate.load_adapter(store.get_bytes(run["outer_momentum_ref"]))
            except Exception:
                momentum = None  # rebuilt from zeros; costs one round of momentum

        combined, new_momentum = aggregate.combine(
            base, adapters, weights,
            mode=run["combine_mode"],
            lr_outer=float(run["lr_outer"]),
            beta=float(run["outer_beta"]),
            momentum=momentum,
        )
        divergence = aggregate.adapter_divergence(adapters)

        result_ref = base_adapter_key(run_id, round_idx + 1)
        store.put_bytes(result_ref, aggregate.save_adapter(combined))
        if new_momentum is not None:
            mref = momentum_key(run_id)
            store.put_bytes(mref, aggregate.save_adapter(new_momentum))
            with immediate(conn):
                conn.execute(
                    "UPDATE runs SET outer_momentum_ref = ? WHERE id = ?", (mref, run_id)
                )

    with immediate(conn):
        conn.execute(
            """UPDATE rounds
               SET status = 'closed', closed_at = ?, result_adapter_ref = ?,
                   distinct_contributors = ?, adapter_divergence = ?
               WHERE run_id = ? AND idx = ?""",
            (rounds._iso(now), result_ref, contributors, divergence, run_id, round_idx),
        )
        # Any lease still outstanding belongs to a round that no longer exists.
        conn.execute(
            """UPDATE tasks SET status = 'expired', lease_expires_at = NULL
               WHERE run_id = ? AND round_idx = ? AND status = 'leased'""",
            (run_id, round_idx),
        )

    # Fold observed throughput back in, so the next round's budgets are measured
    # rather than guessed. This is what makes the cold-start default cheap.
    for s in kept:
        metrics = json.loads(s["metrics_json"] or "{}")
        spm, gpu = metrics.get("steps_per_min"), metrics.get("gpu_model")
        if spm and gpu:
            rounds.update_throughput(conn, run_id, gpu, float(spm), now)

    next_opened = False
    if result_ref is not None and round_idx + 1 < run["target_rounds"]:
        rounds.open_round(
            conn, run_id, round_idx + 1, result_ref,
            target_steps=rnd["target_steps"],
            min_round_sec=rnd["min_round_sec"],
            max_round_sec=rnd["max_round_sec"],
        )
        next_opened = True
    elif result_ref is not None:
        with immediate(conn):
            conn.execute("UPDATE runs SET status = 'done' WHERE id = ?", (run_id,))

    return CloseResult(
        run_id=run_id, round_idx=round_idx, reason=reason,
        accepted=len(kept), rejected=rejected, total_steps=total_steps,
        distinct_contributors=int(contributors), result_adapter_ref=result_ref,
        divergence=divergence, next_round_opened=next_opened,
    )


def maybe_close(
    conn: sqlite3.Connection,
    store: Store,
    run_id: str,
    now: datetime | None = None,
    settings=None,
) -> CloseResult | None:
    """Evaluate the close rule for a run's current round and act on it.

    Called opportunistically from the request path (after a submit) rather than
    from a background scheduler: with rounds measured in tens of minutes, a
    close that lands a few seconds late costs nothing, and one fewer moving
    part is worth more than the precision.
    """
    now = now or rounds.utcnow()
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["status"] != "active":
        return None

    idx = int(run["current_round"])
    close, reason = rounds.should_close(conn, run_id, idx, now)
    if close:
        return close_round(conn, store, run_id, idx, reason, now, settings)

    # Backstop reached with nothing submitted: restart the clock, silently.
    rounds.reopen_empty_round(conn, run_id, idx, now)
    return None
