"""Round lifecycle for ``collab_lora_finetune`` (ex-``coordinator/rounds.py``).

Relocated verbatim in Phase A (docs/10-jobtype-sdk.md §3). A round is this
type's ``reduce`` checkpoint (§5): the generic layer knows jobs, tasks, and an
optional reduce-epoch counter; everything about rounds is private here.

The load-bearing idea is unchanged: a round closes on **accumulated work**,
never on a worker count. A one-machine round is a legitimate round.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime

from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, _parse, utcnow

__all__ = [
    "open_round",
    "current_round",
    "round_progress",
    "should_close",
    "reopen_empty_round",
    "task_seed",
    "update_throughput",
]


# --------------------------------------------------------------------------
# Round lifecycle
# --------------------------------------------------------------------------


def open_round(
    conn: sqlite3.Connection,
    run_id: str,
    idx: int,
    base_adapter_ref: str,
    target_steps: int,
    min_round_sec: int,
    max_round_sec: int,
) -> None:
    now = _iso(utcnow())
    with immediate(conn):
        conn.execute(
            """INSERT INTO rounds
                 (run_id, idx, base_adapter_ref, status, target_steps,
                  min_round_sec, max_round_sec, opened_at)
               VALUES (?, ?, ?, 'open', ?, ?, ?, ?)""",
            (run_id, idx, base_adapter_ref, target_steps, min_round_sec, max_round_sec, now),
        )
        conn.execute("UPDATE runs SET current_round = ? WHERE id = ?", (idx, run_id))


def current_round(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT r.* FROM rounds r
           JOIN runs ON runs.id = r.run_id AND runs.current_round = r.idx
           WHERE r.run_id = ?""",
        (run_id,),
    ).fetchone()


def round_progress(conn: sqlite3.Connection, run_id: str, round_idx: int) -> dict:
    """Accumulated work in a round: what the close decision is made against.

    Counts only *accepted-or-ungated* submissions. A submission already rejected
    by the structural gates contributed nothing and must not push the round over
    its target, or a fleet with a broken worker would close rounds on garbage.
    """
    row = conn.execute(
        """SELECT COALESCE(SUM(s.steps_completed), 0) AS steps,
                  COUNT(*)                            AS n_subs,
                  COUNT(DISTINCT w.contributor_id)    AS n_contributors
           FROM submissions s
           JOIN tasks   t ON t.id = s.task_id
           LEFT JOIN workers w ON w.id = t.worker_id
           WHERE t.run_id = ? AND t.round_idx = ?
             AND (s.accepted IS NULL OR s.accepted = 1)""",
        (run_id, round_idx),
    ).fetchone()
    return {
        "steps": int(row["steps"]),
        "submissions": int(row["n_subs"]),
        "distinct_contributors": int(row["n_contributors"]),
    }


def should_close(
    conn: sqlite3.Connection,
    run_id: str,
    round_idx: int,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """The 3.2 close rule. Returns (close?, reason).

        close when:  steps >= target AND elapsed >= min_round_sec
                 OR  elapsed >= max_round_sec AND submissions >= 1

    The ``min_round_sec`` floor on the first branch stops a single very fast
    worker from closing a round in ninety seconds and burning through the
    round budget before anyone else has finished downloading the base adapter.

    The ``submissions >= 1`` guard on the second stops an empty round from
    "closing" into an aggregation over nothing. An empty round at the backstop
    reopens quietly -- see ``reopen_empty_round``. That is the expected state
    for an unscheduled fleet overnight, not an incident.
    """
    now = now or utcnow()
    rnd = conn.execute(
        "SELECT * FROM rounds WHERE run_id = ? AND idx = ?", (run_id, round_idx)
    ).fetchone()
    if rnd is None or rnd["status"] != "open":
        return False, "not open"

    elapsed = (now - _parse(rnd["opened_at"])).total_seconds()
    prog = round_progress(conn, run_id, round_idx)

    if prog["steps"] >= rnd["target_steps"] and elapsed >= rnd["min_round_sec"]:
        return True, "target_steps"
    if elapsed >= rnd["max_round_sec"] and prog["submissions"] >= 1:
        return True, "max_round_sec"
    return False, "in progress"


def reopen_empty_round(
    conn: sqlite3.Connection, run_id: str, round_idx: int, now: datetime | None = None
) -> bool:
    """Restart the clock on a round that hit its backstop with no submissions.

    Deliberately silent. With no reliable machine schedule, "nobody was around
    for forty minutes" is normal operation. Alerting on it would train the
    operator to ignore alerts.
    """
    now = now or utcnow()
    with immediate(conn):
        rnd = conn.execute(
            "SELECT * FROM rounds WHERE run_id = ? AND idx = ?", (run_id, round_idx)
        ).fetchone()
        if rnd is None or rnd["status"] != "open":
            return False
        elapsed = (now - _parse(rnd["opened_at"])).total_seconds()
        if elapsed < rnd["max_round_sec"]:
            return False
        prog = round_progress(conn, run_id, round_idx)
        if prog["submissions"] > 0:
            return False
        conn.execute(
            "UPDATE rounds SET opened_at = ? WHERE run_id = ? AND idx = ?",
            (_iso(now), run_id, round_idx),
        )
    return True


def _pick_buckets(
    conn: sqlite3.Connection, run_id: str, n: int, total_buckets: int
) -> list[int]:
    """Least-trained-first, so coverage stays even as workers come and go.

    Ties break on bucket index rather than randomly: with a single worker doing
    consecutive rounds, random tie-breaking would revisit the same buckets by
    chance while others went untouched.
    """
    rows = conn.execute(
        """SELECT bucket_idx FROM buckets WHERE run_id = ?
           ORDER BY times_trained ASC, last_round IS NOT NULL, last_round ASC, bucket_idx ASC
           LIMIT ?""",
        (run_id, n),
    ).fetchall()
    picked = [int(r["bucket_idx"]) for r in rows]
    # A run whose buckets table was never seeded still has to be workable.
    if not picked:
        picked = list(range(min(n, total_buckets)))
    return picked


def task_seed(run_id: str, round_idx: int, task_id: str) -> int:
    """Deterministic per-task seed for shuffling within the assigned buckets.

    Derived rather than stored, and derived from the task id rather than the
    round: two workers in the same round must not walk their shards in a
    correlated order, and a task retried after a lease expiry should reproduce
    the ordering the first attempt used. blake2b rather than hash() because
    Python salts str hashing per process, which would make this irreproducible
    across a coordinator restart.
    """
    digest = hashlib.blake2b(
        f"{run_id}/{round_idx}/{task_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**31)


def update_throughput(
    conn: sqlite3.Connection,
    run_id: str,
    gpu_model: str,
    steps_per_min: float,
    now: datetime | None = None,
) -> None:
    """Fold an observation into the running throughput estimate.

    An exponentially weighted mean rather than a plain average: a machine that
    was thermally throttled last week should not hold down its own estimate
    forever, and a shared desktop's speed genuinely varies with what else is
    running on it.
    """
    now = now or utcnow()
    alpha = 0.3
    with immediate(conn):
        row = conn.execute(
            "SELECT steps_per_min, samples FROM throughput WHERE run_id = ? AND gpu_model = ?",
            (run_id, gpu_model),
        ).fetchone()
        if row is None:
            conn.execute(
                """INSERT INTO throughput (run_id, gpu_model, steps_per_min, samples, updated_at)
                   VALUES (?, ?, ?, 1, ?)""",
                (run_id, gpu_model, steps_per_min, _iso(now)),
            )
        else:
            blended = (1 - alpha) * row["steps_per_min"] + alpha * steps_per_min
            conn.execute(
                """UPDATE throughput SET steps_per_min = ?, samples = samples + 1, updated_at = ?
                   WHERE run_id = ? AND gpu_model = ?""",
                (blended, _iso(now), run_id, gpu_model),
            )
