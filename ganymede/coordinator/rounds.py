"""Round lifecycle, lease management, and the close decision.

docs/02-architecture-v2.md sections 3.1, 3.2 and 6.4.

The load-bearing idea here is that a round closes on **accumulated work**, never
on a worker count. The fleet is unscheduled -- machines appear and vanish -- so
any quorum rule would need retuning every time the fleet's shape changed, and
would deadlock whenever it did not. A one-machine round is a legitimate round,
not a degenerate case.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ganymede.coordinator import budget as budget_mod
from ganymede.coordinator.config import COLD_START_STEPS_PER_MIN
from ganymede.coordinator.db import immediate

# Fallback wall-clock ceiling for task rows written before max_runtime_sec
# existed. One hour matches the default lease, so the two bounds agree.
DEFAULT_MAX_RUNTIME_SEC = 3600


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class RoundClosed(Exception):
    """The round moved on while a worker was still holding a lease. -> HTTP 409."""


class LeaseLost(Exception):
    """The lease expired or was reassigned. The worker must stop. -> HTTP 410."""


class NotEligible(Exception):
    """Worker or contributor may not work this run. Carries a reason for the log."""


@dataclass(frozen=True)
class TaskSpec:
    id: str
    run_id: str
    round_idx: int
    buckets: list[int]
    # Total bucket count for the run. The worker receives bucket *indices* and
    # has to turn them into rows itself -- the coordinator never sees the data --
    # which it cannot do without knowing how many buckets the dataset was cut
    # into. Sending the indices alone is not a shard assignment.
    num_buckets: int
    local_steps: int
    # Wall-clock safety net (8). The worker stops at whichever comes first, this
    # or lease_expires_at -- they answer different questions: this one is "how
    # long was this work budgeted", the lease is "when does the coordinator stop
    # believing you".
    max_runtime_sec: int
    lease_expires_at: datetime
    base_adapter_ref: str
    base_model: str
    base_precision: str
    lora_cfg: dict
    hyperparams: dict
    dataset_ref: str
    # Image tag the worker must be running, or None for no requirement (4.2
    # step 5). A worker that cannot honour it abandons before downloading
    # anything rather than submitting an artifact from the wrong stack.
    required_image: str | None


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


# --------------------------------------------------------------------------
# Leases
# --------------------------------------------------------------------------


def expire_leases(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Reclaim leases whose holder stopped heartbeating. Returns the count.

    A machine that was shut down mid-task is the common case, not an anomaly,
    so its shard has to become available again without operator involvement.
    """
    now = now or utcnow()
    with immediate(conn):
        cur = conn.execute(
            """UPDATE tasks SET status = 'expired'
               WHERE status = 'leased' AND lease_expires_at IS NOT NULL
                 AND lease_expires_at < ?""",
            (_iso(now),),
        )
        return cur.rowcount


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


def claim_task(
    conn: sqlite3.Connection,
    run_id: str,
    worker_id: str,
    contributor_clearance: str,
    profile: dict,
    settings,
    now: datetime | None = None,
    worker_image_tag: str | None = None,
) -> TaskSpec | None:
    """Lease one task, or return None meaning 204 No Content.

    None is a legitimate answer, not a failure: no open round, nothing this
    worker is eligible for, or too little time left in the round to finish
    anything worth aggregating.
    """
    now = now or utcnow()

    with immediate(conn):
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None or run["status"] != "active":
            return None

        if not budget_mod.clearance_permits(contributor_clearance, run["data_classification"]):
            raise NotEligible(
                f"clearance {contributor_clearance!r} < classification "
                f"{run['data_classification']!r}"
            )

        ok, why = budget_mod.is_eligible(profile, json.loads(run["requires_json"]))
        if not ok:
            raise NotEligible(why or "profile does not meet run requirements")

        # The image requirement is eligibility, not a worker-side courtesy check.
        # The worker checks it too (4.2 step 5), but only as defense in depth
        # against a mismatch that appeared after registration -- if the filter
        # lived *only* there, every ineligible worker would be handed a lease it
        # must immediately abandon, marking a shard spoken for and churning the
        # bucket counters once per poll interval, forever. Observed live before
        # this check existed.
        required_image = run["required_image"]
        if required_image and required_image != worker_image_tag:
            raise NotEligible(
                f"run requires image {required_image!r}, "
                f"worker reports {worker_image_tag!r}"
            )

        rnd = conn.execute(
            "SELECT * FROM rounds WHERE run_id = ? AND idx = ? AND status = 'open'",
            (run_id, run["current_round"]),
        ).fetchone()
        if rnd is None:
            return None

        # One lease per worker. A second claim returns the lease it already
        # holds rather than a new one -- a worker that retries after a network
        # blip should resume, not fork its own work into two shards.
        held = conn.execute(
            """SELECT * FROM tasks
               WHERE worker_id = ? AND status = 'leased' AND run_id = ? AND round_idx = ?""",
            (worker_id, run_id, rnd["idx"]),
        ).fetchone()
        if held is not None:
            return _task_spec(held, run, rnd)

        remaining = rnd["max_round_sec"] - (now - _parse(rnd["opened_at"])).total_seconds()
        hp = json.loads(run["hyperparams_json"])
        gpu_model = profile.get("device_name", "unknown")

        tp_row = conn.execute(
            "SELECT steps_per_min FROM throughput WHERE run_id = ? AND gpu_model = ?",
            (run_id, gpu_model),
        ).fetchone()
        cal_row = conn.execute(
            "SELECT calibration_json FROM calibration WHERE run_id = ?", (run_id,)
        ).fetchone()
        calibrated = None
        if cal_row is not None:
            cal = json.loads(cal_row["calibration_json"])
            calibrated = cal.get("throughput", {}).get(gpu_model)

        plan = budget_mod.plan_budget(
            remaining_sec=int(remaining),
            measured=tp_row["steps_per_min"] if tp_row else None,
            calibrated=calibrated,
            cold_start=hp.get("cold_start_steps_per_min", COLD_START_STEPS_PER_MIN),
            # `micro_batch` is the name the task spec (8) and the trainer both
            # use; `batch_size` is accepted only so run configs written before
            # the trainer existed keep working. They must agree: the trainer
            # reads micro_batch, so a run that set only batch_size would have
            # the coordinator sizing budgets for one effective batch while the
            # worker trained with another -- an error of exactly the ratio
            # between them, in a number nothing else cross-checks.
            samples_per_step=(
                int(hp.get("micro_batch", hp.get("batch_size", 8)))
                * int(hp.get("grad_accum", 1))
            ),
            # No default: samples_per_bucket is a fact about the dataset, and
            # the coordinator never sees the dataset. newrun.py derives it from
            # the same plan_partition the workers use and stores it here, so a
            # run missing it is misconfigured rather than merely undecided --
            # and a wrong bucket size silently mis-sizes every budget in the run.
            samples_per_bucket=int(hp["samples_per_bucket"]),
            total_buckets=int(run["num_buckets"]),
            est_download_sec=settings.est_download_sec,
            est_upload_sec=settings.est_upload_sec,
            est_setup_sec=settings.est_setup_sec,
            safety_margin_sec=settings.safety_margin_sec,
            min_usable_sec=settings.min_usable_sec,
            target_passes=float(hp.get("target_passes", 1.0)),
        )
        if plan is None:
            return None

        # Minimum viable throughput (3.5). Below some speed a worker costs more
        # than it contributes: it holds a lease and a full ~25 MB round trip to
        # add noise-level weight. The comparison is against what this round's
        # other workers were actually budgeted, so the floor adapts to whatever
        # hardware showed up rather than encoding an absolute steps/min.
        peers = [
            int(r["local_steps"])
            for r in conn.execute(
                """SELECT local_steps FROM tasks
                   WHERE run_id = ? AND round_idx = ? AND status != 'expired'""",
                (run_id, rnd["idx"]),
            ).fetchall()
        ]
        if peers:
            peers.sort()
            mid = len(peers) // 2
            median = (peers[mid] if len(peers) % 2 else (peers[mid - 1] + peers[mid]) / 2)
            if not budget_mod.meets_floor(
                plan.local_steps, median, settings.throughput_floor_frac
            ):
                # Not an error: this worker stays eligible for other runs.
                raise NotEligible(
                    f"below throughput floor for this run: {plan.local_steps} steps "
                    f"< {settings.throughput_floor_frac:.0%} of median {median:.0f}"
                )

        buckets = _pick_buckets(conn, run_id, plan.n_buckets, int(run["num_buckets"]))
        task_id = uuid.uuid4().hex
        expires = now + timedelta(seconds=settings.lease_duration_sec)

        conn.execute(
            """INSERT INTO tasks
                 (id, run_id, round_idx, buckets_json, local_steps, status,
                  worker_id, lease_expires_at, attempts, max_runtime_sec, created_at)
               VALUES (?, ?, ?, ?, ?, 'leased', ?, ?, 1, ?, ?)""",
            (task_id, run_id, rnd["idx"], json.dumps(buckets), plan.local_steps,
             worker_id, _iso(expires), plan.usable_sec, _iso(now)),
        )
        # Mark the shard as spoken for now, not at submit. If this worker
        # vanishes the lease expires and the buckets come back round on
        # least-trained-first anyway, so the worst case is one round of slightly
        # uneven coverage -- much better than two workers training the same
        # shard because the counter had not moved yet.
        for b in buckets:
            conn.execute(
                """UPDATE buckets SET times_trained = times_trained + 1, last_round = ?
                   WHERE run_id = ? AND bucket_idx = ?""",
                (rnd["idx"], run_id, b),
            )
        conn.execute(
            "UPDATE workers SET last_seen = ?, rounds_joined = rounds_joined + 1 WHERE id = ?",
            (_iso(now), worker_id),
        )
        task_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task_spec(task_row, run, rnd)


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


def _task_spec(task: sqlite3.Row, run: sqlite3.Row, rnd: sqlite3.Row) -> TaskSpec:
    return TaskSpec(
        id=task["id"],
        run_id=task["run_id"],
        round_idx=task["round_idx"],
        buckets=json.loads(task["buckets_json"]),
        num_buckets=int(run["num_buckets"]),
        local_steps=task["local_steps"],
        # Defaulted for rows written before the column existed; the lease still
        # bounds them, so the fallback is a belt rather than the braces.
        max_runtime_sec=int(task["max_runtime_sec"] or DEFAULT_MAX_RUNTIME_SEC),
        lease_expires_at=_parse(task["lease_expires_at"]),
        base_adapter_ref=rnd["base_adapter_ref"],
        base_model=run["base_model"],
        base_precision=run["base_precision"],
        lora_cfg=json.loads(run["lora_cfg_json"]),
        hyperparams=json.loads(run["hyperparams_json"]),
        dataset_ref=run["dataset_ref"],
        required_image=run["required_image"],
    )


def heartbeat(
    conn: sqlite3.Connection,
    task_id: str,
    worker_id: str,
    steps_completed: int,
    settings,
    now: datetime | None = None,
) -> datetime:
    """Extend a lease. Raises RoundClosed (409) or LeaseLost (410)."""
    now = now or utcnow()
    with immediate(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None or task["worker_id"] != worker_id:
            raise LeaseLost("no such task for this worker")
        if task["status"] != "leased":
            raise LeaseLost(f"task is {task['status']}")

        rnd = conn.execute(
            "SELECT status FROM rounds WHERE run_id = ? AND idx = ?",
            (task["run_id"], task["round_idx"]),
        ).fetchone()
        if rnd is None or rnd["status"] != "open":
            raise RoundClosed("round is no longer open")

        expires = now + timedelta(seconds=settings.lease_duration_sec)
        # Progress only ever moves forward. A worker that restarts mid-task and
        # resumes from a checkpoint may report a lower figure than its own last
        # heartbeat; taking the max stops that from being read as the worker
        # having un-trained steps, which gate 5 would reject.
        conn.execute(
            """UPDATE tasks
               SET lease_expires_at = ?,
                   last_heartbeat_steps = MAX(COALESCE(last_heartbeat_steps, 0), ?)
               WHERE id = ?""",
            (_iso(expires), steps_completed, task_id),
        )
        conn.execute(
            """INSERT INTO audit (at, worker_id, event, detail_json)
               VALUES (?, ?, 'heartbeat', ?)""",
            (_iso(now), worker_id, json.dumps({"task": task_id, "steps": steps_completed})),
        )
        conn.execute("UPDATE workers SET last_seen = ? WHERE id = ?", (_iso(now), worker_id))
        return expires


def last_heartbeat_steps(conn: sqlite3.Connection, task_id: str) -> int | None:
    """Highest progress figure the worker reported, for acceptance gate 5."""
    row = conn.execute(
        "SELECT last_heartbeat_steps FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    return None if row is None else row["last_heartbeat_steps"]


def abandon(conn: sqlite3.Connection, task_id: str, worker_id: str) -> None:
    """Voluntary release. The host is going away; give the shard back cleanly."""
    with immediate(conn):
        conn.execute(
            """UPDATE tasks SET status = 'abandoned', lease_expires_at = NULL
               WHERE id = ? AND worker_id = ? AND status = 'leased'""",
            (task_id, worker_id),
        )


def record_submission(
    conn: sqlite3.Connection,
    task_id: str,
    worker_id: str,
    artifact_ref: str,
    steps_completed: int,
    tokens_seen: int,
    metrics: dict,
    now: datetime | None = None,
) -> None:
    """Persist a submission and close out its lease. Gating happens separately.

    Submission and gating are deliberately separate: the bytes are durably
    recorded before any validation runs, so a coordinator crash inside the
    gates cannot lose work a worker already did.
    """
    now = now or utcnow()
    with immediate(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None or task["worker_id"] != worker_id:
            raise LeaseLost("no such task for this worker")
        if task["status"] != "leased":
            raise LeaseLost(f"task is {task['status']}")

        rnd = conn.execute(
            "SELECT status FROM rounds WHERE run_id = ? AND idx = ?",
            (task["run_id"], task["round_idx"]),
        ).fetchone()
        if rnd is None or rnd["status"] != "open":
            raise RoundClosed("round is no longer open")

        conn.execute(
            """INSERT OR REPLACE INTO submissions
                 (task_id, artifact_ref, steps_completed, tokens_seen,
                  metrics_json, accepted, reject_reason, received_at)
               VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)""",
            (task_id, artifact_ref, steps_completed, tokens_seen,
             json.dumps(metrics), _iso(now)),
        )
        conn.execute(
            "UPDATE tasks SET status = 'submitted', lease_expires_at = NULL WHERE id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE workers SET steps_total = steps_total + ?, last_seen = ? WHERE id = ?",
            (steps_completed, _iso(now), worker_id),
        )


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
