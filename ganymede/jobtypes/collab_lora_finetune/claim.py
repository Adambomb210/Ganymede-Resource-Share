"""Per-machine task sizing for ``collab_lora_finetune`` (ex-``rounds.claim_task``).

Relocated verbatim in Phase A (docs/10-jobtype-sdk.md §3, "The claim seam").
``plan(job, conn)`` has no machine profile, and ``local_steps`` / bucket count
depend on the claiming machine's measured throughput (3.5), so this body is the
optional ``shape_claim`` hook. Refusal is still signalled by raising
``coordinator.rounds.NotEligible`` -- the generic claim path records it in
``worker_eligibility`` exactly as it does today.

``budget.plan_budget`` stays generic (it is not in ``05``'s move table); this
body calls it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta

from ganymede.coordinator import budget as budget_mod
from ganymede.coordinator.config import COLD_START_STEPS_PER_MIN
from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import NotEligible, _iso, _parse, utcnow
from ganymede.jobtypes.base import TaskSpec
from ganymede.jobtypes.collab_lora_finetune.plan import _pick_buckets

# Fallback wall-clock ceiling for task rows written before max_runtime_sec
# existed. One hour matches the default lease, so the two bounds agree.
DEFAULT_MAX_RUNTIME_SEC = 3600

log = logging.getLogger("ganymede.jobtypes.collab_lora_finetune.claim")


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

        if plan.data_limited:
            # The run has less data than this machine can chew through in a
            # round. Not an error -- the budget was cut to fit and the work is
            # honest -- but it means every worker in the round is being handed
            # most or all of the same dataset, so the shards have stopped being
            # shards. The fix is the run's shape (more buckets, a bigger
            # dataset, or shorter rounds), which only an operator can make, so
            # it has to be said out loud somewhere they will see it.
            log.warning(
                "run %s: worker %s is data-limited -- budget cut to %d steps over "
                "%d/%d buckets. The dataset is small relative to this fleet's speed.",
                run_id, worker_id, plan.local_steps, plan.n_buckets, run["num_buckets"],
            )

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
