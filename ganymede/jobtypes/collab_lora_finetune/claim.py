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
from ganymede.coordinator import devices as devices_mod
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
    agreed_at: str | None = None,
    *,
    free_devices: list[int],
    gpu_count: int = 1,
    active_task_ids: object = None,
) -> TaskSpec | None:
    """Lease one task, or return None meaning 204 No Content.

    None is a legitimate answer, not a failure: no open round, nothing this
    worker is eligible for, too little time left in the round to finish
    anything worth aggregating, or (docs/14 §5.3) a device-allocation race
    this call lost -- transient, not a capability mismatch, and expected to
    succeed on the worker's next poll.

    ``free_devices`` is required and has no default. A caller that omits it is a bug at that call site, not a machine with no
    cards -- and the two must not look alike. Defaulting it to an empty list
    would turn a forgotten argument into a silent, permanent refusal: the
    worker gets 204 forever and nothing anywhere errors. Required, so the
    mistake is a TypeError at the call site instead.

    ``free_devices`` / ``gpu_count`` come from the generic walk's capacity
    gate (docs/14 §5.2), already validated there (``gpu_count`` is never
    non-positive by the time it reaches here -- ``devices.allocate`` raises on
    that, and an exception raised mid-walk would be a ``break``, so the walk
    checks first). ``active_task_ids`` is the worker's own claimed set,
    forwarded from the request; see the held-lease comment below for why this
    function needs it now.
    """
    now = now or utcnow()
    active_ids = active_task_ids or ()

    try:
        with immediate(conn):
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None or run["status"] != "active":
                return None

            if not budget_mod.clearance_and_terms_permit(
                contributor_clearance, run["data_classification"], agreed_at
            ):
                raise NotEligible(
                    f"clearance {contributor_clearance!r} or unagreed terms < "
                    f"classification {run['data_classification']!r}"
                )

            ok, why = budget_mod.is_eligible(profile, json.loads(run["requires_json"]))
            if not ok:
                raise NotEligible(why or "profile does not meet run requirements")

            # The image requirement is eligibility, not a worker-side courtesy
            # check. The worker checks it too (4.2 step 5), but only as defense
            # in depth against a mismatch that appeared after registration -- if
            # the filter lived *only* there, every ineligible worker would be
            # handed a lease it must immediately abandon, marking a shard spoken
            # for and churning the bucket counters once per poll interval,
            # forever. Observed live before this check existed.
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

            # docs/14 §1 replaces "one lease per machine" with "one lease per
            # *device*", so a held lease on this exact run+round no longer
            # blocks a new one by itself -- a multi-GPU host taking several
            # collab_lora_finetune leases on the same run is the intended
            # shape (docs/14 §7), not a race to prevent. What is still a race
            # to prevent: this exact HTTP call being retried (a network blip,
            # the crash-recovery case) before the worker ever learned the
            # first attempt's task id. ``active_task_ids`` is how the caller
            # tells this call what the worker already knows; a held row *not*
            # in it is unknown to the worker and gets resumed rather than
            # forked. app.py's outer reconcile makes this same check before
            # the walk even starts (docs/14 §5.1); repeating it here,
            # authoritatively, under the write lock, closes the race window
            # between that read and this transaction's start -- the same
            # reason the old global held-check lived here at all. A held row
            # that *is* in ``active_task_ids`` is simply not this function's
            # business: it is a lease the worker is knowingly keeping while it
            # asks for one more device, and the section below mints a new one
            # for it rather than mistaking it for a retry.
            held_rows = conn.execute(
                """SELECT * FROM tasks WHERE worker_id = ? AND run_id = ?
                    AND round_idx = ? AND status = 'leased'
                    ORDER BY leased_at, id""",
                (worker_id, run_id, rnd["idx"]),
            ).fetchall()
            resumable = next(
                (t for t in held_rows if t["id"] not in active_ids), None
            )
            if resumable is not None:
                return _task_spec(
                    resumable, run, rnd, devices_mod.held_devices(conn, resumable["id"])
                )

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

            # ``job_id`` is stamped on the task so ``GET /v1/admin/queue`` can count
            # leased tasks per job (docs/07 §4) and ``worker_eligibility`` stays
            # keyed by job. ``run`` is ``SELECT *`` so it carries the column added
            # in migration 003.
            job_id = run["job_id"] if "job_id" in run.keys() else None
            conn.execute(
                # ``leased_at`` duplicates ``created_at`` on this path and does so
                # on purpose: a dynamic type inserts its task rows already
                # ``leased``, so the two *are* the same instant here, while a static
                # type plans at enqueue and leases much later. Share accounting
                # (docs/13 §1.2) reads one column for both.
                """INSERT INTO tasks
                     (id, run_id, round_idx, job_id, buckets_json, local_steps, status,
                      worker_id, lease_expires_at, leased_at, attempts, max_runtime_sec,
                      gpu_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'leased', ?, ?, ?, 1, ?, ?, ?)""",
                (task_id, run_id, rnd["idx"], job_id, json.dumps(buckets), plan.local_steps,
                 worker_id, _iso(expires), _iso(now), plan.usable_sec, gpu_count, _iso(now)),
            )
            # The allocation lives inside this same transaction (docs/14 §5.3),
            # not around it -- the partial unique index is what makes two
            # concurrent claims on this worker safe even if the free set each
            # one read was already stale. A lost race raises rather than
            # returning: the task row above and the bucket counters below must
            # not survive a call that could not actually back them with a
            # device (``devices.AllocationRaced``'s own docstring).
            allocated = devices_mod.allocate(
                conn, worker_id, task_id, gpu_count, free_devices
            )
            if allocated is None:
                raise devices_mod.AllocationRaced()
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
            return _task_spec(task_row, run, rnd, allocated)
    except devices_mod.AllocationRaced:
        return None


def _task_spec(task: sqlite3.Row, run: sqlite3.Row, rnd: sqlite3.Row,
               devices: list[int]) -> TaskSpec:
    return TaskSpec(
        id=task["id"],
        run_id=task["run_id"],
        round_idx=task["round_idx"],
        job_id=run["job_id"] if "job_id" in run.keys() else None,
        buckets=json.loads(task["buckets_json"]),
        num_buckets=int(run["num_buckets"]),
        local_steps=task["local_steps"],
        # Defaulted for rows written before the column existed; the lease still
        # bounds them, so the fallback is a belt rather than the braces.
        max_runtime_sec=int(task["max_runtime_sec"] or DEFAULT_MAX_RUNTIME_SEC),
        lease_expires_at=_parse(task["lease_expires_at"]),
        base_adapter_ref=rnd["base_adapter_ref"],
        base_model=run["base_model"],
        devices=devices,
        base_precision=run["base_precision"],
        lora_cfg=json.loads(run["lora_cfg_json"]),
        hyperparams=json.loads(run["hyperparams_json"]),
        dataset_ref=run["dataset_ref"],
        required_image=run["required_image"],
    )
