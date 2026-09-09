"""Known-answer probes (docs/13 §5, docs/09 §5.1 input 3).

A spot-check is **an already-accepted shard re-issued to a different machine**.
That is the whole design, and it exists because of one requirement in the
sentence that states it: docs/09 §5.1 wants known-answer tasks "issued
indistinguishably from real work". A synthesized task with a canned expected
output fails that -- it is either not on any real job, or it is on one and
pollutes the aggregation, and either way a machine looking for the tell finds
one. A re-issued real shard has no tell, because there is nothing to detect: it
is genuine work, on a genuine job, with a genuine payload.

This makes a probe a near-relative of the ``attempt_group`` redundancy that
already exists, and the difference is the interesting one. Redundancy issues N
copies concurrently and compares them against each other, so a disagreement is
*ambiguous* -- a minority is suspected, not convicted, which is why docs/09 §5.1
rates it a hard hit rather than the maximum. A probe compares one new answer
against one **already-accepted** one. There is no question about which side is
wrong, and that asymmetry is why a failure is the largest single penalty.

Only deterministic types, and they opt in explicitly via
``JobType.spot_checkable`` (default off). "Known answer" is meaningless for
``collab_lora_finetune``: training is stochastic, two honest machines produce
different adapters, and that is the entire reason docs/05 has a norm gate and a
divergence metric instead of an equality check. It is equally meaningless for
``contained_batch``, whose body is an image the coordinator did not build.
Today the only type that opts in is ``batch_inference``, which decodes greedily
-- which is what makes ``judge``'s use of that type's comparator correct rather
than merely convenient.

Off unless ``settings.spotcheck_rate``, which defaults to 0.0.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import uuid
from datetime import datetime, timedelta

from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, utcnow

log = logging.getLogger("ganymede.coordinator.spotcheck")

PASSED = "passed"
FAILED = "failed"
# A probe whose task never came back. Not evidence of anything -- the machine
# went away, which the ordinary abandoned / expired path already counts against
# it -- but recorded rather than left NULL so the unjudged population does not
# grow into something an operator has to interpret.
VOID = "void"


def is_probe(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM spot_check_issues WHERE task_id = ?", (task_id,)
    ).fetchone() is not None


def _spot_checkable(job_type: str) -> bool:
    """Does this type opt in to known-answer probes? (docs/13 §5.2.)

    Fail-closed: a type that says nothing is not probed. The polarity matters
    because of what a probe costs when it is wrong -- docs/09 §5.1 rates a
    failed one the largest single penalty in the system, so the safe default
    for an unknown type is to leave it alone.

    Until ``contained_batch`` there was no gate here at all, and none was
    needed: probes are issued from the static-task reserve path, and
    ``batch_inference`` was the only type on it. That made "static" an accurate
    proxy for "deterministic" by accident. ``contained_batch`` is static too
    and runs a submitter's image, so the proxy stopped holding: an honest
    machine on a job that samples, threads or stamps a timestamp would have
    been convicted, and ``judge`` would have reached that verdict through
    ``batch_inference``'s comparator regardless of the job's own type.

    Gating issuance rather than making ``judge`` polymorphic is the smaller
    fix, and it makes that hardcoded import correct *by construction* rather
    than by coincidence: nothing but ``batch_inference`` is ever judged.
    """
    from ganymede.jobtypes import REGISTRY

    cls = REGISTRY.get(job_type)
    return bool(getattr(cls, "spot_checkable", False))


def _source_for(conn: sqlite3.Connection, job_id: str,
                worker_id: str) -> sqlite3.Row | None:
    """An accepted shard of this job, done by a *different* machine.

    Different machine is load-bearing and not a nicety: a machine checked
    against its own earlier answer agrees with itself, so the probe would
    measure determinism rather than honesty.

    A probe is never itself a source. Chaining probes would compare an unverified
    answer against another unverified answer and call the result a known-answer
    check.
    """
    return conn.execute(
        """SELECT t.id, t.input_ref_json, t.buckets_json, t.local_steps,
                  t.max_runtime_sec
             FROM tasks t
             JOIN submissions s ON s.task_id = t.id AND s.accepted = 1
            WHERE t.job_id = ? AND t.worker_id IS NOT NULL AND t.worker_id <> ?
              AND t.id NOT IN (SELECT task_id FROM spot_check_issues)
            ORDER BY s.received_at DESC LIMIT 1""",
        (job_id, worker_id),
    ).fetchone()


def maybe_issue(conn: sqlite3.Connection, job: sqlite3.Row, worker_id: str,
                settings, now: datetime | None = None,
                rng: random.Random | None = None) -> str | None:
    """Roll for a probe; on a hit, insert and lease one. Returns its task id.

    Called *instead of* the ordinary reserve, not alongside it -- a machine gets
    one task, and a probe has to be able to be that task or it is
    distinguishable by the shape of the response.
    """
    rate = float(getattr(settings, "spotcheck_rate", 0.0) or 0.0)
    if rate <= 0.0:
        return None
    if not _spot_checkable(job["job_type"]):
        return None
    rng = rng or random
    if rng.random() >= rate:
        return None
    source = _source_for(conn, job["id"], worker_id)
    if source is None:
        # Nothing accepted yet, or only this machine's own work. Not an error:
        # early in a job there is nothing to check anyone against.
        return None

    now = now or utcnow()
    task_id = uuid.uuid4().hex
    lease_sec = int(getattr(settings, "lease_duration_sec", 900))
    expires = _iso(now + timedelta(seconds=lease_sec))
    with immediate(conn):
        conn.execute(
            """INSERT INTO tasks
                 (id, job_id, buckets_json, input_ref_json, local_steps, status,
                  worker_id, lease_expires_at, leased_at, attempts,
                  max_runtime_sec, created_at)
               VALUES (?, ?, ?, ?, ?, 'leased', ?, ?, ?, 1, ?, ?)""",
            (task_id, job["id"], source["buckets_json"], source["input_ref_json"],
             source["local_steps"], worker_id, expires, _iso(now),
             source["max_runtime_sec"], _iso(now)),
        )
        conn.execute(
            "INSERT INTO spot_check_issues (task_id, source_task_id, issued_at) "
            "VALUES (?, ?, ?)",
            (task_id, source["id"], _iso(now)),
        )
    return task_id


def judge(conn: sqlite3.Connection, store, task_id: str, job: sqlite3.Row,
          now: datetime | None = None) -> str | None:
    """Compare a probe's answer against the accepted one. ``None`` if not a probe.

    Uses the **type's own comparator** -- for ``batch_inference``,
    ``validate.sample_agreement``, the exact function the ``attempt_group`` path
    calls, with the job's own ``agree_on`` and ``sample_rows``. One comparator
    and one definition of "the same answer": a probe that applied a stricter test
    than redundancy would convict machines redundancy would acquit, and the
    penalty here is the largest one in the system.
    """
    row = conn.execute(
        "SELECT source_task_id, outcome FROM spot_check_issues WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None or row["outcome"] is not None:
        return None
    now = now or utcnow()

    spec = json.loads(job["spec_json"] or "{}")
    redundancy = spec.get("redundancy") or {}
    agree_on = redundancy.get("agree_on", "output")
    sample_rows = int(redundancy.get("sample_rows", 8))

    from ganymede.jobtypes.batch_inference import run as bi_run
    from ganymede.jobtypes.batch_inference import validate as bi_validate

    outputs = []
    for tid in (row["source_task_id"], task_id):
        sub = conn.execute(
            "SELECT artifact_ref FROM submissions WHERE task_id = ?", (tid,)
        ).fetchone()
        if sub is None:
            return None
        try:
            outputs.append(bi_run.parse_jsonl(store.get_bytes(sub["artifact_ref"])))
        except Exception as exc:  # noqa: BLE001
            # "We could not read it" is not "they got it wrong", and this
            # penalty is the largest in the system. Leave the probe unjudged.
            log.warning("spot-check %s: unreadable artifact for %s: %s",
                        task_id, tid, exc)
            return None

    outcome = PASSED if bi_validate.sample_agreement(
        outputs, sample_rows, agree_on) else FAILED
    worker_id = conn.execute(
        "SELECT worker_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()["worker_id"]
    with immediate(conn):
        conn.execute(
            "UPDATE spot_check_issues SET outcome = ?, decided_at = ? "
            "WHERE task_id = ?",
            (outcome, _iso(now), task_id),
        )
        conn.execute(
            "INSERT INTO audit (at, worker_id, event, detail_json) VALUES (?, ?, ?, ?)",
            (_iso(now), worker_id, f"spot_check_{outcome}", json.dumps({
                "task_id": task_id, "source_task_id": row["source_task_id"],
                "job_id": job["id"],
            })),
        )
    return outcome


def void_stale(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Retire probes whose task ended without a submission. For the sweep.

    A probe that never came back says nothing about the machine -- it went away,
    and the ordinary ``abandoned`` / ``expired`` path already counts that. What
    it must not do is sit at ``outcome IS NULL`` forever: the reputation query
    reads only ``passed`` / ``failed``, so a growing NULL population is not
    *wrong*, it is just a column an operator cannot interpret.
    """
    now = now or utcnow()
    with immediate(conn):
        return conn.execute(
            """UPDATE spot_check_issues SET outcome = ?, decided_at = ?
                WHERE outcome IS NULL AND task_id IN (
                      SELECT id FROM tasks
                       WHERE status IN ('expired', 'abandoned', 'cancelled',
                                        'preempted'))""",
            (VOID, _iso(now)),
        ).rowcount


def outcomes_for(conn: sqlite3.Connection, machine_id: str,
                 since: str) -> tuple[int, int]:
    """``(passed, failed)`` probes for this machine since ``since``."""
    row = conn.execute(
        """SELECT
             SUM(CASE WHEN i.outcome = 'passed' THEN 1 ELSE 0 END) AS p,
             SUM(CASE WHEN i.outcome = 'failed' THEN 1 ELSE 0 END) AS f
           FROM spot_check_issues i
           JOIN tasks t ON t.id = i.task_id
          WHERE t.worker_id = ? AND i.decided_at >= ?""",
        (machine_id, since),
    ).fetchone()
    return int(row["p"] or 0), int(row["f"] or 0)


def failures_since(conn: sqlite3.Connection, machine_id: str, since: str) -> int:
    return outcomes_for(conn, machine_id, since)[1]
