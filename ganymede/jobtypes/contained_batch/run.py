"""The worker body for ``contained_batch`` (docs/10 §7, docs/11 §2).

The first caller ``sandbox.JobContainer`` has ever had. Everything confinement
does -- the caps, the read-only rootfs, the one bind mount, ``--network none``,
the digest check before ``docker load``, the soft/hard kill -- was built and
unit-tested against an injectable runner and had no consumer; this is it.

The shape of a task:

1. stage ``/scratch/in`` from the presigned shard GET,
2. pull the image archive, **hash it, then** ``docker load`` it,
3. start the container detached and supervise it,
4. read ``/scratch/out``, PUT it, return a reference.

The container does no object-store I/O of its own (docs/11 §2.3) -- it has no
network at all. The worker is the only thing that talks to the store, which is
what lets ``--network none`` be unconditional.

Signature note (docs/10 "Spine deviations"): ``should_stop`` returns
``"soft" | "hard" | None``, the same shape ``batch_inference`` uses. Here the
distinction is finally real rather than collapsed: ``soft`` is
``docker stop --time <grace>``, so the job gets a SIGTERM and a grace period to
checkpoint, and ``hard`` is ``docker kill``. docs/11 §3 says that distinction
only ever mattered for submitter code, and this is the submitter code.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# What the worker writes into /scratch/in and reads back from /scratch/out.
# Part of the published container contract (docs/10 §7) -- changing one of these
# strings breaks every image a submitter has already built.
INPUT_NAME = "input.jsonl"
PARAMS_NAME = "params.json"
OUTPUT_NAME = "output.jsonl"

# How often the supervisor asks the runtime whether the container is still up.
# Liveness does not depend on this -- the Heartbeater is a thread on its own
# timer -- so this only bounds how quickly a cancel is acted on and how fresh
# the progress figure is. Each tick is a `docker inspect`, so a job that runs an
# hour costs ~720 of them at this interval, which is cheap enough to leave low.
DEFAULT_POLL_SEC = 5.0


@dataclass(frozen=True)
class ContainedTask:
    """The claim payload for one contained shard, parsed once.

    The image fields are the coordinator's (docs/06, ``_image_handles``); they
    are ``None`` for a first-party built-in and required here.
    """

    task_id: str
    job_id: str | None
    shard_ref: str
    shard_rows: int
    output_key: str
    output_schema: dict[str, str]
    params: dict[str, Any]
    image_ref: str | None
    image_digest: str | None
    image_pull_url: str | None
    max_runtime_sec: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ContainedTask":
        params = payload.get("params") or {}
        desc: dict[str, Any] = {}
        raw = payload.get("input_ref")
        if isinstance(raw, str) and raw:
            try:
                desc = json.loads(raw)
            except ValueError:
                desc = {}
        elif isinstance(raw, dict):
            desc = raw
        get = lambda k, d=None: params.get(k, desc.get(k, d))  # noqa: E731
        return cls(
            task_id=payload["task_id"],
            job_id=payload.get("job_id"),
            shard_ref=get("shard_ref"),
            shard_rows=int(get("shard_rows") or 0),
            output_key=get("output_key") or "",
            output_schema=get("output_schema") or {},
            params=get("params", {}) or {},
            image_ref=payload.get("image_ref"),
            image_digest=payload.get("image_digest"),
            image_pull_url=payload.get("image_pull_url"),
            max_runtime_sec=payload.get("max_runtime_sec"),
        )


@dataclass(frozen=True)
class ContainedResult:
    """Deliberately the same field set as ``InferResult``.

    Not a coincidence and not laziness: the worker's ``_submit_shard`` reads
    ``rows`` / ``digest`` / ``output_ref`` / ``seconds`` off whatever a static
    type's ``run`` returned, and a third type that reused that path without
    changing it is the cheapest real evidence that the seam generalises. If this
    had needed a fourth submit path, that would have been the finding.
    """

    rows: int
    output_ref: str
    digest: str
    seconds: float
    exit_code: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ContainedFailure(RuntimeError):
    """The container ran and exited non-zero.

    A verdict on the *job*, unlike ``SandboxError``, which says the machine
    could not run it. Kept distinct so the worker can tell "this submitter's
    code is broken" from "this host is", and abandon rather than fail in the
    second case.
    """

    def __init__(self, exit_code: int, tail: str = "") -> None:
        super().__init__(
            f"job container exited {exit_code}" + (f": {tail}" if tail else "")
        )
        self.exit_code = exit_code


class ContainedCancelled(RuntimeError):
    """The supervisor stopped the container because it was asked to.

    Its own type rather than a ``ContainedFailure`` with a sentinel exit code:
    a cancel and a genuine non-zero exit are different events with different
    handling, and telling them apart by a side channel -- "did *we* ask for a
    stop just now?" -- is the kind of thing that reads fine and goes wrong when
    a second stop source appears.
    """

    def __init__(self, signal: str) -> None:
        super().__init__(f"job container cancelled ({signal})")
        self.signal = signal


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    """sha256 over the ``(id, row)`` pairs, order-independent.

    A fingerprint for the operator and for ``submissions.metrics_json``, **not**
    a comparator: ``plan.validate_spec`` refuses ``redundancy`` and this type
    opts out of spot-checks, precisely because nothing here may assume two runs
    of a submitter image agree. It is here so a human can tell whether two
    outputs differ, not so the coordinator can penalise a machine for it.
    """
    pairs = sorted((str(r.get("id")), json.dumps(r, sort_keys=True)) for r in rows)
    return hashlib.sha256(
        json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def parse_jsonl(raw: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _default_download(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 -- presigned
        return resp.read()


def _default_upload(url: str, blob: bytes) -> None:
    req = urllib.request.Request(url, data=blob, method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req):  # noqa: S310 -- presigned
        return


def _stop_signal(should_stop: Callable[[], Any] | None) -> str | None:
    if should_stop is None:
        return None
    sig = should_stop()
    if sig is True:
        return "hard"
    if sig in ("soft", "hard"):
        return sig
    return None


def _rows_written(scratch: Path) -> int:
    """Progress, counted from what the container has actually flushed.

    Newlines rather than parsed records on purpose: a half-written final line
    has no newline yet, so this never counts a row the job has not finished
    emitting, and it never raises on one either. A container that buffers its
    output reports zero until it exits, which is honest -- it *has* produced
    nothing durable -- and costs nothing, because liveness rides the heartbeat
    thread rather than this number.
    """
    path = scratch / "out" / OUTPUT_NAME
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def run(
    task: ContainedTask,
    inputs,
    on_step: Callable[[int, float], None] | None = None,
    should_stop: Callable[[], Any] | None = None,
    *,
    config=None,
    runner=None,
    download: Callable[[str], bytes] | None = None,
    upload: Callable[[bytes], None] | None = None,
    poll_sec: float = DEFAULT_POLL_SEC,
    sleep: Callable[[float], None] | None = None,
) -> ContainedResult:
    """Run one shard inside the submitter's image.

    ``config`` / ``runner`` / ``download`` / ``upload`` / ``sleep`` are the
    injection points, in the repo's established style: left ``None`` they
    resolve the environment's ``SandboxConfig``, shell out to the configured
    runtime binary, and use the presigned URLs on ``inputs``.
    """
    from ganymede.worker import sandbox

    started = time.monotonic()
    params = getattr(inputs, "params", {}) or {}
    artifacts = getattr(inputs, "artifacts", {}) or {}
    on_step = on_step or (lambda _rows, _loss: None)
    download = download or _default_download
    sleep = sleep or time.sleep

    if not task.image_pull_url or not task.image_digest:
        # Unreachable from the claim path -- `_image_handles` returns all three
        # or none, and the claim walk refuses a job whose image is unschedulable
        # -- so this is the belt to that braces. Loud, because the alternative
        # for a *contained* type is running nothing at all and reporting success.
        raise sandbox.SandboxError(
            f"task {task.task_id} has no image to run: contained_batch is "
            "meaningless without one (docs/11 §4)"
        )

    cfg = config or sandbox.SandboxConfig.from_env()
    job = sandbox.JobContainer(task_id=task.task_id, config=cfg, runner=runner)

    try:
        scratch = job.prepare_scratch()

        # -- stage /scratch/in (docs/11 §2.3) -----------------------------
        shard_url = artifacts.get("shard") or params.get("shard_ref")
        (scratch / "in" / INPUT_NAME).write_bytes(download(shard_url))
        (scratch / "in" / PARAMS_NAME).write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "job_id": task.job_id,
                    "shard_ref": task.shard_ref,
                    "shard_rows": task.shard_rows,
                    "output_schema": task.output_schema,
                    "params": task.params,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # -- pull, verify, load (docs/11 §2.3) ----------------------------
        archive = job.fetch_archive(download, task.image_pull_url, task.image_digest)
        image_id = job.load_image(archive)
        # The archive is bind-mounted into the container along with the rest of
        # scratch, and at the 10 GiB upload cap it is the largest thing in it.
        # Dropping it here frees that before the job runs and keeps it out of
        # the job's own view of /scratch.
        archive.unlink(missing_ok=True)

        # -- run ----------------------------------------------------------
        job.start(image_id, max_runtime_sec=task.max_runtime_sec)
        signal = _supervise(job, scratch, on_step, should_stop, poll_sec, sleep)
        if signal is not None:
            # Cancelled. Nothing is uploaded: a partial output is shorter than
            # the shard's declared rows, `validate` rejects on that count, and
            # submitting one would spend an attempt to be told no.
            raise ContainedCancelled(signal)

        code = job.exit_code()
        if code != 0:
            raise ContainedFailure(code if code is not None else -1)

        # -- collect and upload -------------------------------------------
        out_path = scratch / "out" / OUTPUT_NAME
        try:
            raw = out_path.read_bytes()
        except OSError as exc:
            raise ContainedFailure(
                0, f"exited 0 but wrote no {OUTPUT_NAME} ({exc})"
            ) from None
        rows = parse_jsonl(raw)
        on_step(len(rows), 0.0)

        if upload is not None:
            upload(raw)
        elif params.get("output_put_url"):
            _default_upload(params["output_put_url"], raw)

        return ContainedResult(
            rows=len(rows),
            output_ref=task.output_key or params.get("output_key", ""),
            digest=canonical_digest(rows),
            seconds=round(time.monotonic() - started, 3),
            exit_code=code,
            metrics={"rows": len(rows), "image_ref": task.image_ref},
        )
    finally:
        # Every exit path, including the ones raised through. A scratch dir that
        # survives a crash is the disk that fills up over a week of them, and a
        # container left behind holds its image and its cgroup with it.
        job.cleanup()


def _supervise(job, scratch: Path, on_step, should_stop, poll_sec: float,
               sleep) -> str | None:
    """Watch the container until it exits or is told to stop.

    Returns the stop signal that ended it, or ``None`` if it exited on its own.

    The wall-clock ceiling is not enforced here: ``docker run`` was given
    ``--stop-timeout`` and the worker's lease is the outer bound, so a second
    timer in this loop would be a third opinion about when the job is over.
    """
    while True:
        signal = _stop_signal(should_stop)
        if signal is not None:
            # docs/11 §3 step 3: soft is SIGTERM plus the grace the job was
            # promised, hard is SIGKILL now.
            job.cancel(signal)
            return signal
        if not job.is_running():
            return None
        on_step(_rows_written(scratch), 0.0)
        sleep(poll_sec)
