"""The worker entrypoint loop (docs/02-architecture-v2.md 4.2).

```
 1. Read env: GANYMEDE_KEY, COORDINATOR_URL, GANYMEDE_CACHE_DIR
 2. Probe (6.9); a machine with no GPU still registers and simply never matches
 3. POST /v1/workers/register -> worker_id
 4. POST /v1/tasks/claim
      -> 204 : sleep with jittered backoff, goto 4
      -> 200 : task spec (8)
 5. Verify we can honor the task: precision supported, memory sufficient,
    required image matches. If not -> abandon, keep polling
 6. Ensure base model in the host-persistent HF cache (pull once, reuse forever)
 7. GET base adapter via presigned URL (safetensors)
 8. Train local_steps, or until the deadline
      - heartbeat every 60 s with step progress (renews the lease)
      - 409 -> round closed, drop the work, goto 4
      - stop/pause -> abandon, exit or wait (4.4)
 9. Save adapter -> presigned PUT -> submit
10. goto 4
```

The honor-check at step 5 is the part that earns its place. A worker that claims
work it cannot do has already cost the round: the shard is marked leased, and it
stays leased until the lease expires or the worker abandons. Checking *before*
downloading a base model turns a wasted round into a wasted second.

Heartbeats run on a background thread rather than between optimizer steps. A step
on a slow card can take longer than the lease-renewal interval, so a worker that
only checked between steps could lose a lease it was actively working on -- and
the failure would look like a flaky network rather than a scheduling mistake.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ganymede.worker import probe as probe_mod
from ganymede.worker.client import (
    CoordinatorClient,
    CoordinatorError,
    GateRejected,
    LeaseLost,
    RoundClosed,
)
from ganymede.worker.control import ControlFiles

log = logging.getLogger("ganymede.worker")

# How long to wait after a 204 when the coordinator sends no Retry-After. Jittered
# so a fleet that all woke together does not re-converge into a synchronized poll.
IDLE_SLEEP_SEC = 30
IDLE_SLEEP_JITTER = 0.5

# How long to sit out a pause before re-checking (7.1). Long, because a
# contributor pausing to play a game is thinking in hours.
PAUSE_POLL_SEC = 60

# Floor on the heartbeat interval. A zero or tiny value from a misconfigured
# coordinator would turn the heartbeat thread into a busy loop against the API,
# from every worker in the fleet at once.
MIN_HEARTBEAT_INTERVAL_SEC = 5

# Reasons a worker declines a task at step 5. Reported on abandon so a
# contributor asking "why do I never get work" has an answer.
DECLINE_PRECISION = "precision_unsupported"
DECLINE_IMAGE = "image_mismatch"
DECLINE_MEMORY = "insufficient_memory"


@dataclass
class WorkerConfig:
    coordinator_url: str
    key: str
    image_tag: str | None = None
    run_id: str | None = None
    state_dir: str | None = None
    cache_dir: str | None = None
    backend: str | None = None
    once: bool = False
    max_rounds: int | None = None
    verify_tls: bool = True
    skip_bench: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "WorkerConfig":
        cfg = cls(
            coordinator_url=os.environ.get("GANYMEDE_COORDINATOR_URL", ""),
            key=os.environ.get("GANYMEDE_KEY", ""),
            # Set by the container build, absent on a native install -- which is
            # exactly how a run's required_image excludes native workers (6.10).
            image_tag=os.environ.get("GANYMEDE_IMAGE_TAG"),
            run_id=os.environ.get("GANYMEDE_RUN_ID"),
            state_dir=os.environ.get("GANYMEDE_STATE"),
            cache_dir=os.environ.get("GANYMEDE_CACHE_DIR"),
            backend=os.environ.get("GANYMEDE_BACKEND"),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


class Heartbeater:
    """Renews the lease on a background thread while training runs.

    Owns the round-closed signal: when the coordinator answers 409 or 410 the
    heartbeat thread sets a flag, the trainer's ``should_stop`` sees it on its
    next optimizer step, and the loop unwinds without submitting. That is the
    3.2 contract -- a closed round's work is dropped, not argued with.
    """

    def __init__(self, client: CoordinatorClient, task_id: str, interval_sec: int):
        self.client = client
        self.task_id = task_id
        self.interval = max(MIN_HEARTBEAT_INTERVAL_SEC, interval_sec)
        self.steps = 0
        self.loss: float | None = None
        self.round_closed = False
        self.lease_lost = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def record(self, steps: int, loss: float | None = None) -> None:
        self.steps = steps
        self.loss = loss

    def start(self) -> "Heartbeater":
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"heartbeat-{self.task_id[:8]}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def should_drop(self) -> bool:
        return self.round_closed or self.lease_lost

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.client.heartbeat(self.task_id, self.steps, self.loss)
            except RoundClosed:
                log.info("round closed under task %s; dropping the work", self.task_id)
                self.round_closed = True
                return
            except LeaseLost:
                log.warning("lease lost on task %s; dropping the work", self.task_id)
                self.lease_lost = True
                return
            except CoordinatorError as exc:
                # A heartbeat that cannot reach the coordinator is not fatal on
                # its own -- the client already retried, and the lease has slack.
                # Losing it entirely surfaces as a 410 on a later beat.
                log.warning("heartbeat failed for %s: %s", self.task_id, exc)


@dataclass
class Worker:
    config: WorkerConfig
    client: CoordinatorClient
    control: ControlFiles
    profile: dict[str, Any]
    worker_id: str = ""
    heartbeat_interval: int = 60
    tasks_done: int = 0
    # Coordinator rounds this worker has worked, as (run_id, round_idx). A set
    # rather than a counter because a worker routinely takes several tasks
    # inside one round -- it finishes its step budget, the round is still open,
    # and it claims again -- and `--max-rounds 5` has to mean five rounds. It
    # meant five *tasks* until M4a, where three workers all stopped inside
    # round 0 having taken twenty tiny tasks each and the run never advanced.
    rounds_worked: set[tuple[str, int]] = field(default_factory=set)
    cached_base_models: set[str] = field(default_factory=set)

    @property
    def rounds_done(self) -> int:
        """How many distinct rounds this worker has worked."""
        return len(self.rounds_worked)

    # ---------------- setup ----------------

    @classmethod
    def create(cls, config: WorkerConfig) -> "Worker":
        if not config.coordinator_url:
            raise SystemExit("GANYMEDE_COORDINATOR_URL is not set")
        if not config.key:
            raise SystemExit("GANYMEDE_KEY is not set")

        if config.cache_dir:
            # Persisted on the host and shared across container restarts: a base
            # model is pulled once and reused forever (4.2 step 6).
            os.environ.setdefault("HF_HOME", config.cache_dir)

        log.info("probing hardware")
        profile = probe_mod.run_probe(config.backend, skip_bench=config.skip_bench)
        log.info("backend=%s device=%s supports=%s bench=%s",
                 profile["backend"], profile["device_name"], profile["supports"],
                 profile["probe"].get("bench_score"))

        return cls(
            config=config,
            client=CoordinatorClient(config.coordinator_url, config.key,
                                     verify_tls=config.verify_tls),
            control=ControlFiles(config.state_dir),
            profile=profile,
        )

    def register(self) -> str:
        result = self.client.register(self.profile, self.config.image_tag)
        self.worker_id = result["worker_id"]
        self.heartbeat_interval = int(result.get("heartbeat_interval_sec", 60))
        log.info("registered as %s", self.worker_id)
        return self.worker_id

    # ---------------- step 5: can we honor this? ----------------

    def can_honor(self, task: dict[str, Any]) -> tuple[bool, str | None]:
        """Check before downloading anything (4.2 step 5).

        The coordinator's eligibility filter (6.8) already screens on the profile
        this worker reported, so a mismatch here means something changed between
        registration and now -- a driver update, a different image, a run
        retargeted mid-flight. Rare, and precisely why it is worth checking:
        the expensive version of this failure is discovering it after a multi-GB
        base-model download.
        """
        required_image = task.get("required_image")
        if required_image and required_image != self.config.image_tag:
            return False, (
                f"{DECLINE_IMAGE}: run needs {required_image!r}, "
                f"this worker is {self.config.image_tag!r}"
            )

        precision = task.get("base_precision")
        if precision and precision not in self.profile.get("supports", []):
            # Silently training at a different precision would break 5.2's
            # shared-frozen-base assumption without erroring anywhere.
            return False, f"{DECLINE_PRECISION}: {precision} not in {self.profile['supports']}"

        return True, None

    # ---------------- step 8: one round ----------------

    def run_round(self, task: dict[str, Any]) -> dict[str, Any] | None:
        """Train and submit one task. Returns the submit response, or None if
        the work was dropped (round closed, lease lost, or told to stop)."""
        from ganymede.trainer.train import Task, run_task

        task_id = task["task_id"]
        beat = Heartbeater(self.client, task_id, self.heartbeat_interval).start()
        started = time.monotonic()

        try:
            download_started = time.monotonic()
            base_adapter = self.client.download(task["base_adapter_url"])
            download_sec = time.monotonic() - download_started
            log.info("task %s: %d bytes of base adapter in %.1fs",
                     task_id, len(base_adapter), download_sec)

            parsed = Task.from_payload(task)
            self.cached_base_models.add(parsed.base_model)

            def on_step(step: int, loss: float) -> None:
                beat.record(step + 1, loss)

            def should_stop() -> bool:
                return beat.should_drop() or self.control.should_stop() \
                    or self.control.should_pause()

            result = run_task(parsed, base_adapter, on_step=on_step, should_stop=should_stop)

            if beat.should_drop():
                log.info("task %s: work dropped after %d steps", task_id, result.steps)
                return None

            if self.control.should_stop() or self.control.should_pause():
                # 4.4: abandon rather than race to save and upload. Docker's
                # default stop grace is 10 s and a half-uploaded artifact is
                # worse than none -- abandoning releases the shard immediately.
                log.info("task %s: stopping, abandoning after %d steps", task_id, result.steps)
                self._abandon(task_id)
                return None

            if result.steps == 0:
                log.warning("task %s: no steps completed, abandoning", task_id)
                self._abandon(task_id)
                return None

            return self._submit(task_id, result, download_sec, started)

        except (RoundClosed, LeaseLost) as exc:
            log.info("task %s: %s", task_id, exc)
            return None
        except GateRejected as exc:
            # Not retryable and not a crash: the submission was structurally
            # wrong, the coordinator said which gate, and the loop moves on.
            log.error("task %s rejected: %s", task_id, exc.reason)
            return None
        except CoordinatorError as exc:
            # Infrastructure, not this worker: the coordinator or the object
            # store is unreachable, and the client already exhausted its
            # retries. 6.4 says a worker rides this out and retries with
            # backoff -- it must not be fatal.
            #
            # It was, until M4a. The object store went away mid-round and all
            # three workers exited on the spot, on an unhandled exception out
            # of the upload. On a volunteer fleet that is the worst available
            # behaviour: the machines that were contributing stop, quietly and
            # permanently, and nobody is watching to notice. A storage blip
            # should cost the round in progress, not the worker.
            log.error("task %s: %s -- abandoning and backing off", task_id, exc)
            self._abandon(task_id)
            # Pause before claiming again. Without this the loop would take a
            # fresh lease immediately, train a whole round, and fail at the same
            # upload -- burning a contributor's machine for as long as the
            # outage lasts.
            self._idle(IDLE_SLEEP_SEC)
            return None
        except Exception:
            log.exception("task %s failed; abandoning the lease", task_id)
            self._abandon(task_id)
            raise
        finally:
            beat.stop()

    def _submit(self, task_id: str, result, download_sec: float, started: float) -> dict[str, Any]:
        slot = self.client.upload_url(task_id)

        upload_started = time.monotonic()
        self.client.upload(slot["url"], result.adapter_bytes)
        upload_sec = time.monotonic() - upload_started

        # 6.9: transfer rates are measured, not asked for. After one round the
        # coordinator knows this machine's real bandwidth, which feeds the next
        # round's budget instead of a number a contributor would have to look up
        # and would often get wrong.
        metrics = dict(result.metrics)
        metrics.update({
            "download_sec": round(download_sec, 2),
            "upload_sec": round(upload_sec, 2),
            "artifact_bytes": len(result.adapter_bytes),
            "wall_sec": round(time.monotonic() - started, 2),
            "backend": self.profile["backend"],
        })

        response = self.client.submit(
            task_id, slot["key"], result.steps,
            tokens_seen=result.metrics.get("tokens", 0), metrics=metrics,
        )
        log.info("task %s: submitted %d steps, accepted=%s%s",
                 task_id, result.steps, response.get("accepted"),
                 f" ({response['reject_reason']})" if response.get("reject_reason") else "")
        return response

    def _abandon(self, task_id: str) -> None:
        try:
            self.client.abandon(task_id)
        except Exception as exc:  # noqa: BLE001
            # Abandoning is a courtesy that shortens the wait for the next
            # worker; the lease expires on its own regardless, so failing to
            # abandon must never take the worker down with it.
            log.warning("could not abandon %s: %s", task_id, exc)

    # ---------------- the loop ----------------

    def _idle(self, seconds: int) -> None:
        delay = seconds or IDLE_SLEEP_SEC
        time.sleep(delay * (1 - IDLE_SLEEP_JITTER + random.random() * IDLE_SLEEP_JITTER * 2))

    def run(self) -> int:
        self.register()

        while True:
            if self.control.should_stop():
                log.info("stopping: %s", self.control.reason())
                return 0

            if self.control.should_pause():
                log.info("paused: %s", self.control.reason())
                time.sleep(PAUSE_POLL_SEC)
                continue

            try:
                task, retry_after = self.client.claim(
                    self.worker_id,
                    capabilities=self.profile,
                    cached_base_models=sorted(self.cached_base_models),
                    run_id=self.config.run_id,
                )
            except CoordinatorError as exc:
                log.warning("claim failed: %s", exc)
                self._idle(IDLE_SLEEP_SEC)
                continue

            if task is None:
                # No eligible work. On an unscheduled volunteer fleet this is the
                # common state, not an error (3.2).
                log.debug("no work; sleeping %ss", retry_after or IDLE_SLEEP_SEC)
                self._idle(retry_after)
                if self.config.once:
                    return 0
                continue

            honored, reason = self.can_honor(task)
            if not honored:
                # Should be unreachable: the coordinator filters on the same
                # facts at claim time. Reaching it means registration and the
                # run disagree -- a driver update, a rebuilt image, a run
                # retargeted mid-flight -- so abandon and let the next poll
                # re-register the truth.
                log.warning("declining task %s: %s", task["task_id"], reason)
                self._abandon(task["task_id"])
                if self.config.once:
                    return 0
                self._idle(IDLE_SLEEP_SEC)
                continue

            self.run_round(task)
            self.tasks_done += 1
            # Recorded whatever the outcome. A round whose work was dropped
            # because it closed underneath us was still a round this worker
            # took part in, and counting only successes would have `--max-rounds`
            # quietly mean "until N rounds happen to go your way".
            self.rounds_worked.add((task["run_id"], int(task["round_idx"])))

            if self.config.once:
                return 0
            if self.config.max_rounds and self.rounds_done >= self.config.max_rounds:
                log.info("reached max_rounds=%s after %d task(s)",
                         self.config.max_rounds, self.tasks_done)
                return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ganymede-worker",
        description="Join a Ganymede run: claim work, train, submit, repeat.",
    )
    p.add_argument("--coordinator-url", default=None, help="overrides GANYMEDE_COORDINATOR_URL")
    p.add_argument("--key", default=None, help="overrides GANYMEDE_KEY (prefer the env var)")
    p.add_argument("--run-id", default=None, help="restrict to one run")
    p.add_argument("--image-tag", default=None, help="this worker's image tag, if containerized")
    p.add_argument("--state-dir", default=None, help="where the stop/pause sentinels live")
    p.add_argument("--cache-dir", default=None, help="host-persistent HF cache")
    p.add_argument("--backend", default=None, help="pin the compute backend (cuda, rocm, xpu, mps, cpu)")
    p.add_argument("--once", action="store_true", help="one claim then exit; for smoke tests")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="stop after taking part in this many coordinator rounds; "
                        "a worker may take several tasks within one")
    p.add_argument("--skip-bench", action="store_true", help="skip the benchmark during probing")
    p.add_argument("--insecure", action="store_true",
                   help="skip TLS verification -- for a self-signed coordinator only")
    p.add_argument("--probe-only", action="store_true",
                   help="print the compute profile and exit; the first thing to ask for "
                        "when a contributor reports getting no work")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    if args.probe_only:
        import json

        print(json.dumps(probe_mod.run_probe(args.backend, skip_bench=args.skip_bench), indent=2))
        return 0

    config = WorkerConfig.from_env(
        coordinator_url=args.coordinator_url, key=args.key, run_id=args.run_id,
        image_tag=args.image_tag, state_dir=args.state_dir, cache_dir=args.cache_dir,
        backend=args.backend, once=args.once or None, max_rounds=args.max_rounds,
        verify_tls=False if args.insecure else None, skip_bench=args.skip_bench or None,
    )

    try:
        return Worker.create(config).run()
    except KeyboardInterrupt:
        log.info("interrupted")
        return 0


if __name__ == "__main__":
    sys.exit(main())
