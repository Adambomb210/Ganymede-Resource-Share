"""The host agent's tick (docs/02-architecture-v2.md 7).

```
systemd timer, every N minutes:
  1. Already running a Ganymede container for this GPU?     -> exit
  2. IdleBackend.is_idle()?                                 -> if no, exit
  3. GET /v1/manifest; if required image tag != local, pull it
  4. docker run ...
```

One tick, then exit
-------------------
The unit is `Type=oneshot` and this process does one pass and leaves. A daemon
would be less code -- no timer, no unit file per platform -- and it would be the
wrong shape: a long-lived process on a contributor's laptop is a thing that
leaks, wedges, survives a config change it never re-read, and has to be
explained. A tick that exits has no state to corrupt between runs, and "is it
working?" is answerable from the scheduler's own logs on all three platforms.

`--loop` exists anyway, for a machine where a timer is more trouble than it is
worth, and it is deliberately nothing more than this same tick in a sleep loop.

What each step does when it fails
---------------------------------
Nothing here is worth crashing over except a configuration the contributor has
to fix. An unreachable coordinator is a quiet skip -- it is Tuesday, the
coordinator is being restarted, and the timer will fire again in fifteen
minutes. This is the same split 4.2 draws for the worker and for the same
reason: on an unscheduled volunteer fleet, a component that gives up
permanently is a machine that has silently stopped contributing, and nobody is
watching to notice.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from ganymede.host import cache as cache_mod
from ganymede.host import idle as idle_mod
from ganymede.host import manifest as manifest_mod
from ganymede.host import runtime as runtime_mod
from ganymede.host.config import HostConfig, default_config_path

log = logging.getLogger("ganymede.host")

DEFAULT_LOOP_INTERVAL_SEC = 900


@dataclass(frozen=True)
class TickResult:
    """What one pass did, in a form a test can assert on and a log can print.

    Returned rather than logged-and-forgotten because the M3 exit criteria are
    all statements about a single tick -- "machine idles, worker starts within
    one timer interval", "touch pause, running container stops" -- and a tick
    whose only output is log text can only be tested by scraping log text.
    """

    action: str  # started | stopped | running | idle-skip | error | noop
    detail: str
    image: str | None = None

    @property
    def ok(self) -> bool:
        return self.action != "error"


def tick(
    config: HostConfig,
    *,
    runtime: runtime_mod.WorkerRuntime | None = None,
    backend: idle_mod.IdleBackend | None = None,
    fetch: Any = None,
) -> TickResult:
    """One pass of the 7 loop."""
    runtime = runtime if runtime is not None else runtime_mod.for_config(config)
    backend = backend if backend is not None else idle_mod.LocalIdleBackend(config)
    fetch = fetch if fetch is not None else manifest_mod.fetch

    gaps = config.missing()
    if gaps:
        # The one genuinely fatal case, and it is fatal because no amount of
        # waiting fixes it. Non-zero exit puts it in `systemctl status`, which
        # is where a contributor looks first.
        return TickResult("error", f"not configured: {', '.join(gaps)} unset")

    status = _status(runtime)
    if status is None:
        return TickResult("error", "could not ask the container runtime for status")

    # Steps 1 and 2 are inverted relative to the spec block, deliberately.
    # ------------------------------------------------------------------
    # 7 lists "already running?" first and idleness second, which reads as an
    # optimization: skip the idle probe when there is nothing to decide. But
    # taken literally it means a running worker is never re-examined, and the
    # pause sentinel -- the contributor's kill switch, and the thing 7.1 says
    # must work with no network and no explanation -- would then only prevent
    # *new* workers and never stop the one currently holding the GPU. Someone
    # who wants their machine back is not helped by "it will stop after this
    # round". So idleness is evaluated first, and a running worker on a
    # no-longer-idle machine is stopped.
    report = backend.report() if hasattr(backend, "report") else _thin_report(backend)

    if not report.idle:
        if status.running:
            log.info("machine no longer idle (%s); stopping worker", report.reason)
            try:
                runtime.stop()
            except runtime_mod.RuntimeError_ as exc:
                return TickResult("error", f"stopping the worker failed: {exc}")
            return TickResult("stopped", report.reason)
        return TickResult("idle-skip", report.reason)

    # Step 3: what should we be running?
    decision = manifest_mod.ImageDecision(None, "not consulted", constrained=False)
    try:
        manifest = fetch(config)
        decision = manifest_mod.resolve(manifest, config)
        _evict_cache(config, protect=manifest.base_models)
    except manifest_mod.ManifestError as exc:
        # Quiet on purpose. A coordinator that is down is not this machine's
        # problem to solve, and a worker already running against it has its own
        # retry and backoff (4.2). Log and let the timer come back.
        log.info("manifest unavailable (%s)", exc)
        if status.running:
            return TickResult("running", f"manifest unavailable, leaving worker up: {exc}")
        return TickResult("noop", f"manifest unavailable: {exc}")

    image = decision.image
    if image is None:
        # Nothing to reconcile against. If a worker is up, leave it: it polls
        # and will pick work up the moment a run appears. If none is up, there
        # is no run for one to work on, so starting one would just add a
        # process that polls 204 forever.
        if status.running:
            return TickResult("running", decision.reason, image=status.image)
        return TickResult("noop", decision.reason)

    # Step 1, now that we know what should be running.
    if status.running:
        if not decision.differs_from(status.image):
            return TickResult("running", "already running the required image", image=status.image)
        log.info("image bump: running %s, required %s", status.image, image)
        try:
            runtime.stop()
        except runtime_mod.RuntimeError_ as exc:
            return TickResult("error", f"stopping for an image bump failed: {exc}")

    # Step 3's second half, and step 4.
    try:
        runtime.ensure_image(image)
        runtime.start(image, worker_env(config))
    except runtime_mod.RuntimeError_ as exc:
        return TickResult("error", f"starting {image} failed: {exc}")

    return TickResult("started", decision.reason, image=image)


def worker_env(config: HostConfig) -> dict[str, str]:
    """What the worker reads out of its environment (4.2 step 1)."""
    env = {
        "GANYMEDE_COORDINATOR_URL": config.coordinator_url,
        "GANYMEDE_KEY": config.key,
    }
    if config.runtime == "native":
        # Container paths are fixed by the mounts in `runtime.run_argv`; a
        # native worker has to be told where the host actually put things.
        env["GANYMEDE_CACHE_DIR"] = str(config.resolved_cache_dir())
        env["GANYMEDE_STATE"] = str(config.resolved_state_dir())
    if not config.verify_tls:
        env["GANYMEDE_INSECURE"] = "1"
    return env


def _evict_cache(config: HostConfig, *, protect: frozenset[str]) -> None:
    """6.7's cap, applied at the one moment it is safe to apply it.

    Immediately before starting a worker, and never while one is running: the
    running worker holds open file handles into the cache and is the one thing
    on the machine that can be reading a blob at the moment it is unlinked.
    Protecting the active runs' base models on top of that is belt and braces
    -- evicting a model this machine is about to be handed work for means
    re-downloading ~16 GB before anything useful happens, potentially longer
    than the round itself.
    """
    if config.cache_cap_gb <= 0:
        return
    try:
        result = cache_mod.evict_to_cap(
            config.resolved_cache_dir(),
            cap_bytes=int(config.cache_cap_gb * 1024**3),
            protect=protect,
        )
    except OSError as exc:
        # A cache we cannot prune is a disk that may fill, which is bad, but it
        # is not a reason to decline work that would otherwise run.
        log.warning("cache eviction failed: %s", exc)
        return
    if result.removed:
        log.info(
            "evicted %d cached base model(s), freed %.1f GB",
            len(result.removed), result.bytes_freed / 1024**3,
        )


def _status(runtime: runtime_mod.WorkerRuntime) -> runtime_mod.RuntimeStatus | None:
    try:
        return runtime.status()
    except runtime_mod.RuntimeError_ as exc:
        log.error("%s", exc)
        return None


def _thin_report(backend: idle_mod.IdleBackend) -> idle_mod.IdleReport:
    """For a backend that implements only 7.1's two-line protocol.

    The `vast` and `tensordock` backends 7.1 anticipates will be exactly that
    -- one API call, a bool -- and they should not have to carry a reason
    string to be usable here.
    """
    ok = backend.is_idle()
    return idle_mod.IdleReport(idle=ok, reason="idle" if ok else "not idle")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ganymede-host",
        description="Start and stop a Ganymede worker as the machine allows (architecture 7).",
    )
    p.add_argument("--config", default=None, help=f"config file (default {default_config_path()})")
    p.add_argument("--once", action="store_true", help="one tick then exit; this is what a timer runs")
    p.add_argument("--loop", action="store_true", help="tick forever, for a host with no usable timer")
    p.add_argument("--interval-sec", type=int, default=DEFAULT_LOOP_INTERVAL_SEC,
                   help=f"seconds between ticks under --loop (default {DEFAULT_LOOP_INTERVAL_SEC})")
    p.add_argument("--check", action="store_true",
                   help="report configuration and the idle decision without starting anything")
    p.add_argument("--stop", action="store_true", help="stop the worker now")
    p.add_argument("--pause", action="store_true", help="stop, and take no new work until --resume")
    p.add_argument("--resume", action="store_true", help="clear the pause sentinel")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from pathlib import Path

    try:
        config = HostConfig.load(Path(args.config) if args.config else None)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.pause:
        config.pause_path.parent.mkdir(parents=True, exist_ok=True)
        config.pause_path.touch()
        print(f"paused: {config.pause_path}")
        args.stop = True

    if args.resume:
        try:
            config.pause_path.unlink()
            print(f"resumed: removed {config.pause_path}")
        except FileNotFoundError:
            print("not paused")
        # Deliberately falls through into a tick rather than returning. Someone
        # who types --resume means now, not "at some point in the next fifteen
        # minutes when the timer next fires".

    if args.stop:
        # The sentinel is already written by the time we get here, and that is
        # the part that has to work "with no network, no coordinator, and no
        # explanation required" (7.1). A missing Docker, a daemon that is down,
        # a native worker that was never started -- none of those should turn
        # the contributor's kill switch into a traceback. Report and move on:
        # the worker reads the sentinel on its next poll regardless.
        try:
            runtime_mod.for_config(config).stop()
            print("worker stopped")
        except (runtime_mod.RuntimeError_, ValueError) as exc:
            print(f"worker not stopped directly ({exc});"
                  f" it will stop on its next poll of {config.resolved_state_dir() / 'stop'}")
        return 0

    if args.check:
        return _check(config)

    if args.loop:
        while True:
            result = tick(config)
            log.info("%s: %s", result.action, result.detail)
            time.sleep(max(60, args.interval_sec))

    # --once is the default: a bare invocation and a timer's invocation should
    # do the same thing, so that a contributor debugging by hand is exercising
    # the same path the scheduler does.
    result = tick(config)
    log.info("%s: %s", result.action, result.detail)
    return 0 if result.ok else 1


def _check(config: HostConfig) -> int:
    """"Did I install this right?" -- the last step of every install script.

    Prints what it resolved and what it would do, and never starts anything.
    """
    print(f"coordinator : {config.coordinator_url or '(unset)'}")
    print(f"key         : {'set' if config.key else '(unset)'}")
    print(f"runtime     : {config.runtime}")
    print(f"state dir   : {config.resolved_state_dir()}")
    print(f"cache dir   : {config.resolved_cache_dir()}")
    print(f"cache cap   : {config.cache_cap_gb:g} GB")
    free = cache_mod.free_disk_bytes(config.resolved_cache_dir())
    print(f"free disk   : {free / 1024**3:.1f} GB")

    gaps = config.missing()
    if gaps:
        print(f"\nnot configured: {', '.join(gaps)} unset", file=sys.stderr)
        return 2

    report = idle_mod.LocalIdleBackend(config).report()
    print(f"idle now    : {'yes' if report.idle else 'no'} ({report.reason})")

    try:
        manifest = manifest_mod.fetch(config)
    except manifest_mod.ManifestError as exc:
        print(f"\ncoordinator unreachable: {exc}", file=sys.stderr)
        return 1

    decision = manifest_mod.resolve(manifest, config)
    print(f"active runs : {len(manifest.runs)}")
    print(f"image       : {decision.image or '(none)'} -- {decision.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
