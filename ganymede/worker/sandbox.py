"""Runtime confinement for submitter job code (docs/11 §2, §3).

The worker owns the claim / heartbeat / submit loop and now also supervises a
**sibling** job container: it pulls the archive the coordinator named, verifies
the digest before anything is loaded, stages inputs into a scratch directory,
runs the image under §2.2's flags, and signals it on a cancel.

Threat model is Decision 3 -- a trusted author's honest mistake, not an attacker
against the runtime. That buys the flags below and does not buy gVisor / Kata
(docs/11 §5).

**Who launches it.** §2.1 forbids handing the worker the host's Docker socket:
that is root-equivalent on the machine and defeats the §4.6 hardening the worker
itself runs under. What this module does instead is invoke a runtime *binary*
and never a socket path -- so whether that binary talks to a scoped socket proxy
(``DOCKER_HOST`` pointing at it) or is a rootless Podman is a deployment
decision, made where the host agent is configured, not a code path here. The
deviation to record honestly: **Ganymede does not yet ship the socket proxy**, so
an operator who points ``GANYMEDE_JOB_RUNTIME`` at a plain ``docker`` on the
host socket has given the worker more than §2.1 wants it to have. The interface
is the seam that makes fixing that a config change.

Every subprocess goes through an injectable ``runner``, the pattern
``host/runtime.py`` already uses -- so the flag template, the digest check and
the kill path are all unit-testable on a machine with no container runtime at
all, which is the machine this was written on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ganymede.worker.sandbox")

# A pull of a multi-GB archive; everything else is a control-plane call that
# should answer immediately or is wedged.
PULL_TIMEOUT_SEC = 3600
RUNTIME_TIMEOUT_SEC = 30

# docs/11 §3: how long a soft cancel waits for the job to checkpoint and exit
# before it becomes a hard one. Never past the lease -- the caller clamps.
DEFAULT_CANCEL_GRACE_SEC = 120

# Where the job sees its own scratch. The host side is a per-task directory.
CONTAINER_SCRATCH = "/scratch"

# The uid the job runs as. Matches the worker image's own unprivileged user
# (host/config.CONTAINER_UID) so a bind-mounted scratch dir written by one is
# readable by the other.
CONTAINER_UID = 1000


class SandboxError(RuntimeError):
    """The job could not be run. The task is abandoned, not failed."""


class DigestMismatch(SandboxError):
    """The archive did not hash to what the coordinator said (docs/11 §2.3).

    Its own type because its handling is specific: abandon with reason
    ``image_digest_mismatch`` and let the task be re-queued. It is not a
    verdict on the job -- the bytes in flight were wrong, and the next worker
    may pull them intact.
    """


@dataclass(frozen=True)
class SandboxConfig:
    """The half of §2.2 that is a deployment's to choose.

    Defaults are the §4.6 baseline the worker container itself runs under, so a
    job never gets more of the machine than the worker was given.
    """

    scratch_root: Path
    runtime_bin: str = "docker"
    memory: str = "16g"
    cpus: str = "0.9"
    # The operator's override, not a default GPU grant. Before docs/14, one
    # task ever ran on a machine at a time (Decision 4), so "all" was a
    # harmless default -- the one task on the box already owned every card.
    # Now several leases can share a multi-GPU host concurrently, and "all"
    # unconditionally would hand each of their containers every card,
    # including the ones a *sibling* lease holds -- exactly the double-booking
    # docs/14's ledger exists to prevent. ``None`` (unset) therefore no longer
    # means "all devices"; it means "no override", and ``run_argv`` derives
    # the actual flag from the lease's own ``devices`` via ``device_argv``. An
    # operator who sets this explicitly still gets it verbatim -- see
    # ``run_argv``'s docstring for why that is deliberate and what it costs.
    gpus: str | None = None
    pids_limit: int = 512
    scratch_gb: int = 50
    # §2.2: the ceiling on spec.max_runtime_sec, not the value itself.
    job_max_runtime_sec: int = 86_400
    cancel_grace_sec: int = DEFAULT_CANCEL_GRACE_SEC
    # §2.2's storage quota is driver-dependent (`--storage-opt size=` needs
    # devicemapper/btrfs/zfs or xfs with pquota). Off by default because on
    # overlay2 -- the common case -- passing it is a hard error at `run`, and a
    # flag that refuses to start the job is worse than a quota that is a
    # directory the worker wipes.
    storage_opt_size: bool = False

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "SandboxConfig":
        env = os.environ if environ is None else environ
        root = env.get("GANYMEDE_JOB_SCRATCH")
        if not root:
            raise SandboxError(
                "GANYMEDE_JOB_SCRATCH is not set: a contained job needs a "
                "host-visible scratch directory (docs/11 §2.3)"
            )
        return cls(
            scratch_root=Path(root),
            runtime_bin=env.get("GANYMEDE_JOB_RUNTIME", "docker"),
            memory=env.get("GANYMEDE_JOB_MEMORY", "16g"),
            cpus=env.get("GANYMEDE_JOB_CPUS", "0.9"),
            # No default here either (see the field's docstring): unset or
            # empty means "no override", not "all".
            gpus=env.get("GANYMEDE_JOB_GPUS") or None,
            pids_limit=int(env.get("GANYMEDE_JOB_PIDS_LIMIT", "512")),
            scratch_gb=int(env.get("GANYMEDE_JOB_SCRATCH_GB", "50")),
            job_max_runtime_sec=int(env.get("GANYMEDE_JOB_MAX_RUNTIME_SEC", "86400")),
            cancel_grace_sec=int(
                env.get("GANYMEDE_JOB_CANCEL_GRACE_SEC", str(DEFAULT_CANCEL_GRACE_SEC))
            ),
            storage_opt_size=env.get("GANYMEDE_JOB_STORAGE_QUOTA", "").lower()
            in {"1", "true", "yes", "on"},
        )


def detect_runtime(runner=None, runtime_bin: str | None = None) -> str | None:
    """The runtime this machine can launch a job container with, or ``None``.

    Reported in the compute profile, where the coordinator's claim gate reads
    it (docs/11 §4): a machine that answers ``None`` is refused submitter jobs
    with ``no_container_runtime`` rather than handed an image it cannot run.

    Deliberately an actual call, not a ``shutil.which``: inside the worker
    container the binary can be present while the socket it needs is not, and a
    profile that claims a capability the machine does not have costs a real
    task a real lease.
    """
    binary = runtime_bin or os.environ.get("GANYMEDE_JOB_RUNTIME", "docker")
    run = runner or _run
    try:
        result = run([binary, "version", "--format", "{{.Server.Version}}"],
                     timeout=RUNTIME_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return Path(binary).stem


# Backends where a container given no device flags at all gets *no* accelerator,
# as opposed to cpu/mps where there is nothing to hand it in the first place.
# The distinction matters only in one place -- ``run_argv``'s empty-lease
# branch -- but it matters a lot there: on these three, "pin nothing" and "pin
# everything" are opposite outcomes rather than the same one.
_DISCRETE_BACKENDS = frozenset({"cuda", "rocm", "xpu"})


#: The visibility variables whose values are *ordinal lists*, so an ambient
#: restriction has to be composed through rather than overwritten. Keyed by the
#: variable the pin would set; see ``_compose_visible``.
ORDINAL_VISIBILITY_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                            "ZE_AFFINITY_MASK")


def compose_visible(var: str, local: list[int],
                     env: "os._Environ[str] | dict[str, str] | None" = None
                     ) -> str | None:
    """Translate this lease's *worker-local* device indices into values
    meaningful to a child, honouring any restriction already on the worker.

    ``CUDA_VISIBLE_DEVICES`` and its siblings do **not** nest. When a process
    sets one, the driver reads it against the box's *full physical* device
    list -- not against whatever subset an ancestor's own setting had already
    narrowed things to. But ``worker.probe.run_probe`` enumerates
    ``range(torch.cuda.device_count())``, which *is* narrowed: a worker
    launched with ``CUDA_VISIBLE_DEVICES=4,5,6,7`` reports four devices as
    indices 0-3, and those are the numbers the coordinator allocates and hands
    back in a lease. Setting that lease's ``[2]`` straight into the child
    would resolve to physical card **2** -- a card deliberately withheld from
    Ganymede -- rather than the intended card 6.

    So the ambient list, when present, is the translation table: local index
    ``i`` means its ``i``-th entry. With nothing set, every local index is
    already physical and the value is returned unchanged -- which is every
    ordinary deployment, byte for byte.

    ``None`` means refuse: a local index with no entry in the ambient list
    cannot be resolved to any card at all, and guessing is how a lease lands
    on hardware it does not hold.
    """
    env = os.environ if env is None else env
    raw = (env.get(var) or "").strip()
    if not raw:
        return ",".join(str(d) for d in sorted(local))
    # An ambient list may name devices by UUID as well as by ordinal; either
    # way the *position* is what a local index refers to, so no parsing of the
    # entries themselves is needed or wanted.
    ambient = [p.strip() for p in raw.split(",") if p.strip()]
    resolved: list[str] = []
    for i in sorted(local):
        if i < 0 or i >= len(ambient):
            return None
        resolved.append(ambient[i])
    return ",".join(resolved)


def device_argv(backend: str | None, indices: list[int],
                env: "os._Environ[str] | dict[str, str] | None" = None
                ) -> list[str] | None:
    """The container flags that confine a job to exactly this lease's devices
    (docs/14 §2's "container pin" column). The container-launch counterpart of
    ``worker.loop._pin_env``, which does the identical job for an in-process
    child -- read that function's docstring first; this one only differs where
    a container's confinement genuinely differs from a process's.

    Returns ``None`` to mean *refuse*: docs/14 §2 says a backend with no known
    pinning form "refuses to launch rather than falling back to 'all
    devices'". Handing a container every card on the box while the
    coordinator's ledger believes this lease holds only ``indices`` is exactly
    the double-booking the ledger exists to prevent -- and unlike the
    in-process case, a container that gets no device flags at all does not
    fall back to "sees everything" the way an unpinned process does; it falls
    back to "sees nothing", which is a different failure but not a safer one
    to produce silently.

    An empty ``indices`` is not a refusal. It means nothing was named to pin
    to -- unreachable for a real lease once ``devices.allocate`` has run (it
    raises on ``count <= 0``), but reachable from a payload built before
    docs/14 landed (``jobtypes.base.TaskSpec.devices`` defaults to ``[]`` for
    exactly this reason -- see its docstring). Inventing a pin for a device
    that was never named would be worse than pinning none.
    """
    idx = sorted(indices)
    if not idx:
        return []
    if backend == "cuda":
        # Docker's ``--gpus`` value is parsed as a CSV key=value list of its
        # own (count=, capabilities=, driver=, device=), so an *unquoted*
        # multi-index value like ``device=0,2`` is ambiguous with that outer
        # grammar: everything after the first comma reads as further fields
        # rather than more of ``device``'s value, and the failure is silent --
        # a wrong device set, not an error. The fix is embedding **literal**
        # double-quote characters in the flag's value (not shell quoting --
        # these two characters travel inside the single argv element), which
        # tells Docker's CSV parser to treat the whole thing as one quoted
        # field. Confirmed against Docker's own example
        # (docs.docker.com/engine/containers/gpu/): `--gpus '"device=0,2"'`,
        # where the outer `'...'` is the shell's and the inner `"..."` is the
        # value Docker actually receives. A single index does not strictly
        # need it, but a quoted single field parses identically to an
        # unquoted one, so there is no reason to keep two code paths -- one of
        # which is the one this footgun would come back through.
        # Composed against any restriction already on the *worker* process,
        # exactly as the in-process pin is (``compose_visible`` above). The
        # NVIDIA container runtime addresses cards by absolute physical index
        # or UUID and does not inherit the worker's own
        # ``CUDA_VISIBLE_DEVICES``, so a worker launched restricted to
        # ``4,5,6,7`` -- whose probe therefore reported its cards as 0-3 --
        # would otherwise hand the container physical card 2 for a lease
        # holding local index 2, instead of card 6. ``--gpus device=`` accepts
        # a UUID wherever it accepts an index, so a UUID-valued ambient list
        # composes through unchanged too.
        csv = compose_visible("CUDA_VISIBLE_DEVICES", idx, env)
        if csv is None:
            return None
        return ["--gpus", f'"device={csv}"']
    if backend == "rocm":
        # docs/14 §2. ``/dev/kfd`` is the single shared compute-queue device
        # for the whole box; ``/dev/dri/renderD{128+N}`` is the per-card DRM
        # render node, numbered from 128 by kernel convention, so index N is
        # node 128+N *if* index N is also how the kernel orders render nodes --
        # unverified on real ROCm hardware, and the doc's own formula, not
        # something this step can confirm without a card to test on (see the
        # step report). ``--group-add video`` is what makes those
        # group-owned nodes readable by ``--user 1000:1000`` rather than root.
        # Not composed through an ambient ``HIP_VISIBLE_DEVICES`` the way the
        # cuda branch above is: that variable restricts what *torch* enumerates,
        # while what is needed here is a DRM render-node number, and an ambient
        # list may hold UUIDs that no ``renderD`` formula can consume. Since the
        # 128+N formula is itself unverified on real hardware (below), guessing
        # a second unverified mapping on top of it would compound the risk
        # rather than reduce it. A restricted ROCm worker is therefore a known
        # gap, named here the way the formula itself is.
        argv = ["--device=/dev/kfd"]
        argv += [f"--device=/dev/dri/renderD{128 + i}" for i in idx]
        argv += ["--group-add", "video"]
        return argv
    if backend == "xpu":
        # Same render-node convention as ROCm, no ``/dev/kfd`` analogue --
        # Intel's compute stack talks to the render node directly. Same
        # unverified-formula caveat as above.
        return [f"--device=/dev/dri/renderD{128 + i}" for i in idx]
    if backend == "cpu":
        # docs/14 §2's table lists ``--cpuset-cpus`` as cpu's analogue, but
        # nothing feeds it a real value to pin *to*: ``probe._cpu_devices``
        # hands out ``range(slots)`` as pure ordinal labels (its own
        # docstring), not physical core ids, and the module has no topology
        # query anywhere that could turn index N into a real core number.
        # ``loop._pin_env`` already documents this exact gap for the
        # in-process pin and leaves cpu unpinned rather than invent a mapping;
        # this does the same, for the same reason, rather than fabricate a
        # ``--cpuset-cpus`` value that could pin two sibling containers to the
        # same physical core while both believe they are isolated -- worse
        # than the accepted gap it would replace. This also keeps the
        # overwhelming common case (``GANYMEDE_CPU_SLOTS`` at its default of
        # 1, one task, the whole box already exclusively its own) working
        # exactly as before: nothing to add, nothing needed.
        return []
    # mps, or any name this table does not know. mps is always exactly one
    # device by construction (unified memory, no index), and its *in-process*
    # pin is correctly a no-op (``_pin_env`` returns ``{}`` for it) -- torch's
    # MPS backend has no visible-device concept, so an unpinned process
    # already sees the one GPU there is. A *container* is a different
    # question with a different answer: Docker Desktop for Mac runs
    # containers inside a Linux VM with no Metal passthrough at all, so there
    # is no flag -- quoted, unquoted, or otherwise -- that hands a container
    # that GPU. docs/14 §2 spells this "none", distinct from cpu's "n/a": cpu
    # genuinely has nothing to restrict (single implicit tenant); mps has
    # something to restrict to and no mechanism to do it with. Refusing here
    # is therefore not the same kind of refusal as an unrecognised backend's --
    # it is honesty about a platform gap, not a ledger violation -- but the
    # consequence for the caller is identical, and it beats the alternative of
    # a container starting, running the submitter's job with no GPU at all,
    # and nobody finding out until the output looks wrong.
    return None


_LOADED_ID = re.compile(r"sha256:[0-9a-f]{64}")
# `Loaded image: name:tag` -- what a *tagged* archive reports, which is every
# archive docs/11 §1.1's upload path can produce.
_LOADED_TAG = re.compile(r"Loaded image:\s*(\S+)")


def container_name_for(task_id: str) -> str:
    """The name a contained job's container answers to, deterministically.

    Split out so it can be computed by someone who has not started (or does
    not own) the ``JobContainer`` -- the worker's heartbeat thread names the
    target of a cancel this way (``loop.Worker._run_contained``) without
    constructing one, and it is what a lease crumb's ``container`` field
    records so the host agent's reaper (``host.agent.reap_orphaned_jobs``) can
    ask the runtime about the same name later, from a different process,
    knowing only the task id.
    """
    return f"ganymede-job-{task_id[:12]}"


@dataclass
class JobContainer:
    """One contained job: pull, verify, load, run, and the kill path."""

    task_id: str
    config: SandboxConfig
    runner: object = None
    container_name: str = ""
    image_id: str | None = None

    def __post_init__(self) -> None:
        self._run = self.runner or _run
        if not self.container_name:
            self.container_name = container_name_for(self.task_id)

    # -- scratch (§2.3) ---------------------------------------------------

    @property
    def scratch(self) -> Path:
        return self.config.scratch_root / self.task_id

    def prepare_scratch(self) -> Path:
        """``/scratch/in`` for staged inputs, ``/scratch/out`` for results.

        Wiped first, not merged: a retry of the same task id must not see the
        previous attempt's half-written output and mistake it for its own.
        """
        self.wipe_scratch()
        (self.scratch / "in").mkdir(parents=True, exist_ok=True)
        (self.scratch / "out").mkdir(parents=True, exist_ok=True)
        return self.scratch

    def wipe_scratch(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    # -- pull and verify (§2.3) -------------------------------------------

    def fetch_archive(self, download, url: str, expected_digest: str) -> Path:
        """Pull the archive and hash it *before* anything loads it.

        The order is the point. ``docker load`` parses an archive an author
        controls; running it on bytes that failed their check would be trusting
        the thing being checked. The digest is the coordinator's row (docs/11
        §1.1), which is the only value a worker can independently recompute.
        """
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch / "image.tar"
        payload = download(url)
        path.write_bytes(payload)
        actual = hashlib.sha256(payload).hexdigest()
        want = (expected_digest or "").split(":")[-1].lower()
        if not want or actual != want:
            path.unlink(missing_ok=True)
            raise DigestMismatch(
                f"image archive hashed {actual}, coordinator said {want or '(none)'}"
            )
        return path

    def load_image(self, archive: Path) -> str:
        """``docker load`` and return the image **ID**, never the tag.

        The archive carries its own ``RepoTags``, which the author chose and
        which can collide with an image already on the machine -- running by tag
        would let a submitter's archive decide which bytes execute. The id is
        content-addressed and cannot.

        **``docker load`` prints one of two things**, and which one depends on
        the archive rather than on us:

        * ``Loaded image ID: sha256:...`` -- an archive with no repo tags;
        * ``Loaded image: name:tag``      -- an archive that has them.

        Only the first was handled, which meant this raised on every archive a
        submitter could realistically produce: docs/11 §1.1 takes a
        ``docker save`` payload and ``upload-url`` requires a ``repo_tag``, so
        real archives are always the tagged kind. Found the first time this ran
        against a real daemon; no test with an injectable runner could see it,
        because the fake returned the output the parser already expected.

        Resolving the tag through ``inspect`` keeps the property intact. The
        tag is used once, immediately after the load that just re-pointed it at
        these bytes, and only to *learn* the id; what is run is still the id.
        """
        result = self._run(
            [self.config.runtime_bin, "load", "-i", str(archive)],
            timeout=PULL_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise SandboxError(f"image load failed: {result.stderr.strip()}")
        stdout = result.stdout or ""

        match = _LOADED_ID.search(stdout)
        if match is not None:
            self.image_id = match.group(0)
            return self.image_id

        tag_match = _LOADED_TAG.search(stdout)
        if tag_match is None:
            raise SandboxError(
                f"image load reported no image id: {stdout.strip()}"
            )
        self.image_id = self._resolve_tag(tag_match.group(1).strip())
        return self.image_id

    def _resolve_tag(self, tag: str) -> str:
        """The id the tag points at, right after the load that set it."""
        result = self._run(
            [self.config.runtime_bin, "inspect", "--format", "{{.Id}}", tag],
            timeout=RUNTIME_TIMEOUT_SEC,
        )
        digest = (result.stdout or "").strip()
        if result.returncode != 0 or not _LOADED_ID.fullmatch(digest):
            raise SandboxError(
                f"loaded image {tag!r} but could not resolve it to an id: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        return digest

    # -- run (§2.2, §2.3, §2.4) -------------------------------------------

    def runtime_ceiling(self, max_runtime_sec: int | None) -> int:
        """The job's ask, clamped by the operator's ceiling (§2.2).

        Its own method rather than a value stashed on the instance by whichever
        of ``run_argv`` / ``start`` ran last: a caller that builds argv once and
        starts twice would otherwise inherit the previous task's clamp, and
        nothing would say so.
        """
        return min(max_runtime_sec or self.config.job_max_runtime_sec,
                   self.config.job_max_runtime_sec)

    def run_argv(self, image_id: str, *, env: dict[str, str] | None = None,
                 max_runtime_sec: int | None = None, backend: str | None = None,
                 devices: list[int] | None = None) -> list[str]:
        """The flag template. §4.6's baseline plus §2.2's additions.

        Notes on the ones that are not obvious:

        - ``--memory`` and ``--memory-swap`` are set **equal**, which is how
          Docker spells "no swap". Without it a job over its memory cap slows to
          a crawl against the host's disk instead of dying, and a task that
          takes twenty times its budget is worse for the fleet than one that
          fails.
        - ``--network none`` is unconditional here. §2.4's per-job allowlist and
          its CONNECT proxy are a separate piece; the common job reads
          ``/scratch/in``, computes, writes ``/scratch/out`` and needs no network
          at all, because the worker does every transfer.
        - The worker's own state dir is **not** mounted. §4.6's reasoning is
          that a process which can write there can forge the kill switch, and
          the job is exactly the process that must not.
        - ``--user`` is forced to a non-root uid regardless of what the image's
          own ``USER`` says. The scan flags a root image (docs/11 §1.3) but a
          flag is advice; this is the enforcement.
        - ``--gpus`` (or its ROCm/XPU equivalent): ``config.gpus`` is an
          **explicit operator override** and wins outright over ``devices``
          when set -- the same relationship every other §2.2 field has to the
          task (``job_max_runtime_sec`` clamps ``max_runtime_sec``, never the
          other way around), and the deliberate escape hatch docs/14 §2 keeps
          open for a deployment that knows better. That trust cuts both ways:
          an operator who sets ``GANYMEDE_JOB_GPUS=all`` on a host running
          several concurrent leases re-opens the exact double-booking docs/14
          exists to close, on purpose, and this only warns about it rather
          than refusing -- refusing an explicit operator setting is not this
          module's call to make. Left unset (the default, see the field's own
          docstring), ``devices`` -- this lease's actual allocation -- decides,
          through ``device_argv``, which raises ``SandboxError`` rather than
          start the container if the backend has no known way to honour it.
        """
        cfg = self.config
        if cfg.gpus and devices:
            # ``cfg.gpus`` empty-but-not-``None`` is an explicit *withhold*
            # (see the ``if cfg.gpus:`` below), not an override that
            # contradicts the ledger, so it does not warrant this warning.
            log.warning(
                "task %s: GANYMEDE_JOB_GPUS=%r overrides the lease's own "
                "devices %s -- this container is not confined to what the "
                "coordinator's ledger believes it holds",
                self.task_id, cfg.gpus, sorted(devices),
            )
        argv = [
            cfg.runtime_bin, "run",
            "--detach",
            "--name", self.container_name,
            # §4.6 baseline, inherited whole.
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--user", f"{CONTAINER_UID}:{CONTAINER_UID}",
            # §2.2 additions.
            "--memory", cfg.memory,
            "--memory-swap", cfg.memory,
            "--cpus", cfg.cpus,
            "--pids-limit", str(cfg.pids_limit),
            "--ipc=private",
            "--ulimit", "core=0",
            "--network", "none",
            "--stop-timeout", str(cfg.cancel_grace_sec),
            # §2.3: scratch, /tmp and /run. Nothing else is writable.
            "-v", f"{self.scratch}:{CONTAINER_SCRATCH}",
            "--tmpfs", "/tmp",
            "--tmpfs", "/run",
        ]
        if cfg.storage_opt_size:
            argv += ["--storage-opt", f"size={cfg.scratch_gb}G"]
        if cfg.gpus is not None:
            # The operator's override, verbatim -- see the docstring above.
            if cfg.gpus:
                argv += ["--gpus", cfg.gpus]
        elif not devices and backend in _DISCRETE_BACKENDS:
            # No devices named, on a backend where that would mean handing the
            # container no accelerator at all.
            #
            # This is **version skew, not a ledger violation**, and the two
            # want opposite answers. A coordinator that allocates devices
            # always names at least one (``devices.allocate`` raises on a
            # non-positive count), so an empty list here says the coordinator
            # predates docs/14 -- and a coordinator that predates docs/14 is
            # still enforcing one lease per machine, which makes ``all`` both
            # safe and exactly what this host used to get. Emitting nothing
            # instead would silently run a submitter's GPU job on CPU: it
            # "works", 50x slower, and nobody finds out.
            #
            # The refusal below is for the genuinely different case -- devices
            # *were* named and this backend has no way to honour them, where
            # obeying loosely would double-book a card the ledger has already
            # promised to someone else. If a NEW coordinator ever reaches here
            # it is a bug on its side, and the ``lease_without_device``
            # invariant catches it there rather than this branch papering over
            # it silently.
            log.warning(
                "task %s: the coordinator named no devices for a %s host; "
                "falling back to every device, as a pre-docs/14 coordinator "
                "would have. If this coordinator does allocate devices, this "
                "is a bug on its side.", self.task_id, backend,
            )
            argv += ["--gpus", "all"]
        else:
            pin = device_argv(backend, devices or [])
            if pin is None:
                raise SandboxError(
                    f"backend {backend!r} has no known way to confine a "
                    f"container to devices {devices} (docs/14 §2); refusing "
                    f"to start task {self.task_id} unconfined"
                )
            argv += pin
        for name in sorted(env or {}):
            # Name only, never NAME=value: a value here lands in the host
            # process table and in `inspect` output for the life of the
            # container. Same reasoning as host/runtime.run_argv.
            argv += ["-e", name]
        argv += [
            "-e", "GANYMEDE_SCRATCH",
            "-e", "GANYMEDE_MAX_RUNTIME_SEC",
            image_id,
        ]
        return argv

    def start(self, image_id: str, *, env: dict[str, str] | None = None,
              max_runtime_sec: int | None = None, backend: str | None = None,
              devices: list[int] | None = None) -> str:
        argv = self.run_argv(image_id, env=env, max_runtime_sec=max_runtime_sec,
                             backend=backend, devices=devices)
        child_env = dict(os.environ)
        child_env.update(env or {})
        child_env["GANYMEDE_SCRATCH"] = CONTAINER_SCRATCH
        child_env["GANYMEDE_MAX_RUNTIME_SEC"] = str(
            self.runtime_ceiling(max_runtime_sec)
        )
        result = self._run(argv, timeout=RUNTIME_TIMEOUT_SEC, env=child_env)
        if result.returncode != 0:
            raise SandboxError(f"could not start job container: {result.stderr.strip()}")
        return self.container_name

    def is_running(self) -> bool:
        result = self._run(
            [self.config.runtime_bin, "inspect", "-f", "{{.State.Running}}",
             self.container_name],
            timeout=RUNTIME_TIMEOUT_SEC,
        )
        return result.returncode == 0 and (result.stdout or "").strip() == "true"

    def exit_code(self) -> int | None:
        result = self._run(
            [self.config.runtime_bin, "inspect", "-f", "{{.State.ExitCode}}",
             self.container_name],
            timeout=RUNTIME_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            return None
        try:
            return int((result.stdout or "").strip())
        except ValueError:
            return None

    # -- the kill path (§3) -----------------------------------------------

    def cancel(self, mode: str, *, grace_sec: int | None = None) -> None:
        """Act on a cancel that arrived on a heartbeat (docs/11 §3 step 3).

        ``soft`` is SIGTERM with a deadline: the job is expected to trap it,
        checkpoint the unit it is on, and exit. One that ignores it is SIGKILLed
        at the timeout -- the grace is a promise about how long the swarm waits,
        not about whether the container stops.

        ``hard`` is SIGKILL now, and takes no view on what the job was doing.
        """
        grace = grace_sec if grace_sec is not None else self.config.cancel_grace_sec
        if mode == "hard":
            self._run([self.config.runtime_bin, "kill", self.container_name],
                      timeout=RUNTIME_TIMEOUT_SEC)
            return
        self._run(
            [self.config.runtime_bin, "stop", "--time", str(grace),
             self.container_name],
            timeout=grace + RUNTIME_TIMEOUT_SEC,
        )

    def remove(self) -> None:
        self._run([self.config.runtime_bin, "rm", "-f", self.container_name],
                  timeout=RUNTIME_TIMEOUT_SEC)

    def cleanup(self) -> None:
        """Everything this task left on the machine. Called on every exit path,
        including the failures -- a scratch dir that survives a crash is a disk
        that fills up over a week of them."""
        self.remove()
        self.wipe_scratch()


# --------------------------------------------------------------------------
# The lease crumb (docs/11 §3, wedged-worker path)
# --------------------------------------------------------------------------

# One task, one crumb file, under this subdirectory of the job scratch root.
# Originally a single ``lease.json`` was enough -- one task per host, one
# lease to renew. A host that runs several tasks concurrently needs each
# task's renewal recorded separately: a shared file would have the second
# task's heartbeat overwrite the first's record, and the reaper would then see
# only one task's liveness for two containers, with no way to tell which.
_LEASE_CRUMB_DIR = "leases"


def _crumb_path(scratch_root: Path, task_id: str) -> Path:
    return scratch_root / _LEASE_CRUMB_DIR / f"{task_id}.json"


def write_lease_crumb(scratch_root: Path, task_id: str, renewed_at: str,
                      container: str | None = None) -> Path:
    """Record that this task's lease was renewed, where the *host agent* can
    read it.

    Not in the state dir, which is mounted read-only into the worker precisely
    so the worker cannot forge the contributor's kill switch (host/runtime
    ``run_argv``). The job scratch root is the one directory both the worker and
    the host agent can see and the worker may write -- and it exists already,
    because §2.3's bind mount needs it.
    """
    path = _crumb_path(scratch_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task_id": task_id, "renewed_at": renewed_at, "container": container}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return path


def read_lease_crumb(scratch_root: Path, task_id: str) -> dict | None:
    """This task's own crumb, or ``None`` if it has none (or it does not parse)."""
    try:
        return json.loads(_crumb_path(scratch_root, task_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_all_lease_crumbs(scratch_root: Path) -> list[tuple[str | None, dict]]:
    """Every crumb on disk, each paired with the key ``clear_lease_crumb``
    needs to remove exactly that one file.

    Used by the reaper, which must weigh every task's liveness, not just one
    (concurrent contained jobs mean concurrent orphans). The pairing key is the
    crumb file's own name (``path.stem``), not the ``task_id`` field inside its
    JSON -- the two should always agree, but the name is what
    ``clear_lease_crumb`` actually needs, and trusting the file over its own
    content is the cheaper invariant to keep.

    Tolerates a legacy single-file ``lease.json`` a worker built before crumbs
    were split per task -- paired with ``None``, which tells
    ``clear_lease_crumb`` to remove that file specifically rather than a
    per-task one. A crumb that fails to parse names no container to act on
    (there is nothing here to weigh it against), so it is dropped rather than
    surfaced as an unresolvable entry -- the same fate an unparseable crumb met
    under the single-file scheme.
    """
    crumbs: list[tuple[str | None, dict]] = []
    crumb_dir = scratch_root / _LEASE_CRUMB_DIR
    if crumb_dir.is_dir():
        for path in sorted(crumb_dir.glob("*.json")):
            try:
                crumb = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            crumbs.append((path.stem, crumb))
    legacy = scratch_root / "lease.json"
    if legacy.is_file():
        try:
            crumbs.append((None, json.loads(legacy.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            pass
    return crumbs


def clear_lease_crumb(scratch_root: Path, task_id: str | None = None) -> None:
    """Remove one task's crumb -- or, with ``task_id=None``, the legacy
    single-file crumb (see ``read_all_lease_crumbs``). Best-effort: the crumb
    is bookkeeping for the reaper, not state anything else depends on being
    gone, and a task whose crumb could not be cleared is simply a crumb the
    next sweep looks at again.
    """
    path = _crumb_path(scratch_root, task_id) if task_id is not None \
        else scratch_root / "lease.json"
    try:
        path.unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------
# subprocess
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _run(argv: list[str], *, timeout: int, env: dict[str, str] | None = None) -> Completed:
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built here, never a shell string
            argv, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        # A control-plane call that does not answer means a wedged daemon. The
        # caller's next tick tries again; blocking on it would pile up.
        return Completed(returncode=124, stderr=f"timeout after {timeout}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return Completed(returncode=127, stderr=str(exc))
    return Completed(proc.returncode, proc.stdout or "", proc.stderr or "")
