"""Capability probing (docs/02-architecture-v2.md 6.9).

Supporting almost any hardware rules out the obvious implementation. A table
mapping device names to capabilities would be wrong the first time someone shows
up with a card nobody anticipated, and the failure would be a silently-excluded
contributor rather than a loud error. So the worker **measures and reports**, and
the coordinator believes it.

Registration always succeeds
----------------------------
Every function here returns a *result*, never raises. A CPU-only laptop, a 6 GB
card, a machine whose torch build has no working bf16 -- all of them register
fine and simply never match a run's requirements. Turning someone away at the
door is the wrong behavior for a platform whose goal is broad compatibility, and
the stored profile is also the support answer when a contributor asks why they
never get work.

Adding a backend is one entry in one table
------------------------------------------
`BACKENDS` is ordered, and detection walks it. Supporting AMD or Intel properly
is a matter of filling in an entry, not of touching the loop, the client, the
coordinator, or anything the coordinator stores -- 6.9 promises that new hardware
needs no coordinator change, and this table is where that promise is kept.

**Order is load-bearing for ROCm.** A PyTorch ROCm build reports
``torch.cuda.is_available() == True`` and exposes the whole ``torch.cuda`` API;
AMD chose that so CUDA code runs unmodified. The only reliable discriminator is
``torch.version.hip``, so ROCm is checked *before* CUDA -- reverse them and every
AMD machine in the fleet reports itself as NVIDIA, and the throughput table keys
two different architectures under one name.

A multi-GPU host (docs/14 §2)
------------------------------
Each ``Backend`` also carries ``list_devices``, which answers "how many, and
which ``torch.device`` objects" the way ``describe`` answers "what is this one
like". ROCm and CUDA share an implementation -- same reason ``describe`` does --
XPU calls its own ``device_count``, MPS is always exactly one device (unified
memory has no index to enumerate), and CPU defaults to one and is raisable by an
operator (``GANYMEDE_CPU_SLOTS``) because ``_describe_cpu`` reports ``vram_mb``
as total system RAM: N slots would each claim the whole machine's memory, so
raising it is a deliberate act, never a default.

``run_probe`` measures every device ``list_devices`` returns, in a plain
sequential loop -- never concurrently. ``allocation_ceiling_mb`` doubles a real
allocation until it fails; two such searches racing on the same box would each
see a fraction of the true ceiling, and that understated number is exactly what
``budget.is_eligible`` and ``constraints._vram_mb`` prefer over ``vram_mb``, so
a concurrent probe would silently make a machine ineligible for work it could
actually do. Device 0's measurements populate the profile's flat fields, byte-
identical to what a single-GPU probe has always produced -- see ``run_probe``.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ganymede.device import device_name

# Bump when the benchmark's shape, dtype or iteration count changes. Scores are
# stored on the coordinator and compared against a run's `min_bench_score`, so a
# silent change to what the number means would silently re-rank the whole fleet.
BENCH_VERSION = 1

# The fixed shape 6.9 calls for. Small enough to run on a laptop CPU in a second,
# large enough that a real GPU is not measuring launch overhead.
BENCH_BATCH, BENCH_SEQ, BENCH_DIM, BENCH_HEADS = 4, 128, 256, 4
BENCH_ITERS = 8
BENCH_WARMUP = 2

# Allocation-ceiling search bounds, in MiB.
ALLOC_PROBE_START_MB = 256
ALLOC_PROBE_MAX_MB = 512 * 1024


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """One compute backend, and everything the probe needs to drive it.

    ``allocation_is_catchable`` is the field that matters most and is easiest to
    overlook. On CUDA, ROCm and XPU an over-allocation raises a catchable
    exception, so the ceiling can be *measured* by climbing until it fails. On
    CPU -- and on MPS, whose memory is the system's -- exhausting memory invokes
    the kernel's OOM killer, which sends SIGKILL. A probe that measured the
    ceiling there would kill the worker instead of reporting a number, so those
    backends report what the OS says is available and do not climb.
    """

    name: str
    detect: Callable[[], bool]
    device: Callable[[], torch.device]
    describe: Callable[[torch.device], dict[str, Any]]
    # Every ``torch.device`` this backend has right now, index order (docs/14
    # §2). ``device`` above stays what it always was -- this backend's default
    # device, used by every caller that predates multi-GPU reporting -- so
    # ``list_devices`` is additive rather than a replacement, and a machine
    # whose enumeration errors (module docstring's "registration always
    # succeeds") falls back to ``[device()]`` in ``run_probe``, never to zero
    # devices.
    list_devices: Callable[[], list[torch.device]]
    # True when every device this backend reports draws on ONE pool of memory
    # rather than each having its own -- CPU slots and Apple's unified memory.
    # It decides whether the per-device ``vram_mb`` figures may be summed:
    # four CPU slots each reporting the machine's whole RAM would have
    # ``constraints.total_vram_gb`` advertise four times the memory that
    # exists, and a submitter predicate like ``total_vram_gb >= 100`` would
    # match a 32 GB box.
    allocation_is_catchable: bool
    # Defaults False: a card's VRAM is normally its own, which is both the
    # common case and the safe one to get by default -- summing discrete
    # memory is exactly right, while wrongly summing a shared pool
    # over-advertises it.
    shares_memory_pool: bool = False
    synchronize: Callable[[torch.device], None] = lambda device: None
    empty_cache: Callable[[], None] = lambda: None
    # "supported" backends are exercised in CI or on real hardware we have.
    # "untested" ones are believed correct and have never been run; the honest
    # thing is to say so in the profile rather than to imply equal confidence.
    maturity: str = "supported"
    notes: str = ""


def _noop_sync(device: torch.device) -> None:
    pass


def _is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()


def _is_cuda() -> bool:
    return not getattr(torch.version, "hip", None) and torch.cuda.is_available()


def _has_xpu() -> bool:
    xpu = getattr(torch, "xpu", None)
    try:
        return xpu is not None and xpu.is_available()
    except Exception:  # noqa: BLE001 - a backend that errors on probe is absent
        return False


def _has_mps() -> bool:
    mps = getattr(torch.backends, "mps", None)
    try:
        return mps is not None and mps.is_available()
    except Exception:  # noqa: BLE001
        return False


def _describe_cuda(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "device_name": device_name(device),
        "vram_mb": int(props.total_memory / 1024**2),
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": torch.version.cuda,
    }


def _describe_rocm(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "device_name": device_name(device),
        "vram_mb": int(props.total_memory / 1024**2),
        # gfx target, e.g. "gfx1100". The CUDA-shaped major.minor is meaningless
        # on AMD, and reporting it would invite comparisons that do not hold.
        "compute_capability": getattr(props, "gcnArchName", None),
        "driver": getattr(torch.version, "hip", None),
    }


def _describe_xpu(device: torch.device) -> dict[str, Any]:
    props = torch.xpu.get_device_properties(device)
    return {
        "device_name": device_name(device),
        "vram_mb": int(getattr(props, "total_memory", 0) / 1024**2),
        "compute_capability": getattr(props, "device_arch", None),
        "driver": getattr(props, "driver_version", None),
    }


def _describe_mps(device: torch.device) -> dict[str, Any]:
    return {
        "device_name": device_name(device),
        # Apple silicon has no dedicated VRAM: the GPU shares system memory, so
        # the total is the machine's RAM and the *usable* figure is whatever the
        # OS is not already holding. alloc_max_mb is the number to trust here.
        "vram_mb": int(_system_memory_mb()),
        "compute_capability": None,
        "driver": None,
    }


def _describe_cpu(device: torch.device) -> dict[str, Any]:
    return {
        "device_name": device_name(device),
        "vram_mb": int(_system_memory_mb()),
        "compute_capability": None,
        "driver": None,
    }


# --------------------------------------------------------------------------
# Per-backend device enumeration (docs/14 §2)
# --------------------------------------------------------------------------


def _cuda_devices() -> list[torch.device]:
    """One entry per ``torch.cuda.device_count()``, in index order.

    Shared by ``cuda`` and ``rocm`` for the same reason ``_describe_rocm``
    exists as its own function rather than an alias of ``_describe_cuda``: the
    *count* is identical either way (AMD's ROCm build exposes the same
    ``torch.cuda`` API), even though what each index *is* differs enough to
    need its own describe.
    """
    return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]


def _xpu_devices() -> list[torch.device]:
    return [torch.device("xpu", i) for i in range(torch.xpu.device_count())]


def _mps_devices() -> list[torch.device]:
    """Always exactly one, unindexed -- unified memory, no per-card identity
    (docs/14 §2's table)."""
    return [torch.device("mps")]


def _cpu_slot_count() -> int:
    """1 by default; raisable by an operator via ``GANYMEDE_CPU_SLOTS``.

    A CPU box could genuinely run several tasks at once, but ``_describe_cpu``
    reports ``vram_mb`` as total system RAM -- there is no per-slot figure to
    report instead -- so N slots would each claim the whole machine's memory
    as their own budget. That is a real overcommit, not a rounding error, so
    raising it has to be an operator's deliberate act rather than something
    this probe guesses at. An unset or malformed value is read as 1: the same
    "never refuse to register" doctrine as everything else in this module
    applies to a misconfigured env var as much as to a broken driver.
    """
    raw = os.environ.get("GANYMEDE_CPU_SLOTS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _cpu_devices() -> list[torch.device]:
    return [torch.device("cpu") for _ in range(_cpu_slot_count())]


# Ordered. ROCm precedes CUDA deliberately -- see the module docstring.
BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="rocm", detect=_is_rocm, device=lambda: torch.device("cuda"),
        describe=_describe_rocm, list_devices=_cuda_devices,
        allocation_is_catchable=True,
        synchronize=torch.cuda.synchronize, empty_cache=torch.cuda.empty_cache,
        maturity="untested",
        notes="AMD via a PyTorch ROCm build. Detected by torch.version.hip; the "
              "torch.cuda API is the correct one to drive it.",
    ),
    Backend(
        name="cuda", detect=_is_cuda, device=lambda: torch.device("cuda"),
        describe=_describe_cuda, list_devices=_cuda_devices,
        allocation_is_catchable=True,
        synchronize=torch.cuda.synchronize, empty_cache=torch.cuda.empty_cache,
    ),
    Backend(
        name="xpu", detect=_has_xpu, device=lambda: torch.device("xpu"),
        describe=_describe_xpu, list_devices=_xpu_devices,
        allocation_is_catchable=True,
        synchronize=lambda device: torch.xpu.synchronize(device),
        empty_cache=lambda: torch.xpu.empty_cache(),
        maturity="untested",
        notes="Intel Arc / Data Center GPU via torch.xpu.",
    ),
    Backend(
        name="mps", detect=_has_mps, device=lambda: torch.device("mps"),
        describe=_describe_mps, list_devices=_mps_devices,
        # Unified memory: exhausting it is exhausting the machine's RAM, and the
        # OOM killer is not catchable.
        shares_memory_pool=True,
        allocation_is_catchable=False,
        synchronize=lambda device: torch.mps.synchronize(),
        empty_cache=lambda: torch.mps.empty_cache(),
        notes="Apple silicon. No container path (6.8) -- native install only.",
    ),
    Backend(
        name="cpu", detect=lambda: True, device=lambda: torch.device("cpu"),
        describe=_describe_cpu, list_devices=_cpu_devices,
        shares_memory_pool=True,
        allocation_is_catchable=False,
        synchronize=_noop_sync,
        notes="Always available. Rarely above a run's throughput floor, but it "
              "registers, and 6.8 notes such a machine may still be able to run evals.",
    ),
)


def detect_backend(prefer: str | None = None) -> Backend:
    """First backend in BACKENDS that reports itself present; cpu is the floor.

    ``prefer`` forces a specific one by name, which is what
    ``GANYMEDE_BACKEND`` exists for: a machine with both an Arc card and a
    working CPU path may want to be told which to use, and a bug report is much
    easier to reproduce when the backend can be pinned.
    """
    if prefer:
        for backend in BACKENDS:
            if backend.name == prefer:
                return backend
        raise ValueError(
            f"unknown backend {prefer!r}; have {[b.name for b in BACKENDS]}"
        )
    for backend in BACKENDS:
        try:
            if backend.detect():
                return backend
        except Exception:  # noqa: BLE001 - a backend that errors while detecting is absent
            continue
    return BACKENDS[-1]


def _system_memory_mb() -> float:
    """Total system RAM, by whatever means this platform offers."""
    try:  # Linux, macOS
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**2
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1024**2
    except Exception:  # noqa: BLE001
        return 0.0


def _available_memory_mb() -> float:
    """RAM the OS says is actually free, falling back to the total."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return _system_memory_mb()


# --------------------------------------------------------------------------
# The three measurements
# --------------------------------------------------------------------------


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, (RuntimeError, MemoryError)) and (
        "out of memory" in text or "allocat" in text
    )


def allocation_ceiling_mb(backend: Backend, device: torch.device | None = None) -> dict[str, Any]:
    """How much device memory can actually be reserved (6.9 #1).

    This is the number that matters, not the spec-sheet figure: unified memory is
    shared with the OS, and a desktop card may already be driving displays.

    On a backend whose OOM is catchable, it is measured -- double until failure,
    then bisect. On one whose OOM is not, it is *asked for*, because the
    alternative is a probe that kills the worker it is describing.
    """
    device = device or backend.device()

    if not backend.allocation_is_catchable:
        return {
            "alloc_max_mb": int(_available_memory_mb()),
            "method": "reported",
            "note": "queried from the OS; this backend's OOM is not catchable, so "
                    "climbing until failure would kill the process rather than "
                    "return a number",
        }

    def fits(mb: int) -> bool:
        block = None
        try:
            block = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=device)
            backend.synchronize(device)
            return True
        except Exception as exc:  # noqa: BLE001
            if not _is_oom(exc):
                raise
            return False
        finally:
            del block
            backend.empty_cache()

    low = 0
    high = ALLOC_PROBE_START_MB
    try:
        while high <= ALLOC_PROBE_MAX_MB and fits(high):
            low = high
            high *= 2
        # A single block is a stricter test than the total free memory, since it
        # also has to be contiguous -- which is the allocation a model weight
        # actually needs.
        while high - low > 128:
            mid = (low + high) // 2
            if fits(mid):
                low = mid
            else:
                high = mid
    except Exception as exc:  # noqa: BLE001 - never let the probe fail registration
        return {"alloc_max_mb": low or None, "method": "measured",
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        backend.empty_cache()

    return {"alloc_max_mb": low, "method": "measured"}


def precision_support(backend: Backend, device: torch.device | None = None) -> dict[str, Any]:
    """What genuinely worked, not what the hardware theoretically supports (6.9 #2).

    Every entry is an executed operation. bf16 on an older card, fp16 on CPU, nf4
    without a CUDA bitsandbytes -- each of these is a case where the library
    imports fine and the arithmetic then fails or silently falls back, and a
    worker that claimed support it does not have would abandon its lease mid-round
    after downloading a base model.
    """
    device = device or backend.device()
    supports: list[str] = []
    detail: dict[str, str] = {}

    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            a = torch.ones((32, 32), dtype=dtype, device=device)
            result = (a @ a).float().sum().item()
            backend.synchronize(device)
            if result != 32 * 32 * 32:
                detail[name] = f"matmul produced {result}, expected {32 ** 3}"
                continue
            supports.append(name)
        except Exception as exc:  # noqa: BLE001
            detail[name] = f"{type(exc).__name__}: {exc}"

    try:
        import bitsandbytes  # noqa: F401

        from transformers import BitsAndBytesConfig  # noqa: F401

        if backend.name in ("cuda", "rocm"):
            supports.append("nf4")
        else:
            detail["nf4"] = f"bitsandbytes present but not supported on {backend.name}"
    except ImportError as exc:
        # Expected on worker-core, which has no transformers layer, and on every
        # non-CUDA machine. Not an error: a run that needs nf4 simply will not
        # match this worker (6.9), which is the designed outcome.
        detail["nf4"] = f"unavailable: {exc}"

    return {"supports": supports, "detail": detail}


class _BenchBlock(torch.nn.Module):
    """A fixed transformer block, hand-written so worker-core needs no `transformers`.

    Deliberately not imported from anywhere: the benchmark's whole value is that
    it is *the same shape everywhere, forever*. Depending on a library whose
    internals change across versions would make scores from different weeks
    incomparable while looking identical.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), torch.nn.GELU(), torch.nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


def bench_score(backend: Backend, device: torch.device | None = None) -> dict[str, Any]:
    """A normalized throughput number from a fixed forward+backward (6.9 #3).

    This does real work for 3.4: it replaces "conservative default for an unseen
    GPU class" with a measurement, so a brand-new device gets a sensible step
    budget on its *first* round rather than after one wasted one.

    The score is iterations per second at the fixed shape. It is only meaningful
    against other scores of the same ``bench_version`` -- which is why the
    version travels with it.
    """
    device = device or backend.device()
    try:
        torch.manual_seed(0)
        block = _BenchBlock(BENCH_DIM, BENCH_HEADS).to(device)
        x = torch.randn(BENCH_BATCH, BENCH_SEQ, BENCH_DIM, device=device)

        for _ in range(BENCH_WARMUP):
            block(x).sum().backward()
        backend.synchronize(device)

        started = time.monotonic()
        for _ in range(BENCH_ITERS):
            block.zero_grad(set_to_none=True)
            block(x).sum().backward()
        backend.synchronize(device)
        elapsed = time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 - a machine that cannot run it is not eligible
        return {"bench_score": None, "bench_version": BENCH_VERSION,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        backend.empty_cache()

    return {
        "bench_score": round(BENCH_ITERS / elapsed, 3) if elapsed > 0 else None,
        "bench_version": BENCH_VERSION,
        "bench_seconds": round(elapsed, 4),
    }


# --------------------------------------------------------------------------
# The whole profile
# --------------------------------------------------------------------------


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("ganymede")
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_share(value: object, divisor: int) -> int | None:
    """One device's share of a pooled figure, or the figure itself when the
    pool is not shared (``divisor == 1``). ``None`` in, ``None`` out: a
    measurement that was skipped or failed stays absent rather than becoming
    a fabricated zero, which ``is_eligible`` would read as a real capacity
    claim of nothing.
    """
    if value is None:
        return None
    try:
        return int(int(value) // max(1, divisor))
    except (TypeError, ValueError):
        return None


def _probe_one_device(backend: Backend, device: torch.device, *,
                      skip_bench: bool, skip_alloc: bool) -> dict[str, Any]:
    """describe + precision_support + allocation_ceiling_mb + bench_score for
    one ``device``, in the exact shape ``run_probe`` has always returned for
    its single (flat-field) device. Factored out so that device 0 of a multi-
    GPU box goes through *this same code*, unmodified, and the flat profile
    fields it feeds stay byte-identical to a pre-multi-GPU probe (docs/14 §2).

    Every one of the four measurements runs here, for every device --
    including precision and bench, which is 4x the work on a 4-card box. That
    is a deliberate choice, not an oversight: this module's whole premise
    (module docstring) is that the worker *measures* and the coordinator
    *believes it*, and copying device 0's numbers onto devices 1..N would
    quietly violate that for exactly the boxes where it matters most --
    several cards sharing one power and cooling budget throttle
    independently, so an untested assumption that "the same card model means
    the same number" is precisely the kind of claim this module exists not to
    make. ``skip_bench`` / ``skip_alloc`` (now reaching every device, not just
    one) are the operator's existing lever for that cost, not a new one.
    """
    try:
        described = backend.describe(device)
    except Exception as exc:  # noqa: BLE001
        described = {"device_name": f"{backend.name}:unknown", "vram_mb": 0,
                     "describe_error": f"{type(exc).__name__}: {exc}"}

    precision = precision_support(backend, device)
    alloc = {"alloc_max_mb": None, "method": "skipped"} if skip_alloc \
        else allocation_ceiling_mb(backend, device)
    bench = {"bench_score": None, "bench_version": BENCH_VERSION} if skip_bench \
        else bench_score(backend, device)

    return {
        "device_name": described.get("device_name", "unknown"),
        "vram_mb": int(described.get("vram_mb") or 0),
        "compute_capability": described.get("compute_capability"),
        "driver": described.get("driver"),
        "supports": precision["supports"],
        "probe": {
            **alloc,
            **bench,
            "precision_detail": precision["detail"],
            "backend_maturity": backend.maturity,
            "platform": f"{platform.system()} {platform.machine()}",
            **({"describe_error": described["describe_error"]} if "describe_error" in described else {}),
        },
    }


def run_probe(
    prefer_backend: str | None = None,
    *,
    skip_bench: bool = False,
    skip_alloc: bool = False,
) -> dict[str, Any]:
    """The full self-test, in the shape the coordinator's ComputeProfile expects.

    Target is under a minute per device. Nothing here raises: a machine whose
    every measurement failed still produces a profile, registers, and is
    simply never eligible -- which is the designed outcome, and a far better
    support experience than silence.

    Measures every device ``backend.list_devices()`` reports, **strictly
    sequentially** (docs/14 §2) -- a plain loop, never a thread pool or async
    gather. ``allocation_ceiling_mb`` finds the real ceiling by doubling a
    live allocation until it fails; run two of those searches concurrently on
    one box and each only sees whatever fraction of memory the other left
    behind, understating the true ceiling. That understated number is exactly
    what ``budget.is_eligible`` and ``constraints._vram_mb`` prefer over the
    spec-sheet ``vram_mb``, so a concurrent probe would silently make a
    machine ineligible for work it could actually do -- the opposite of this
    module's "registration always succeeds" doctrine.
    """
    backend = detect_backend(prefer_backend or os.environ.get("GANYMEDE_BACKEND"))

    try:
        targets = backend.list_devices()
    except Exception:  # noqa: BLE001 - an enumeration that errors registers as one device
        targets = []
    if not targets:
        targets = [backend.device()]

    devices = [
        _probe_one_device(backend, device, skip_bench=skip_bench, skip_alloc=skip_alloc)
        for device in targets
    ]

    # The flat fields, populated from device 0 (docs/14 §2). They feed the
    # uuid5 fingerprint that derives worker_id in app.register; changing them
    # for a machine that already exists in the fleet would re-register it as
    # a new worker and orphan its reputation, enrollment and accrual history.
    # ``_probe_one_device`` is the same function (unmodified) a single-GPU
    # probe has always called, so this block is byte-identical to the profile
    # this function produced before multi-GPU reporting existed.
    primary = devices[0]
    # One pool split N ways, or N independent pools. See the ``devices`` block.
    pool_divisor = len(devices) if backend.shares_memory_pool else 1
    return {
        "backend": backend.name,
        "device_name": primary["device_name"],
        "vram_mb": primary["vram_mb"],
        "compute_capability": primary["compute_capability"],
        "driver": primary["driver"],
        "torch_ver": torch.__version__,
        "package_version": _package_version(),
        "supports": primary["supports"],
        "probe": primary["probe"],
        # Per-device inventory (docs/14 §2). Key names are load-bearing --
        # cross-checked against ``coordinator.devices.reconcile_inventory``,
        # the consumer, both in its fallback synthesis and in what it writes
        # to ``worker_devices``.
        #
        # ``vram_mb`` here is a *share*, not a copy of the flat figure, on a
        # backend whose devices draw on one pool (``shares_memory_pool``).
        # The flat field above describes the machine and must not change --
        # it is a uuid5 fingerprint input -- but these are summed by
        # ``constraints.total_vram_gb``, so four CPU slots each reporting the
        # machine's whole RAM would advertise four times the memory that
        # exists and a ``total_vram_gb >= 100`` predicate would match a 32 GB
        # box. Dividing is also the honest per-slot budget: a slot really can
        # only use its share without starving its siblings. Discrete-memory
        # backends (CUDA, ROCm, XPU) are untouched -- each card's VRAM is
        # genuinely its own, and summing those is exactly right.
        "devices": [
            {
                "index": i,
                "name": d["device_name"],
                "vram_mb": _device_share(d["vram_mb"], pool_divisor),
                "compute_capability": d["compute_capability"],
                "supports": d["supports"],
                "alloc_max_mb": _device_share(
                    d["probe"].get("alloc_max_mb"), pool_divisor),
                "bench_score": d["probe"].get("bench_score"),
            }
            for i, d in enumerate(devices)
        ],
    }
