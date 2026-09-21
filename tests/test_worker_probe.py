"""The capability probe (§6.9): what it measures, and that it never refuses.

The governing rule is that **registration always succeeds**. A CPU-only laptop, a
6 GB card, a torch build whose bf16 is broken — every one of them must produce a
profile and register, then simply never match a run. Turning someone away at the
door is the wrong behavior for a platform whose goal is broad compatibility, and
the stored profile is also the answer when a contributor asks why they never get
work. So most of these tests are about failing *softly*.
"""

from __future__ import annotations

import platform
import time

import pytest
import torch

from ganymede.coordinator.app import ComputeProfile
from ganymede.device import device_name
from ganymede.worker import probe

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# Backend detection
# --------------------------------------------------------------------------


def test_rocm_is_checked_before_cuda():
    """A PyTorch ROCm build reports ``torch.cuda.is_available() == True``.

    AMD made ``torch.cuda`` drive their GPUs so that CUDA code runs unmodified,
    which means availability cannot distinguish them — only ``torch.version.hip``
    can. Reverse the order and every AMD machine in the fleet registers as
    NVIDIA, and the coordinator's throughput table keys two architectures under
    one entry, averaging them into a number describing neither.
    """
    names = [b.name for b in probe.BACKENDS]
    assert names.index("rocm") < names.index("cuda")


def test_hip_selects_rocm_over_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert probe.detect_backend().name == "rocm"


def test_no_hip_selects_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert probe.detect_backend().name == "cuda"


def test_cpu_is_the_floor_when_nothing_else_is_present():
    assert probe.detect_backend().name in {b.name for b in probe.BACKENDS}
    assert probe.BACKENDS[-1].name == "cpu"
    assert probe.BACKENDS[-1].detect() is True


def test_a_backend_that_raises_while_detecting_is_treated_as_absent(monkeypatch):
    """Some torch builds raise rather than return False from is_available.

    An exception during detection must not stop the walk — otherwise one broken
    optional backend makes an otherwise fine machine unable to register at all.
    """
    def explode():
        raise RuntimeError("driver mismatch")

    monkeypatch.setattr(torch.cuda, "is_available", explode)
    assert probe.detect_backend().name == "cpu"


def test_backend_can_be_pinned_and_an_unknown_one_names_the_options():
    assert probe.detect_backend("cpu").name == "cpu"
    with pytest.raises(ValueError, match="rocm"):
        probe.detect_backend("cuda-but-faster")


def test_amd_and_intel_are_declared_but_honestly_labelled():
    """They are wired in so that supporting them is filling in an entry rather
    than a refactor — but neither has run on real hardware, and the profile says
    so rather than implying equal confidence."""
    by_name = {b.name: b for b in probe.BACKENDS}
    assert by_name["rocm"].maturity == "untested"
    assert by_name["xpu"].maturity == "untested"
    assert by_name["cuda"].maturity == "supported"


# --------------------------------------------------------------------------
# The three measurements
# --------------------------------------------------------------------------


def test_allocation_ceiling_is_asked_for_when_oom_is_not_catchable():
    """On CPU and MPS, exhausting memory invokes the kernel's OOM killer.

    SIGKILL is not catchable, so a probe that climbed until failure would kill
    the worker it was describing instead of returning a number. This is not
    hypothetical — it is exactly how the M0 calibration harness died.
    """
    cpu = probe.detect_backend("cpu")
    assert cpu.allocation_is_catchable is False

    result = probe.allocation_ceiling_mb(cpu, CPU)
    assert result["method"] == "reported"
    assert result["alloc_max_mb"] > 0


def test_catchable_backends_measure_instead():
    by_name = {b.name: b for b in probe.BACKENDS}
    assert by_name["cuda"].allocation_is_catchable
    assert by_name["rocm"].allocation_is_catchable
    assert by_name["xpu"].allocation_is_catchable
    assert by_name["mps"].allocation_is_catchable is False  # unified memory


def test_precision_support_records_what_actually_ran():
    result = probe.precision_support(probe.detect_backend("cpu"), CPU)
    assert "fp32" in result["supports"]
    # nf4 needs a CUDA bitsandbytes; its absence is a reason, not a crash.
    assert "nf4" not in result["supports"]
    assert "nf4" in result["detail"]


def test_a_broken_dtype_is_reported_rather_than_raised(monkeypatch):
    """A torch build whose bf16 matmul errors must still register.

    Claiming support the machine does not have is the expensive failure: the
    worker would win a lease, download a base model, and only then discover it
    cannot honor the run's precision.
    """
    real_ones = torch.ones

    def selective(*args, **kwargs):
        if kwargs.get("dtype") is torch.bfloat16:
            raise RuntimeError("bf16 not supported on this device")
        return real_ones(*args, **kwargs)

    monkeypatch.setattr(torch, "ones", selective)
    result = probe.precision_support(probe.detect_backend("cpu"), CPU)

    assert "bf16" not in result["supports"]
    assert "bf16 not supported" in result["detail"]["bf16"]
    assert "fp32" in result["supports"]


def test_bench_score_is_a_number_and_carries_its_version():
    """The score is stored and compared against a run's ``min_bench_score``, so a
    silent change to what it means would re-rank the whole fleet."""
    result = probe.bench_score(probe.detect_backend("cpu"), CPU)
    assert result["bench_score"] > 0
    assert result["bench_version"] == probe.BENCH_VERSION


def test_bench_failure_is_a_null_score_not_an_exception(monkeypatch):
    monkeypatch.setattr(probe, "_BenchBlock", None)
    result = probe.bench_score(probe.detect_backend("cpu"), CPU)
    assert result["bench_score"] is None
    assert "error" in result


# --------------------------------------------------------------------------
# The whole profile
# --------------------------------------------------------------------------


def test_the_profile_validates_against_the_coordinators_own_model():
    """The probe's output is a request body. Checking it against the real
    pydantic model is the only way to know the two agree without a live server."""
    profile = probe.run_probe()
    validated = ComputeProfile(**profile)

    assert validated.backend == profile["backend"]
    assert validated.device_name == profile["device_name"]
    assert validated.vram_mb > 0
    assert "fp32" in validated.supports
    assert validated.probe["bench_version"] == probe.BENCH_VERSION


def test_the_probe_names_the_device_the_way_everything_else_does():
    """This string joins the worker's registration to the coordinator's
    throughput table to the trainer's submitted metrics. It has already gone
    wrong once between two of those; one implementation is why it cannot again."""
    from ganymede.trainer import calibrate as C

    profile = probe.run_probe("cpu")
    assert profile["device_name"] == device_name(CPU)
    assert profile["device_name"] == C.describe_device(CPU)["name"]


def test_run_probe_survives_every_measurement_failing(monkeypatch):
    """Registration always succeeds (§6.9). A machine where nothing worked still
    produces a profile, registers, and is simply never eligible."""
    def broken_describe(device):
        raise RuntimeError("no driver")

    monkeypatch.setattr(
        probe, "BACKENDS",
        (probe.Backend(name="cpu", detect=lambda: True, device=lambda: CPU,
                       describe=broken_describe, list_devices=lambda: [CPU],
                       allocation_is_catchable=False),),
    )
    profile = probe.run_probe()

    assert profile["backend"] == "cpu"
    assert profile["vram_mb"] == 0
    assert "describe_error" in profile["probe"]
    ComputeProfile(**profile)  # still a valid registration body


def test_skips_exist_for_a_fast_path():
    """Registration re-runs on every start; a machine that has already been
    measured should not pay a minute for it again."""
    profile = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    assert profile["probe"]["bench_score"] is None
    assert profile["probe"]["method"] == "skipped"
    assert profile["supports"]  # precision is cheap and always measured


# --------------------------------------------------------------------------
# Multi-GPU device inventory (docs/14 §2)
# --------------------------------------------------------------------------


def test_cpu_backend_reports_exactly_one_device_by_default(monkeypatch):
    monkeypatch.delenv("GANYMEDE_CPU_SLOTS", raising=False)
    profile = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    assert len(profile["devices"]) == 1
    assert profile["devices"][0]["index"] == 0


def test_flat_fields_are_unchanged_by_multi_gpu_reporting():
    """§2's emphatic point: the flat fields feed the uuid5 fingerprint behind
    ``worker_id`` in ``app.register``, and any drift here re-registers every
    single-GPU machine in the fleet as a new worker, orphaning its
    reputation, enrollment and accrual history.

    This reconstructs, field by field, exactly what ``run_probe`` assembled
    *before* this file grew a ``devices`` list -- using the same building
    blocks (``backend.describe``, ``precision_support``) this change left
    untouched -- and checks the new output's flat half against it with
    ``devices`` stripped off. ``skip_bench`` / ``skip_alloc`` pin the two
    measurements that are otherwise non-deterministic (wall-clock timing),
    so this is a real equality check, not an approximate one.
    """
    backend = probe.detect_backend("cpu")
    device = backend.device()
    described = backend.describe(device)
    precision = probe.precision_support(backend, device)
    alloc = {"alloc_max_mb": None, "method": "skipped"}
    bench = {"bench_score": None, "bench_version": probe.BENCH_VERSION}
    expected_flat = {
        "backend": backend.name,
        "device_name": described.get("device_name", "unknown"),
        "vram_mb": int(described.get("vram_mb") or 0),
        "compute_capability": described.get("compute_capability"),
        "driver": described.get("driver"),
        "torch_ver": torch.__version__,
        "package_version": probe._package_version(),
        "supports": precision["supports"],
        "probe": {
            **alloc,
            **bench,
            "precision_detail": precision["detail"],
            "backend_maturity": backend.maturity,
            "platform": f"{platform.system()} {platform.machine()}",
        },
    }

    actual = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    actual_flat = {k: v for k, v in actual.items() if k != "devices"}
    assert actual_flat == expected_flat


def test_flat_fields_are_populated_from_device_zero():
    """§2: "the flat fields stay, populated from device 0" -- checked against
    the ``devices`` entry itself, under the key renaming ``devices``' schema
    uses (``name`` there, ``device_name`` on the flat profile)."""
    profile = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    zero = profile["devices"][0]
    assert profile["device_name"] == zero["name"]
    assert profile["vram_mb"] == zero["vram_mb"]
    assert profile["compute_capability"] == zero["compute_capability"]
    assert profile["supports"] == zero["supports"]
    assert profile["probe"]["alloc_max_mb"] == zero["alloc_max_mb"]
    assert profile["probe"]["bench_score"] == zero["bench_score"]


def test_cpu_slots_env_var_raises_the_device_count(monkeypatch):
    monkeypatch.setenv("GANYMEDE_CPU_SLOTS", "3")
    profile = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    assert [d["index"] for d in profile["devices"]] == [0, 1, 2]
    # Every slot is the same physical machine, not a divided fraction of it
    # (§2's overcommit warning) -- so the RAM figure is identical across
    # slots, never split N ways.
    assert len({d["vram_mb"] for d in profile["devices"]}) == 1


def test_malformed_cpu_slots_env_var_falls_back_to_one(monkeypatch):
    """Never let a typo in an env var turn into a registration failure --
    the same doctrine as every other measurement in this module."""
    monkeypatch.setenv("GANYMEDE_CPU_SLOTS", "not-a-number")
    assert probe._cpu_slot_count() == 1


def test_skip_alloc_reaches_every_device(monkeypatch):
    """The flag was previously unreachable from the CLI at all (docs/14 §2);
    now that it exists, it has to actually reach a probe run against every
    device, not just the one the flat fields are drawn from."""
    monkeypatch.setenv("GANYMEDE_CPU_SLOTS", "3")
    profile = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    assert profile["probe"]["method"] == "skipped"
    assert all(d["alloc_max_mb"] is None for d in profile["devices"])


def test_devices_are_probed_strictly_sequentially(monkeypatch):
    """docs/14 §2: two allocation-ceiling searches racing on one box each see
    only a fraction of the real ceiling. A slow fake ``allocation_ceiling_mb``
    proves the per-device probes never overlap in time -- a naive threaded or
    async loop would violate this, a plain ``for`` loop cannot."""
    monkeypatch.setenv("GANYMEDE_CPU_SLOTS", "3")
    intervals: list[tuple[float, float]] = []

    def slow_alloc(backend, device=None):
        start = time.monotonic()
        time.sleep(0.05)
        intervals.append((start, time.monotonic()))
        return {"alloc_max_mb": 1024, "method": "measured"}

    monkeypatch.setattr(probe, "allocation_ceiling_mb", slow_alloc)
    probe.run_probe("cpu", skip_bench=True, skip_alloc=False)

    assert len(intervals) == 3
    intervals.sort()
    for (_, end), (next_start, _) in zip(intervals, intervals[1:]):
        assert next_start >= end, "two device probes overlapped in time"


def test_devices_round_trip_through_register_into_worker_devices(
    client, conn, make_contributor, monkeypatch
):
    """The exact shape ``run_probe`` now emits has to survive
    ``POST /v1/workers/register`` into ``worker_devices`` unchanged --
    ``devices.reconcile_inventory`` is the consumer this key naming
    (``index``, ``name``, ``vram_mb``, ``compute_capability``, ``supports``,
    ``alloc_max_mb``, ``bench_score``) is cross-checked against."""
    monkeypatch.setenv("GANYMEDE_CPU_SLOTS", "3")
    profile = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)
    assert len(profile["devices"]) == 3

    _, key = make_contributor()
    resp = client.post(
        "/v1/workers/register", headers={"Authorization": f"Bearer {key}"},
        json={"compute_profile": profile},
    )
    assert resp.status_code == 200, resp.text
    worker_id = resp.json()["worker_id"]

    rows = conn.execute(
        "SELECT * FROM worker_devices WHERE worker_id = ? ORDER BY device_index",
        (worker_id,),
    ).fetchall()
    assert [r["device_index"] for r in rows] == [0, 1, 2]
    for row, d in zip(rows, profile["devices"]):
        assert row["device_name"] == d["name"]
        assert row["vram_mb"] == d["vram_mb"]
        assert row["compute_capability"] == d["compute_capability"]
        assert row["alloc_max_mb"] == d["alloc_max_mb"]
        assert row["bench_score"] == d["bench_score"]


def test_cpu_slots_split_the_shared_memory_pool_rather_than_each_claiming_it_all(
    monkeypatch,
):
    """``constraints.total_vram_gb`` sums the per-device figures, so slots that
    each reported the machine's whole RAM would advertise N times the memory
    that exists -- and a submitter predicate like ``total_vram_gb >= 100``
    would match a 32 GB box. The flat field is deliberately NOT divided: it
    describes the machine, and it is a uuid5 fingerprint input that must not
    move for a worker already in the fleet."""
    monkeypatch.setenv("GANYMEDE_CPU_SLOTS", "4")
    p = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)

    assert len(p["devices"]) == 4
    per_slot = [d["vram_mb"] for d in p["devices"]]
    assert len(set(per_slot)) == 1, "slots share one pool evenly"
    assert sum(per_slot) <= p["vram_mb"], (
        "the slots must not add up to more memory than the machine has"
    )
    assert per_slot[0] == p["vram_mb"] // 4


def test_a_single_cpu_slot_reports_the_whole_pool_undivided(monkeypatch):
    """The divisor is the device count, so the default one-slot machine is
    untouched -- device 0 still equals the flat field exactly, which is the
    backward-compatibility property the whole flat-field freeze rests on."""
    monkeypatch.delenv("GANYMEDE_CPU_SLOTS", raising=False)
    p = probe.run_probe("cpu", skip_bench=True, skip_alloc=True)

    assert len(p["devices"]) == 1
    assert p["devices"][0]["vram_mb"] == p["vram_mb"]


def test_discrete_memory_backends_are_never_divided():
    """A CUDA card's VRAM is genuinely its own; summing four of them is exactly
    right. Only ``shares_memory_pool`` backends split."""
    pooled = {b.name for b in probe.BACKENDS if b.shares_memory_pool}
    assert pooled == {"cpu", "mps"}
