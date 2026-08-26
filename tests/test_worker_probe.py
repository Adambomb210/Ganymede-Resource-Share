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
                       describe=broken_describe, allocation_is_catchable=False),),
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
