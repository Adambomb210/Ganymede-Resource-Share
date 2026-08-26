"""The device name, in one place, because it is a join key in four.

The string this returns is not a label. It is what:

- the worker registers as ``compute_profile.device_name`` (6.9),
- ``rounds.claim_task`` looks a run's throughput estimate up by,
- the trainer reports as ``metrics.gpu_model``, which ``closer.close_round``
  folds that estimate back in under (3.2), and
- ``ganymede-calibrate`` writes ``throughput[<name>]`` into calibration.json.

All four must produce the identical string for the same machine. A mismatch has
no symptom: the lookup simply misses, the coordinator falls back to the
cold-start guess forever, and the only sign is step budgets that never improve.
That has already happened once in this codebase, between the trainer and the
calibration harness, which is why this module exists rather than a convention.

It lives at the package root rather than under `trainer/` or `worker/` because
both need it and neither should import the other: worker-core has torch and the
standard library only (4.1), and the trainer's stack is a layer on top.
"""

from __future__ import annotations

import platform

import torch


def device_name(device: torch.device) -> str:
    """A stable identifier for the compute device, keyed on what torch reports.

    ``cuda`` covers AMD too: a PyTorch ROCm build exposes AMD devices through the
    ``torch.cuda`` API, and ``get_device_name`` returns the AMD marketing name
    ("AMD Radeon RX 7900 XTX"), so the two vendors' cards land under naturally
    distinct keys without needing to be told apart here. Telling them apart
    matters for *capabilities*, which is the probe's job, not for naming.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "xpu":
        return torch.xpu.get_device_name(device)
    if device.type == "mps":
        # Apple silicon exposes no per-model identifier through torch, so the
        # machine architecture is the most specific thing available.
        return f"mps:{platform.machine()}"
    return f"cpu:{platform.processor() or platform.machine()}"
