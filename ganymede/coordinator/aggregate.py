"""Round-close aggregation: acceptance gates and the outer combine step
(docs/02-architecture-v2.md section 5).

Everything here is pure CPU tensor math over small LoRA adapters (dict[str,
Tensor], ~224 tensors / ~25 MB). Nothing in this module talks to the DB or to
object storage -- callers pass bytes/tensors in and get GateResult/tensors
back, and are responsible for persisting the outcome (submissions.accepted,
submissions.reject_reason, rounds.result_adapter_ref, runs.outer_momentum_ref).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from safetensors.torch import load as _safetensors_load
from safetensors.torch import save as _safetensors_save

# Stable reject reason slugs. These are persisted to the DB and shown to
# workers, so they must not change once shipped.
REJECT_NOT_SAFETENSORS = "not_safetensors"
REJECT_KEY_MISMATCH = "key_mismatch"
REJECT_SHAPE_MISMATCH = "shape_mismatch"
REJECT_DTYPE_MISMATCH = "dtype_mismatch"
REJECT_NON_FINITE = "non_finite"
REJECT_NORM_OUTLIER = "norm_outlier"
REJECT_NO_STEPS = "no_steps"
REJECT_STEPS_INCONSISTENT = "steps_inconsistent"

# name -> (shape, dtype_str). "expected" is always A_base's manifest -- the
# round's base_adapter_ref, per architecture 5.1 #2 -- so there is no separate
# schema to maintain and the coordinator never needs to hold a base model.
Manifest = dict[str, tuple[tuple[int, ...], str]]

_MAX_DETAIL_NAMES = 5


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str | None = None  # None iff accepted; else one of the REJECT_* slugs
    detail: str | None = None  # short human-readable elaboration


def load_adapter(raw: bytes) -> dict[str, torch.Tensor]:
    """Deserialize safetensors bytes into a tensor dict.

    Worker artifacts are untrusted input arriving over the network. We use
    ``safetensors.torch.load`` and never ``torch.load``: torch's pickle-based
    format executes arbitrary code on deserialization (Finding G), which would
    hand every worker code execution on the coordinator. safetensors' format
    is a flat header + raw tensor bytes with no executable content, so a
    malformed or hostile submission can only fail to parse, not run.

    Raises on malformed input -- callers (check_structural) catch it.
    """
    return _safetensors_load(raw)


def save_adapter(adapter: dict[str, torch.Tensor]) -> bytes:
    """Serialize a tensor dict to safetensors bytes.

    The inverse of load_adapter, and the only way the coordinator writes an
    adapter. safetensors requires contiguous tensors, and the arithmetic in
    combine() can produce non-contiguous views, so contiguity is forced here
    rather than left for a caller to remember.
    """
    return _safetensors_save({k: v.contiguous() for k, v in adapter.items()})


def manifest_of(adapter: dict[str, torch.Tensor]) -> Manifest:
    return {name: (tuple(t.shape), str(t.dtype)) for name, t in adapter.items()}


def _truncated_names(names: set[str]) -> str:
    shown = sorted(names)[:_MAX_DETAIL_NAMES]
    suffix = "" if len(names) <= _MAX_DETAIL_NAMES else f", +{len(names) - _MAX_DETAIL_NAMES} more"
    return ", ".join(shown) + suffix


def check_structural(
    raw: bytes,
    expected: Manifest,
    steps_completed: int,
    heartbeat_steps: int | None = None,
) -> tuple[GateResult, dict[str, torch.Tensor] | None]:
    """Gates 1, 2, 3, 5 -- everything that can be checked on one submission in
    isolation, without the rest of the round's cohort. Runs in the order the
    spec fixes (structural before numeric before bookkeeping) and returns the
    first failure, since later checks are meaningless once an earlier one has
    failed (e.g. comparing NaN-free-ness of a tensor whose shape is wrong).
    """
    # Gate 1: parses as safetensors at all. A pickle file (torch.save output)
    # or any other garbage must land here rather than raising past this
    # function -- see load_adapter's comment on why we never try torch.load.
    try:
        adapter = load_adapter(raw)
    except Exception as e:
        return GateResult(False, REJECT_NOT_SAFETENSORS, str(e)), None

    # Gate 2a: exact key set.
    got_keys = set(adapter.keys())
    exp_keys = set(expected.keys())
    if got_keys != exp_keys:
        missing = exp_keys - got_keys
        extra = got_keys - exp_keys
        parts = []
        if missing:
            parts.append(f"missing: {_truncated_names(missing)}")
        if extra:
            parts.append(f"extra: {_truncated_names(extra)}")
        return GateResult(False, REJECT_KEY_MISMATCH, "; ".join(parts)), None

    # Gate 2b: shapes.
    for name, (exp_shape, _exp_dtype) in expected.items():
        got_shape = tuple(adapter[name].shape)
        if got_shape != exp_shape:
            detail = f"{name}: expected {exp_shape}, got {got_shape}"
            return GateResult(False, REJECT_SHAPE_MISMATCH, detail), None

    # Gate 3a: dtypes.
    for name, (_exp_shape, exp_dtype) in expected.items():
        got_dtype = str(adapter[name].dtype)
        if got_dtype != exp_dtype:
            detail = f"{name}: expected {exp_dtype}, got {got_dtype}"
            return GateResult(False, REJECT_DTYPE_MISMATCH, detail), None

    # Gate 3b: finite. Cast to float32 first -- torch.isfinite on bf16 is a
    # narrower, less-tested path than on fp32, and the cast is cheap at this
    # size, so we don't rely on bf16 NaN/Inf handling being exactly right.
    for name, tensor in adapter.items():
        if not torch.isfinite(tensor.float()).all():
            return GateResult(False, REJECT_NON_FINITE, f"{name} has NaN/Inf"), None

    # Gate 5a: made any progress at all.
    if steps_completed <= 0:
        return GateResult(False, REJECT_NO_STEPS, f"steps_completed={steps_completed}"), None

    # Gate 5b: steps_completed must be plausible relative to the last
    # heartbeat -- a worker can't un-train steps it already reported, and a
    # huge jump past the last heartbeat means the heartbeat is stale (or the
    # report is fabricated) rather than real progress.
    if heartbeat_steps is not None:
        if steps_completed < heartbeat_steps:
            detail = f"steps_completed={steps_completed} < heartbeat_steps={heartbeat_steps}"
            return GateResult(False, REJECT_STEPS_INCONSISTENT, detail), None
        implausible_ceiling = heartbeat_steps * 3 + 100
        if steps_completed > implausible_ceiling:
            detail = (
                f"steps_completed={steps_completed} > "
                f"3*heartbeat_steps+100={implausible_ceiling} "
                f"(heartbeat_steps={heartbeat_steps})"
            )
            return GateResult(False, REJECT_STEPS_INCONSISTENT, detail), None

    return GateResult(True), adapter


def check_norms(
    adapters: list[dict[str, torch.Tensor]],
    k: float = 5.0,
) -> list[GateResult]:
    """Gate 4 -- cohort-relative. Runs once per round close over every
    structurally-accepted adapter, not per submission, because "outlier"
    only means something relative to the other submissions this round.
    """
    n = len(adapters)

    # A median over 1 or 2 samples isn't a cohort: with one worker the
    # "median" is that worker's own norm (nothing to compare against), and
    # with two, either one could be the outlier -- there's no way to tell
    # which side is wrong. Accept everyone rather than reject arbitrarily.
    if n < 3:
        return [GateResult(True) for _ in adapters]

    names = adapters[0].keys()
    norms: dict[str, list[float]] = {}
    for name in names:
        norms[name] = [torch.linalg.norm(a[name].float()).item() for a in adapters]

    results: list[GateResult] = []
    for i in range(n):
        offending: str | None = None
        for name, vals in norms.items():
            median = sorted(vals)[n // 2] if n % 2 else (sorted(vals)[n // 2 - 1] + sorted(vals)[n // 2]) / 2
            # A tensor with zero median (every worker left it untouched) has
            # nothing to be an outlier relative to -- skip it rather than
            # dividing by zero, which would flag anything nonzero as +inf x.
            if median == 0.0:
                continue
            if vals[i] > k * median:
                offending = f"{name}: norm={vals[i]:.4g} > {k}x median={median:.4g}"
                break
        if offending is None:
            results.append(GateResult(True))
        else:
            results.append(GateResult(False, REJECT_NORM_OUTLIER, offending))
    return results


def dense_weights(
    steps: list[int],
    keys: list[str],
    cap: float | None = 2.0,
) -> list[dict[str, float]]:
    """Per-tensor weights for the dense (non-MoE) case: every tensor of a
    given worker gets the SAME scalar share of total steps. The per-tensor
    shape (rather than a flat list[float]) exists so MoE can later swap in
    per-expert routed-token weights (architecture 5.4) without changing this
    function's signature or combine()'s -- it costs nothing today and saves
    an API break later.
    """
    total = sum(steps)
    if total == 0:
        raise ValueError("sum(steps) == 0: no worker reported any steps")

    shares = [s / total for s in steps]
    n = len(shares)

    # With fewer than 3 workers the cap is meaningless: with 2 workers the
    # median share IS the midpoint between them, so capping "the share above
    # cap*median" either does nothing or clamps both to the same value.
    if cap is not None and n >= 3:
        # Clamp-then-renormalize can push a clamped value back over the cap:
        # taking weight away from the dominant worker and redistributing it
        # raises everyone else's share, including the median's, so the limit
        # itself moves. We recompute the median from the CURRENT shares each
        # pass (not the pre-cap originals) so the loop converges to a fixed
        # point where the cap holds against the final distribution rather
        # than against a stale one -- otherwise a fixed limit computed once
        # keeps re-triggering on the very shares it just produced and the
        # whole vector collapses to being exactly equal, which caps far more
        # aggressively than "no more than cap x median" requires.
        for _ in range(10):
            sorted_shares = sorted(shares)
            m = len(sorted_shares)
            median_share = (
                sorted_shares[m // 2]
                if m % 2
                else (sorted_shares[m // 2 - 1] + sorted_shares[m // 2]) / 2
            )
            limit = cap * median_share
            clamped = [min(s, limit) for s in shares]
            total_clamped = sum(clamped)
            renormalized = [s / total_clamped for s in clamped]
            if all(abs(a - b) < 1e-9 for a, b in zip(renormalized, shares)):
                shares = renormalized
                break
            shares = renormalized

    return [{key: shares[i] for key in keys} for i in range(n)]


def _zeros_like_manifest(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(t, dtype=torch.float32) for name, t in base.items()}


def combine(
    base: dict[str, torch.Tensor],
    adapters: list[dict[str, torch.Tensor]],
    weights: list[dict[str, float]],
    mode: str = "mean",
    lr_outer: float = 1.0,
    beta: float = 0.0,
    momentum: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """The outer step (architecture 5.2).

    Both "mean" and "diloco" run the identical Nesterov-outer-step formula;
    they differ only in which (lr_outer, beta) the caller passes. mode="mean"
    with lr_outer=1, beta=0 algebraically reduces to a plain weighted mean
    (outer_g cancels, A_next = A_base - outer_g = A_mean) -- we do NOT
    special-case that reduction, because M4's A/B needs the two modes to
    differ only by parameters, not by code path.
    """
    if mode not in ("mean", "diloco"):
        raise ValueError(f"unknown combine mode: {mode!r}")

    # beta == 0 makes the momentum term (outer_g + beta*m) collapse to
    # outer_g regardless of m's value, so momentum bookkeeping is pointless
    # and we return None for it, per spec, rather than a dict of zeros.
    track_momentum = beta != 0.0
    if track_momentum and momentum is None:
        momentum = _zeros_like_manifest(base)

    next_adapter: dict[str, torch.Tensor] = {}
    next_momentum: dict[str, torch.Tensor] | None = {} if track_momentum else None

    for name, base_t in base.items():
        w_sum = sum(w.get(name, 0.0) for w in weights)

        # Critical zero-guard: nobody trained this tensor this round. Pass
        # A_base through unchanged and leave its momentum untouched -- never
        # divide by zero, and never let an untouched tensor's momentum decay
        # it toward zero just because it happened to sit in the same dict as
        # tensors that DID get updates.
        if w_sum == 0.0:
            next_adapter[name] = base_t.clone()
            if track_momentum:
                next_momentum[name] = momentum[name].clone()
            continue

        # Accumulate in float32: bf16 accumulation across many workers loses
        # real precision (each add rounds), and this is cheap at adapter size.
        base_f = base_t.float()
        a_mean = torch.zeros_like(base_f)
        for adapter, w in zip(adapters, weights):
            wt = w.get(name, 0.0)
            if wt == 0.0:
                continue
            a_mean = a_mean + wt * adapter[name].float()
        a_mean = a_mean / w_sum  # normalize in case this tensor's weights don't sum to 1

        outer_g = base_f - a_mean

        if track_momentum:
            m_next = beta * momentum[name].float() + outer_g
            next_momentum[name] = m_next
        else:
            m_next = torch.zeros_like(base_f)  # unused: beta == 0 zeroes its term below

        a_next = base_f - lr_outer * (outer_g + beta * m_next)
        next_adapter[name] = a_next.to(base_t.dtype)

    return next_adapter, next_momentum


def adapter_divergence(adapters: list[dict[str, torch.Tensor]]) -> float:
    """Mean pairwise cosine distance (1 - cos_sim) between adapters.

    This is the DiLoCo diagnostic (architecture 5.2): it measures exactly
    what more local steps trades away -- more drift between workers' deltas
    means a less meaningful mean, and growing divergence across rounds is the
    signal that local_steps is set too high.
    """
    n = len(adapters)
    if n < 2:
        return 0.0

    # Sorted key order makes the flattening deterministic regardless of dict
    # insertion order (which safetensors round-trips can otherwise vary).
    keys = sorted(adapters[0].keys())
    flat = [torch.cat([a[k].float().reshape(-1) for k in keys]) for a in adapters]

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            ni = torch.linalg.norm(flat[i])
            nj = torch.linalg.norm(flat[j])
            if ni == 0.0 or nj == 0.0:
                # A zero-norm adapter has no direction to compare -- treat it
                # as contributing no divergence rather than producing NaN.
                cos_dist = 0.0
            else:
                cos_sim = torch.dot(flat[i], flat[j]) / (ni * nj)
                cos_sim = float(torch.clamp(cos_sim, -1.0, 1.0))
                cos_dist = 1.0 - cos_sim
            total += cos_dist
            pairs += 1

    return total / pairs
