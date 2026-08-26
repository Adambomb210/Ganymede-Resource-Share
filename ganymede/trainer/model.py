"""Base-model loading, LoRA attachment, and adapter serialization.

The one hard requirement running through this module: **the key names the
trainer emits must be byte-identical to the ones `scripts/newrun.py` minted for
round 0.** Acceptance gate 2 (5.1) checks a submission's key set and shapes
against the round's ``base_adapter_ref``, and that reference is the seed adapter
for round 0 and an aggregate of accepted submissions thereafter. A trainer whose
naming differs by so much as a ``.default`` fails every gate on every round.

That is why :func:`lora_state` reads ``named_parameters()`` directly and does not
go through ``peft``'s ``save_pretrained`` / ``get_peft_model_state_dict``. Those
rewrite key names for the hub's benefit -- stripping the adapter name, re-rooting
the prefix -- and the rewrite has changed across `peft` releases. Raw parameter
names are what ``newrun.py`` enumerates from the meta model, so raw parameter
names are what we write.
"""

from __future__ import annotations

import platform
from typing import Any

import torch

# `base_precision` describes the frozen base model only. The adapter is always
# fp32 -- see build_seed_adapter in scripts/newrun.py for why.
ADAPTER_DTYPE = torch.float32

_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def pick_device(prefer: str | None = None) -> torch.device:
    """CUDA, then MPS, then CPU -- unless overridden."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """The string throughput is keyed by, everywhere it is keyed.

    This is a join key, not a label. ``closer.close_round`` folds a submission's
    ``steps_per_min`` into ``throughput[gpu_model]``; ``rounds.claim_task`` looks
    the next budget up by the worker's reported ``device_name``; and
    ``ganymede-calibrate`` writes ``throughput[<name>]`` into calibration.json.
    All three must produce the identical string for the same card or the lookups
    silently miss -- and a miss is invisible, because the coordinator simply
    falls back to the cold-start guess forever and the only symptom is step
    budgets that never improve.

    One function, so they agree by construction rather than by convention.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "mps:apple-silicon"
    return f"cpu:{platform.processor() or platform.machine()}"


def load_base(base_model: str, precision: str, device: torch.device | None = None):
    """Load the frozen base model at the run's pinned precision (8, J).

    ``base_precision`` is pinned per run rather than chosen per worker, because a
    fleet where some workers quantize and others don't is a fleet whose adapters
    were trained against measurably different forward passes.
    """
    from transformers import AutoModelForCausalLM

    device = device or pick_device()

    if precision == "nf4":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "base_precision 'nf4' needs bitsandbytes, which is CUDA-only. A "
                "worker that cannot honor the run's pinned precision must abandon "
                "its lease rather than silently train at a different one (4.2 step 5)."
            ) from exc
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, device_map={"": device.type}
        )
    else:
        try:
            dtype = _TORCH_DTYPES[precision]
        except KeyError:
            raise ValueError(
                f"unknown base_precision {precision!r}; have {sorted(_TORCH_DTYPES)} + 'nf4'"
            ) from None
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        model.to(device)

    model.config.use_cache = False  # incompatible with gradient checkpointing, useless in training
    return model


def load_tokenizer(base_model: str):
    """Tokenizer with a guaranteed pad token.

    Base models frequently ship without one, and a missing pad token surfaces as
    a collate-time crash rather than anything informative. Falling back to EOS is
    safe here because padded positions are masked out of both attention and the
    loss (see data.collate).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def attach_lora(model, lora_cfg: dict[str, Any], init_from: dict[str, torch.Tensor] | None = None):
    """Wrap the base model in the run's LoRA config, optionally loading weights.

    LoRA parameters are forced to fp32 even when the base is bf16 or nf4. `peft`
    casts the layer input to the adapter's dtype on the way in, so a fp32 adapter
    over a bf16 base is a supported configuration and not a dtype mismatch -- and
    it keeps the optimizer's arithmetic, which is the only arithmetic that
    actually accumulates across the run, out of bf16's three decimal digits.
    """
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    peft_model = get_peft_model(model, peft_cfg)

    for _, param in _lora_named_parameters(peft_model):
        param.data = param.data.to(ADAPTER_DTYPE)

    if init_from is not None:
        load_lora_state(peft_model, init_from)

    return peft_model


def _lora_named_parameters(peft_model):
    return [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]


def lora_params(peft_model) -> list[torch.nn.Parameter]:
    return [p for _, p in _lora_named_parameters(peft_model) if p.requires_grad]


def lora_state(peft_model) -> dict[str, torch.Tensor]:
    """The adapter, as the coordinator expects to receive it: fp32, on CPU."""
    return {
        name: param.detach().to(device="cpu", dtype=ADAPTER_DTYPE).contiguous()
        for name, param in _lora_named_parameters(peft_model)
    }


def load_lora_state(peft_model, adapter: dict[str, torch.Tensor]) -> None:
    """Copy a downloaded adapter into an attached LoRA, strictly.

    Strict on both sides deliberately. A key in the adapter that the model does
    not have means the run's ``lora_cfg`` and its base adapter disagree -- the
    kind of mismatch that would otherwise train happily against a partly-random
    adapter and produce a submission that fails gate 2 after the compute is
    already spent.
    """
    params = dict(_lora_named_parameters(peft_model))

    missing = sorted(set(params) - set(adapter))
    extra = sorted(set(adapter) - set(params))
    if missing or extra:
        raise ValueError(
            f"adapter does not match this lora_cfg: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})"
        )

    with torch.no_grad():
        for name, param in params.items():
            incoming = adapter[name]
            if tuple(incoming.shape) != tuple(param.shape):
                raise ValueError(
                    f"{name}: adapter has shape {tuple(incoming.shape)}, "
                    f"model expects {tuple(param.shape)}"
                )
            param.copy_(incoming.to(device=param.device, dtype=param.dtype))
