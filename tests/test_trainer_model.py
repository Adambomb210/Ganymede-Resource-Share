"""LoRA attachment, adapter naming, and the strictness that guards gate 2."""

from __future__ import annotations

import pytest
import torch

from ganymede.coordinator.aggregate import manifest_of
from ganymede.trainer import model as M
from scripts.newrun import build_seed_adapter

CPU = torch.device("cpu")


def test_trainer_and_newrun_agree_on_every_key_and_shape(tiny_model_dir, tiny_lora_cfg):
    """The single most load-bearing agreement in the system.

    ``newrun.py`` mints round 0's adapter from ``config.json`` on a meta device,
    never loading weights. The trainer produces its adapter from a real, loaded
    model. Acceptance gate 2 compares a submission's keys and shapes against the
    round's base adapter -- so if these two naming paths ever diverge, *every*
    submission on *every* round fails, and the failure surfaces as a rejection
    slug rather than as anything pointing at naming.

    This is also why ``lora_state`` reads ``named_parameters()`` directly instead
    of going through peft's ``get_peft_model_state_dict``, which rewrites key
    names and has changed how it does so across releases.
    """
    seed = build_seed_adapter(tiny_model_dir, tiny_lora_cfg)

    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    live = M.lora_state(M.attach_lora(base, tiny_lora_cfg))

    assert set(live) == set(seed)
    assert manifest_of(live) == manifest_of(seed)


def test_init_from_actually_loads_the_downloaded_weights(tiny_model_dir, tiny_lora_cfg):
    seed = build_seed_adapter(tiny_model_dir, tiny_lora_cfg)
    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    live = M.lora_state(M.attach_lora(base, tiny_lora_cfg, init_from=seed))
    assert all(torch.equal(live[k], seed[k]) for k in seed)


def test_a_freshly_seeded_adapter_is_a_no_op_on_the_base_model(tiny_model_dir, tiny_lora_cfg):
    """lora_B == 0 means B @ A == 0, so round 0 starts from the unmodified base.

    If it did not, round 0 would begin from noise injected into every targeted
    projection, and the first round's loss would be worse than the base model's
    for reasons that look exactly like a broken aggregation.
    """
    seed = build_seed_adapter(tiny_model_dir, tiny_lora_cfg)
    assert all(t.abs().max() == 0 for k, t in seed.items() if "lora_B" in k)
    assert all(t.abs().max() > 0 for k, t in seed.items() if "lora_A" in k)


def test_adapter_stays_fp32_over_a_bf16_base(tiny_model_dir, tiny_lora_cfg):
    """`base_precision` describes the frozen base only.

    The adapter is the one tensor set that accumulates across the whole run --
    each round's output is the next round's input -- so keeping it in bf16 would
    round-trip the result through three decimal digits twenty-odd times.
    """
    base = M.load_base(tiny_model_dir, "bf16", device=CPU)
    assert base.model.embed_tokens.weight.dtype is torch.bfloat16

    state = M.lora_state(M.attach_lora(base, tiny_lora_cfg))
    assert all(t.dtype is torch.float32 for t in state.values())


def test_lora_state_is_cpu_and_contiguous(tiny_model_dir, tiny_lora_cfg):
    # safetensors refuses non-contiguous tensors, and it refuses them at the
    # point of upload -- after the round's compute has already been spent.
    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    state = M.lora_state(M.attach_lora(base, tiny_lora_cfg))
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in state.values())


def test_only_lora_parameters_are_trainable(tiny_model_dir, tiny_lora_cfg):
    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    peft_model = M.attach_lora(base, tiny_lora_cfg)
    trainable = {n for n, p in peft_model.named_parameters() if p.requires_grad}
    assert trainable
    assert all("lora_" in n for n in trainable)
    assert len(M.lora_params(peft_model)) == len(trainable)


def test_loading_an_adapter_for_a_different_lora_cfg_is_refused(tiny_model_dir, tiny_lora_cfg):
    """Better to fail at attach than to train against a half-random adapter and
    discover the mismatch at gate 2, after the compute is spent."""
    other = {**tiny_lora_cfg, "target_modules": ["q_proj"]}
    narrow = build_seed_adapter(tiny_model_dir, other)

    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    with pytest.raises(ValueError, match="missing"):
        M.attach_lora(base, tiny_lora_cfg, init_from=narrow)


def test_loading_an_adapter_of_the_wrong_rank_is_refused(tiny_model_dir, tiny_lora_cfg):
    wrong_rank = build_seed_adapter(tiny_model_dir, {**tiny_lora_cfg, "rank": 8})
    base = M.load_base(tiny_model_dir, "fp32", device=CPU)
    with pytest.raises(ValueError, match="shape"):
        M.attach_lora(base, tiny_lora_cfg, init_from=wrong_rank)


def test_unknown_precision_names_the_ones_that_exist(tiny_model_dir):
    with pytest.raises(ValueError, match="nf4"):
        M.load_base(tiny_model_dir, "int3", device=CPU)


def test_tokenizer_always_has_a_pad_token(tiny_model_dir):
    assert M.load_tokenizer(tiny_model_dir).pad_token_id is not None


def test_pick_device_prefers_cuda_then_mps_then_cpu():
    assert M.pick_device("cpu") == torch.device("cpu")
    chosen = M.pick_device()
    assert chosen.type in {"cuda", "mps", "cpu"}
    if not torch.cuda.is_available():
        assert chosen.type != "cuda"
