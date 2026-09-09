"""A process-lifetime cache for loaded models, so a worker pays the load once.

docs/03 flagged this under "Two things worth knowing before M4b": *a worker
re-loads the base model for every task, and takes several tasks per round when
its budget is small relative to the round. At the measured M2 setup cost of
110 s on a 1.7B model, three tasks in a round is 330 s of setup against one
round of work.* On a laptop that is untidy. On a rental it is billed, and it
lands on wall-clock -- which is itself an M4b exit criterion, so an uncached
worker would have M4b measuring the safetensors reader rather than the
aggregation.

Not to be confused with ``Worker.cached_base_models``, which is a *claim-time
affinity hint* about what is on disk (docs/02 §6.2). This is the same model
already deserialized, converted and resident on the device.

Two tiers, deliberately separate entries even for the same base:

* :meth:`base` -- a bare model for a **read-only** consumer. ``batch_inference``
  only ever calls ``.eval()`` and generates.
* :meth:`peft_model` -- the fully assembled training stack (checkpointing hooks
  attached, LoRA injected). This is where the flagged 330 s actually lives:
  within a run every task shares ``base_model``, ``base_precision`` and
  ``lora_cfg`` exactly, and only the adapter weights and the seed differ.

Sharing one underlying base between the tiers would be tempting and wrong: the
training path mutates what it is given (``enable_input_require_grads`` registers
a forward hook, ``get_peft_model`` *replaces target modules in place*), so a
second ``attach_lora`` on a cached base would wrap the wrappers. Keeping the
assembled stack whole, and resetting only the weights, avoids ever needing to
un-mutate anything.

**Why reuse is safe.** :func:`~ganymede.trainer.model.load_lora_state` is
strict in both directions -- it raises unless the adapter's key set matches the
model's exactly -- and ``run_task`` always supplies a real adapter. So every
trainable parameter is overwritten on every task; there is no path where task 2
inherits a weight from task 1. A ``lora_cfg`` that disagrees with the cached
stack does not train against stale structure, it raises. The strictness that
already existed to catch a bad ``base_adapter_ref`` is what makes the cache
safe, which is why the reset lives in here rather than at the call site where
it could be forgotten.
"""

from __future__ import annotations

import gc
import json
import logging
from collections import OrderedDict
from typing import Any

from ganymede.trainer import model as model_mod

log = logging.getLogger(__name__)

# One model. docs/02 §6.2 puts base models at ~16 GB, and the failure mode of
# guessing high is an OOM on a box that was correctly sized for one model --
# which on a rental is a dead afternoon rather than a slow one. A worker that
# alternates between two runs thrashes and pays what it pays today, plus an
# eviction; it does not get slower than the uncached path by more than that.
DEFAULT_CAPACITY = 1


def _fingerprint(lora_cfg: dict[str, Any]) -> str:
    """A cache key for a LoRA config. Any difference at all must miss -- the
    fallback is a full reload, and the alternative is training against the
    wrong structure."""
    return json.dumps(lora_cfg, sort_keys=True, default=str)


def _peft_key(model_ref: str, precision: str, device,
              lora_cfg: dict[str, Any], checkpointing: bool) -> tuple:
    """Built in one place because two call sites need it and they must not
    drift: ``peft_model`` looks the entry up and ``would_hit`` predicts whether
    it will. A second copy that fell behind a new key field would not fail --
    ``would_hit`` would simply start answering ``False`` always, and the
    ``setup_cached`` metric an operator sizes ``safety_margin_sec`` from would
    quietly become a constant."""
    return ("peft", model_ref, precision, str(device),
            _fingerprint(lora_cfg), bool(checkpointing))


class ModelCache:
    """Keyed on everything that changes what gets loaded. Not thread-safe: a
    worker runs one task at a time, and the heartbeat thread never touches a
    model."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self._models: OrderedDict[tuple, Any] = OrderedDict()
        self._tokenizers: dict[str, Any] = {}
        # What has been loaded successfully at least once. The worker reports
        # this as ``cached_base_models`` on claim: a model is only "cached" once
        # its files are on disk *and* it actually loaded, which is a claim the
        # old call site could not make -- it added the ref before the load that
        # might fail.
        self.loaded: set[str] = set()
        self.hits = 0
        self.misses = 0

    # ---------------- tier 1: a read-only base ----------------

    def tokenizer(self, model_ref: str):
        """Tokenizers are megabytes, not gigabytes; they do not count against
        ``capacity`` and are never evicted."""
        if model_ref not in self._tokenizers:
            self._tokenizers[model_ref] = model_mod.load_tokenizer(model_ref)
        return self._tokenizers[model_ref]

    def base(self, model_ref: str, precision: str, device) -> Any:
        """A bare base model, for a consumer that will not mutate it.

        ``batch_inference`` sets ``.eval()`` and generates; both are idempotent
        and safe to repeat on a shared object.
        """
        return self._get_or_load(
            ("base", model_ref, precision, str(device)),
            lambda: model_mod.load_base(model_ref, precision, device=device),
            model_ref,
        )

    # ---------------- tier 2: the assembled training stack ----------------

    def peft_model(self, *, model_ref: str, precision: str, device,
                   lora_cfg: dict[str, Any], checkpointing: bool,
                   adapter: dict[str, Any]) -> Any:
        """Base + checkpointing hooks + LoRA, with ``adapter`` copied in.

        ``checkpointing`` is part of the key rather than merely an argument
        because it comes from the task's ``hp``, not the run: two tasks against
        the same base can disagree, and a cached stack whose hooks were attached
        for the other answer would silently train under the wrong setting.
        """
        key = _peft_key(model_ref, precision, device, lora_cfg, checkpointing)

        cached = self._models.get(key)
        if cached is not None:
            self._models.move_to_end(key)
            self.hits += 1
            # The whole safety argument, in one call: strict in both directions,
            # so every trainable parameter is overwritten or this raises.
            model_mod.load_lora_state(cached, adapter)
            _zero_grads(cached)
            return cached

        def build():
            base = model_mod.load_base(model_ref, precision, device=device)
            if checkpointing:
                # Same order as run_task's: peft's inputs come from a frozen
                # embedding, so without enable_input_require_grads the
                # checkpointed segment has no input requiring grad.
                base.enable_input_require_grads()
                base.gradient_checkpointing_enable()
            return model_mod.attach_lora(base, lora_cfg, init_from=adapter)

        return self._get_or_load(key, build, model_ref)

    def would_hit(self, *, model_ref: str, precision: str, device,
                  lora_cfg: dict[str, Any], checkpointing: bool) -> bool:
        """Whether :meth:`peft_model` is about to reuse rather than load.

        Asked *before* the call, because the trainer reports it as
        ``metrics["setup_cached"]`` and by then the answer has been overwritten.
        """
        return _peft_key(model_ref, precision, device, lora_cfg,
                         checkpointing) in self._models

    # ---------------- plumbing ----------------

    def _get_or_load(self, key: tuple, build, model_ref: str) -> Any:
        cached = self._models.get(key)
        if cached is not None:
            self._models.move_to_end(key)
            self.hits += 1
            return cached

        self.misses += 1
        # Evict *before* building, never after. Holding the outgoing model while
        # the incoming one allocates needs room for two, and a box sized for one
        # 16 GB model OOMs there -- turning a cache that was meant to save time
        # into the reason the round died.
        self._evict_to(self.capacity - 1)
        obj = build()
        self._models[key] = obj
        self.loaded.add(model_ref)
        return obj

    def _evict_to(self, size: int) -> None:
        while len(self._models) > max(0, size):
            key, _ = self._models.popitem(last=False)
            log.info("model cache: evicting %s", key[:2])
        _release()

    def clear(self) -> None:
        self._models.clear()
        self._tokenizers.clear()
        _release()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "resident": len(self._models)}


def _zero_grads(peft_model) -> None:
    """Belt to ``train_loop``'s braces.

    ``train_loop`` already opens each step with ``opt.zero_grad(set_to_none=True)``
    over these same tensors, so a task that stopped early cannot in fact leak a
    gradient into the next one's first backward. That is a property of a loop
    two modules away, though, and the cost of not depending on it is this.
    """
    for param in peft_model.parameters():
        param.grad = None


def _release() -> None:
    """Drop what the evicted entry was holding.

    ``empty_cache`` does nothing while any Python reference survives, so the
    ``gc`` pass is the part that matters: the model is a cycle-rich object graph
    and refcounting alone does not always get it.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- releasing memory must never fail a round
        pass
