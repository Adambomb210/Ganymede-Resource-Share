"""Held-out loss and the generation smoke set (docs/03-roadmap.md, *Eval metric*).

Two instruments, deliberately, because they catch different failures and neither
substitutes for the other.

**Held-out loss** answers "did distributed training preserve the training
signal?" It is cheap, deterministic, and sensitive to small degradation --
which is the point, since a distributed run that is quietly slightly worse than
single-node looks identical to a healthy one on a *training* curve.

**The greedy smoke set** catches what loss won't. Generation quality can degrade
while held-out loss looks fine; prompt-format corruption is the specific case
already flagged, and it barely moves loss at all. Twenty fixed prompts decoded at
temperature 0 and diffed against the previous round is not a metric -- it is a
tripwire, and it is the only thing in the system watching that particular wire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ganymede.trainer import data as data_mod
from ganymede.trainer import model as model_mod


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    tokens: int
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": round(self.loss, 5),
            "perplexity": round(self.perplexity, 4),
            "tokens": self.tokens,
            "examples": self.examples,
        }


@torch.no_grad()
def held_out_loss(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    eval_indices: Sequence[int],
    *,
    seq_len: int = 1024,
    micro_batch: int = 4,
    prompt_format: str = "dolly-v1",
    completion_only: bool = True,
    device: torch.device | None = None,
    limit: int | None = None,
) -> EvalResult:
    """Token-weighted mean cross-entropy over the held-out split.

    **Token-weighted, not batch-weighted**, and the difference is not cosmetic.
    Averaging per-batch losses gives every batch equal say regardless of how many
    supervised tokens it contained, so a run whose eval batches happen to sort
    differently reports a different number for an identical model. Dolly's
    responses vary from a few tokens to several hundred, so the effect is large
    enough to swamp the differences M4 is trying to measure.
    """
    device = device or model_mod.pick_device()
    formatter = data_mod.get_format(prompt_format)
    was_training = model.training
    model.eval()

    indices = list(eval_indices)[:limit] if limit else list(eval_indices)

    total_nll = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(indices), micro_batch):
            chunk = indices[start : start + micro_batch]
            encoded = []
            for i in chunk:
                prompt, completion = formatter(rows[i])
                encoded.append(
                    data_mod.encode_example(
                        tokenizer, prompt, completion,
                        seq_len=seq_len, completion_only=completion_only,
                    )
                )
            batch = data_mod.collate(encoded, tokenizer.pad_token_id)
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Recompute the shift ourselves rather than passing `labels=` and
            # reading `out.loss`: that value is a mean over the batch's own
            # supervised tokens, and re-weighting a batch of means back into a
            # token-weighted total needs the token count anyway.
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=data_mod.IGNORE_INDEX,
                reduction="sum",
            )
            n_tokens = int((shift_labels != data_mod.IGNORE_INDEX).sum().item())
            total_nll += float(nll.item())
            total_tokens += n_tokens
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("no supervised tokens in the eval split")

    loss = total_nll / total_tokens
    return EvalResult(
        loss=loss,
        perplexity=math.exp(min(loss, 20.0)),  # clamp: exp(700) is inf, and a
        tokens=total_tokens,                   # loss that high is meaningless anyway
        examples=len(indices),
    )


# --------------------------------------------------------------------------
# The generation smoke set
# --------------------------------------------------------------------------

# Twenty fixed prompts, never resampled. They are deliberately mundane and
# deliberately varied in shape -- open question, closed question, instruction
# with context, list request, refusal-adjacent, and a couple of formatting
# probes. What is being watched is not whether the answers are *good*; it is
# whether they are still the same *kind of thing* they were last round. Mode
# collapse, prompt-format corruption and EOS breakage all announce themselves
# here long before they move a loss curve.
SMOKE_PROMPTS: tuple[dict[str, str], ...] = (
    {"instruction": "What is the capital of France?", "context": ""},
    {"instruction": "Explain photosynthesis in two sentences.", "context": ""},
    {"instruction": "List three uses for baking soda.", "context": ""},
    {"instruction": "Who wrote the novel Moby-Dick?", "context": ""},
    {"instruction": "Summarize the passage.", "context": "The Kuiper belt is a circumstellar disc in the outer Solar System, extending from the orbit of Neptune to approximately 50 AU from the Sun."},
    {"instruction": "Translate to French: The weather is cold today.", "context": ""},
    {"instruction": "What is 17 multiplied by 4?", "context": ""},
    {"instruction": "Write a one-line description of a lighthouse.", "context": ""},
    {"instruction": "Give me a recipe for scrambled eggs.", "context": ""},
    {"instruction": "Classify the sentiment: 'The service was slow but the food was excellent.'", "context": ""},
    {"instruction": "What are the primary colors?", "context": ""},
    {"instruction": "Extract the year mentioned in the text.", "context": "The bridge was completed in 1937 after four years of construction."},
    {"instruction": "Name two programming languages used for data science.", "context": ""},
    {"instruction": "Why do leaves change color in autumn?", "context": ""},
    {"instruction": "Rewrite this sentence to be more concise.", "context": "Due to the fact that it was raining, we made the decision to stay indoors."},
    {"instruction": "What is the boiling point of water at sea level?", "context": ""},
    {"instruction": "Describe the difference between a comet and an asteroid.", "context": ""},
    {"instruction": "Suggest a name for a coffee shop near a university.", "context": ""},
    {"instruction": "How many continents are there?", "context": ""},
    {"instruction": "Write a haiku about winter.", "context": ""},
)


@torch.no_grad()
def smoke_generate(
    model,
    tokenizer,
    *,
    prompts: Sequence[dict[str, str]] = SMOKE_PROMPTS,
    prompt_format: str = "dolly-v1",
    max_new_tokens: int = 64,
    device: torch.device | None = None,
) -> list[dict[str, str]]:
    """Greedy (temperature 0) decode of the fixed prompt set.

    Greedy and nothing else. Sampling would make round-to-round output differ for
    reasons that have nothing to do with the model changing, and a tripwire that
    trips at random is a tripwire people learn to ignore.
    """
    device = device or model_mod.pick_device()
    formatter = data_mod.get_format(prompt_format)
    was_training = model.training
    model.eval()

    out: list[dict[str, str]] = []
    try:
        for row in prompts:
            prompt, _ = formatter({**row, "response": ""})
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion = tokenizer.decode(
                generated[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
            )
            out.append({"instruction": row["instruction"], "completion": completion})
    finally:
        model.train(was_training)
    return out


def diff_smoke(previous: Sequence[dict[str, str]], current: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Which smoke prompts changed answer since the last checkpoint.

    Returned rather than printed so a caller can decide whether "17 of 20
    changed" is expected (early rounds: yes) or alarming (late rounds: yes).
    """
    changed = []
    for prev, cur in zip(previous, current):
        if prev.get("completion") != cur.get("completion"):
            changed.append({
                "instruction": cur.get("instruction", ""),
                "before": prev.get("completion", ""),
                "after": cur.get("completion", ""),
            })
    return changed
