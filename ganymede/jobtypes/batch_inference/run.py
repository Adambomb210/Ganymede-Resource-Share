"""The worker body for ``batch_inference`` (docs/10-jobtype-sdk.md §4).

Load the model once, iterate the shard's rows in batches, emit exactly one
output row per input row, upload the JSONL to the presigned PUT. ``on_step`` per
batch drives the heartbeat; ``should_stop`` carries the soft/hard kill.

Signature note (docs/10 "Spine deviations"): ``StopCb`` is not fixed to a
return shape. ``collab_lora_finetune``'s is ``() -> bool``; this one is
``() -> str | None`` returning ``"soft"`` | ``"hard"`` | ``None`` so the two
kills stay distinct -- ``"soft"`` flushes the in-flight batch, uploads and
exits; ``"hard"`` aborts with nothing uploaded. A bare ``True`` is treated as
``"hard"``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class InferTask:
    """The claim payload for one shard, parsed once (docs/10 §4).

    A nominal per-type name, the way ``trainer.train.Task`` is
    ``collab_lora_finetune``'s.
    """

    task_id: str
    job_id: str | None
    model_ref: str
    shard_ref: str
    shard_rows: int
    prompt_template: str
    decode: dict[str, Any]
    output_schema: dict[str, str]
    output_key: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "InferTask":
        params = payload.get("params") or {}
        # ``input_ref`` carries the same descriptor ``plan`` packed; ``params``
        # is what ``inputs_for`` derived from it. Prefer ``params`` (it has the
        # presigned URLs) and fall back to the descriptor for the static fields.
        desc: dict[str, Any] = {}
        raw = payload.get("input_ref")
        if isinstance(raw, str) and raw:
            try:
                desc = json.loads(raw)
            except ValueError:
                desc = {}
        elif isinstance(raw, dict):
            desc = raw
        get = lambda k: params.get(k, desc.get(k))  # noqa: E731
        return cls(
            task_id=payload["task_id"],
            job_id=payload.get("job_id"),
            model_ref=desc.get("model_ref") or payload.get("artifacts", {}).get("model", ""),
            shard_ref=get("shard_ref"),
            shard_rows=int(desc.get("shard_rows") or params.get("shard_rows") or 0),
            prompt_template=get("prompt_template"),
            decode=get("decode") or {"mode": "greedy", "max_new_tokens": 256},
            output_schema=get("output_schema") or {},
            output_key=desc.get("output_key") or params.get("output_key", ""),
        )


@dataclass(frozen=True)
class InferResult:
    """What ``run`` returns and ``validate`` gates (docs/10 §4).

    ``digest`` is the sha256 of the canonicalised ``(id, output)`` pairs -- the
    coordinator's redundant-execution comparator (``compare_digest``) is derived
    from it; the type never compares across an ``attempt_group`` itself.
    """

    rows: int
    output_ref: str
    digest: str
    seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)


def canonical_digest(rows: Sequence[dict[str, Any]]) -> str:
    """sha256 over the ``(id, output)`` pairs, order-independent."""
    pairs = sorted((str(r.get("id")), r.get("output")) for r in rows)
    blob = json.dumps(pairs, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


_PY = {"str": str, "int": int, "float": float, "bool": bool}


def coerce_row(raw: dict[str, Any], schema: dict[str, str]) -> dict[str, Any]:
    """One output row, keyed to the schema. Values pass through; ``validate``
    is the gate, this only drops fields the schema does not name."""
    return {k: raw.get(k) for k in schema}


def _default_upload(url: str, blob: bytes) -> None:
    req = urllib.request.Request(url, data=blob, method="PUT")
    req.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(req):  # noqa: S310 -- presigned, pre-authorised
        return


def _download_shard(url: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        raw = resp.read()
    return parse_jsonl(raw)


def parse_jsonl(raw: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _hf_id(model_ref: str) -> str:
    """``hf://org/name`` -> ``org/name``.

    ``inputs_for`` passes an ``hf://`` ref through untouched, on the grounds
    that the worker pulls it from the Hub itself -- and ``from_pretrained`` has
    never heard of the scheme. Stripped here, at the one place that loads.
    """
    return model_ref[len("hf://"):] if model_ref.startswith("hf://") else model_ref


def _chunks(seq: Sequence[Any], size: int):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _stop_signal(should_stop: Callable[[], Any] | None) -> str | None:
    if should_stop is None:
        return None
    sig = should_stop()
    if sig is True:
        return "hard"
    if sig in ("soft", "hard"):
        return sig
    return None


def run(
    task: InferTask,
    inputs,
    on_step: Callable[[int, float], None] | None = None,
    should_stop: Callable[[], Any] | None = None,
    *,
    rows: Sequence[dict[str, Any]] | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: Any | None = None,
    upload: Callable[[bytes], None] | None = None,
    batch_size: int = 8,
    cache: Any | None = None,
) -> InferResult:
    """Run the shard. ``rows`` / ``model`` / ``tokenizer`` / ``upload`` are the
    injection points -- left ``None`` they resolve the presigned URLs in
    ``inputs`` and load the model from ``task.model_ref`` (the repo's
    established ``run_task(rows=..., device=...)`` seam).

    ``cache`` is a :class:`~ganymede.trainer.modelcache.ModelCache`. Shards are
    small and numerous, so the per-task load dominates this type even harder
    than it does training -- a worker handed a run's worth of shards would
    otherwise pay the full model load for each one. Reuse is unconditionally
    safe here because nothing below mutates the model: ``.eval()`` and
    ``generate`` are read-only and idempotent."""
    import torch

    from ganymede.trainer import model as model_mod

    started = time.monotonic()
    params = getattr(inputs, "params", {}) or {}
    artifacts = getattr(inputs, "artifacts", {}) or {}
    on_step = on_step or (lambda _rows, _loss: None)

    if rows is None:
        rows = _download_shard(params["shard_ref"] if "://" in str(params.get("shard_ref", ""))
                               else artifacts.get("shard") or params["shard_ref"])
    rows = list(rows)

    device = device or model_mod.pick_device()
    model_id = _hf_id(task.model_ref)
    if tokenizer is None:
        tokenizer = (cache.tokenizer(model_id) if cache is not None
                     else model_mod.load_tokenizer(model_id))
    if model is None:
        model = (cache.base(model_id, "fp32", device) if cache is not None
                 else model_mod.load_base(model_id, "fp32", device=device))
    model.eval()

    # Generation pads on the LEFT. A decoder-only model continues from the last
    # position, so right-padding a batch makes every short prompt generate from
    # after its own padding -- the row count still comes out right and the
    # digest is still stable, which is exactly why this can ship broken and
    # look fine. It also makes the ``g[in_len:]`` slice below correct for every
    # row rather than only the longest, because left padding puts all the
    # prompts' ends at the same index.
    tokenizer.padding_side = "left"

    max_new_tokens = int(task.decode.get("max_new_tokens", 256))
    do_sample = task.decode.get("mode") != "greedy"

    out_rows: list[dict[str, Any]] = []
    aborted = False
    with torch.no_grad():
        for batch in _chunks(rows, max(1, batch_size)):
            sig = _stop_signal(should_stop)
            if sig == "hard":
                aborted = True
                break
            prompts = [_render(task.prompt_template, r) for r in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            in_len = enc["input_ids"].shape[1]
            for r, g in zip(batch, gen):
                text = tokenizer.decode(g[in_len:], skip_special_tokens=True)
                out_rows.append(_build_output(r, text, task.output_schema))
            on_step(len(out_rows), 0.0)
            if sig == "soft":
                break

    if aborted:
        raise RuntimeError("batch_inference run aborted by a hard stop")

    blob = ("\n".join(json.dumps(o, sort_keys=True, ensure_ascii=False)
                      for o in out_rows)).encode("utf-8")
    if upload is not None:
        upload(blob)
    elif params.get("output_put_url"):
        _default_upload(params["output_put_url"], blob)

    return InferResult(
        rows=len(out_rows),
        output_ref=task.output_key or params.get("output_key", ""),
        digest=canonical_digest(out_rows),
        seconds=round(time.monotonic() - started, 3),
        metrics={"rows": len(out_rows)},
    )


def _render(template: str, row: dict[str, Any]) -> str:
    """``prompt_template`` is ``"...{input}..."``; a row supplies ``input`` (and
    whatever else the template names). Missing keys are left as the literal
    placeholder rather than crashing a whole shard on one malformed row."""
    class _D(dict):
        def __missing__(self, key):  # noqa: D401
            return "{" + key + "}"

    fields = {**row}
    if "input" not in fields:
        # Common shapes: a single text column under another name.
        for alt in ("text", "prompt", "instruction", "content"):
            if alt in fields:
                fields["input"] = fields[alt]
                break
    return template.format_map(_D(fields))


def _build_output(row: dict[str, Any], text: str, schema: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in schema:
        if name == "output":
            out[name] = text
        elif name == "id":
            out[name] = row.get("id")
        else:
            out[name] = row.get(name)
    return out
