"""``inputs_for`` for ``batch_inference`` (docs/10-jobtype-sdk.md §4).

``artifacts`` names the model (a presigned GET); ``params`` carries the shard
ref, a presigned PUT for the output, and the small inline decode settings. **No
base adapter key** -- that absence is the seam this type exists to prove.

Everything comes off the task row's ``input_ref_json`` descriptor that
``plan`` packed, because the frozen ``inputs_for(task, store)`` signature has no
``conn`` to read ``jobs.spec_json``.
"""

from __future__ import annotations

import json

from ganymede.jobtypes.base import InputRefs


def _model_get_url(store, model_ref: str) -> str:
    """A presigned GET for the model. An ``hf://`` (or bare HF repo) ref is not
    an object-store key -- the worker pulls those from the Hub directly -- so it
    is passed through unchanged."""
    if "://" in model_ref and not model_ref.startswith(("s3://",)):
        return model_ref
    if model_ref.startswith("hf://") or model_ref.count("/") == 1:
        return model_ref
    url, _ = store.presign_get(model_ref)
    return url


def inputs_for(task, store) -> InputRefs:
    """``task``: a ``tasks`` row (needs ``input_ref_json``); ``store``: the object store."""
    desc = json.loads(task["input_ref_json"])
    put_url, _ = store.presign_put(desc["output_key"])
    return InputRefs(
        artifacts={"model": _model_get_url(store, desc["model_ref"])},
        params={
            "shard_ref": desc["shard_ref"],
            "shard_rows": desc["shard_rows"],
            "output_put_url": put_url,
            "output_key": desc["output_key"],
            "decode": desc["decode"],
            "prompt_template": desc["prompt_template"],
            "output_schema": desc["output_schema"],
        },
    )
