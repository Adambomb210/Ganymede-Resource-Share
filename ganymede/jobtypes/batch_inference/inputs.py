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

from ganymede.coordinator import store as store_mod
from ganymede.jobtypes.base import InputRefs


def _shard_get_url(store, shard_ref: str) -> str:
    """A URL a worker can GET the shard from.

    Deliberately *not* ``_model_get_url``'s rule. That one treats a bare
    ``org/name`` as a Hub repo id, which is right for a model and wrong here:
    ``shards/0`` has exactly that shape and is an object-store key. A shard is
    data, never a repo, so only an already-addressable URL passes through and
    everything else is presigned.
    """
    if shard_ref.startswith(("http://", "https://")):
        return shard_ref
    if store_mod.is_reserved_key(shard_ref):
        # Defence in depth: validate_spec already refused this at submission.
        # Reaching here means a row predates that check or bypassed it, and
        # signing the read anyway would be the whole bug.
        raise ValueError(
            "refusing to presign a shard ref inside the coordinator's own "
            f"storage namespace: {shard_ref!r}")
    return store.presign_get(shard_ref)[0]


def _model_get_url(store, model_ref: str) -> str:
    """A presigned GET for the model. An ``hf://`` (or bare HF repo) ref is not
    an object-store key -- the worker pulls those from the Hub directly -- so it
    is passed through unchanged."""
    if "://" in model_ref and not model_ref.startswith(("s3://",)):
        return model_ref
    if model_ref.startswith("hf://") or model_ref.count("/") == 1:
        return model_ref
    if store_mod.is_reserved_key(model_ref):
        raise ValueError(
            "refusing to presign a model ref inside the coordinator's "
            f"own storage namespace: {model_ref!r}")
    url, _ = store.presign_get(model_ref)
    return url


def inputs_for(task, store) -> InputRefs:
    """``task``: a ``tasks`` row (needs ``input_ref_json``); ``store``: the object store."""
    desc = json.loads(task["input_ref_json"])
    put_url, _ = store.presign_put(desc["output_key"])

    # The shard. ``params["shard_ref"]`` stays the submitter's literal ref --
    # it is what the descriptor declared and what a log or a support question
    # will quote -- and the fetchable URL goes in ``artifacts``, which is what
    # ``InputRefs`` says artifacts are: a logical name to a presigned GET.
    #
    # This was the gap that made the type unrunnable off a real store. ``run``
    # was written against exactly this shape (``artifacts["shard"]`` when the
    # ref is not itself a URL) and ``inputs_for`` never supplied it, so a real
    # worker urlopen()'d the literal string ``"shard/0"``. Nothing caught it:
    # every test injected ``rows=`` and skipped the download.
    artifacts = {
        "model": _model_get_url(store, desc["model_ref"]),
        "shard": _shard_get_url(store, desc["shard_ref"]),
    }

    return InputRefs(
        artifacts=artifacts,
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
