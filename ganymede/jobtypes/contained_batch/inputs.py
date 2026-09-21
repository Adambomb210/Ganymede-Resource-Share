"""``inputs_for`` for ``contained_batch`` (docs/10 §7).

``artifacts`` names the shard (a presigned GET); ``params`` carries the output
PUT and the small inline values the worker stages into the container.

**No image handle here.** The archive's ref, digest and pull URL come from
``jobs.image_id`` via the coordinator's ``_image_handles`` and land on the task
payload as ``image_ref`` / ``image_digest`` / ``image_pull_url`` (docs/06). That
is the frozen path, it is the same one the claim gate reads to refuse a machine
with no runtime, and duplicating it into ``params`` would create a second answer
to "which image runs" that could disagree with the first.
"""

from __future__ import annotations

import json

from ganymede.coordinator import store as store_mod
from ganymede.jobtypes.base import InputRefs


def _shard_get_url(store, shard_ref: str) -> str:
    """A URL a worker can GET the shard from.

    Same rule as ``batch_inference``'s and for the same reason: a shard is data,
    never a repo, so an already-addressable URL passes through and everything
    else is presigned.
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


def inputs_for(task, store) -> InputRefs:
    """``task``: a ``tasks`` row (needs ``input_ref_json``); ``store``: the object store."""
    desc = json.loads(task["input_ref_json"])
    put_url, _ = store.presign_put(desc["output_key"])

    return InputRefs(
        artifacts={"shard": _shard_get_url(store, desc["shard_ref"])},
        params={
            "shard_ref": desc["shard_ref"],
            "shard_rows": desc["shard_rows"],
            "output_put_url": put_url,
            "output_key": desc["output_key"],
            "output_schema": desc["output_schema"],
            # The submitter's opaque settings, handed to the container as a
            # file. Ganymede does not read it.
            "params": desc.get("params", {}),
        },
    )
