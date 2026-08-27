"""``inputs_for`` for ``collab_lora_finetune`` (docs/10-jobtype-sdk.md §1).

Names what a worker fetches for one task: the round's base adapter (a presigned
GET) plus the small inline params it trains from.

Phase A note: the live claim payload is still assembled by
``coordinator/app.py:_task_payload`` (kept byte-for-byte). This module is the
declared seam that assembly moves behind once the ``jobs`` table exists; the
LoRA-specific ``num_buckets`` / ``dataset_ref`` are run-level and are threaded
in then.
"""

from __future__ import annotations

import json

from ganymede.coordinator.store import base_adapter_key
from ganymede.jobtypes.base import InputRefs
from ganymede.jobtypes.collab_lora_finetune import plan


def inputs_for(task, store) -> InputRefs:
    """``task``: a ``tasks`` row; ``store``: the object store."""
    base_ref = base_adapter_key(task["run_id"], task["round_idx"])
    url, _ = store.presign_get(base_ref)
    buckets = json.loads(task["buckets_json"]) if task["buckets_json"] else []
    keys = task.keys() if hasattr(task, "keys") else ()
    return InputRefs(
        artifacts={"base_adapter": url},
        params={
            "buckets": buckets,
            "num_buckets": task["num_buckets"] if "num_buckets" in keys else None,
            "dataset_ref": task["dataset_ref"] if "dataset_ref" in keys else None,
            "seed": plan.task_seed(task["run_id"], task["round_idx"], task["id"]),
        },
    )
