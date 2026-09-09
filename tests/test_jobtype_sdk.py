"""The job-type SDK registry, versioning and discovery (docs/10-jobtype-sdk.md §2).

``resolve`` asserts the pinned version; ``POST /v1/jobs`` freezes
``spec_json.sdk = {job_type, version}`` and never mutates it; a spec that pins a
version newer than this build ships is refused -- at ``POST`` (422) and, for a
job that somehow reached the queue, in the claim walk as a
``job_type_version_unsupported`` verdict recorded in ``worker_eligibility``,
exactly as the ``required_image`` mismatch is.
"""

from __future__ import annotations

import json
import uuid

import pytest

from ganymede.coordinator import eligibility, rounds
from ganymede.jobtypes import REGISTRY, resolve
from ganymede.jobtypes.base import TaskSpec
from ganymede.worker.loop import DECLINE_JOBTYPE_VERSION, Worker, WorkerConfig
from tests.fake_worker import FakeWorker


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def make_submitter(conn, make_contributor):
    def _make(name: str = "submitter", status: str = "approved"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, status, rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


# --------------------------------------------------------------------------
# REGISTRY / resolve
# --------------------------------------------------------------------------


def test_registry_holds_the_first_party_types():
    assert set(REGISTRY) == {"collab_lora_finetune", "batch_inference",
                             "contained_batch"}
    for name, cls in REGISTRY.items():
        inst = cls()
        assert inst.name == name
        assert isinstance(inst.version, int) and inst.version >= 1


def test_every_type_answers_the_two_questions_that_gate_it():
    """``spot_checkable`` and ``requires_image`` are read off the class by code
    that is not the type -- ``spotcheck.maybe_issue`` and ``Worker.can_honor``
    -- and both fail closed on a missing attribute. Fail-closed is right, and
    it also means a type that simply forgot to answer looks identical to one
    that answered "no". Asserted explicitly so the forgetting is visible."""
    for name, cls in REGISTRY.items():
        assert isinstance(getattr(cls, "spot_checkable", None), bool), name
        assert isinstance(getattr(cls, "requires_image", None), bool), name


def test_only_a_deterministic_type_opts_in_to_spot_checks():
    """docs/13 §5.2. A probe compares one answer against an already-accepted
    one and docs/09 §5.1 rates a failure the largest single penalty in the
    system, so the set that opts in is worth pinning: training is stochastic,
    and a submitter's image is not something the coordinator can make any
    determinism claim about at all."""
    opted_in = {n for n, c in REGISTRY.items() if c.spot_checkable}
    assert opted_in == {"batch_inference"}


def test_only_the_contained_type_requires_an_image():
    """docs/11 §4's split. The other direction matters as much: a first-party
    type must *not* require one, because there is no confined body to run it
    in and an in-tree body would run submitter code unconfined."""
    assert {n for n, c in REGISTRY.items() if c.requires_image} == {"contained_batch"}


def test_resolve_returns_a_fresh_instance_each_call():
    a = resolve("batch_inference")
    b = resolve("batch_inference")
    assert a is not b and type(a) is type(b)


def test_resolve_unknown_type_raises_keyerror():
    with pytest.raises(KeyError):
        resolve("no_such_type")


def test_resolve_asserts_pinned_version():
    # At or below the build's version is fine; above it is not.
    assert resolve("batch_inference", 1).version == 1
    with pytest.raises(AssertionError):
        resolve("batch_inference", 2)
    # None skips the check entirely.
    assert resolve("batch_inference", None).version == 1


# --------------------------------------------------------------------------
# spec_json.sdk freezing at POST /v1/jobs
# --------------------------------------------------------------------------


def test_post_jobs_freezes_the_resolved_sdk_pair(client, conn, make_submitter):
    _, skey = make_submitter()
    spec = {
        "model_ref": "hf://m", "shards": [{"ref": "s0", "rows": 4}],
        "output_prefix": "out/", "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 8},
        "output_schema": {"id": "str", "output": "str"},
    }
    jid = client.post("/v1/jobs", headers=_hdr(skey),
                      json={"job_type": "batch_inference", "spec": spec}).json()["job_id"]

    stored = json.loads(conn.execute(
        "SELECT spec_json FROM jobs WHERE id = ?", (jid,)
    ).fetchone()["spec_json"])
    assert stored["sdk"] == {"job_type": "batch_inference", "version": 1}
    # The rest of the spec is preserved untouched.
    assert stored["shards"] == spec["shards"]


def test_post_jobs_rejects_a_spec_that_pins_a_newer_version(client, make_submitter):
    _, skey = make_submitter()
    r = client.post("/v1/jobs", headers=_hdr(skey), json={
        "job_type": "collab_lora_finetune",
        "spec": {"sdk": {"job_type": "collab_lora_finetune", "version": 99}},
    })
    assert r.status_code == 422
    assert "99" in r.json()["detail"] or "newer" in r.json()["detail"]


# --------------------------------------------------------------------------
# job_type_version_unsupported in the claim walk (recorded like required_image)
# --------------------------------------------------------------------------


def test_claim_walk_refuses_a_job_pinned_to_an_unsupported_version(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    spec = {
        "model_ref": "hf://m", "shards": [{"ref": "s0", "rows": 2}],
        "output_prefix": "out/", "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
    }
    jid = client.post("/v1/jobs", headers=_hdr(skey),
                      json={"job_type": "batch_inference", "spec": spec}).json()["job_id"]
    client.post(f"/v1/jobs/{jid}/enqueue", headers=_hdr(skey))
    # Tamper the frozen pin to a version this build cannot serve. (A real spec
    # can never reach the queue this way -- POST would 422 -- but the claim walk
    # is the belt to that braces, and it must record, not crash.)
    conn.execute(
        "UPDATE jobs SET spec_json = ? WHERE id = ?",
        (json.dumps({**spec, "sdk": {"job_type": "batch_inference", "version": 7}}), jid),
    )
    conn.commit()

    _, key = make_contributor()
    fw = FakeWorker(client, store, key)
    assert fw.claim() is None  # 204: the only walkable job was refused

    verdicts = {v.job_id: v for v in eligibility.explain(conn, fw.worker_id).verdicts}
    assert verdicts[jid].outcome == eligibility.REFUSED
    assert verdicts[jid].reason == "job_type_version_unsupported"


# --------------------------------------------------------------------------
# worker-side abandon-before-download (can_honor), mirroring required_image
# --------------------------------------------------------------------------


def _worker() -> Worker:
    return Worker(
        config=WorkerConfig(coordinator_url="http://c", key="k", image_tag=None),
        client=None, control=None,
        profile={"backend": "cpu", "device_name": "cpu", "supports": ["fp32"]},
    )


def test_can_honor_declines_a_task_pinned_to_a_newer_sdk_version():
    task = {
        "task_id": "t", "job_type": "batch_inference",
        "sdk": {"job_type": "batch_inference", "version": 9},
    }
    honored, reason = _worker().can_honor(task)
    assert honored is False
    assert reason.startswith(DECLINE_JOBTYPE_VERSION)


def test_can_honor_accepts_a_task_at_the_supported_sdk_version():
    task = {
        "task_id": "t", "job_type": "batch_inference",
        "sdk": {"job_type": "batch_inference", "version": 1},
    }
    assert _worker().can_honor(task) == (True, None)


def test_can_honor_ignores_a_task_with_no_sdk_block():
    # A collab task from scripts/newrun carries no sdk; the check is inert.
    assert _worker().can_honor({"task_id": "t"}) == (True, None)


# --------------------------------------------------------------------------
# TaskSpec widening (docs/10 "Spine deviations" #1)
# --------------------------------------------------------------------------


def test_taskspec_builds_from_id_alone_with_generic_fields():
    spec = TaskSpec(id="t", job_id="j", input_ref="shard://0", attempt_group="g")
    assert spec.run_id is None and spec.round_idx is None
    assert spec.base_adapter_ref is None and spec.lora_cfg is None
    assert spec.buckets == [] and spec.attempt_group == "g"
