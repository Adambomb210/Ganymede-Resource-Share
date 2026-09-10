"""``batch_inference`` against the seven-method protocol (docs/10-jobtype-sdk.md §4)
plus the generic claim / submit / close dispatch it exercises.

The seam's proof it is not welded to LoRA: no base adapter, no ``reduce``, no
round. The fast paths run a genuine (tiny) Qwen3 from the shared
``tiny_model_dir`` fixture; ``test_qwen_0_6b_cpu_end_to_end`` is ``slow`` and
mirrors ``test_trainer_cpu.py``.
"""

from __future__ import annotations

import json
import uuid

import pytest
import torch

from ganymede.coordinator import close, eligibility, rounds
from ganymede.jobtypes import resolve
from ganymede.jobtypes.batch_inference import BatchInference
from ganymede.jobtypes.batch_inference import plan as bi_plan
from ganymede.jobtypes.batch_inference import validate as bi_validate
from ganymede.jobtypes.batch_inference.run import InferResult, InferTask, canonical_digest
from ganymede.trainer import model as M
from tests.fake_worker import FakeWorker

CPU = torch.device("cpu")


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _spec(shards, *, redundancy=None, prefix="out/run1"):
    spec = {
        "model_ref": "hf://test-model",
        "shards": shards,
        "output_prefix": prefix,
        "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
    }
    if redundancy is not None:
        spec["redundancy"] = redundancy
    return spec


@pytest.fixture
def make_submitter(conn, make_contributor):
    def _make(name: str = "submitter"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, "approved", rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


@pytest.fixture(scope="module")
def tiny_lm(tiny_model_dir):
    tok = M.load_tokenizer(tiny_model_dir)
    mdl = M.load_base(tiny_model_dir, "fp32", device=CPU)
    return mdl, tok


# ==========================================================================
# validate_spec (submit-time shape check, docs/10 §4)
# ==========================================================================


def test_validate_spec_accepts_a_well_formed_spec():
    BatchInference().validate_spec(_spec([{"ref": "s0", "rows": 4096}]))


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s.pop("model_ref"), "model_ref"),
    (lambda s: s.__setitem__("shards", []), "shards"),
    (lambda s: s.__setitem__("shards", [{"ref": "s0", "rows": 0}]), "rows"),
    (lambda s: s.__setitem__("prompt_template", "no placeholder"), "prompt_template"),
    (lambda s: s.__setitem__("decode", {"mode": "greedy"}), "max_new_tokens"),
    (lambda s: s.__setitem__("output_schema", {}), "output_schema"),
])
def test_validate_spec_rejects_a_malformed_spec(mutate, msg):
    spec = _spec([{"ref": "s0", "rows": 4}])
    mutate(spec)
    with pytest.raises(ValueError) as exc:
        BatchInference().validate_spec(spec)
    assert msg in str(exc.value)


def test_a_redundancy_job_must_decode_greedily():
    spec = _spec([{"ref": "s0", "rows": 4}],
                 redundancy={"fraction": 0.5, "n": 2, "sample_rows": 2, "agree_on": "output"})
    spec["decode"]["mode"] = "sample"
    with pytest.raises(ValueError) as exc:
        BatchInference().validate_spec(spec)
    assert "greedy" in str(exc.value)


def test_redundancy_agree_on_must_name_an_output_field():
    spec = _spec([{"ref": "s0", "rows": 4}],
                 redundancy={"fraction": 0.5, "n": 2, "sample_rows": 2, "agree_on": "nope"})
    with pytest.raises(ValueError):
        BatchInference().validate_spec(spec)


# ==========================================================================
# plan (called once at enqueue; one task per shard, n copies for a fraction)
# ==========================================================================


def _job_row(spec: dict, job_id: str = "job1"):
    return {"id": job_id, "spec_json": json.dumps(spec)}


def test_plan_emits_one_taskspec_per_shard():
    spec = _spec([{"ref": "s0", "rows": 10}, {"ref": "s1", "rows": 20}])
    specs = bi_plan.plan(_job_row(spec), conn=None)
    assert len(specs) == 2
    for s, shard in zip(specs, spec["shards"]):
        assert s.job_id == "job1"
        assert json.loads(s.input_ref)["shard_ref"] == shard["ref"]
        assert s.run_id is None and s.round_idx is None
        assert s.attempt_group is None
        assert s.base_adapter_ref is None and s.lora_cfg is None


def test_plan_emits_n_copies_sharing_an_attempt_group_for_a_fraction():
    spec = _spec([{"ref": f"s{i}", "rows": 8} for i in range(4)],
                 redundancy={"fraction": 0.5, "n": 3, "sample_rows": 2, "agree_on": "output"})
    specs = bi_plan.plan(_job_row(spec), conn=None)
    # ceil(0.5 * 4) = 2 shards get 3 copies; 2 shards get 1.  -> 8 tasks
    assert len(specs) == 2 * 3 + 2
    groups = {}
    for s in specs:
        ref = json.loads(s.input_ref)["shard_ref"]
        groups.setdefault(ref, []).append(s)
    redundant = [g for g in groups.values() if len(g) > 1]
    singletons = [g for g in groups.values() if len(g) == 1]
    assert len(redundant) == 2 and len(singletons) == 2
    for g in redundant:
        gid = {s.attempt_group for s in g}
        assert len(gid) == 1 and None not in gid
    for g in singletons:
        assert g[0].attempt_group is None


# ==========================================================================
# inputs_for (model GET + output PUT; NO base adapter key)
# ==========================================================================


def test_inputs_for_names_the_model_and_an_output_put_no_base_adapter(store):
    spec = _spec([{"ref": "shard/0", "rows": 5}])
    task_spec = bi_plan.plan(_job_row(spec), conn=None)[0]
    task_row = {"input_ref_json": task_spec.input_ref}
    refs = BatchInference().inputs_for(task_row, store)
    assert set(refs.artifacts) == {"model", "shard"}
    assert "base_adapter" not in refs.artifacts
    assert refs.params["shard_ref"] == "shard/0"
    assert "output_put_url" in refs.params and "sig=put" in refs.params["output_put_url"]
    assert refs.params["decode"] == spec["decode"]
    assert refs.params["output_schema"] == spec["output_schema"]
    assert refs.params["output_key"].startswith("out/run1/")


# ==========================================================================
# run (tiny real model), digest stability
# ==========================================================================


class _Inputs:
    def __init__(self, artifacts=None, params=None):
        self.artifacts = artifacts or {}
        self.params = params or {}


def _infer_task(spec, task_spec):
    desc = json.loads(task_spec.input_ref)
    return InferTask(
        task_id=task_spec.id, job_id=task_spec.job_id, model_ref=desc["model_ref"],
        shard_ref=desc["shard_ref"], shard_rows=desc["shard_rows"],
        prompt_template=desc["prompt_template"], decode=desc["decode"],
        output_schema=desc["output_schema"], output_key=desc["output_key"],
    )


def test_run_emits_one_output_row_per_input_row_and_a_stable_digest(tiny_lm):
    mdl, tok = tiny_lm
    spec = _spec([{"ref": "s0", "rows": 6}])
    task_spec = bi_plan.plan(_job_row(spec), conn=None)[0]
    it = _infer_task(spec, task_spec)
    rows = [{"id": f"r{i}", "input": f"w{i} w{i + 1} w{i + 2}"} for i in range(6)]

    captured: dict[str, bytes] = {}
    steps: list[int] = []
    res = BatchInference().run(
        it, _Inputs(), on_step=lambda n, _l: steps.append(n),
        rows=rows, model=mdl, tokenizer=tok, device=CPU,
        upload=lambda b: captured.__setitem__("blob", b), batch_size=4,
    )
    assert isinstance(res, InferResult)
    assert res.rows == 6
    assert res.output_ref == it.output_key
    out_rows = [json.loads(line) for line in captured["blob"].decode().splitlines()]
    assert len(out_rows) == 6
    assert all(set(r) == {"id", "output"} for r in out_rows)
    assert [r["id"] for r in out_rows] == [f"r{i}" for i in range(6)]
    assert res.digest == canonical_digest(out_rows)
    assert steps and steps[-1] == 6

    # Greedy decode is deterministic: a second run digests identically.
    res2 = BatchInference().run(
        it, _Inputs(), rows=rows, model=mdl, tokenizer=tok, device=CPU,
        upload=lambda b: None, batch_size=2,
    )
    assert res2.digest == res.digest


def test_run_soft_stop_flushes_and_uploads_a_partial(tiny_lm):
    mdl, tok = tiny_lm
    spec = _spec([{"ref": "s0", "rows": 8}])
    task_spec = bi_plan.plan(_job_row(spec), conn=None)[0]
    it = _infer_task(spec, task_spec)
    rows = [{"id": f"r{i}", "input": f"w{i}"} for i in range(8)]

    calls = {"n": 0}
    def should_stop():
        calls["n"] += 1
        return "soft" if calls["n"] >= 2 else None

    captured: dict[str, bytes] = {}
    res = BatchInference().run(
        it, _Inputs(), should_stop=should_stop, rows=rows, model=mdl, tokenizer=tok,
        device=CPU, upload=lambda b: captured.__setitem__("blob", b), batch_size=2,
    )
    # first batch runs (sig None), second batch runs then soft breaks -> 4 rows.
    assert 0 < res.rows < 8
    assert len(captured["blob"].decode().splitlines()) == res.rows


def test_run_hard_stop_aborts_with_nothing_uploaded(tiny_lm):
    mdl, tok = tiny_lm
    spec = _spec([{"ref": "s0", "rows": 4}])
    it = _infer_task(spec, bi_plan.plan(_job_row(spec), conn=None)[0])
    rows = [{"id": f"r{i}", "input": f"w{i}"} for i in range(4)]
    uploaded = []
    with pytest.raises(RuntimeError):
        BatchInference().run(
            it, _Inputs(), should_stop=lambda: "hard", rows=rows,
            model=mdl, tokenizer=tok, device=CPU, upload=uploaded.append,
        )
    assert uploaded == []


# ==========================================================================
# validate (per-submission; row count + schema; attaches compare_digest)
# ==========================================================================


def _task_row(spec, task_spec):
    return {"input_ref_json": task_spec.input_ref}


def _put_output(store, key, rows):
    blob = "\n".join(json.dumps(r, sort_keys=True) for r in rows).encode()
    store.put_bytes(key, blob)


def test_validate_accepts_a_conformant_output_and_attaches_the_digest(store):
    spec = _spec([{"ref": "s0", "rows": 3}])
    ts = bi_plan.plan(_job_row(spec), conn=None)[0]
    key = json.loads(ts.input_ref)["output_key"]
    rows = [{"id": f"r{i}", "output": f"a{i}"} for i in range(3)]
    _put_output(store, key, rows)
    result = InferResult(rows=3, output_ref=key, digest="deadbeef", seconds=0.1)

    v = BatchInference().validate(_task_row(spec, ts), result, conn=None, store=store)
    assert v.accepted is True
    assert v.compare_digest == "deadbeef"


def test_validate_rejects_a_row_count_mismatch(store):
    spec = _spec([{"ref": "s0", "rows": 5}])
    ts = bi_plan.plan(_job_row(spec), conn=None)[0]
    key = json.loads(ts.input_ref)["output_key"]
    _put_output(store, key, [{"id": "r0", "output": "a"}])
    result = InferResult(rows=1, output_ref=key, digest="d", seconds=0.0)
    v = BatchInference().validate(_task_row(spec, ts), result, conn=None, store=store)
    assert v.accepted is False and v.reason == "row_count_mismatch"


def test_validate_rejects_a_schema_mismatch(store):
    spec = _spec([{"ref": "s0", "rows": 2}])
    ts = bi_plan.plan(_job_row(spec), conn=None)[0]
    key = json.loads(ts.input_ref)["output_key"]
    _put_output(store, key, [{"id": "r0", "output": "a"}, {"id": "r1", "extra": 1}])
    result = InferResult(rows=2, output_ref=key, digest="d", seconds=0.0)
    v = BatchInference().validate(_task_row(spec, ts), result, conn=None, store=store)
    assert v.accepted is False and v.reason == "schema_mismatch"


def test_validate_does_not_compare_across_the_attempt_group(store):
    # validate() only ever sees one result; the cross-task comparison is the
    # coordinator's (sample_agreement, exercised below).
    spec = _spec([{"ref": "s0", "rows": 1}])
    ts = bi_plan.plan(_job_row(spec), conn=None)[0]
    key = json.loads(ts.input_ref)["output_key"]
    _put_output(store, key, [{"id": "r0", "output": "x"}])
    v = BatchInference().validate(
        _task_row(spec, ts),
        InferResult(rows=1, output_ref=key, digest="d", seconds=0.0),
        conn=None, store=store,
    )
    assert v.accepted is True


# ==========================================================================
# reduce / is_complete / credit
# ==========================================================================


def test_reduce_is_always_none_and_is_complete_is_not_consulted():
    jt = BatchInference()
    assert jt.reduce(job=None, results=[], conn=None, store=None) is None
    assert jt.is_complete(job=None, state=None) is True


def test_credit_reports_rows():
    w = BatchInference().credit(task=None, result=InferResult(7, "k", "d", 1.0))
    assert (w.unit, w.count) == ("rows", 7)


# ==========================================================================
# sample_agreement (coordinator-owned attempt-group gate)
# ==========================================================================


def test_sample_agreement_true_when_the_sampled_field_matches():
    a = [{"id": "r0", "output": "x"}, {"id": "r1", "output": "y"}]
    b = [{"id": "r1", "output": "y"}, {"id": "r0", "output": "x"}]
    assert bi_validate.sample_agreement([a, b], sample_rows=2, agree_on="output") is True


def test_sample_agreement_false_on_a_disagreement():
    a = [{"id": "r0", "output": "x"}]
    b = [{"id": "r0", "output": "DIFFERENT"}]
    assert bi_validate.sample_agreement([a, b], sample_rows=4, agree_on="output") is False


def test_sample_agreement_false_when_the_id_sets_are_disjoint():
    a = [{"id": "r0", "output": "x"}]
    b = [{"id": "r9", "output": "x"}]
    assert bi_validate.sample_agreement([a, b], sample_rows=4, agree_on="output") is False


# ==========================================================================
# Generic dispatch: POST /v1/jobs -> enqueue -> claim -> run -> submit -> done
# ==========================================================================


def _enqueue_batch(client, skey, spec):
    jid = client.post("/v1/jobs", headers=_hdr(skey),
                      json={"job_type": "batch_inference", "spec": spec}).json()["job_id"]
    r = client.post(f"/v1/jobs/{jid}/enqueue", headers=_hdr(skey))
    assert r.status_code == 200, r.text
    return jid


def _run_and_submit(client, store, wkey, task, rows, tiny_lm):
    mdl, tok = tiny_lm
    it = InferTask.from_payload(task)
    key = task["params"]["output_key"]
    res = BatchInference().run(
        it, _Inputs(task["artifacts"], task["params"]),
        rows=rows, model=mdl, tokenizer=tok, device=CPU,
        upload=lambda b: store.put_bytes(key, b), batch_size=4,
    )
    up = client.post(f"/v1/tasks/{task['task_id']}/upload-url", headers=_hdr(wkey)).json()
    assert up["key"] == key
    return client.post(
        f"/v1/tasks/{task['task_id']}/submit", headers=_hdr(wkey),
        json={"artifact_key": key, "steps_completed": res.rows,
              "metrics": {"digest": res.digest, "rows": res.rows}},
    )


def test_batch_inference_runs_a_job_end_to_end(
    client, store, conn, make_contributor, make_submitter, tiny_lm
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="worker-owner")
    spec = _spec([{"ref": "shard/0", "rows": 6}])
    jid = _enqueue_batch(client, skey, spec)

    planned = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE job_id = ? AND status = 'planned'", (jid,)
    ).fetchone()["n"]
    assert planned == 1

    fw = FakeWorker(client, store, wkey)
    task = fw.claim()
    assert task is not None
    assert task["job_type"] == "batch_inference"
    assert task["job_id"] == jid
    assert task["sdk"] == {"job_type": "batch_inference", "version": 1}
    assert "model" in task["artifacts"]
    assert "output_put_url" in task["params"] and "base_adapter_url" not in task

    rows = [{"id": f"r{i}", "input": f"w{i} w{i + 1}"} for i in range(6)]
    r = _run_and_submit(client, store, wkey, task, rows, tiny_lm)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["reject_reason"] is None
    assert body["next_action"] == "claim"
    assert body["round_closed"] is False  # meaningless for a type with no reduce

    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (jid,)
    ).fetchone()["status"] == "done"
    assert conn.execute(
        "SELECT status FROM tasks WHERE job_id = ?", (jid,)
    ).fetchone()["status"] == "submitted"

    # A second claim from the same machine gets 204 -- the job is done.
    assert fw.claim() is None

    # /v1/runs/{id}/rounds/current is collab_lora_finetune-specific (docs/10 §5):
    # a batch job has no runs row, so its id 404s like a missing run.
    assert client.get(f"/v1/runs/{jid}/rounds/current", headers=_hdr(wkey)).status_code == 404


def test_one_lease_per_machine_holds_across_a_batch_job(
    client, store, conn, make_contributor, make_submitter, tiny_lm
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="w2")
    jid = _enqueue_batch(client, skey, _spec(
        [{"ref": f"shard/{i}", "rows": 4} for i in range(3)]
    ))
    fw = FakeWorker(client, store, wkey)
    first = fw.claim()
    assert first is not None
    # Holding a lease: a second poll re-serves the SAME task, never a new one.
    again = fw.claim()
    assert again is not None and again["task_id"] == first["task_id"]
    leased = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE worker_id = ? AND status = 'leased'",
        (fw.worker_id,),
    ).fetchone()["n"]
    assert leased == 1


def test_submit_rejects_an_artifact_key_the_worker_made_up(
    client, store, conn, make_contributor, make_submitter, tiny_lm
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="w3")
    jid = _enqueue_batch(client, skey, _spec([{"ref": "shard/0", "rows": 2}]))
    fw = FakeWorker(client, store, wkey)
    task = fw.claim()
    store.put_bytes("out/run1/somewhere-else.jsonl", b'{"id": "r0", "output": "x"}')
    r = client.post(
        f"/v1/tasks/{task['task_id']}/submit", headers=_hdr(wkey),
        json={"artifact_key": "out/run1/somewhere-else.jsonl", "steps_completed": 2,
              "metrics": {}},
    )
    assert r.status_code == 422


# ==========================================================================
# The redundancy agreement gate, via close.advance_job
# ==========================================================================


def _seed_agreeing_group(conn, store, jid, spec, *, outputs):
    """Insert an attempt_group of len(outputs) submitted+accepted tasks whose
    stored outputs are ``outputs`` (a list of row-lists)."""
    group = uuid.uuid4().hex
    shard = spec["shards"][0]
    for rows in outputs:
        tid = uuid.uuid4().hex
        key = f"{spec['output_prefix'].rstrip('/')}/{tid}.jsonl"
        desc = {
            "shard_ref": shard["ref"], "shard_rows": shard["rows"],
            "model_ref": spec["model_ref"], "prompt_template": spec["prompt_template"],
            "decode": spec["decode"], "output_schema": spec["output_schema"],
            "output_prefix": spec["output_prefix"], "output_key": key,
        }
        conn.execute(
            """INSERT INTO tasks (id, run_id, round_idx, job_id, buckets_json,
                 input_ref_json, attempt_group, local_steps, status, worker_id,
                 lease_expires_at, attempts, max_runtime_sec, created_at)
               VALUES (?, NULL, NULL, ?, '[]', ?, ?, 0, 'submitted', NULL, NULL, 1,
                       0, ?)""",
            (tid, jid, json.dumps(desc), group, rounds._iso(rounds.utcnow())),
        )
        conn.execute(
            """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
                 tokens_seen, metrics_json, accepted, reject_reason, received_at)
               VALUES (?, ?, ?, 0, '{}', 1, NULL, ?)""",
            (tid, key, shard["rows"], rounds._iso(rounds.utcnow())),
        )
        _put_output(store, key, rows)
    conn.commit()
    return group


def _bare_batch_job(conn, spec, status="running"):
    jid = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO jobs (id, owner_id, job_type, spec_json, image_id, status,
             priority_rank, constraints_json, cancel_mode, created_at)
           VALUES (?, 'system', 'batch_inference', ?, NULL, ?, 100, '{}', NULL, ?)""",
        (jid, json.dumps(spec), status, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    return jid


def test_agreement_gate_completes_the_job_when_the_group_agrees(conn, store):
    spec = _spec([{"ref": "s0", "rows": 2}],
                 redundancy={"fraction": 1.0, "n": 2, "sample_rows": 2, "agree_on": "output"})
    jid = _bare_batch_job(conn, spec)
    rows = [{"id": "r0", "output": "x"}, {"id": "r1", "output": "y"}]
    _seed_agreeing_group(conn, store, jid, spec, outputs=[rows, list(rows)])

    close.advance_job(conn, store, job_id=jid)
    assert conn.execute("SELECT status FROM jobs WHERE id = ?", (jid,)).fetchone()["status"] == "done"


def test_agreement_gate_redispatches_a_disagreeing_group_and_holds_the_job(conn, store):
    spec = _spec([{"ref": "s0", "rows": 2}],
                 redundancy={"fraction": 1.0, "n": 2, "sample_rows": 2, "agree_on": "output"})
    jid = _bare_batch_job(conn, spec)
    a = [{"id": "r0", "output": "x"}, {"id": "r1", "output": "y"}]
    b = [{"id": "r0", "output": "x"}, {"id": "r1", "output": "DIFFERENT"}]
    _seed_agreeing_group(conn, store, jid, spec, outputs=[a, b])

    close.advance_job(conn, store, job_id=jid)

    job_status = conn.execute("SELECT status FROM jobs WHERE id = ?", (jid,)).fetchone()["status"]
    assert job_status == "running"  # not done
    planned = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE job_id = ? AND status = 'planned'", (jid,)
    ).fetchone()["n"]
    assert planned == 2  # both members re-dispatched
    # Submissions are kept (durability-before-validation) but marked uncredited.
    subs = conn.execute(
        "SELECT s.accepted, s.reject_reason FROM submissions s "
        "JOIN tasks t ON t.id = s.task_id WHERE t.job_id = ?", (jid,)
    ).fetchall()
    assert len(subs) == 2
    assert all(s["accepted"] == 0 for s in subs)
    assert all(s["reject_reason"] == "attempt_group_disagreement" for s in subs)
    audit = conn.execute(
        "SELECT detail_json FROM audit WHERE event = 'attempt_group_disagreement'"
    ).fetchall()
    assert len(audit) == 1
    # The offending machines are recoverable from the audit event (Phase D input).
    members = json.loads(audit[0]["detail_json"])["members"]
    assert len(members) == 2 and all("worker_id" in m for m in members)


def test_an_expired_batch_lease_is_re_served_and_the_job_can_still_finish(
    client, store, conn, make_contributor, make_submitter, tiny_lm
):
    from datetime import timedelta

    _, skey = make_submitter()
    _, wkey = make_contributor(name="w-exp")
    jid = _enqueue_batch(client, skey, _spec([{"ref": "shard/0", "rows": 4}]))

    fw = FakeWorker(client, store, wkey)
    first = fw.claim()
    assert first is not None
    tid = first["task_id"]

    # The machine vanishes; its lease expires.
    rounds.expire_leases(conn, now=rounds.utcnow() + timedelta(hours=2))
    assert conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["status"] == "expired"
    fw.task = None  # the FakeWorker no longer holds it

    # A fresh poll re-serves the same shard, and the job completes on it.
    again = fw.claim()
    assert again is not None and again["task_id"] == tid
    rows = [{"id": f"r{i}", "input": f"w{i}"} for i in range(4)]
    r = _run_and_submit(client, store, wkey, again, rows, tiny_lm)
    assert r.status_code == 200 and r.json()["accepted"] is True
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (jid,)
    ).fetchone()["status"] == "done"


def test_an_abandoned_batch_task_is_re_served(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="w-ab")
    jid = _enqueue_batch(client, skey, _spec([{"ref": "shard/0", "rows": 4}]))
    fw = FakeWorker(client, store, wkey)
    first = fw.claim()
    client.post(f"/v1/tasks/{first['task_id']}/abandon", headers=_hdr(wkey))
    fw.task = None
    again = fw.claim()
    assert again is not None and again["task_id"] == first["task_id"]


def test_a_rejected_batch_submission_is_recycled_for_another_attempt(
    client, store, conn, make_contributor, make_submitter
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="w-rej")
    jid = _enqueue_batch(client, skey, _spec([{"ref": "shard/0", "rows": 3}]))
    fw = FakeWorker(client, store, wkey)
    task = fw.claim()
    key = task["params"]["output_key"]
    # Upload a wrong-row-count output -> validate() rejects it.
    _put_output(store, key, [{"id": "r0", "output": "a"}])
    r = client.post(
        f"/v1/tasks/{task['task_id']}/submit", headers=_hdr(wkey),
        json={"artifact_key": key, "steps_completed": 1, "metrics": {"digest": "d"}},
    )
    assert r.status_code == 200 and r.json()["accepted"] is False
    assert r.json()["reject_reason"] == "row_count_mismatch"

    # The shard is not orphaned: the next poll hands it back.
    fw.task = None
    again = fw.claim()
    assert again is not None and again["task_id"] == task["task_id"]
    assert conn.execute(
        "SELECT attempts FROM tasks WHERE id = ?", (task["task_id"],)
    ).fetchone()["attempts"] == 2


def test_parallel_job_not_done_until_every_task_has_an_accepted_verdict(conn, store):
    spec = _spec([{"ref": "s0", "rows": 2}, {"ref": "s1", "rows": 2}])
    jid = _bare_batch_job(conn, spec)
    # one shard done, one still planned
    _seed_agreeing_group(conn, store, jid, spec, outputs=[[{"id": "r0", "output": "x"},
                                                          {"id": "r1", "output": "y"}]])
    conn.execute(
        """INSERT INTO tasks (id, run_id, round_idx, job_id, buckets_json,
             input_ref_json, attempt_group, local_steps, status, worker_id,
             lease_expires_at, attempts, max_runtime_sec, created_at)
           VALUES (?, NULL, NULL, ?, '[]', '{}', NULL, 0, 'planned', NULL, NULL, 1, 0, ?)""",
        (uuid.uuid4().hex, jid, rounds._iso(rounds.utcnow())),
    )
    conn.commit()
    close.advance_job(conn, store, job_id=jid)
    assert conn.execute("SELECT status FROM jobs WHERE id = ?", (jid,)).fetchone()["status"] == "running"


# ==========================================================================
# slow: the real bring-up-class model on CPU (mirrors test_trainer_cpu.py)
# ==========================================================================


@pytest.mark.slow
def test_qwen_0_6b_cpu_end_to_end():
    """A genuine ``Qwen/Qwen3-0.6B-Base`` generation shard, structural asserts
    only -- CPU generation quality is not the point, the protocol path is."""
    base_model = "Qwen/Qwen3-0.6B-Base"
    try:
        tok = M.load_tokenizer(base_model)
        mdl = M.load_base(base_model, "fp32", device=CPU)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model unreachable: {exc}")

    spec = _spec([{"ref": "s0", "rows": 4}])
    ts = bi_plan.plan(_job_row(spec), conn=None)[0]
    it = _infer_task(spec, ts)
    rows = [{"id": f"q{i}", "input": f"The capital of country {i} is"} for i in range(4)]

    captured: dict[str, bytes] = {}
    res = BatchInference().run(
        it, _Inputs(), rows=rows, model=mdl, tokenizer=tok, device=CPU,
        upload=lambda b: captured.__setitem__("blob", b), batch_size=2,
    )
    out = [json.loads(line) for line in captured["blob"].decode().splitlines()]
    assert len(out) == 4
    assert all(set(r) == {"id", "output"} for r in out)
    assert res.digest == canonical_digest(out)


# ==========================================================================
# The shard has to be fetchable (Phase E)
# ==========================================================================


def test_inputs_for_presigns_the_shard(store):
    """``params["shard_ref"]`` keeps the submitter's literal ref -- it is what
    a log or a support question quotes -- and the fetchable URL goes in
    ``artifacts``, which is what ``InputRefs`` says artifacts are.

    This was the gap that made the type unrunnable off a real store. ``run``
    was written against exactly this shape and ``inputs_for`` never supplied
    it, so a real worker urlopen()'d the literal string ``"shard/0"``. Nothing
    caught it: every test above injects ``rows=`` and skips the download."""
    task_spec = bi_plan.plan(_job_row(_spec([{"ref": "shard/0", "rows": 5}])), conn=None)[0]
    refs = BatchInference().inputs_for({"input_ref_json": task_spec.input_ref}, store)

    assert refs.params["shard_ref"] == "shard/0"
    assert refs.artifacts["shard"].endswith("/shard/0?sig=get")


@pytest.mark.parametrize("ref", [
    "https://data.example/shards/0.jsonl",
    "http://data.example/shards/0.jsonl",
])
def test_a_shard_already_addressable_is_passed_through(store, ref):
    """A submitter whose data is not in our bucket names a URL. Presigning it
    against our own store would produce a key that does not exist."""
    task_spec = bi_plan.plan(_job_row(_spec([{"ref": ref, "rows": 5}])), conn=None)[0]
    refs = BatchInference().inputs_for({"input_ref_json": task_spec.input_ref}, store)
    assert refs.artifacts["shard"] == ref


def test_run_downloads_the_shard_from_the_artifacts_url(tiny_lm, monkeypatch):
    """The path a real worker takes: no ``rows=`` injection, so ``run`` has to
    resolve the shard itself from what ``inputs_for`` handed it."""
    from ganymede.jobtypes.batch_inference import run as run_mod

    mdl, tok = tiny_lm
    spec = _spec([{"ref": "shard/0", "rows": 3}])
    task_spec = bi_plan.plan(_job_row(spec), conn=None)[0]
    it = _infer_task(spec, task_spec)

    fetched: list[str] = []
    payload = b"\n".join(
        json.dumps({"id": f"r{i}", "input": f"w{i} w{i + 1}"}).encode() for i in range(3)
    )

    def fake_urlopen(url, *a, **kw):
        fetched.append(url if isinstance(url, str) else url.full_url)

        class _R:
            def read(self_inner):
                return payload

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _R()

    monkeypatch.setattr(run_mod.urllib.request, "urlopen", fake_urlopen)
    res = BatchInference().run(
        it, _Inputs({"model": "hf://x", "shard": "http://storage/x?sig=get"},
                    {"shard_ref": "shard/0"}),
        model=mdl, tokenizer=tok, device=CPU, upload=lambda b: None,
    )
    assert fetched == ["http://storage/x?sig=get"]
    assert res.rows == 3


def test_the_digest_does_not_depend_on_the_worker_s_batch_size(tiny_lm):
    """``batch_size`` is a worker-side knob, and redundancy compares digests
    across *different machines*. So a batch of eight and a batch of two have to
    agree, or an agreement gate is measuring the fleet's configuration.

    They did not. Generation pads on the left; the body padded on the right,
    which makes every short prompt continue from after its own padding -- so
    the output depended on which rows a batch happened to contain. The row
    count and the digest's *stability* were both unaffected, which is exactly
    why this could ship looking fine (transformers even warns about it, into a
    log nothing reads)."""
    mdl, tok = tiny_lm
    spec = _spec([{"ref": "s0", "rows": 6}])
    task_spec = bi_plan.plan(_job_row(spec), conn=None)[0]
    it = _infer_task(spec, task_spec)
    # Deliberately ragged: with equal-length prompts the bug is invisible.
    rows = [{"id": f"r{i}", "input": " ".join(f"w{j}" for j in range(i + 1))}
            for i in range(6)]

    digests = {
        n: BatchInference().run(it, _Inputs(), rows=rows, model=mdl, tokenizer=tok,
                                device=CPU, upload=lambda b: None, batch_size=n).digest
        for n in (1, 2, 6)
    }
    assert len(set(digests.values())) == 1, digests


def test_two_shards_share_one_loaded_model_and_agree_with_an_uncached_run(
    tiny_model_dir, monkeypatch
):
    """The cache matters more here than on the training side: shards are small
    and numerous, so a worker handed a run's worth of them would otherwise pay
    the full model load for every one (docs/03, "Two things worth knowing
    before M4b").

    Reuse is unconditionally safe for this type -- ``.eval()`` and ``generate``
    are read-only and idempotent -- but "safe by inspection" is what the right
    padding was too, so the digests are compared against uncached runs rather
    than assumed.
    """
    from ganymede.trainer.modelcache import ModelCache

    loads = []
    real_load = M.load_base
    monkeypatch.setattr(
        M, "load_base",
        lambda *a, **kw: (loads.append(a[0]), real_load(*a, **kw))[1],
    )

    spec = _spec([{"ref": "s0", "rows": 3}, {"ref": "s1", "rows": 3}])
    spec["model_ref"] = tiny_model_dir
    task_specs = bi_plan.plan(_job_row(spec), conn=None)[:2]
    rows = [[{"id": f"s{n}r{i}", "input": " ".join(f"w{j}" for j in range(i + 1))}
             for i in range(3)] for n in (0, 1)]

    cache = ModelCache()
    shared = [
        BatchInference().run(_infer_task(spec, ts), _Inputs(), rows=r, device=CPU,
                             upload=lambda b: None, cache=cache).digest
        for ts, r in zip(task_specs, rows)
    ]
    assert len(loads) == 1, "the second shard reloaded the model"

    uncached = [
        BatchInference().run(_infer_task(spec, ts), _Inputs(), rows=r, device=CPU,
                             upload=lambda b: None).digest
        for ts, r in zip(task_specs, rows)
    ]
    assert shared == uncached
    assert len(loads) == 3
    assert cache.stats() == {"hits": 1, "misses": 1, "resident": 1}


def test_an_hf_ref_is_stripped_before_it_reaches_from_pretrained():
    """``inputs_for`` passes an ``hf://`` ref through on the grounds that the
    worker pulls it from the Hub itself -- and ``from_pretrained`` has never
    heard of the scheme."""
    from ganymede.jobtypes.batch_inference.run import _hf_id

    assert _hf_id("hf://Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"
    assert _hf_id("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"
    assert _hf_id("/local/path/to/model") == "/local/path/to/model"


# ==========================================================================
# The real worker (Phase E)
#
# Everything above this line either drives the *coordinator* with a FakeWorker
# or calls the type's functions directly. Neither sees ``ganymede/worker/loop.py``,
# which is where a batch claim used to raise KeyError on ``base_adapter_url``
# and kill the worker on the spot. The whole suite was green through all of it.
#
# So: the real ``Worker``, the real ``_run_shard``, the real ``run``, real
# urllib against a real socket for the shard download and the output PUT, and
# the real coordinator gates on the way back. What is stubbed is the transport
# to the coordinator (a TestClient rather than uvicorn) and the model, which is
# the tiny local Qwen3 -- both of which have their own coverage elsewhere.
# ==========================================================================


@pytest.fixture
def served_store(store, monkeypatch):
    """``store``, with presigned URLs that a real urllib can actually fetch.

    FakeStore hands out ``http://storage.test:9000/...``, which is fine for
    every test that writes bytes into the dict by hand and exactly no use to a
    worker that does the transfer itself."""
    import threading
    from datetime import datetime, timedelta, timezone
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlparse

    class Handler(BaseHTTPRequestHandler):
        def _key(self):
            return unquote(urlparse(self.path).path.lstrip("/"))

        def do_GET(self):
            blob = store.objects.get(self._key())
            if blob is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_PUT(self):
            n = int(self.headers.get("Content-Length") or 0)
            store.objects[self._key()] = self.rfile.read(n)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    monkeypatch.setattr(store, "presign_get",
                        lambda key, expires_in=None: (f"{base}/{key}", later))
    monkeypatch.setattr(store, "presign_put",
                        lambda key, expires_in=None, content_length=None:
                        (f"{base}/{key}", later))
    try:
        yield store
    finally:
        srv.shutdown()


class _LoopbackClient:
    """The ``CoordinatorClient`` surface a Worker uses, over a TestClient."""

    def __init__(self, client, key):
        self.c = client
        self.h = {"Authorization": f"Bearer {key}"}
        self.calls: list[str] = []

    # `node_id` is accepted rather than swallowed by **kwargs on purpose:
    # this stub failing loudly when the real client's signature changed is
    # the suite noticing, and a stand-in that accepts anything notices
    # nothing.
    def register(self, profile, image_tag=None, node_id=None):
        self.calls.append("register")
        r = self.c.post("/v1/workers/register", headers=self.h,
                        json={"compute_profile": profile})
        assert r.status_code == 200, r.text
        return r.json()

    def claim(self, worker_id, **kwargs):
        self.calls.append("claim")
        r = self.c.post("/v1/tasks/claim", headers=self.h, json={"worker_id": worker_id})
        if r.status_code == 204:
            return None, 1
        assert r.status_code == 200, r.text
        return r.json(), 0

    def heartbeat(self, task_id, steps, loss=None):
        self.calls.append("heartbeat")
        r = self.c.post(f"/v1/tasks/{task_id}/heartbeat", headers=self.h,
                        json={"steps_completed": steps})
        return r.json() if r.content else {}

    def upload_url(self, task_id):
        self.calls.append("upload_url")
        r = self.c.post(f"/v1/tasks/{task_id}/upload-url", headers=self.h)
        assert r.status_code == 200, r.text
        return r.json()

    def upload(self, url, data, content_type="application/octet-stream"):
        raise AssertionError("the batch body must not upload: run() already did")

    def download(self, url):
        raise AssertionError("a batch task has nothing for the loop to download")

    def submit(self, task_id, key, steps, tokens_seen=0, metrics=None):
        self.calls.append("submit")
        r = self.c.post(f"/v1/tasks/{task_id}/submit", headers=self.h,
                        json={"artifact_key": key, "steps_completed": steps,
                              "tokens_seen": tokens_seen, "metrics": metrics or {}})
        assert r.status_code == 200, r.text
        return r.json()

    def abandon(self, task_id):
        self.calls.append("abandon")
        return self.c.post(f"/v1/tasks/{task_id}/abandon", headers=self.h).json()


def _real_worker(client, key, tmp_path, **cfg):
    from ganymede.worker.control import ControlFiles
    from ganymede.worker.loop import Worker, WorkerConfig

    return Worker(
        config=WorkerConfig(coordinator_url="http://c", key=key, once=True, **cfg),
        client=_LoopbackClient(client, key),
        control=ControlFiles(tmp_path, install_signal_handlers=False),
        profile={"backend": "cpu", "device_name": "cpu:test",
                 "supports": ["fp32"], "probe": {}, "vram_mb": 8000},
    )


def test_a_real_worker_carries_a_batch_shard_end_to_end(
    client, served_store, conn, make_contributor, make_submitter, tiny_model_dir, tmp_path
):
    _, skey = make_submitter()
    _, wkey = make_contributor(name="real-worker-owner")

    rows = [{"id": f"r{i}", "input": " ".join(f"w{j}" for j in range(i + 1))}
            for i in range(4)]
    served_store.put_bytes(
        "shards/0.jsonl",
        b"\n".join(json.dumps(r).encode() for r in rows),
    )

    spec = _spec([{"ref": "shards/0.jsonl", "rows": 4}])
    spec["model_ref"] = str(tiny_model_dir)
    jid = _enqueue_batch(client, skey, spec)

    # ``served_store`` patches the store *instance*, so this only works because
    # the ``client`` fixture is built from the same object. Pinned here rather
    # than assumed: if that ever stops holding, the presigned URLs revert to
    # unfetchable and this test fails as a confusing 404 inside urllib instead
    # of saying what actually broke.
    refs = BatchInference().inputs_for(
        conn.execute("SELECT * FROM tasks WHERE job_id = ?", (jid,)).fetchone(),
        served_store,
    )
    assert refs.artifacts["shard"].startswith("http://127.0.0.1")

    worker = _real_worker(client, wkey, tmp_path)
    assert worker.run() == 0

    assert worker.client.calls == ["register", "claim", "upload_url", "submit"]

    task = conn.execute("SELECT * FROM tasks WHERE job_id = ?", (jid,)).fetchone()
    assert task["status"] == "submitted"
    assert conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (jid,)
    ).fetchone()["status"] == "done"

    # The output the worker PUT, read back through the coordinator's own key.
    out = json.loads(conn.execute(
        "SELECT input_ref_json FROM tasks WHERE id = ?", (task["id"],)
    ).fetchone()["input_ref_json"])["output_key"]
    written = [json.loads(line) for line in
               served_store.get_bytes(out).decode().splitlines() if line.strip()]
    assert [r["id"] for r in written] == [r["id"] for r in rows]
    assert all(set(r) == {"id", "output"} for r in written)

    # The digest the worker computed is the one the coordinator kept, which is
    # what an attempt group is later compared on.
    from ganymede.jobtypes.batch_inference.run import canonical_digest

    sub = conn.execute(
        "SELECT * FROM submissions WHERE task_id = ?", (task["id"],)
    ).fetchone()
    assert sub["accepted"] == 1 and sub["reject_reason"] is None
    assert sub["steps_completed"] == 4  # rows are this type's steps on the wire
    assert json.loads(sub["metrics_json"])["compare_digest"] == canonical_digest(written)

    # And the trusted work signal, which only a type with ``credit`` records
    # (docs/09 §4): banked at zero weighted hours, carrying the WorkUnits
    # scalar. Nothing above this line would have noticed if it never landed.
    work = conn.execute(
        "SELECT * FROM credit_events WHERE kind = \'work\'"
    ).fetchall()
    assert len(work) == 1
    assert work[0]["raw_seconds"] == 4 and work[0]["weighted_hours"] == 0.0


def test_a_real_worker_declines_nothing_and_takes_no_lease_when_the_job_is_done(
    client, served_store, conn, make_contributor, make_submitter, tiny_model_dir, tmp_path
):
    """The second poll. A worker that finished the only shard must idle, not
    take a lease it cannot use -- and `--once` must still exit 0."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="second-poll")
    served_store.put_bytes("shards/0.jsonl", json.dumps({"id": "r0", "input": "w1"}).encode())
    spec = _spec([{"ref": "shards/0.jsonl", "rows": 1}])
    spec["model_ref"] = str(tiny_model_dir)
    _enqueue_batch(client, skey, spec)

    assert _real_worker(client, wkey, tmp_path).run() == 0

    second = _real_worker(client, wkey, tmp_path)
    second._idle = lambda seconds=0: None
    assert second.run() == 0
    assert second.client.calls == ["register", "claim"]
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status = 'leased'"
    ).fetchone()["n"] == 0
