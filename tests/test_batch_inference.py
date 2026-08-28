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
    assert set(refs.artifacts) == {"model"}
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
