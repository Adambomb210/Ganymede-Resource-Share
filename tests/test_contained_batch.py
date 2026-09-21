"""``contained_batch`` -- the third job type, and the sandbox's first consumer.

**What these tests cannot see, stated up front.** There is no container runtime
on the machine that wrote them, so nothing here runs a real container. What that
does *not* leave untested is the confinement itself: ``run_argv``,
``fetch_archive``, ``load_image`` and ``cancel`` all have real coverage in
``test_sandbox.py`` against the same injectable runner, and the wiring above
them is covered here. What remains genuinely unproven is exactly two claims:

1. a real daemon accepts this flag set and honours ``--network none`` and the
   caps, and
2. a real image's ENTRYPOINT reads ``/scratch/in`` and writes ``/scratch/out``
   per the contract docs/10 §7 publishes.

The second is a contract with no implementor yet, which is the honest shape of
the risk -- not "the sandbox is untested".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ganymede.jobtypes import resolve
from ganymede.jobtypes.base import InputRefs
from ganymede.jobtypes.contained_batch import plan as cb_plan
from ganymede.jobtypes.contained_batch import run as cb_run
from ganymede.jobtypes.contained_batch import validate as cb_validate
from ganymede.jobtypes.contained_batch.run import (
    ContainedCancelled,
    ContainedFailure,
    ContainedTask,
)
from ganymede.worker import sandbox

ARCHIVE = b"a docker save archive, near enough"
ARCHIVE_DIGEST = hashlib.sha256(ARCHIVE).hexdigest()
IMAGE_ID = "sha256:" + "ab" * 32


def _spec(shards=None, **over):
    # `is None`, not `or`: an empty shard list is a case under test, and a
    # truthiness default would quietly hand it the valid one instead.
    spec = {
        "shards": [{"ref": "in/0.jsonl", "rows": 2}] if shards is None else shards,
        "output_prefix": "out/j1",
        "output_schema": {"id": "str", "label": "str"},
        "params": {"threshold": 0.5},
    }
    spec.update(over)
    return spec


def _job_row(spec, job_id="j1"):
    return {"id": job_id, "spec_json": json.dumps(spec)}


def _task(spec, task_spec, **over):
    desc = json.loads(task_spec.input_ref)
    payload = {
        "task_id": task_spec.id,
        "job_id": task_spec.job_id,
        "image_ref": "img1",
        "image_digest": ARCHIVE_DIGEST,
        "image_pull_url": "http://store/img1",
        "max_runtime_sec": 600,
        "input_ref": task_spec.input_ref,
        "params": {**desc, "output_put_url": "http://store/put"},
    }
    payload.update(over)
    return ContainedTask.from_payload(payload)


class FakeRunner:
    """Stands in for the runtime binary. Records argv; scripts ``inspect``.

    Deliberately the same seam ``test_sandbox.py`` drives, so the argv these
    tests assert on is the argv that file already checks the flags of.
    """

    def __init__(self, running_polls=1, exit_code=0, on_start=None):
        self.calls: list[list[str]] = []
        self.running_polls = running_polls
        self.exit_code = exit_code
        self.on_start = on_start
        self.loaded = False

    def __call__(self, argv, timeout=None, env=None):
        self.calls.append(list(argv))
        verb = argv[1] if len(argv) > 1 else ""
        out = ""
        if verb == "load":
            out = f"Loaded image ID: {IMAGE_ID}"
            self.loaded = True
        elif verb == "run" and self.on_start is not None:
            self.on_start()
        elif verb == "inspect":
            fmt = argv[argv.index("-f") + 1]
            if "Running" in fmt:
                out = "true" if self.running_polls > 0 else "false"
                self.running_polls -= 1
            else:
                out = str(self.exit_code)
        return sandbox.Completed(returncode=0, stdout=out, stderr="")

    def verbs(self) -> list[str]:
        return [c[1] for c in self.calls if len(c) > 1]

    def argv_for(self, verb) -> list[str] | None:
        for c in self.calls:
            if len(c) > 1 and c[1] == verb:
                return c
        return None


@pytest.fixture
def make_submitter(conn, make_contributor):
    """The same per-file fixture ``test_batch_inference`` / ``test_images``
    carry. Duplicated rather than shared because that is where it already
    lives; hoisting it into conftest is a separate change."""
    from ganymede.coordinator import rounds

    def _make(name: str = "submitter"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, ?, ?)",
            (cid, "approved", rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


@pytest.fixture
def cfg(tmp_path):
    return sandbox.SandboxConfig(scratch_root=tmp_path / "scratch",
                                 cancel_grace_sec=90)


def _download(url):
    if "img" in url:
        return ARCHIVE
    return b'{"id": "r0", "text": "a"}\n{"id": "r1", "text": "b"}\n'


def _writes_output(cfg, task_id, rows=2):
    """The container's side of the contract, faked: called at ``docker run``."""
    def _write():
        out = cfg.scratch_root / task_id / "out" / cb_run.OUTPUT_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps({"id": f"r{i}", "label": "yes"})
                      for i in range(rows)),
            encoding="utf-8",
        )
    return _write


# ==========================================================================
# validate_spec -- what this type refuses to be asked for
# ==========================================================================


def test_a_well_formed_spec_is_accepted():
    cb_plan.validate_spec(_spec())


def test_redundancy_is_refused_because_the_body_is_not_ours():
    """docs/10 §4's redundancy and docs/13 §5's spot-checks both compare two
    machines' outputs for exact equality. Both are claims about determinism,
    and this type cannot make one -- the body is an image the coordinator did
    not build. An honest machine on a job that samples or stamps a timestamp
    would be convicted, and docs/09 §5.1 rates a failed probe the largest
    single penalty in the system.

    Refused at submit rather than ignored at plan: a submitter who asked for
    redundancy and silently did not get it is owed the error."""
    with pytest.raises(ValueError, match="redundancy"):
        cb_plan.validate_spec(_spec(redundancy={"n": 2, "fraction": 1.0}))


def test_the_schema_must_carry_an_id():
    """``validate`` aligns the container's output against the shard by id.
    Without one there is no way to say a row came back at all -- only that the
    right *number* of rows did."""
    with pytest.raises(ValueError, match="id"):
        cb_plan.validate_spec(_spec(output_schema={"label": "str"}))


@pytest.mark.parametrize("bad", [
    {"shards": []},
    {"shards": [{"ref": "", "rows": 2}]},
    {"shards": [{"ref": "a", "rows": 0}]},
    {"output_prefix": ""},
    {"output_schema": {"id": "str", "x": "complex"}},
    {"params": "not an object"},
])
def test_a_malformed_spec_names_its_field(bad):
    with pytest.raises(ValueError):
        cb_plan.validate_spec(_spec(**bad))


def test_plan_makes_one_task_per_shard_and_never_a_group():
    """No ``attempt_group`` is ever set, because ``validate_spec`` refused the
    only thing that would create one."""
    spec = _spec([{"ref": "in/0", "rows": 2}, {"ref": "in/1", "rows": 3}])
    specs = cb_plan.plan(_job_row(spec), conn=None)
    assert len(specs) == 2
    assert all(s.attempt_group is None for s in specs)
    assert {json.loads(s.input_ref)["shard_ref"] for s in specs} == {"in/0", "in/1"}


# ==========================================================================
# The body
# ==========================================================================


def test_a_shard_runs_end_to_end_through_the_sandbox(cfg):
    """Stage, pull, verify, load, run, collect, upload -- the whole path, with
    a fake runtime standing in for the daemon."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts)
    runner = FakeRunner(on_start=_writes_output(cfg, task.task_id))
    uploaded = []

    result = resolve("contained_batch").run(
        task, InputRefs(artifacts={"shard": "http://store/shard"}, params={}),
        None, None, config=cfg, runner=runner, download=_download,
        upload=uploaded.append, sleep=lambda _s: None,
    )

    assert runner.verbs()[:3] == ["load", "run", "inspect"]
    assert result.rows == 2
    assert result.exit_code == 0
    assert [json.loads(l)["id"] for l in uploaded[0].decode().splitlines()] == ["r0", "r1"]
    assert result.metrics["image_ref"] == "img1"


def test_the_leases_devices_reach_the_container_launch(cfg):
    """docs/14 §5.4's ``devices`` field on the task payload, threaded end to
    end: ``ContainedTask.from_payload`` reads it off the task, and ``backend``
    -- the *worker's*, not the task's -- comes in through ``run``'s own kwarg
    the way ``worker.loop._run_contained`` supplies it from ``self.profile``.
    The ``docker run`` argv the fake runtime actually saw must carry the
    lease's own device, never 'all'."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts, devices=[2])
    runner = FakeRunner(on_start=_writes_output(cfg, task.task_id))

    resolve("contained_batch").run(
        task, InputRefs(artifacts={"shard": "http://store/shard"}, params={}),
        None, None, config=cfg, runner=runner, download=_download,
        upload=lambda b: None, sleep=lambda _s: None, backend="cuda",
    )

    run_argv = runner.argv_for("run")
    assert run_argv is not None
    assert "--gpus" in run_argv
    assert run_argv[run_argv.index("--gpus") + 1] == '"device=2"'


def test_the_archive_is_hashed_before_anything_loads_it(cfg):
    """docs/11 §2.3. ``docker load`` parses an archive the submitter controls,
    so running it on bytes that failed their check would be trusting the thing
    being checked."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts, image_digest="sha256:" + "00" * 32)
    runner = FakeRunner()

    with pytest.raises(sandbox.DigestMismatch):
        resolve("contained_batch").run(
            task, InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, None, config=cfg, runner=runner, download=_download,
            sleep=lambda _s: None,
        )
    assert "load" not in runner.verbs(), "loaded an archive that failed its digest"


def test_the_image_archive_does_not_stay_in_the_job_s_scratch(cfg):
    """Scratch is bind-mounted into the container, and at docs/11 §1.1's 10 GiB
    upload cap the archive is the largest thing in it. Dropped after load: it
    frees the space before the job runs and keeps it out of the job's view."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts)
    seen = {}

    def peek():
        seen["archive"] = (cfg.scratch_root / task.task_id / "image.tar").exists()
        _writes_output(cfg, task.task_id)()

    resolve("contained_batch").run(
        task, InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
        None, None, config=cfg, runner=FakeRunner(on_start=peek),
        download=_download, upload=lambda b: None, sleep=lambda _s: None,
    )
    assert seen["archive"] is False


def test_the_container_gets_the_shard_and_the_params_as_files(cfg):
    """The published contract (docs/10 §7): ``/scratch/in/input.jsonl`` and
    ``/scratch/in/params.json`` in, ``/scratch/out/output.jsonl`` out. The job
    has no network, so this is the only way anything reaches it."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts)
    staged = {}

    def peek():
        d = cfg.scratch_root / task.task_id / "in"
        staged["input"] = (d / cb_run.INPUT_NAME).read_bytes()
        staged["params"] = json.loads((d / cb_run.PARAMS_NAME).read_text())
        _writes_output(cfg, task.task_id)()

    resolve("contained_batch").run(
        task, InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
        None, None, config=cfg, runner=FakeRunner(on_start=peek),
        download=_download, upload=lambda b: None, sleep=lambda _s: None,
    )
    assert staged["input"] == _download("shard")
    assert staged["params"]["params"] == {"threshold": 0.5}
    assert staged["params"]["shard_rows"] == 2
    assert staged["params"]["task_id"] == task.task_id


def test_a_non_zero_exit_is_the_job_s_fault_and_uploads_nothing(cfg):
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts)
    uploaded = []

    with pytest.raises(ContainedFailure) as exc:
        resolve("contained_batch").run(
            task, InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, None, config=cfg, runner=FakeRunner(exit_code=3),
            download=_download, upload=uploaded.append, sleep=lambda _s: None,
        )
    assert exc.value.exit_code == 3
    assert uploaded == []


def test_exiting_zero_with_no_output_is_still_a_failure(cfg):
    """A container that ran, said it succeeded and wrote nothing has not done
    the task. Silently submitting zero rows would spend an attempt and read as
    the job's answer."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    with pytest.raises(ContainedFailure, match=cb_run.OUTPUT_NAME):
        resolve("contained_batch").run(
            _task(spec, ts),
            InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, None, config=cfg, runner=FakeRunner(),
            download=_download, sleep=lambda _s: None,
        )


# ==========================================================================
# The soft / hard kill -- the distinction this type exists to make real
# ==========================================================================


@pytest.mark.parametrize("signal,verb,extra", [
    ("soft", "stop", ["--time", "90"]),
    ("hard", "kill", []),
])
def test_soft_and_hard_finally_mean_different_things(cfg, signal, verb, extra):
    """docs/11 §3 step 3. For every previous type the two collapsed to "stop
    the loop and give the shard back" -- there was no container to signal and
    nothing to checkpoint. This is the submitter code that distinction was
    written for: ``soft`` is a SIGTERM plus the grace the job was promised,
    ``hard`` is SIGKILL now.
    """
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    runner = FakeRunner(running_polls=10)

    with pytest.raises(ContainedCancelled) as exc:
        resolve("contained_batch").run(
            _task(spec, ts),
            InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, lambda: signal, config=cfg, runner=runner,
            download=_download, sleep=lambda _s: None,
        )
    assert exc.value.signal == signal
    argv = runner.argv_for(verb)
    assert argv is not None, f"expected a {verb!r} call, saw {runner.verbs()}"
    for token in extra:
        assert token in argv


def test_a_cancel_uploads_nothing(cfg):
    """A partial output is shorter than the shard's declared rows, ``validate``
    rejects on that count, and submitting one would spend an attempt to be told
    no. The partial itself is harmless -- the next attempt writes the key."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    uploaded = []
    with pytest.raises(ContainedCancelled):
        resolve("contained_batch").run(
            _task(spec, ts),
            InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, lambda: "hard", config=cfg, runner=FakeRunner(running_polls=5),
            download=_download, upload=uploaded.append, sleep=lambda _s: None,
        )
    assert uploaded == []


# ==========================================================================
# Supervision and cleanup
# ==========================================================================


def test_progress_is_reported_from_what_the_container_flushed(cfg):
    """``on_step`` fires each poll with the rows actually on disk. Liveness
    does not depend on it -- the Heartbeater is a thread on its own timer -- so
    a job that buffers reports zero without going stale."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts)
    steps = []
    runner = FakeRunner(running_polls=3, on_start=_writes_output(cfg, task.task_id))

    resolve("contained_batch").run(
        task, InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
        lambda rows, _loss: steps.append(rows), None,
        config=cfg, runner=runner, download=_download,
        upload=lambda b: None, sleep=lambda _s: None,
    )
    # Three polls while running, then the final count after collection.
    assert len(steps) == 4
    assert steps[-1] == 2


@pytest.mark.parametrize("kind", ["exit", "digest", "cancel", "ok"])
def test_scratch_is_wiped_on_every_exit_path(cfg, kind):
    """A scratch dir that survives a crash is the disk that fills up over a
    week of them, so the cleanup is in a ``finally`` rather than on the happy
    path. Asserted on the directory, not on whether ``rm -f`` was issued."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    task = _task(spec, ts,
                 **({"image_digest": "sha256:" + "00" * 32} if kind == "digest" else {}))
    runner = FakeRunner(
        exit_code=3 if kind == "exit" else 0,
        running_polls=5 if kind == "cancel" else 1,
        on_start=_writes_output(cfg, task.task_id) if kind == "ok" else None,
    )
    stop = (lambda: "hard") if kind == "cancel" else None

    try:
        resolve("contained_batch").run(
            task, InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, stop, config=cfg, runner=runner, download=_download,
            upload=lambda b: None, sleep=lambda _s: None,
        )
    except RuntimeError:
        pass

    assert not (cfg.scratch_root / task.task_id).exists()
    assert "rm" in runner.verbs()


def test_a_task_with_no_image_is_refused_rather_than_run_empty(cfg):
    """Unreachable from the claim path -- ``_image_handles`` returns all three
    fields or none -- so this is the belt to that braces. Loud, because the
    alternative for a *contained* type is running nothing and reporting
    success."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    with pytest.raises(sandbox.SandboxError, match="no image"):
        resolve("contained_batch").run(
            _task(spec, ts, image_pull_url=None),
            InputRefs(artifacts={"shard": "http://s/shard"}, params={}),
            None, None, config=cfg, runner=FakeRunner(), download=_download,
        )


# ==========================================================================
# validate
# ==========================================================================


class _Store:
    def __init__(self, blob: bytes):
        self.blob = blob

    def get_bytes(self, ref):
        return self.blob


class _Result:
    def __init__(self, rows, output_ref="out/j1/t.jsonl"):
        self.rows = rows
        self.output_ref = output_ref


def _task_row(spec, ts):
    return {"input_ref_json": ts.input_ref}


def _rows_blob(rows):
    return "\n".join(json.dumps(r) for r in rows).encode()


def test_validate_accepts_a_conforming_output():
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    blob = _rows_blob([{"id": "r0", "label": "a"}, {"id": "r1", "label": "b"}])
    v = cb_validate.validate(_task_row(spec, ts), _Result(2), None, _Store(blob))
    assert v.accepted
    assert v.compare_digest is None, "a submitter image makes no determinism claim"


def test_validate_reads_the_artifact_rather_than_the_worker_s_count():
    """The worker is the thing being checked. A gate that trusted its summary
    would only be checking its arithmetic."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    blob = _rows_blob([{"id": "r0", "label": "a"}])   # one row, worker claims two
    v = cb_validate.validate(_task_row(spec, ts), _Result(2), None, _Store(blob))
    assert not v.accepted and v.reason == "row_count_mismatch"


def test_validate_rejects_a_row_that_misses_the_schema():
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    blob = _rows_blob([{"id": "r0", "label": 7}, {"id": "r1", "label": "b"}])
    v = cb_validate.validate(_task_row(spec, ts), _Result(2), None, _Store(blob))
    assert not v.accepted and v.reason == "schema_mismatch"


def test_validate_rejects_repeated_ids():
    """One row per input row, but the same id twice, answers a different
    question than the one asked -- and nothing that joins on id would notice."""
    spec = _spec()
    ts = cb_plan.plan(_job_row(spec), conn=None)[0]
    blob = _rows_blob([{"id": "r0", "label": "a"}, {"id": "r0", "label": "b"}])
    v = cb_validate.validate(_task_row(spec, ts), _Result(2), None, _Store(blob))
    assert not v.accepted and v.reason == "duplicate_ids"


# ==========================================================================
# Through the real coordinator: POST /v1/jobs -> enqueue -> claim -> submit
# ==========================================================================
#
# The unit tests above build a ``ContainedTask`` by hand. That is exactly the
# blind spot that hid two real defects in ``batch_inference`` -- a shard that
# ``inputs_for`` never presigned, and an ``hf://`` prefix that never reached
# ``from_pretrained`` -- because every test injected what the coordinator was
# supposed to supply. These drive the actual endpoints instead.


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _clean_image(client, conn, store, key) -> str:
    """Upload, finalize, and mark clean.

    The scan itself is docs/11 §1.3's and is tested in ``test_images.py``;
    what matters here is that a *schedulable* image produces the three payload
    fields a contained task needs.
    """
    import hashlib

    from ganymede.coordinator.store import image_key

    body = {"repo_tag": "job:latest", "digest": hashlib.sha256(ARCHIVE).hexdigest(),
            "size_bytes": len(ARCHIVE)}
    image_id = client.post("/v1/images/upload-url", json=body,
                           headers=_hdr(key)).json()["image_id"]
    store.put_bytes(image_key(image_id), ARCHIVE)
    assert client.post(f"/v1/images/{image_id}/finalize",
                       headers=_hdr(key)).status_code == 200
    conn.execute("UPDATE images SET scan_status = 'clean' WHERE id = ?", (image_id,))
    conn.commit()
    return image_id


def _enqueue(client, skey, spec, image_id):
    r = client.post("/v1/jobs", headers=_hdr(skey),
                    json={"job_type": "contained_batch", "spec": spec,
                          "image_id": image_id})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert client.post(f"/v1/jobs/{jid}/enqueue",
                       headers=_hdr(skey)).status_code == 200
    return jid


def test_the_claim_payload_carries_everything_the_contained_body_needs(
    client, store, conn, make_contributor, make_submitter
):
    """The seam the hand-built ``ContainedTask`` above cannot check.

    Three of these come from three different places -- ``_image_handles`` walks
    ``jobs.image_id``, ``inputs_for`` presigns the shard, ``plan`` packed the
    descriptor -- and a contained task is unrunnable if any one is missing.
    """
    _, skey = make_submitter()
    _, wkey = make_contributor(name="worker-owner")
    image_id = _clean_image(client, conn, store, skey)
    store.put_bytes("in/0.jsonl", _download("shard"))
    _enqueue(client, skey, _spec(), image_id)

    reg = client.post("/v1/workers/register", headers=_hdr(wkey), json={
        "compute_profile": {"backend": "cpu", "vram_gb": 8, "supports": ["fp32"],
                            "container_runtime": "docker"}}).json()
    payload = client.post("/v1/tasks/claim", headers=_hdr(wkey),
                          json={"worker_id": reg["worker_id"]}).json()

    assert payload["job_type"] == "contained_batch"
    # The image, from jobs.image_id (docs/06, _image_handles).
    assert payload["image_ref"] == image_id
    assert payload["image_digest"] and payload["image_pull_url"]
    # The shard, presigned rather than left as the literal key. This is the
    # exact defect that made batch_inference unrunnable off a real store.
    assert payload["artifacts"]["shard"] != "in/0.jsonl"
    assert "in/0.jsonl" in payload["artifacts"]["shard"]

    parsed = ContainedTask.from_payload(payload)
    assert parsed.shard_rows == 2
    assert parsed.params == {"threshold": 0.5}
    assert parsed.output_key.startswith("out/j1/")
    assert parsed.image_pull_url == payload["image_pull_url"]


def test_a_machine_with_no_runtime_is_not_offered_the_job(
    client, store, conn, make_contributor, make_submitter
):
    """docs/11 §4: submitter code only ever runs contained, on a host that
    opted into a container runtime. Refused at claim, before the worker's own
    ``can_honor`` ever sees it."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="no-docker")
    image_id = _clean_image(client, conn, store, skey)
    store.put_bytes("in/0.jsonl", _download("shard"))
    _enqueue(client, skey, _spec(), image_id)

    reg = client.post("/v1/workers/register", headers=_hdr(wkey), json={
        "compute_profile": {"backend": "cpu", "vram_gb": 8,
                            "supports": ["fp32"]}}).json()
    r = client.post("/v1/tasks/claim", headers=_hdr(wkey),
                    json={"worker_id": reg["worker_id"]})
    # 204: no work for this machine, which is the normal "nothing eligible"
    # answer rather than an error (docs/06).
    assert r.status_code == 204, r.text

    reason = conn.execute(
        "SELECT reason FROM worker_eligibility ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert reason is not None and reason["reason"] == "no_container_runtime"


def test_a_submitted_shard_is_gated_and_credited(
    client, store, conn, make_contributor, make_submitter
):
    """Submit through the real endpoint: ``validate`` reads the stored artifact,
    the verdict lands, and the work is credited in the units this type
    declares."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="worker-owner")
    image_id = _clean_image(client, conn, store, skey)
    store.put_bytes("in/0.jsonl", _download("shard"))
    _enqueue(client, skey, _spec(), image_id)

    reg = client.post("/v1/workers/register", headers=_hdr(wkey), json={
        "compute_profile": {"backend": "cpu", "vram_gb": 8, "supports": ["fp32"],
                            "container_runtime": "docker"}}).json()
    payload = client.post("/v1/tasks/claim", headers=_hdr(wkey),
                          json={"worker_id": reg["worker_id"]}).json()
    tid = payload["task_id"]

    key = client.post(f"/v1/tasks/{tid}/upload-url",
                      headers=_hdr(wkey)).json()["key"]
    store.put_bytes(key, _rows_blob([{"id": "r0", "label": "a"},
                                     {"id": "r1", "label": "b"}]))
    r = client.post(f"/v1/tasks/{tid}/submit", headers=_hdr(wkey),
                    json={"artifact_key": key, "steps_completed": 2,
                          "metrics": {"rows": 2, "digest": "f" * 64}})
    assert r.status_code == 200, r.text
    assert r.json()["accepted"] is True, r.json()

    # The work signal, which nothing above this line would have noticed the
    # absence of. ``credit`` returns rows; the ledger records the *seconds* the
    # lease ran and leaves weighting to the accrual engine (docs/09).
    work = conn.execute(
        "SELECT * FROM credit_events WHERE kind = 'work'").fetchall()
    assert len(work) == 1


def test_a_short_output_is_rejected_by_the_real_gate(
    client, store, conn, make_contributor, make_submitter
):
    """The row-count gate, through the endpoint. A cancelled container's partial
    output would land here, which is why the worker never submits one."""
    _, skey = make_submitter()
    _, wkey = make_contributor(name="worker-owner")
    image_id = _clean_image(client, conn, store, skey)
    store.put_bytes("in/0.jsonl", _download("shard"))
    _enqueue(client, skey, _spec(), image_id)

    reg = client.post("/v1/workers/register", headers=_hdr(wkey), json={
        "compute_profile": {"backend": "cpu", "vram_gb": 8, "supports": ["fp32"],
                            "container_runtime": "docker"}}).json()
    tid = client.post("/v1/tasks/claim", headers=_hdr(wkey),
                      json={"worker_id": reg["worker_id"]}).json()["task_id"]

    key = client.post(f"/v1/tasks/{tid}/upload-url",
                      headers=_hdr(wkey)).json()["key"]
    store.put_bytes(key, _rows_blob([{"id": "r0", "label": "a"}]))
    r = client.post(f"/v1/tasks/{tid}/submit", headers=_hdr(wkey),
                    json={"artifact_key": key, "steps_completed": 1,
                          "metrics": {"rows": 1}})
    assert r.status_code == 200
    assert r.json()["accepted"] is False
    assert r.json()["reject_reason"] == "row_count_mismatch"
