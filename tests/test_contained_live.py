"""``contained_batch`` against a **real** container runtime.

Everything in ``test_contained_batch.py`` drives an injectable runner. That is
the right default -- it runs everywhere in milliseconds -- and it is exactly the
kind of test that cannot see the class of defect this file exists to catch: a
fake returns the output the parser already expects, so a parser that is wrong
about what the real tool prints looks correct forever.

It caught one immediately. ``docker load`` prints ``Loaded image ID: sha256:...``
only for an archive with **no repo tags**; a tagged archive prints
``Loaded image: name:tag``. ``load_image`` handled only the first, and docs/11
§1.1's upload path takes a ``docker save`` payload with a required ``repo_tag``
-- so every archive a submitter could realistically produce would have raised.
The confinement was fine; the thing that reads the daemon's own output was not.

Skipped, not failed, where no daemon is reachable: a contributor without Docker
still runs the rest of the suite, and docs/11 §4 says such a machine is refused
submitter jobs anyway.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from ganymede.jobtypes import resolve
from ganymede.jobtypes.base import InputRefs
from ganymede.jobtypes.contained_batch import plan as cb_plan
from ganymede.jobtypes.contained_batch.run import (
    ContainedCancelled,
    ContainedFailure,
    ContainedTask,
)
from ganymede.worker import sandbox

pytestmark = pytest.mark.slow

BASE = "busybox:1.36"
TAG = "ganymede-test-job:1"
ROWS = b'{"id": "r0", "text": "a"}\n{"id": "r1", "text": "b"}\n'

# The container's half of docs/10 §7's contract, plus three probes reported in
# the output row. Asserting confinement from *inside* the container is the only
# place the claim actually means anything: `run_argv` passing `--network none`
# is a string, `wget` failing is the property.
ENTRYPOINT = """#!/bin/sh
set -e
IN="$GANYMEDE_SCRATCH/in/input.jsonl"
OUT="$GANYMEDE_SCRATCH/out/output.jsonl"

if grep -q '"mode": "fail"' "$GANYMEDE_SCRATCH/in/params.json" 2>/dev/null; then
    exit 7
fi
if grep -q '"mode": "hang"' "$GANYMEDE_SCRATCH/in/params.json" 2>/dev/null; then
    # Traps SIGTERM the way docs/11 §3 expects a well-behaved job to, so a soft
    # cancel is distinguishable from a kill.
    trap 'echo trapped; exit 0' TERM
    sleep 300 &
    wait
    exit 0
fi

net=blocked
wget -q -T 2 -O /dev/null http://1.1.1.1/ 2>/dev/null && net=open
root=ro
touch /probe 2>/dev/null && root=rw
: > "$OUT"
while IFS= read -r line; do
    [ -z "$line" ] && continue
    id=$(printf '%s' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p')
    printf '{"id": "%s", "label": "net=%s;root=%s;uid=%s"}\\n' \\
        "$id" "$net" "$root" "$(id -u)" >> "$OUT"
done < "$IN"
"""

DOCKERFILE = f"""FROM {BASE}
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
"""


def _runtime() -> str | None:
    binary = shutil.which("docker")
    if binary is None:
        return None
    try:
        r = subprocess.run([binary, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return binary if r.returncode == 0 and r.stdout.strip() else None


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """Build the test image and ``docker save`` it -- the same shape docs/11
    §1.1 says a submitter uploads, **tagged**, which is the point."""
    binary = _runtime()
    if binary is None:
        pytest.skip("no reachable container runtime")

    build = tmp_path_factory.mktemp("image")
    (build / "entrypoint.sh").write_text(ENTRYPOINT, encoding="utf-8", newline="\n")
    (build / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8", newline="\n")

    r = subprocess.run([binary, "build", "-q", "-t", TAG, "."],
                       cwd=build, capture_output=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"could not build the test image: {r.stderr.decode()[-400:]}")

    tar = build / "job.tar"
    r = subprocess.run([binary, "save", TAG, "-o", str(tar)],
                       capture_output=True, timeout=600)
    assert r.returncode == 0, r.stderr.decode()
    return tar.read_bytes()


@pytest.fixture
def cfg(tmp_path):
    # No GPU: the flag is the host's to set (docs/11 §2.2) and asking for one
    # here would make the test fail on any box without the container toolkit,
    # which is not what it is testing.
    return sandbox.SandboxConfig(scratch_root=tmp_path / "scratch", gpus=None,
                                 memory="512m", cpus="1.0", cancel_grace_sec=5)


def _spec(**over):
    spec = {"shards": [{"ref": "in/0.jsonl", "rows": 2}],
            "output_prefix": "out/j1",
            "output_schema": {"id": "str", "label": "str"},
            "params": {}}
    spec.update(over)
    return spec


def _task(archive, spec=None, **over):
    spec = spec or _spec()
    ts = cb_plan.plan({"id": "j1", "spec_json": json.dumps(spec)}, conn=None)[0]
    payload = {
        "task_id": ts.id, "job_id": "j1", "input_ref": ts.input_ref,
        "image_ref": "img1",
        "image_digest": hashlib.sha256(archive).hexdigest(),
        "image_pull_url": "http://store/archive",
        "max_runtime_sec": 120,
        "params": dict(json.loads(ts.input_ref)),
    }
    payload.update(over)
    return ContainedTask.from_payload(payload)


def _download(archive):
    def _get(url):
        return archive if "archive" in url else ROWS
    return _get


def _run(task, archive, cfg, **kw):
    kw.setdefault("upload", lambda b: None)
    return resolve("contained_batch").run(
        task, InputRefs(artifacts={"shard": "http://store/shard"}, params={}),
        kw.pop("on_step", None), kw.pop("should_stop", None),
        config=cfg, download=_download(archive), poll_sec=0.3, **kw,
    )


# ==========================================================================


def test_a_real_container_runs_a_shard_end_to_end(archive, cfg):
    """The two claims the fake-runner suite could not make: a real daemon
    accepts this flag set, and a real image's ENTRYPOINT honours docs/10 §7's
    ``/scratch`` contract."""
    task = _task(archive)
    uploaded = []
    result = _run(task, archive, cfg, upload=uploaded.append)

    assert result.exit_code == 0
    assert result.rows == 2
    rows = [json.loads(l) for l in uploaded[0].decode().splitlines()]
    assert [r["id"] for r in rows] == ["r0", "r1"]
    # Nothing survives the task (docs/11 §2.3, and `run`'s finally).
    assert not (cfg.scratch_root / task.task_id).exists()


def test_a_tagged_archive_still_resolves_to_an_image_id(archive, cfg):
    """The regression for the bug this file found on its first run.

    ``docker save`` of a tagged image makes ``docker load`` print
    ``Loaded image: name:tag`` rather than ``Loaded image ID: sha256:...``, and
    only the second was parsed -- so every archive docs/11 §1.1's upload path
    can produce would have raised ``SandboxError`` before the container ever
    started. What is *run* is still the id, never the tag."""
    job = sandbox.JobContainer(task_id="probe", config=cfg)
    job.prepare_scratch()
    try:
        path = job.fetch_archive(_download(archive), "http://store/archive",
                                 hashlib.sha256(archive).hexdigest())
        image_id = job.load_image(path)
        assert image_id.startswith("sha256:") and len(image_id) == 71
        assert TAG not in job.run_argv(image_id)
        assert image_id in job.run_argv(image_id)
    finally:
        job.cleanup()


def test_the_confinement_holds_from_inside_the_container(archive, cfg):
    """docs/11 §2.2-§2.4, asserted where it counts. ``run_argv`` carrying
    ``--network none`` is a string in a list; ``wget`` failing inside the
    container is the property. Same for the read-only rootfs and the forced
    non-root uid -- the scan only *flags* a root image (§1.3), this is the
    enforcement."""
    uploaded = []
    _run(_task(archive), archive, cfg, upload=uploaded.append)
    label = json.loads(uploaded[0].decode().splitlines()[0])["label"]

    assert "net=blocked" in label, f"the job reached the network: {label}"
    assert "root=ro" in label, f"the job wrote to its rootfs: {label}"
    assert f"uid={sandbox.CONTAINER_UID}" in label, label


def test_a_job_that_exits_non_zero_is_the_job_s_failure(archive, cfg):
    spec = _spec(params={"mode": "fail"})
    with pytest.raises(ContainedFailure) as exc:
        _run(_task(archive, spec), archive, cfg)
    assert exc.value.exit_code == 7


def test_a_soft_cancel_stops_a_real_container(archive, cfg):
    """docs/11 §3 step 3, against a job that actually traps SIGTERM. This is
    the distinction every earlier type collapsed -- and the reason the worker
    asks ``cancelled()`` before ``should_drop()``, since promoting a soft
    cancel here would SIGKILL a job mid-checkpoint."""
    spec = _spec(params={"mode": "hang"})
    task = _task(archive, spec)
    polls = {"n": 0}

    def should_stop():
        polls["n"] += 1
        return "soft" if polls["n"] > 1 else None

    with pytest.raises(ContainedCancelled) as exc:
        _run(task, archive, cfg, should_stop=should_stop)
    assert exc.value.signal == "soft"
    assert not (cfg.scratch_root / task.task_id).exists()
