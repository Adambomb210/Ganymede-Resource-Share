"""The submitter's image pipeline against archives a **real** daemon wrote.

``test_images.py`` builds its archives in memory, which is the right default --
it runs everywhere in milliseconds and covers the hostile cases a daemon will
never produce. It is also exactly the kind of test that cannot see the class of
defect this file exists to catch: a fixture emits the shape the parser already
expects, so a parser that is wrong about what Docker actually writes looks
correct forever. ``test_contained_live.py`` says the same thing about the other
half of docs/11, and found a defect in its first run.

This one did too. ``_resolve_config`` dereferenced ``index.json``'s
``manifests[0]`` and expected an image manifest there. A real multi-platform
export -- plain ``docker buildx build --platform linux/arm64,linux/amd64
--output type=oci`` -- puts a nested *index* at that position, so a perfectly
ordinary image was flagged "no readable image config" and, per docs/11 §1.4,
could never be leased. The synthetic ``oci_archive()`` fixture could not
express that shape. Neither could it express buildx's attestation manifests,
which sit in the index beside the real ones marked ``platform: unknown/unknown``.

The last test is the one the unit suite structurally cannot write: real bytes
through every real endpoint, from ``upload-url`` to a lease. Everywhere else
that chain is cut somewhere -- ``test_contained_batch.py`` sets
``scan_status = 'clean'`` with a SQL UPDATE over an ``ARCHIVE`` that is a
bytestring reading "a docker save archive, near enough", and ``test_images.py``
runs the real scan over a synthetic tar. Both halves are sound; they had just
never been joined.

Skipped, not failed, where no daemon is reachable, matching
``test_contained_live.py``. The OCI cases can skip on their own too -- the
multi-platform one needs the network for the arm64 base, and it is the case
most likely to be missing on a cold runner. That is survivable rather than a
hole, and deliberately so: the same nested-index and attestation shapes are
covered unconditionally by ``oci_multiplatform_archive()`` in
``test_images.py``, which needs no daemon at all. These tests corroborate that
the synthetic shapes match what real tooling emits; they are not the only thing
standing between the defect and a release. Every skip names its own reason, so
a case that vanishes says so.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess

import pytest

from ganymede.coordinator import images, rounds
from ganymede.coordinator.store import image_key

pytestmark = pytest.mark.slow

TAG = "ganymede-image-live:1"

# No RUN: the image is only ever read by the scanner, and a build that needs no
# execution is one qemu cannot make fail on the multi-platform export.
DOCKERFILE = """FROM busybox:1.36
USER 65534
ENTRYPOINT ["/bin/sh", "-c", "echo hi"]
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
def built(tmp_path_factory):
    """Build once, then export every shape a submitter can actually hand us.

    ``docker save`` and ``buildx --output type=oci`` do not write the same
    archive, and which one a submitter produces is a property of their tooling,
    not of their image. The OCI exports are optional -- an older buildx, or no
    network for the arm64 base -- and their absence skips the case rather than
    the module, because the ``docker save`` path is the one docs/11 §1.1 names.
    """
    binary = _runtime()
    if binary is None:
        pytest.skip("no reachable container runtime")

    build = tmp_path_factory.mktemp("image")
    (build / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8", newline="\n")

    r = subprocess.run([binary, "build", "-q", "-t", TAG, "."],
                       cwd=build, capture_output=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"could not build the test image: {r.stderr.decode()[-400:]}")

    out: dict[str, bytes | str | None] = {}

    saved = build / "save.tar"
    r = subprocess.run([binary, "save", TAG, "-o", str(saved)],
                       capture_output=True, timeout=600)
    assert r.returncode == 0, r.stderr.decode()
    out["save"] = saved.read_bytes()

    # The bottom diff_id, so the scan can come out *clean* rather than flagged
    # on a policy the deployment has not configured. RootFS.Layers is the
    # config's rootfs.diff_ids, which is what base provenance walks.
    r = subprocess.run(
        [binary, "image", "inspect", TAG, "--format", "{{index .RootFS.Layers 0}}"],
        capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode()
    out["base"] = r.stdout.decode().strip()

    for name, platforms in (("oci", None), ("multi", "linux/arm64,linux/amd64")):
        dest = build / f"{name}.tar"
        argv = [binary, "buildx", "build", "-t", TAG,
                "--output", f"type=oci,dest={dest}"]
        if platforms is not None:
            argv += ["--platform", platforms]
        r = subprocess.run(argv + ["."], cwd=build, capture_output=True, timeout=900)
        if r.returncode == 0 and dest.exists():
            out[name] = dest.read_bytes()
            out[f"{name}_why"] = None
        else:
            # Kept, so the skip says which of the two it was: an older buildx,
            # or no network for the arm64 base. A skip whose reason is "the
            # bytes are absent" tells the next reader nothing.
            out[name] = None
            out[f"{name}_why"] = (r.stderr.decode(errors="replace")[-300:]
                                  or f"exit {r.returncode}, no {dest.name}")

    return out


def _scan(payload: bytes, base: str) -> images.ScanResult:
    return images.scan_archive(io.BytesIO(payload), images.ScanLimits(
        vetted_base_diff_ids=frozenset({base})))


def _checks(result: images.ScanResult) -> dict[str, images.Check]:
    return {c.name: c for c in result.checks}


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def make_submitter(conn, make_contributor):
    def _make(name: str = "submitter"):
        cid, key = make_contributor(name=name)
        conn.execute(
            "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, 'approved', ?)",
            (cid, rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


# ==========================================================================
# The three archive shapes, as a daemon writes them
# ==========================================================================


def test_a_real_docker_save_archive_is_read(built):
    """Docker 29 with the containerd image store writes a *hybrid*: an OCI
    layout plus a legacy ``manifest.json``. The legacy member is what carries
    this archive, and it is the member Docker is progressively retiring."""
    result = _scan(built["save"], built["base"])
    assert result.status == "clean", result.detail()
    assert "docker-archive" in _checks(result)["manifest_sanity"].detail


def test_a_real_single_platform_oci_export_is_read(built):
    if built["oci"] is None:
        pytest.skip(f"buildx produced no OCI export: {built['oci_why']}")
    result = _scan(built["oci"], built["base"])
    assert result.status == "clean", result.detail()
    assert "oci-layout" in _checks(result)["manifest_sanity"].detail


def test_a_real_multi_platform_export_selects_the_amd64_variant(built):
    """The defect this file was written to find.

    ``clean`` is itself the assertion that amd64 was chosen: base provenance is
    pinned to the amd64 base's diff_id, so selecting the arm64 variant would
    flag on a base it cannot recognise even if the platform check passed.
    """
    if built["multi"] is None:
        pytest.skip(f"no multi-platform export (the arm64 base needs the "
                    f"network): {built['multi_why']}")
    result = _scan(built["multi"], built["base"])
    assert result.status == "clean", result.detail()
    assert "linux/amd64" in _checks(result)["manifest_sanity"].detail


def test_the_two_formats_agree_about_the_same_image(built):
    """A submitter's verdict must not depend on which exporter they ran.

    Same image, two archives, and the scan reads the layer count out of two
    entirely different places -- ``manifest.json``'s ``Layers`` versus the
    selected OCI manifest's ``layers``.
    """
    if built["oci"] is None:
        pytest.skip(f"buildx produced no OCI export: {built['oci_why']}")
    save = _checks(_scan(built["save"], built["base"]))
    oci = _checks(_scan(built["oci"], built["base"]))
    assert save["entrypoint"].detail == oci["entrypoint"].detail
    assert save["base_provenance"].ok and oci["base_provenance"].ok


# ==========================================================================
# Real bytes, every real endpoint, upload-url to lease
# ==========================================================================


def test_a_real_image_travels_the_whole_submitter_path_to_a_lease(
        client, conn, store, built, make_contributor, make_submitter):
    """docs/11 §1.1 -> §1.3 -> §1.4 -> docs/06's claim payload, in one piece.

    The gate is asserted in both directions against the *same* image, because
    "not leasable" is only meaningful next to the lease that follows it: the
    scan is the thing that changed, not the job, the worker or the pin.
    """
    archive: bytes = built["save"]
    digest = hashlib.sha256(archive).hexdigest()

    _, skey = make_submitter()
    _, wkey = make_contributor(name="worker-owner")

    r = client.post("/v1/images/upload-url", headers=_hdr(skey), json={
        "repo_tag": TAG, "digest": digest, "size_bytes": len(archive)})
    assert r.status_code == 200, r.text
    image_id = r.json()["image_id"]

    store.put_bytes(image_key(image_id), archive)
    r = client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(skey))
    assert r.status_code == 200, r.text
    # finalize trusts the store's own accounting, not the declaration.
    assert r.json()["size_bytes"] == len(archive)

    store.put_bytes("in/0.jsonl", b'{"id": "r0", "text": "a"}\n'
                                  b'{"id": "r1", "text": "b"}\n')
    r = client.post("/v1/jobs", headers=_hdr(skey), json={
        "job_type": "contained_batch", "image_id": image_id,
        "spec": {"shards": [{"ref": "in/0.jsonl", "rows": 2}],
                 "output_prefix": "out/live",
                 "output_schema": {"id": "str", "label": "str"},
                 "params": {}}})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/enqueue",
                       headers=_hdr(skey)).status_code == 200

    reg = client.post("/v1/workers/register", headers=_hdr(wkey), json={
        "compute_profile": {"backend": "cpu", "vram_gb": 8, "supports": ["fp32"],
                            "container_runtime": "docker"}}).json()
    claim = {"worker_id": reg["worker_id"]}

    # §1.4: unscanned code is unreachable, not merely unlikely.
    assert client.post("/v1/tasks/claim", headers=_hdr(wkey),
                       json=claim).status_code == 204

    assert images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({built["base"]}))) == [(image_id, "clean")]

    r = client.post("/v1/tasks/claim", headers=_hdr(wkey), json=claim)
    assert r.status_code == 200, r.text
    payload = r.json()

    assert payload["job_type"] == "contained_batch"
    assert payload["image_ref"] == image_id
    # docs/11 §2.3: the coordinator never hashes these bytes, so this row is the
    # only thing the worker can check its pull against. If the two ever
    # disagreed, every contained job would fail `image_digest_mismatch` at the
    # far end of a download nobody would think to blame on the coordinator.
    assert payload["image_digest"] == digest
    pulled = store.objects[image_key(image_id)]
    assert hashlib.sha256(pulled).hexdigest() == payload["image_digest"]
    assert payload["image_pull_url"] and image_id in payload["image_pull_url"]


def test_a_real_archive_is_scanned_by_the_sweep_the_way_cron_runs_it(
        client, conn, store, built, make_submitter):
    """``drain_pending`` over real bytes, including the part that matters most
    when it is wrong: the verdict is written to the row, not merely returned."""
    archive: bytes = built["save"]
    _, skey = make_submitter()
    r = client.post("/v1/images/upload-url", headers=_hdr(skey), json={
        "repo_tag": TAG, "digest": hashlib.sha256(archive).hexdigest(),
        "size_bytes": len(archive)})
    image_id = r.json()["image_id"]
    store.put_bytes(image_key(image_id), archive)
    client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(skey))

    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({built["base"]})))

    view = client.get(f"/v1/images/{image_id}", headers=_hdr(skey)).json()
    assert view["scan_status"] == "clean"
    names = [c["name"] for c in view["scan_detail"]["checks"]]
    assert names == ["manifest_sanity", "base_provenance", "secrets", "entrypoint"]
    assert view["scan_detail"]["failed"] == []
    # The scan read real layers rather than skipping them: busybox's own
    # filesystem went past the secrets sweep.
    detail = json.dumps(view["scan_detail"])
    assert "uncompressed bytes" in detail
