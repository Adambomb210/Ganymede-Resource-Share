"""The image pipeline: upload -> finalize -> scan -> gate (docs/11 §1).

Two halves. The scan half builds real ``docker save`` archives in memory and
asserts the four §1.3 checks against them -- including the one that has to hold
under an actively hostile archive, the decompression bomb, where the assertion
is not just "flagged" but "flagged without reading the bomb".

The gate half is the load-bearing one: docs/11 §1.4 says a job pinned to an
image that is not ``clean`` is never leased. That is asserted through the real
claim endpoint with a real worker, not by calling the selector directly -- the
selector is an implementation detail and a future fair-share is expected to
rewrite it, while "no lease against an unscanned image" must survive that.
"""

from __future__ import annotations

import gzip
import io
import json
import tarfile
import uuid

import pytest

from ganymede.coordinator import images, rounds
from ganymede.coordinator.store import image_key
from tests.fake_worker import FakeWorker

VETTED = "sha256:" + "a1" * 32
OTHER_BASE = "sha256:" + "b2" * 32


# --------------------------------------------------------------------------
# archive builders
# --------------------------------------------------------------------------


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    """A tar of ``path -> content``, as a layer blob is."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in entries.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _config(diff_ids: list[str] | None = None, *, os_name: str = "linux",
            arch: str = "amd64", entrypoint: list[str] | None = None,
            cmd: list[str] | None = None, user: str = "1000") -> bytes:
    cfg = {
        "architecture": arch,
        "os": os_name,
        "config": {
            "Entrypoint": entrypoint if entrypoint is not None else ["/bin/run"],
            "User": user,
        },
        "rootfs": {"type": "layers", "diff_ids": diff_ids or [VETTED]},
    }
    if cmd is not None:
        cfg["config"]["Cmd"] = cmd
    return json.dumps(cfg).encode()


def docker_archive(*, layers: list[bytes] | None = None, **cfg_kwargs) -> bytes:
    """A classic ``docker save`` archive: manifest.json + config + layer tars."""
    layers = layers if layers is not None else [_tar_bytes({"app/main.py": b"print(1)"})]
    members: dict[str, bytes] = {}
    layer_names = []
    for i, blob in enumerate(layers):
        name = f"layer{i}/layer.tar"
        members[name] = blob
        layer_names.append(name)
    members["cfg.json"] = _config(**cfg_kwargs)
    members["manifest.json"] = json.dumps([
        {"Config": "cfg.json", "RepoTags": ["job:latest"], "Layers": layer_names}
    ]).encode()
    return _tar_bytes(members)


def oci_archive(**cfg_kwargs) -> bytes:
    """An OCI layout, which is what ``docker save`` emits under the containerd
    image store. Same four checks, different indirection."""
    layer = gzip.compress(_tar_bytes({"app/main.py": b"print(1)"}))
    cfg = _config(**cfg_kwargs)
    layer_digest = "sha256:" + "c3" * 32
    cfg_digest = "sha256:" + "d4" * 32
    manifest = json.dumps({
        "schemaVersion": 2,
        "config": {"digest": cfg_digest, "size": len(cfg)},
        "layers": [{"digest": layer_digest, "size": len(layer)}],
    }).encode()
    man_digest = "sha256:" + "e5" * 32
    members = {
        "oci-layout": b'{"imageLayoutVersion": "1.0.0"}',
        "index.json": json.dumps(
            {"schemaVersion": 2, "manifests": [{"digest": man_digest}]}).encode(),
        f"blobs/sha256/{man_digest.split(':')[1]}": manifest,
        f"blobs/sha256/{cfg_digest.split(':')[1]}": cfg,
        f"blobs/sha256/{layer_digest.split(':')[1]}": layer,
    }
    return _tar_bytes(members)


def _scan(payload: bytes, **limit_kwargs) -> images.ScanResult:
    limits = images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED}), **limit_kwargs)
    return images.scan_archive(io.BytesIO(payload), limits)


def _checks(result: images.ScanResult) -> dict[str, images.Check]:
    return {c.name: c for c in result.checks}


# ==========================================================================
# The scan (docs/11 §1.3)
# ==========================================================================


def test_a_well_formed_archive_is_clean():
    result = _scan(docker_archive())
    assert result.status == "clean", result.detail()


def test_oci_layout_is_read_too():
    """A modern ``docker save`` emits an OCI layout rather than manifest.json.
    Flagging every such image as unreadable would be a scan that mostly reports
    on Docker versions."""
    result = _scan(oci_archive())
    assert result.status == "clean", result.detail()


def test_unrecognised_base_is_flagged_not_denied():
    result = _scan(docker_archive(diff_ids=[OTHER_BASE]))
    assert result.status == "flagged"
    assert "unrecognised base" in _checks(result)["base_provenance"].detail


def test_no_vetted_set_configured_flags_everything():
    """Fail-closed: with no vetted base set there is no such thing as a
    recognised base, so an operator who has not configured one dispositions by
    hand rather than getting a silent pass."""
    result = images.scan_archive(io.BytesIO(docker_archive()), images.ScanLimits())
    assert result.status == "flagged"
    assert "no vetted base set" in _checks(result)["base_provenance"].detail


def test_secret_by_filename_is_found():
    layer = _tar_bytes({"root/.ssh/id_rsa": b"whatever", "app/main.py": b"x"})
    result = _scan(docker_archive(layers=[layer]))
    assert result.status == "flagged"
    secrets = _checks(result)["secrets"]
    assert any("private ssh key" in f for f in secrets.findings)


def test_secret_by_content_is_found():
    layer = _tar_bytes({"app/settings.py": b"KEY = 'AKIAIOSFODNN7EXAMPLE'"})
    result = _scan(docker_archive(layers=[layer]))
    assert result.status == "flagged"
    assert any("aws access key" in f for f in _checks(result)["secrets"].findings)


def test_whiteout_markers_are_not_findings():
    """``.wh.`` entries are deletion markers from a lower layer, not files. An
    image that *removes* an id_rsa must not be flagged for having had one."""
    layer = _tar_bytes({"root/.ssh/.wh.id_rsa": b"", "app/main.py": b"x"})
    result = _scan(docker_archive(layers=[layer]))
    assert _checks(result)["secrets"].ok, _checks(result)["secrets"].findings


def test_wrong_platform_is_flagged():
    result = _scan(docker_archive(os_name="windows", arch="arm64"))
    assert result.status == "flagged"
    assert "not linux/amd64" in _checks(result)["manifest_sanity"].detail


def test_missing_entrypoint_is_flagged():
    result = _scan(docker_archive(entrypoint=[]))
    assert result.status == "flagged"
    assert "neither ENTRYPOINT nor CMD" in _checks(result)["entrypoint"].detail


def test_root_user_is_flagged_but_cmd_alone_is_enough():
    result = _scan(docker_archive(entrypoint=[], cmd=["/bin/sh"], user="root"))
    entry = _checks(result)["entrypoint"]
    assert not entry.ok
    assert "expects root" in entry.detail


def test_garbage_is_flagged_rather_than_raising():
    result = _scan(b"this is not a tar archive at all")
    assert result.status == "flagged"
    assert not _checks(result)["manifest_sanity"].ok


def test_decompression_bomb_trips_the_budget_mid_read():
    """The guard that has to hold. A gzipped layer that unpacks to far more
    than the cap must be flagged *without* the scan reading it out: the budget
    is counted on the decompressed side, so the trip happens inside ``read``.

    One megabyte of zeros compresses to about a kilobyte, so an archive well
    under any size limit expands past a 64 KiB cap. The assertion is the
    verdict; the proof that nothing was buffered is that this test returns at
    all under a cap this small.
    """
    bomb = gzip.compress(_tar_bytes({"big": b"\0" * (1024 * 1024)}))
    payload = docker_archive(layers=[bomb])
    result = _scan(payload, max_uncompressed_bytes=64 * 1024)
    assert result.status == "flagged"
    assert "uncompressed content exceeded" in _checks(result)["manifest_sanity"].detail


def test_too_many_layers_is_flagged():
    layers = [_tar_bytes({f"f{i}": b"x"}) for i in range(5)]
    result = _scan(docker_archive(layers=layers), max_layers=3)
    assert result.status == "flagged"
    assert _checks(result)["manifest_sanity"].detail == (
        "5 layers exceeds the 3 layer cap")


@pytest.mark.parametrize("manifest", [
    [{"Config": 123, "Layers": "not-a-list"}],
    [{"Config": "cfg.json", "Layers": [{"nested": "object"}]}],
    {"not": "even a list"},
    [[]],
])
def test_hostile_manifest_types_are_a_verdict_not_a_raise(manifest):
    """A well-formed tar carrying malformed JSON must not raise. The sweep
    scans a batch, so an archive that breaks the scanner would otherwise take
    every other pending image down with it -- and it is exactly the archive a
    human should be looking at."""
    members = {
        "layer0/layer.tar": _tar_bytes({"app/main.py": b"x"}),
        "cfg.json": _config(),
        "manifest.json": json.dumps(manifest).encode(),
    }
    result = _scan(_tar_bytes(members))
    assert result.status == "flagged"


def test_a_config_that_is_not_an_object_is_a_verdict_not_a_raise():
    members = {
        "cfg.json": b'"a string, not an object"',
        "manifest.json": json.dumps(
            [{"Config": "cfg.json", "Layers": []}]).encode(),
    }
    result = _scan(_tar_bytes(members))
    assert result.status == "flagged"


def test_absolute_paths_are_reported():
    """Nothing here extracts, so a traversal cannot happen -- but a build that
    emits one is broken in a way the submitter wants told about."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
        for name, content in {"cfg.json": _config(),
                              "manifest.json": json.dumps([
                                  {"Config": "cfg.json", "Layers": []}]).encode()}.items():
            i2 = tarfile.TarInfo(name)
            i2.size = len(content)
            tar.addfile(i2, io.BytesIO(content))
    result = _scan(buf.getvalue())
    assert result.status == "flagged"
    assert "traversing paths" in _checks(result)["manifest_sanity"].detail


# ==========================================================================
# Fixtures for the API half
# ==========================================================================


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


@pytest.fixture
def make_admin(conn):
    def _make(name: str = "admin"):
        from ganymede.coordinator.auth import generate_key, hash_key

        cid, key = uuid.uuid4().hex, generate_key()
        conn.execute(
            """INSERT INTO contributors
                 (id, name, key_hash, enabled, clearance, is_admin, created_at)
               VALUES (?, ?, ?, 1, 'open', 1, ?)""",
            (cid, name, hash_key(key), rounds._iso(rounds.utcnow())),
        )
        conn.commit()
        return cid, key
    return _make


def _hdr(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _batch_spec() -> dict:
    """A valid ``batch_inference`` spec. The job type validates its own spec
    before the image pin is looked at, so an empty one would 422 for the wrong
    reason and prove nothing about the pin."""
    return {
        "model_ref": "hf://test-model",
        "shards": [{"ref": "s0", "rows": 4096}],
        "output_prefix": "out/run1",
        "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
    }


def _upload(client, key: str, payload: bytes, store, *, finalize: bool = True) -> str:
    """Walk the real endpoints: upload-url, put the bytes, finalize."""
    import hashlib

    body = {"repo_tag": "job:latest",
            "digest": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}
    r = client.post("/v1/images/upload-url", json=body, headers=_hdr(key))
    assert r.status_code == 200, r.text
    image_id = r.json()["image_id"]
    store.put_bytes(image_key(image_id), payload)
    if finalize:
        r = client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(key))
        assert r.status_code == 200, r.text
    return image_id


# ==========================================================================
# Upload and finalize (docs/11 §1.1)
# ==========================================================================


def test_upload_url_requires_an_approved_submitter(client, make_contributor):
    _, key = make_contributor()
    r = client.post("/v1/images/upload-url", headers=_hdr(key),
                    json={"repo_tag": "x:1", "digest": "a" * 64, "size_bytes": 10})
    assert r.status_code == 404


def test_upload_url_rejects_a_non_digest(client, make_submitter):
    _, key = make_submitter()
    r = client.post("/v1/images/upload-url", headers=_hdr(key),
                    json={"repo_tag": "x:1", "digest": "not-a-digest", "size_bytes": 10})
    assert r.status_code == 422


def test_the_presigned_put_carries_the_length_ceiling():
    """The upload ceiling rides on the signature, so it has to be *in* the
    signature. boto3 signs offline, so this needs no MinIO -- what it cannot
    show is that a given store enforces the signed header, which is why
    finalize re-checks with a HEAD and is the guard that actually holds."""
    from urllib.parse import parse_qs, urlparse

    from ganymede.coordinator.config import StorageConfig
    from ganymede.coordinator.store import Store

    real = Store(StorageConfig(endpoint_url="http://storage.test:9000",
                               bucket="ganymede", region="us-east-1",
                               access_key="k", secret_key="s"))
    url, _ = real.presign_put(image_key("img1"), content_length=4096)
    signed = parse_qs(urlparse(url).query)["X-Amz-SignedHeaders"][0]
    assert "content-length" in signed

    plain, _ = real.presign_put("runs/x/base.safetensors")
    assert "content-length" not in parse_qs(
        urlparse(plain).query)["X-Amz-SignedHeaders"][0]


def test_upload_url_rejects_an_oversized_declaration(client, make_submitter, settings):
    _, key = make_submitter()
    r = client.post("/v1/images/upload-url", headers=_hdr(key),
                    json={"repo_tag": "x:1", "digest": "a" * 64,
                          "size_bytes": settings.image_max_bytes + 1})
    assert r.status_code == 422
    assert "cap" in r.json()["detail"]


def test_the_coordinator_signs_the_declared_length(client, store, make_submitter):
    _, key = make_submitter()
    payload = docker_archive()
    image_id = _upload(client, key, payload, store, finalize=False)
    assert store.signed_lengths[image_key(image_id)] == len(payload)


def test_a_row_is_not_worker_visible_until_finalize(client, conn, store, make_submitter):
    _, key = make_submitter()
    image_id = _upload(client, key, docker_archive(), store, finalize=False)
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    assert row["finalized_at"] is None
    assert row["scan_status"] == "pending"

    client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(key))
    row = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    assert row["finalized_at"] is not None


def test_finalize_without_an_upload_is_422_and_leaves_the_row_open(
        client, conn, make_submitter):
    """The coordinator never sees the body, so it asks the store what landed.
    An interrupted upload must not produce a finalized row."""
    _, key = make_submitter()
    r = client.post("/v1/images/upload-url", headers=_hdr(key),
                    json={"repo_tag": "x:1", "digest": "a" * 64, "size_bytes": 10})
    image_id = r.json()["image_id"]
    r = client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(key))
    assert r.status_code == 422
    row = conn.execute("SELECT finalized_at FROM images WHERE id = ?",
                       (image_id,)).fetchone()
    assert row["finalized_at"] is None


def test_finalize_is_not_repeatable(client, store, make_submitter):
    """An images row is immutable once finalized (§1.2): a rebuild is a new
    row with a new digest, never a second finalize over the same one."""
    _, key = make_submitter()
    image_id = _upload(client, key, docker_archive(), store)
    r = client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(key))
    assert r.status_code == 409


def test_finalize_refuses_a_body_over_the_cap(client, store, conn, make_submitter,
                                              settings, monkeypatch):
    """The signed content-length is the first guard; this is the second. A
    store that ignores the signature still cannot produce a schedulable row."""
    _, key = make_submitter()
    image_id = _upload(client, key, docker_archive(), store, finalize=False)
    monkeypatch.setattr(
        store, "head",
        lambda key_: {"size": settings.image_max_bytes + 1, "etag": "x"})
    r = client.post(f"/v1/images/{image_id}/finalize", headers=_hdr(key))
    assert r.status_code == 422
    row = conn.execute("SELECT finalized_at FROM images WHERE id = ?",
                       (image_id,)).fetchone()
    assert row["finalized_at"] is None


def test_a_submitter_sees_only_their_own_images(client, store, make_submitter):
    _, key_a = make_submitter("alice")
    _, key_b = make_submitter("bob")
    mine = _upload(client, key_a, docker_archive(), store)
    _upload(client, key_b, docker_archive(), store)

    listed = client.get("/v1/images", headers=_hdr(key_a)).json()["images"]
    assert [i["image_id"] for i in listed] == [mine]
    assert client.get(f"/v1/images/{mine}", headers=_hdr(key_b)).status_code == 404


def test_a_job_cannot_pin_someone_elses_image(client, store, make_submitter):
    """A pin that can never be leased is a 422 at submit rather than a job
    that sits in the queue forever."""
    _, key_a = make_submitter("alice")
    _, key_b = make_submitter("bob")
    theirs = _upload(client, key_a, docker_archive(), store)
    r = client.post("/v1/jobs", headers=_hdr(key_b), json={
        "job_type": "batch_inference", "spec": _batch_spec(), "image_id": theirs})
    assert r.status_code == 422
    assert "unknown image" in r.json()["detail"]


# ==========================================================================
# The sweep and the admin disposition
# ==========================================================================


def test_the_sweep_scans_finalized_images_only(client, conn, store, make_submitter):
    _, key = make_submitter()
    open_upload = _upload(client, key, docker_archive(), store, finalize=False)
    finalized = _upload(client, key, docker_archive(), store)

    done = images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))
    assert done == [(finalized, "clean")]
    row = conn.execute("SELECT * FROM images WHERE id = ?", (open_upload,)).fetchone()
    assert row["scan_status"] == "pending" and row["scanned_at"] is None


def test_a_verdict_lands_on_the_row_with_its_reasons(client, conn, store,
                                                     make_submitter):
    _, key = make_submitter()
    image_id = _upload(client, key, docker_archive(diff_ids=[OTHER_BASE]), store)
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    view = client.get(f"/v1/images/{image_id}", headers=_hdr(key)).json()
    assert view["scan_status"] == "flagged"
    assert view["scan_detail"]["failed"] == ["base_provenance"]


def test_unreadable_bytes_leave_the_row_pending_for_the_next_sweep(
        client, conn, store, make_submitter):
    """"We could not look" is not "we looked and it was bad": a store blip must
    not turn into a verdict a human then has to overturn."""
    _, key = make_submitter()
    image_id = _upload(client, key, docker_archive(), store)
    store.objects.pop(image_key(image_id))

    assert images.drain_pending(conn, store) == []
    row = conn.execute("SELECT scan_status FROM images WHERE id = ?",
                       (image_id,)).fetchone()
    assert row["scan_status"] == "pending"


def test_only_an_admin_dispositions(client, store, make_submitter):
    _, key = make_submitter()
    image_id = _upload(client, key, docker_archive(), store)
    r = client.post(f"/v1/admin/images/{image_id}/scan", headers=_hdr(key),
                    json={"disposition": "clean"})
    assert r.status_code == 404


def test_disposition_records_the_human_beside_the_machine(client, conn, store,
                                                          make_submitter, make_admin):
    """An override does not erase what the scan found -- ``checks`` stays put
    and the disposition lands next to it, with who and why."""
    _, key = make_submitter()
    admin_id, admin_key = make_admin()
    image_id = _upload(client, key, docker_archive(diff_ids=[OTHER_BASE]), store)
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    r = client.post(f"/v1/admin/images/{image_id}/scan", headers=_hdr(admin_key),
                    json={"disposition": "clean", "note": "base is ours, unpinned"})
    assert r.status_code == 200 and r.json()["scan_status"] == "clean"

    view = client.get(f"/v1/images/{image_id}", headers=_hdr(key)).json()
    assert view["scan_status"] == "clean"
    assert view["scan_detail"]["failed"] == ["base_provenance"]
    assert view["scan_detail"]["disposition"]["by"] == admin_id
    assert view["scan_detail"]["disposition"]["note"] == "base is ours, unpinned"
    audit = conn.execute(
        "SELECT * FROM audit WHERE event = 'image_scan_disposition'").fetchone()
    assert json.loads(audit["detail_json"])["image_id"] == image_id


def test_rescan_puts_the_image_back_in_the_queue(client, conn, store,
                                                 make_submitter, make_admin):
    _, key = make_submitter()
    _, admin_key = make_admin()
    image_id = _upload(client, key, docker_archive(), store)
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    r = client.post(f"/v1/admin/images/{image_id}/scan", headers=_hdr(admin_key),
                    json={"disposition": "rescan"})
    assert r.json()["scan_status"] == "pending"
    assert images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED}))) == [(image_id, "clean")]


# ==========================================================================
# The gate (docs/11 §1.4) -- the load-bearing assertion
# ==========================================================================


def _pin_image(conn, run_id: str, image_id: str | None) -> None:
    conn.execute(
        "UPDATE jobs SET image_id = ? WHERE id = (SELECT job_id FROM runs WHERE id = ?)",
        (image_id, run_id),
    )
    conn.commit()


def test_a_pending_image_yields_no_lease(client, conn, store, make_contributor,
                                         make_submitter, seeded_run):
    """The whole point of §1.4: unscanned code is not merely unlikely to be
    picked, it is unreachable. Asserted through the claim endpoint, because
    that is the promise -- not through the selector, which fair-share will
    one day rewrite."""
    _, sub_key = make_submitter()
    seeded_run(run_id="pinned-job")
    image_id = _upload(client, sub_key, docker_archive(), store)
    _pin_image(conn, "pinned-job", image_id)

    _, key = make_contributor()
    assert FakeWorker(client, store, key).claim() is None


def test_a_flagged_image_yields_no_lease(client, conn, store, make_contributor,
                                         make_submitter, seeded_run):
    _, sub_key = make_submitter()
    seeded_run(run_id="pinned-job")
    image_id = _upload(client, sub_key, docker_archive(diff_ids=[OTHER_BASE]), store)
    _pin_image(conn, "pinned-job", image_id)
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    _, key = make_contributor()
    assert FakeWorker(client, store, key).claim() is None


def test_a_clean_image_is_leased(client, conn, store, make_contributor,
                                 make_submitter, seeded_run):
    """The other half of the gate: it has to open. Same job, same worker, one
    scan later."""
    _, sub_key = make_submitter()
    seeded_run(run_id="pinned-job")
    image_id = _upload(client, sub_key, docker_archive(), store)
    _pin_image(conn, "pinned-job", image_id)

    _, key = make_contributor()
    fw = FakeWorker(client, store, key, container_runtime="docker")
    assert fw.claim() is None

    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))
    assert fw.claim() is not None


def test_an_admin_disposition_opens_the_gate(client, conn, store, make_contributor,
                                             make_submitter, make_admin, seeded_run):
    _, sub_key = make_submitter()
    _, admin_key = make_admin()
    seeded_run(run_id="pinned-job")
    image_id = _upload(client, sub_key, docker_archive(diff_ids=[OTHER_BASE]), store)
    _pin_image(conn, "pinned-job", image_id)
    images.drain_pending(conn, store, images.ScanLimits(
        vetted_base_diff_ids=frozenset({VETTED})))

    _, key = make_contributor()
    fw = FakeWorker(client, store, key, container_runtime="docker")
    assert fw.claim() is None

    client.post(f"/v1/admin/images/{image_id}/scan", headers=_hdr(admin_key),
                json={"disposition": "clean", "note": "reviewed"})
    assert fw.claim() is not None


def test_a_dangling_image_reference_fails_closed(client, conn, store,
                                                 make_contributor, seeded_run):
    """A job whose image row has vanished is not treated as a built-in. Failing
    closed costs an operator a puzzled look; failing open runs unscanned code."""
    seeded_run(run_id="pinned-job")
    # The foreign key makes this unreachable through the API -- which is why
    # the pragma comes off to write it. What is being tested is the selector's
    # behaviour against a database that holds one anyway: a restore from a
    # backup taken mid-GC, or a hand-edited row.
    conn.execute("PRAGMA foreign_keys = OFF")
    _pin_image(conn, "pinned-job", "no-such-image")
    conn.execute("PRAGMA foreign_keys = ON")

    _, key = make_contributor()
    assert FakeWorker(client, store, key).claim() is None


def test_a_built_in_job_is_untouched_by_the_gate(client, store, make_contributor,
                                                 seeded_run):
    """docs/11 §4: first-party job types carry ``image_id IS NULL`` and take
    none of this path."""
    seeded_run(run_id="builtin")
    _, key = make_contributor()
    assert FakeWorker(client, store, key).claim() is not None
