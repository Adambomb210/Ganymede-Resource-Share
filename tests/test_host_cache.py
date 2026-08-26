"""The HF base-model cache cap and LRU eviction (docs/02-architecture-v2.md 6.7).

Every cache tree here is built by hand under ``tmp_path`` -- no real HF
download, no network -- so these tests run in milliseconds and prove the
selection logic (which repo is oldest, whether `protect` holds, whether
`dry_run` genuinely deletes nothing) rather than anything about HF's library.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ganymede.host import cache


def _make_repo(
    hub_dir: Path,
    name: str,
    *,
    total_bytes: int = 1000,
    last_used: float | None = None,
    revision: str = "abc123",
) -> Path:
    """A minimal but structurally real HF cache repo: blobs/ + snapshots/<rev>/ + refs/main.

    The snapshot entry (a symlink into blobs/) is what real HF cache layouts
    use to expose content under a human path, and it is deliberately a symlink
    here too, so `repo_size` has something to correctly *not* double-count.

    ``total_bytes`` is the size of the whole repo as `repo_size` will report it,
    not the size of the blob -- `refs/main` is a real file and really does count
    toward the cache's footprint. Sizing the blob to compensate keeps every cap
    and byte assertion below a round number that means what it says; the
    alternative is arithmetic in each test that quietly encodes how long a
    revision hash happens to be.
    """
    repo = hub_dir / name
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    blob_file = blobs / ("0" * 40)
    blob_file.write_bytes(b"x" * (total_bytes - len(revision)))

    snap_dir = repo / "snapshots" / revision
    snap_dir.mkdir(parents=True)
    (snap_dir / "config.json").symlink_to(blob_file)

    refs = repo / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(revision)

    if last_used is not None:
        # Back-date the whole tree, not just the snapshot directory. `_last_used`
        # deliberately takes the max over several entries, because no single
        # timestamp survives every mount option -- so a helper that aged only one
        # of them would leave the others reading as "now", flatten the ordering,
        # and make the LRU assertions below pass or fail on directory-walk order.
        # A repo that really has not been touched in a year is old *everywhere*.
        for path in sorted(repo.rglob("*"), reverse=True):
            try:
                os.utime(path, (last_used, last_used), follow_symlinks=False)
            except (OSError, NotImplementedError):
                pass
        os.utime(repo, (last_used, last_used))
    return repo


# --------------------------------------------------------------------------
# model_id: the cache dirname round-tripping back to a task spec's base_model
# --------------------------------------------------------------------------


def test_model_id_round_trips_org_and_name():
    repo = cache.CachedRepo(name="models--Qwen--Qwen3-1.7B", path=Path("/x"), size_bytes=0, last_used=0)
    assert repo.model_id == "Qwen/Qwen3-1.7B"


def test_model_id_handles_no_organization():
    repo = cache.CachedRepo(name="models--gpt2", path=Path("/x"), size_bytes=0, last_used=0)
    assert repo.model_id == "gpt2"


def test_model_id_keeps_a_double_hyphen_inside_the_repo_name_intact():
    repo = cache.CachedRepo(
        name="models--meta-llama--Meta--Llama-3-8B", path=Path("/x"), size_bytes=0, last_used=0
    )
    assert repo.model_id == "meta-llama/Meta--Llama-3-8B"


# --------------------------------------------------------------------------
# repo_size: real bytes only, symlinks never double-counted
# --------------------------------------------------------------------------


def test_repo_size_counts_blobs_not_snapshot_symlinks(tmp_path):
    repo = _make_repo(tmp_path, "models--org--model", total_bytes=4096)
    assert cache.repo_size(repo) == 4096


# --------------------------------------------------------------------------
# scan: layout resolution, symlink and non-repo exclusion, missing dir
# --------------------------------------------------------------------------


def test_scan_of_a_nonexistent_cache_dir_is_empty_not_an_error(tmp_path):
    assert cache.scan(tmp_path / "does-not-exist") == []


def test_scan_resolves_hf_home_style_cache_dir(tmp_path):
    """Contributors set HF_HOME (a directory *containing* hub/)."""
    hf_home = tmp_path / "hf_home"
    hub = hf_home / "hub"
    hub.mkdir(parents=True)
    _make_repo(hub, "models--org--model")

    repos = cache.scan(hf_home)
    assert [r.name for r in repos] == ["models--org--model"]


def test_scan_resolves_huggingface_hub_cache_style_cache_dir(tmp_path):
    """Contributors also set HUGGINGFACE_HUB_CACHE (the hub/ dir itself) --
    getting this wrong must not silently mean 'cache is empty, evict nothing'."""
    hub = tmp_path / "hub_dir_directly"
    hub.mkdir()
    _make_repo(hub, "models--org--model")

    repos = cache.scan(hub)
    assert [r.name for r in repos] == ["models--org--model"]


def test_scan_ignores_non_model_entries(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    _make_repo(hub, "models--org--model")
    (hub / "datasets--org--dset").mkdir()  # not ours
    (hub / "version.txt").write_text("1")  # not even a directory

    repos = cache.scan(hub)
    assert [r.name for r in repos] == ["models--org--model"]


def test_scan_never_follows_a_symlinked_repo_out_of_the_cache(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    outside = tmp_path / "outside" / "models--org--evil"
    outside.mkdir(parents=True)
    (hub / "models--org--evil").symlink_to(outside, target_is_directory=True)

    assert cache.scan(hub) == []


# --------------------------------------------------------------------------
# evict_to_cap: LRU order, protect, dry_run
# --------------------------------------------------------------------------


def _built_cache(tmp_path) -> Path:
    hub = tmp_path / "hub"
    hub.mkdir()
    # Oldest to newest, distinct sizes so eviction order is unambiguous.
    _make_repo(hub, "models--org--oldest", total_bytes=10_000, last_used=100.0)
    _make_repo(hub, "models--org--middle", total_bytes=10_000, last_used=200.0)
    _make_repo(hub, "models--org--newest", total_bytes=10_000, last_used=300.0)
    return hub


def test_eviction_picks_the_least_recently_used_repo_first(tmp_path):
    hub = _built_cache(tmp_path)
    total = 30_000
    # Cap requires freeing exactly one repo's worth.
    result = cache.evict_to_cap(hub, cap_bytes=total - 10_000)

    assert [r.name for r in result.removed] == ["models--org--oldest"]
    assert result.bytes_freed == 10_000
    assert result.size_before == total
    assert result.size_after == total - 10_000
    assert not (hub / "models--org--oldest").exists()
    assert (hub / "models--org--middle").exists()
    assert (hub / "models--org--newest").exists()


def test_eviction_stops_as_soon_as_it_is_under_the_cap(tmp_path):
    hub = _built_cache(tmp_path)
    # Cap already satisfied -- nothing should be touched.
    result = cache.evict_to_cap(hub, cap_bytes=30_000)
    assert result.removed == []
    assert result.bytes_freed == 0
    for name in ("oldest", "middle", "newest"):
        assert (hub / f"models--org--{name}").exists()


def test_eviction_removes_multiple_repos_when_needed(tmp_path):
    hub = _built_cache(tmp_path)
    # 30_000 in three equal repos. A 15_000 cap needs two of them gone: dropping
    # only the oldest leaves 20_000, still over.
    result = cache.evict_to_cap(hub, cap_bytes=15_000)
    assert [r.name for r in result.removed] == ["models--org--oldest", "models--org--middle"]
    assert result.size_after == 10_000


def test_protect_is_honoured_even_when_it_is_the_lru_repo(tmp_path):
    hub = _built_cache(tmp_path)
    result = cache.evict_to_cap(hub, cap_bytes=5_000, protect=frozenset({"org/oldest"}))

    # The oldest repo is protected, so eviction has to reach past it into the
    # next-oldest unprotected ones to make room.
    assert "models--org--oldest" not in [r.name for r in result.removed]
    assert (hub / "models--org--oldest").exists()
    assert [r.name for r in result.removed] == ["models--org--middle", "models--org--newest"]


def test_dry_run_deletes_nothing_but_reports_the_same_selection(tmp_path):
    hub = _built_cache(tmp_path)
    cap = 5_000
    dry = cache.evict_to_cap(hub, cap_bytes=cap, dry_run=True)
    real = cache.evict_to_cap(hub, cap_bytes=cap, dry_run=False)

    assert [r.name for r in dry.removed] == [r.name for r in real.removed]
    assert dry.bytes_freed == real.bytes_freed


def test_dry_run_leaves_every_repo_on_disk(tmp_path):
    hub = _built_cache(tmp_path)
    cache.evict_to_cap(hub, cap_bytes=0, dry_run=True)
    for name in ("oldest", "middle", "newest"):
        assert (hub / f"models--org--{name}").exists()


def test_eviction_on_an_empty_cache_is_a_no_op(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    result = cache.evict_to_cap(hub, cap_bytes=100)
    assert result.removed == []
    assert result.size_before == 0
    assert result.size_after == 0


# --------------------------------------------------------------------------
# free_disk_bytes
# --------------------------------------------------------------------------


def test_free_disk_bytes_on_an_existing_path_is_positive(tmp_path):
    assert cache.free_disk_bytes(tmp_path) > 0


def test_free_disk_bytes_walks_up_to_an_existing_ancestor(tmp_path):
    missing = tmp_path / "not" / "yet" / "created"
    assert cache.free_disk_bytes(missing) == cache.free_disk_bytes(tmp_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_dry_run_json_reports_without_deleting(tmp_path, capsys):
    hub = _built_cache(tmp_path)
    rc = cache.main(["--cache-dir", str(hub), "--cap-gb", "0.00002", "--dry-run", "--json"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert "org/oldest" in out["removed"]
    for name in ("oldest", "middle", "newest"):
        assert (hub / f"models--org--{name}").exists()


def test_cli_protect_flag_is_honoured(tmp_path, capsys):
    hub = _built_cache(tmp_path)
    cache.main([
        "--cache-dir", str(hub), "--cap-gb", "0.00001", "--dry-run",
        "--protect", "org/oldest", "--json",
    ])
    out = json.loads(capsys.readouterr().out)
    assert "org/oldest" not in out["removed"]
