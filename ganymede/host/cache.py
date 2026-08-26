"""The HF base-model cache: size, and keeping it under a cap (docs/02-architecture-v2.md 6.7).

Ten runs across five base models means a contributor's disk fills quietly
until something breaks at an unhelpful moment (6.7). This module is the fix:
a scan of what is cached, and an LRU eviction that keeps the cache under a
configured size without ever leaving a base model half-downloaded.

Standard library only, like the rest of ``ganymede/host`` -- this runs on the
contributor's machine, outside any container, where nothing but a Python
interpreter is guaranteed to exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from ganymede.host.config import DEFAULT_CACHE_CAP_GB, default_cache_dir

# HF's own cache-dir prefix for model repos (as opposed to `datasets--*` or
# `spaces--*`, which this cache never holds -- 4.1's cache is base models only
# -- but excluding them here means a contributor who also uses `huggingface-cli`
# for other things doesn't get their dataset cache mistaken for ours).
_REPO_PREFIX = "models--"


@dataclass(frozen=True)
class CachedRepo:
    """One base model in the cache.

    ``model_id`` is what makes ``protect`` usable: a caller passes the exact
    string a task spec (8) calls ``base_model`` -- ``"Qwen/Qwen3-1.7B"`` -- and
    never needs to know HF's cache-dir naming to protect it from eviction.
    """

    name: str  # e.g. "models--Qwen--Qwen3-1.7B", the cache dirname
    path: Path
    size_bytes: int
    last_used: float  # epoch seconds

    @property
    def model_id(self) -> str:
        """``models--Qwen--Qwen3-1.7B`` -> ``Qwen/Qwen3-1.7B``.

        Split on the *first* ``--`` only. HF's own construction is
        ``"models--" + org + "--" + name``, and while ``org`` never contains a
        literal ``--`` (HF namespace rules forbid it), ``name`` sometimes does
        (e.g. ``Meta-Llama-3-8B`` variants with a doubled hyphen in the wild) --
        splitting greedily on the first occurrence only keeps the rest of
        ``name`` intact instead of mangling it. A repo with no org
        (``models--gpt2``) has no second ``--`` at all and comes back as
        ``"gpt2"``, which matches how such models are actually referred to.
        """
        rest = self.name[len(_REPO_PREFIX):] if self.name.startswith(_REPO_PREFIX) else self.name
        org, sep, repo = rest.partition("--")
        return f"{org}/{repo}" if sep else org


@dataclass(frozen=True)
class EvictionResult:
    removed: list[CachedRepo]
    bytes_freed: int
    size_before: int
    size_after: int
    dry_run: bool


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def _resolve_hub_dir(cache_dir: Path) -> Path:
    """Contributors set ``HF_HOME`` and ``HUGGINGFACE_HUB_CACHE`` interchangeably.

    ``HF_HOME`` points at a directory *containing* ``hub/``; the more specific
    ``HUGGINGFACE_HUB_CACHE`` points at ``hub/`` itself. ``config.cache_dir``
    inherits whichever one a contributor happened to already have set (see
    ``default_cache_dir``), so guessing wrong here must not silently mean "the
    cache is empty, evict nothing" -- that reads as "cache management is
    broken" rather than "config points somewhere unexpected". If a ``hub``
    subdirectory exists, ``cache_dir`` was the HF_HOME form; otherwise assume
    ``cache_dir`` already *is* the hub directory.
    """
    hub = cache_dir / "hub"
    return hub if hub.is_dir() else cache_dir


def repo_size(path: str | Path) -> int:
    """Total bytes of real content under one cached repo directory.

    HF's layout stores each blob once under ``blobs/`` and exposes it under
    ``snapshots/<revision>/...`` as a symlink. Summing every file in the tree
    would double-count (or, if symlinks were resolved, count target sizes
    through paths that are not really "this repo's" own bytes). Counting only
    regular, non-symlink files sums the blobs exactly once and skips the
    snapshot/ref symlinks entirely, without needing to know HF's internal
    directory names or trust that they won't shift between library versions.
    """
    root = Path(path)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
            except OSError:
                # Deleted or permission-denied mid-walk. Skip it rather than
                # fail the whole scan over one file that will not be counted
                # either way.
                continue
    return total


def _last_used(repo_path: Path) -> float:
    """Best-effort last-use time for one cached repo (6.7's LRU eviction).

    ``st_atime`` alone is not trustworthy: `relatime` (the Linux default)
    updates it at most once a day or when it is older than mtime, and `noatime`
    never updates it at all. Either mount option makes a plain atime-based LRU
    pick the wrong repo. So this takes mtime everywhere -- a download sets it,
    and no mount option disables it -- and *adds* atime only where atime is
    meaningful.

    **Atime counts for regular files only, and this is load-bearing.** Reading
    a directory updates that directory's atime, and resolving a symlink updates
    the link's; walking the cache to size it does both. A version of this that
    trusted either would stamp every repo with the time of the last eviction
    scan, flatten the LRU ordering completely, and then evict essentially at
    random while still looking like it worked -- the scan would be destroying
    the signal it was reading. `lstat` on a regular file updates nothing, and
    nothing here opens file contents, so regular-file atime survives the scan.

    Conveniently it is also the signal that matters: `from_pretrained` resolves
    a snapshot symlink and opens the **blob** behind it, so the blob's atime is
    what actually separates "loaded yesterday" from "downloaded a year ago and
    never touched since". That is why this walks the whole repo rather than
    just ``snapshots/`` and ``refs/`` -- on a `noatime` mount it degrades to
    download time, which is the best any filesystem-based answer can do.
    """
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(repo_path, followlinks=False):
        for name in (*dirnames, *filenames):
            try:
                st = (Path(dirpath) / name).lstat()
            except OSError:
                continue
            newest = max(newest, st.st_mtime)
            if stat.S_ISREG(st.st_mode):
                newest = max(newest, st.st_atime)

    if newest == 0.0:
        # An empty repo directory -- an interrupted download that got as far as
        # mkdir and no further. Fall back to the directory's own mtime so it
        # sorts as "recently touched, never finished" rather than as epoch zero,
        # which would make it look like the oldest thing in the cache and
        # monopolize eviction priority forever.
        try:
            newest = repo_path.stat().st_mtime
        except OSError:
            newest = 0.0
    return newest


def scan(cache_dir: str | Path) -> list[CachedRepo]:
    """Every base model currently in the cache. Never raises.

    A cache directory that does not exist yet -- the common case on a
    contributor's very first run, before anything has been downloaded -- comes
    back as an empty list, not an error.
    """
    hub_dir = _resolve_hub_dir(Path(cache_dir))
    if not hub_dir.is_dir():
        return []

    repos = []
    try:
        entries = sorted(hub_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        # Never follow a symlink out of the cache, and never touch anything
        # that is not a model repo directory HF itself created.
        if entry.is_symlink() or not entry.is_dir():
            continue
        if not entry.name.startswith(_REPO_PREFIX):
            continue
        repos.append(
            CachedRepo(
                name=entry.name,
                path=entry,
                size_bytes=repo_size(entry),
                last_used=_last_used(entry),
            )
        )
    return repos


# --------------------------------------------------------------------------
# Eviction
# --------------------------------------------------------------------------


def _remove_repo(path: Path) -> None:
    """Delete one cached repo, defensively.

    Two checks stand between this and ``shutil.rmtree`` reaching somewhere it
    shouldn't: the path must actually be a ``models--*`` directory (never
    trust a caller-constructed ``CachedRepo`` blindly), and it must not itself
    be a symlink -- ``rmtree`` on a symlinked directory deletes through the
    link, which could land outside the cache entirely.
    """
    if path.is_symlink() or not path.is_dir() or not path.name.startswith(_REPO_PREFIX):
        return
    shutil.rmtree(path, ignore_errors=True)


def evict_to_cap(
    cache_dir: str | Path,
    cap_bytes: int,
    *,
    protect: frozenset[str] = frozenset(),
    dry_run: bool = False,
) -> EvictionResult:
    """Evict whole repos, least-recently-used first, until under ``cap_bytes``.

    Whole-repo granularity only (6.7): a half-deleted model is not a smaller
    model, it is a corrupt one the next run re-downloads anyway, so partial
    eviction buys nothing but wasted bandwidth. ``protect`` holds model ids in
    ``org/name`` form -- exactly a task spec's ``base_model`` string (8) -- so
    a caller never needs to know HF's cache-dir naming to keep a run's active
    base model from being evicted out from under it.

    ``dry_run`` walks the identical selection logic and only skips the actual
    delete, so "what would be freed" and "what gets freed" can never diverge.
    """
    repos = scan(cache_dir)
    size_before = sum(r.size_bytes for r in repos)

    protected_paths = {r.path for r in repos if r.model_id in protect}
    # Oldest last_used first; ties broken by name so selection is
    # deterministic -- load-bearing for dry_run/real-run agreement and for
    # tests, not just cosmetic.
    candidates = sorted(
        (r for r in repos if r.path not in protected_paths),
        key=lambda r: (r.last_used, r.name),
    )

    to_remove: list[CachedRepo] = []
    freed = 0
    remaining = size_before
    for repo in candidates:
        if remaining <= cap_bytes:
            break
        to_remove.append(repo)
        freed += repo.size_bytes
        remaining -= repo.size_bytes

    if not dry_run:
        for repo in to_remove:
            _remove_repo(repo.path)

    return EvictionResult(
        removed=to_remove,
        bytes_freed=freed,
        size_before=size_before,
        size_after=size_before - freed,
        dry_run=dry_run,
    )


def free_disk_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem holding ``path`` -- INSTALL.md's minimum-free-disk claim.

    ``path`` may not exist yet (an unconfigured cache dir on a fresh install),
    so this walks up to the nearest existing ancestor instead of raising --
    the free-space number for a directory that doesn't exist yet is the free
    space where it would be created.
    """
    p = Path(path)
    while not p.exists():
        parent = p.parent
        if parent == p:  # reached the filesystem root without finding one
            break
        p = parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m ganymede.host.cache",
        description="Show, and optionally reclaim, disk used by the HuggingFace base-model cache (6.7).",
    )
    p.add_argument(
        "--cache-dir", default=None,
        help="default: $GANYMEDE_CACHE_DIR, $HF_HOME, or ~/.cache/huggingface",
    )
    p.add_argument(
        "--cap-gb", type=float, default=DEFAULT_CACHE_CAP_GB,
        help=f"cap in GB (default {DEFAULT_CACHE_CAP_GB:g}, matching the host agent's own default)",
    )
    p.add_argument("--dry-run", action="store_true", help="report what would be evicted; delete nothing")
    p.add_argument(
        "--protect", action="append", default=[], metavar="ORG/NAME",
        help="model id to never evict, e.g. Qwen/Qwen3-1.7B; repeatable",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    cap_bytes = int(args.cap_gb * 1024**3)
    result = evict_to_cap(cache_dir, cap_bytes, protect=frozenset(args.protect), dry_run=args.dry_run)

    if args.json:
        print(json.dumps({
            "cache_dir": str(cache_dir),
            "dry_run": result.dry_run,
            "size_before_bytes": result.size_before,
            "size_after_bytes": result.size_after,
            "bytes_freed": result.bytes_freed,
            "removed": [r.model_id for r in result.removed],
        }, indent=2))
    else:
        print(f"{cache_dir}: {_human(result.size_before)} used, cap {args.cap_gb:g} GB")
        if not result.removed:
            print("nothing to evict")
        else:
            verb = "would remove" if args.dry_run else "removed"
            for r in result.removed:
                print(f"  {verb} {r.model_id} ({_human(r.size_bytes)}, cached dir {r.name})")
            print(f"{_human(result.bytes_freed)} {'would be freed' if args.dry_run else 'freed'}"
                  f"; {_human(result.size_before)} -> {_human(result.size_after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
