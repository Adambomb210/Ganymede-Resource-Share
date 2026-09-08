"""The image pipeline: upload -> finalize -> scan -> gate (docs/11 §1).

Threat model is Decision 3 -- job code comes from a handful of manually
approved submitters, so the adversary is *a trusted author's honest mistake*: a
baked-in credential, a stale base image, an archive that unpacks to a hundred
times its shipped size. This module is the coordinator's half of that: it reads
a ``docker save`` archive and answers four questions (§1.3), then writes a
verdict onto the ``images`` row. It is not escape analysis and does not pretend
to be -- see docs/11 §5 for what is deliberately out.

**Two deviations from docs/11 §1.3, both deliberate and both narrower than the
design.** The scan there runs "inside a throwaway confined container"; here it
runs in-process, and it reads the archive through one stream rather than
unpacking it. The container is the stronger boundary and is still the target;
what stands in for it today is that *nothing is ever extracted*. Every member is
read through a byte cap, no path from the archive is ever joined to a filesystem
path, and a tripped cap is a ``flagged`` verdict rather than an exception. The
bomb guard is therefore load-bearing rather than advisory, which is why it is
enforced while streaming (``_CappedReader``) instead of checked afterwards.

A verdict is either ``clean`` or ``flagged``; ``flagged`` is "a human should
look", never a permanent deny. An admin dispositions it
(``POST /v1/admin/images/{id}/scan``), and until one does, the claim path will
not lease a task against the image (§1.4).
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sqlite3
import tarfile
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable

from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, utcnow

# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanLimits:
    """Bounds for one scan. Every one of these is a *flag* threshold, not an
    exception: an archive that trips a cap gets a verdict, not a traceback."""

    max_layers: int = 128
    # Cumulative uncompressed bytes across every layer. The decompression-bomb
    # guard: a 200 MB archive that unpacks to 400 GB is the honest mistake this
    # catches (a COPY of a dataset, usually), and the reason the reader below
    # counts as it goes rather than trusting any header.
    max_uncompressed_bytes: int = 64 * 1024**3
    max_members: int = 20_000
    max_entries_per_layer: int = 200_000
    # Content sweep budget. Names are always checked; bytes are read only for
    # small files, and only until the total budget is spent. A secret too big
    # for the per-file cap is not the honest mistake this looks for.
    max_file_scan_bytes: int = 256 * 1024
    max_content_scan_bytes: int = 64 * 1024**2
    # Bottom-of-stack ``diff_id`` values the operator vets (docs/11 §1.3). An
    # empty set flags every image: with no vetted set configured there is no
    # such thing as a recognised base, and fail-closed is the whole point of
    # the check. The operator either configures it or dispositions by hand.
    vetted_base_diff_ids: frozenset[str] = frozenset()

    @classmethod
    def from_settings(cls, settings: Any) -> "ScanLimits":
        return cls(
            max_layers=settings.image_scan_max_layers,
            max_uncompressed_bytes=settings.image_scan_max_uncompressed_bytes,
            vetted_base_diff_ids=frozenset(settings.image_vetted_base_diff_ids),
        )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    findings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "ok": self.ok, "detail": self.detail}
        if self.findings:
            # Bounded: one mislabelled layer can produce thousands of hits, and
            # scan_detail_json is read by a human on a web page.
            d["findings"] = self.findings[:25]
            if len(self.findings) > 25:
                d["findings_truncated"] = len(self.findings) - 25
        return d


@dataclass
class ScanResult:
    checks: list[Check]

    @property
    def status(self) -> str:
        return "clean" if all(c.ok for c in self.checks) else "flagged"

    def detail(self) -> dict:
        return {
            "checks": [c.as_dict() for c in self.checks],
            "failed": [c.name for c in self.checks if not c.ok],
        }


# --------------------------------------------------------------------------
# The bomb guard
# --------------------------------------------------------------------------


class _BudgetExceeded(Exception):
    """Raised inside the walk when a cap trips; caught at the top and turned
    into a ``flagged`` verdict. Never escapes this module."""


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def spend(self, n: int) -> None:
        self.used += n
        if self.used > self.limit:
            raise _BudgetExceeded(f"uncompressed content exceeded {self.limit} bytes")


class _CappedReader(io.RawIOBase):
    """A read-only wrapper that counts bytes against a shared budget.

    Wrapping the *decompressed* stream is the point: a gzip header claims
    nothing useful, and ``tarfile`` will happily read a petabyte if you let it.
    The budget is shared across every layer in the archive, so a thousand small
    bombs cost the same as one big one.
    """

    def __init__(self, raw: BinaryIO, budget: _Budget) -> None:
        self._raw = raw
        self._budget = budget

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        chunk = self._raw.read(len(b))
        if not chunk:
            return 0
        self._budget.spend(len(chunk))
        b[: len(chunk)] = chunk
        return len(chunk)


class _Chain(io.RawIOBase):
    """Put back the bytes a format sniff consumed. ``r|`` streams cannot seek,
    so the only way to look at the first kilobyte and still hand the whole
    member to ``tarfile`` is to splice it back on the front."""

    def __init__(self, head: bytes, rest: BinaryIO) -> None:
        self._head = head
        self._rest = rest

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._head:
            n = min(len(b), len(self._head))
            b[:n] = self._head[:n]
            self._head = self._head[n:]
            return n
        chunk = self._rest.read(len(b))
        if not chunk:
            return 0
        b[: len(chunk)] = chunk
        return len(chunk)


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------

# Paths that are a mistake by their name alone (docs/11 §1.3). Matched against
# the in-layer path, so a leading component is optional.
_SECRET_PATHS = [
    (re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$"), "private ssh key"),
    (re.compile(r"(^|/)\.env$"), ".env file"),
    (re.compile(r"(^|/)\.git/"), ".git directory"),
    (re.compile(r"(^|/)\.docker/config\.json$"), "docker credentials"),
    (re.compile(r"(^|/)\.aws/credentials$"), "aws credentials"),
    (re.compile(r"(^|/)\.netrc$"), ".netrc"),
    (re.compile(r"(^|/)\.npmrc$"), ".npmrc"),
    (re.compile(r"(^|/)\.pypirc$"), ".pypirc"),
    (re.compile(r"(^|/)kubeconfig$"), "kubeconfig"),
]

# Contents worth reading small files for. Deliberately high-precision: a false
# positive costs a human a look, and a check that cries wolf gets dispositioned
# blind, which is worse than not having the check at all.
_SECRET_CONTENT = [
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "aws access key id"),
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(rb'"type"\s*:\s*"service_account"'), "gcp service account key"),
    (re.compile(rb"gh[pousr]_[A-Za-z0-9]{36}"), "github token"),
    (re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"), "slack token"),
    (re.compile(rb"[Aa]uthorization:\s*Bearer\s+[A-Za-z0-9._-]{20,}"), "bearer token"),
]

# Docker's whiteout convention: a deletion marker in an upper layer, not a file.
_WHITEOUT = re.compile(r"(^|/)\.wh\.")


def _sweep_entry(path: str, read: Callable[[], bytes] | None, size: int,
                 limits: ScanLimits, state: dict, findings: list[str]) -> None:
    if _WHITEOUT.search(path):
        return
    for pattern, label in _SECRET_PATHS:
        if pattern.search(path):
            findings.append(f"{label}: {path}")
            return
    if read is None or size > limits.max_file_scan_bytes:
        return
    if state["content"] >= limits.max_content_scan_bytes:
        return
    data = read()
    state["content"] += len(data)
    for pattern, label in _SECRET_CONTENT:
        if pattern.search(data):
            findings.append(f"{label}: {path}")
            return


# --------------------------------------------------------------------------
# Archive walk
# --------------------------------------------------------------------------

_TAR_MAGIC_OFFSET = 257
_GZIP_MAGIC = b"\x1f\x8b"
_SNIFF_BYTES = 1024


def _looks_like_tar(head: bytes) -> bool:
    return head[_TAR_MAGIC_OFFSET:_TAR_MAGIC_OFFSET + 5] == b"ustar"


def _unsafe(name: str) -> bool:
    """An absolute path or a ``..`` component in an archive member. Nothing
    here extracts, so this can never become a traversal -- it is reported
    because a build that produces one is broken in a way the submitter wants
    told about."""
    if name.startswith("/") or name.startswith("\\"):
        return True
    return ".." in re.split(r"[\\/]+", name)


@dataclass
class _Walk:
    """What one pass over the archive collected."""

    small_blobs: dict[str, bytes] = field(default_factory=dict)
    layer_paths: list[str] = field(default_factory=list)
    secret_findings: list[str] = field(default_factory=list)
    unsafe_paths: list[str] = field(default_factory=list)
    members: int = 0
    uncompressed: int = 0
    trip: str | None = None


def _walk(stream: BinaryIO, limits: ScanLimits) -> _Walk:
    """One streaming pass. Nothing is extracted; nothing seeks backwards.

    Small members are kept in memory because the metadata this scan reasons
    about -- ``manifest.json``, ``index.json``, the config blob -- is small,
    sits at an unpredictable offset, and is the only thing worth a second look.
    Anything that sniffs like a layer is streamed through the secrets sweep as
    it goes past, so both formats (a classic ``docker save`` and an OCI layout)
    are handled without knowing which one this is until the walk is over.
    """
    walk = _Walk()
    budget = _Budget(limits.max_uncompressed_bytes)
    state = {"content": 0}
    try:
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                walk.members += 1
                if walk.members > limits.max_members:
                    raise _BudgetExceeded(
                        f"more than {limits.max_members} archive members"
                    )
                if not member.isfile():
                    continue
                if _unsafe(member.name):
                    walk.unsafe_paths.append(member.name)
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                head = handle.read(_SNIFF_BYTES)
                gzipped = head[:2] == _GZIP_MAGIC
                if gzipped or _looks_like_tar(head):
                    walk.layer_paths.append(member.name)
                    _sweep_layer(io.BufferedReader(_Chain(head, handle)), gzipped,
                                 member.name, limits, budget, state, walk)
                else:
                    # Not a layer: charge the declared size and move on. Only
                    # the layer path decompresses, so only it needs the cap
                    # applied per byte read.
                    budget.spend(member.size)
                    if member.size <= limits.max_file_scan_bytes:
                        walk.small_blobs[member.name] = head + handle.read()
    except _BudgetExceeded as exc:
        walk.trip = str(exc)
    except tarfile.TarError as exc:
        walk.trip = f"unreadable archive: {exc}"
    walk.uncompressed = budget.used
    return walk


def _sweep_layer(handle: BinaryIO, gzipped: bool, path: str, limits: ScanLimits,
                 budget: _Budget, state: dict, walk: _Walk) -> None:
    """Stream one layer through the secrets sweep.

    Order matters: the byte cap goes *after* the decompressor, never before it.
    A gzip member's compressed size is bounded by the archive size cap already
    and tells you nothing -- the quantity a bomb inflates is what comes out, so
    that is the side the budget counts, and the trip happens inside ``read``
    rather than after the damage.
    """
    raw: BinaryIO = handle
    if gzipped:
        raw = gzip.GzipFile(fileobj=handle)  # type: ignore[assignment]
    capped = io.BufferedReader(_CappedReader(raw, budget))
    try:
        with tarfile.open(fileobj=capped, mode="r|") as inner:
            entries = 0
            for entry in inner:
                entries += 1
                if entries > limits.max_entries_per_layer:
                    raise _BudgetExceeded(
                        f"layer {path} has more than "
                        f"{limits.max_entries_per_layer} entries"
                    )
                if not entry.isfile():
                    continue
                if _unsafe(entry.name):
                    walk.unsafe_paths.append(f"{path}!{entry.name}")
                    continue

                def _read(entry=entry, inner=inner) -> bytes:
                    fh = inner.extractfile(entry)
                    return fh.read(limits.max_file_scan_bytes) if fh else b""

                _sweep_entry(entry.name, _read, entry.size, limits, state,
                             walk.secret_findings)
    except (tarfile.TarError, gzip.BadGzipFile, EOFError, OSError):
        # A blob that sniffed like a layer and is not one is not a finding: OCI
        # layers are gzipped tars, but a gzipped config blob can be anything.
        # A _BudgetExceeded raised inside the read is *not* caught here -- it
        # propagates to the walk and becomes the verdict.
        return


# --------------------------------------------------------------------------
# The four checks
# --------------------------------------------------------------------------


def _decode(blob: bytes) -> Any:
    try:
        return json.loads(blob)
    except (ValueError, UnicodeDecodeError):
        return None


def _blob(walk: _Walk, digest: str) -> bytes:
    """OCI blobs are addressed ``blobs/<algo>/<hex>``; a reference may or may
    not carry the ``sha256:`` prefix."""
    if not digest:
        return b""
    algo, _, hexpart = digest.partition(":")
    if not hexpart:
        algo, hexpart = "sha256", digest
    return walk.small_blobs.get(f"blobs/{algo}/{hexpart}", b"")


def _resolve_config(walk: _Walk) -> tuple[dict | None, str, list[str]]:
    """Find the image config JSON and the layer list, in either format.

    Returns ``(config, format, layer_refs)``. A ``None`` config means the
    archive is neither a classic ``docker save`` nor an OCI layout -- which is
    itself the manifest-sanity failure, reported there rather than here.
    """
    manifest = walk.small_blobs.get("manifest.json")
    if manifest is not None:
        entries = _decode(manifest)
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            first = entries[0]
            cfg_name = first.get("Config")
            cfg = _decode(walk.small_blobs.get(cfg_name, b"")) if cfg_name else None
            layers = [str(x) for x in first.get("Layers") or []]
            return (cfg if isinstance(cfg, dict) else None), "docker-archive", layers
        return None, "docker-archive", []

    index = walk.small_blobs.get("index.json")
    if index is not None:
        idx = _decode(index)
        manifests = idx.get("manifests") if isinstance(idx, dict) else None
        if isinstance(manifests, list) and manifests:
            man = _decode(_blob(walk, (manifests[0] or {}).get("digest", "")))
            if isinstance(man, dict):
                cfg = _decode(_blob(walk, (man.get("config") or {}).get("digest", "")))
                layers = [
                    str((d or {}).get("digest", "")) for d in man.get("layers") or []
                ]
                return (cfg if isinstance(cfg, dict) else None), "oci-layout", layers
        return None, "oci-layout", []

    return None, "unknown", []


def _check_manifest(walk: _Walk, cfg: dict | None, fmt: str, layers: list[str],
                    limits: ScanLimits) -> Check:
    if walk.trip:
        return Check("manifest_sanity", False, walk.trip)
    if cfg is None:
        return Check(
            "manifest_sanity", False,
            f"no readable image config ({fmt} archive); expected a docker save "
            "archive or an OCI layout",
        )
    os_name = str(cfg.get("os", "")).lower()
    arch = str(cfg.get("architecture", "")).lower()
    if (os_name, arch) != ("linux", "amd64"):
        return Check("manifest_sanity", False,
                     f"platform is {os_name or '?'}/{arch or '?'}, not linux/amd64")
    layer_count = len(layers) or len(walk.layer_paths)
    if layer_count > limits.max_layers:
        return Check("manifest_sanity", False,
                     f"{layer_count} layers exceeds the {limits.max_layers} cap")
    if walk.unsafe_paths:
        return Check("manifest_sanity", False,
                     "archive contains absolute or traversing paths",
                     walk.unsafe_paths)
    return Check("manifest_sanity", True,
                 f"{fmt}, linux/amd64, {layer_count} layers, "
                 f"{walk.uncompressed} uncompressed bytes")


def _check_base(cfg: dict | None, limits: ScanLimits) -> Check:
    diff_ids = ((cfg or {}).get("rootfs") or {}).get("diff_ids") or []
    if not diff_ids:
        return Check("base_provenance", False,
                     "image config carries no rootfs.diff_ids")
    bottom = str(diff_ids[0])
    if not limits.vetted_base_diff_ids:
        return Check(
            "base_provenance", False,
            "no vetted base set is configured (GANYMEDE_VETTED_BASE_DIFF_IDS); "
            f"bottom layer {bottom} cannot be recognised",
        )
    if bottom not in limits.vetted_base_diff_ids:
        return Check("base_provenance", False, f"unrecognised base layer {bottom}")
    return Check("base_provenance", True, f"base {bottom}")


def _check_secrets(walk: _Walk) -> Check:
    if walk.secret_findings:
        return Check("secrets", False,
                     f"{len(walk.secret_findings)} possible secret(s) in the "
                     "image filesystem", walk.secret_findings)
    return Check("secrets", True, "no known credential shapes found")


def _check_entrypoint(cfg: dict | None) -> Check:
    inner = (cfg or {}).get("config") or {}
    entry = inner.get("Entrypoint") or []
    cmd = inner.get("Cmd") or []
    if not entry and not cmd:
        return Check("entrypoint", False, "neither ENTRYPOINT nor CMD is set")
    user = str(inner.get("User") or "").strip()
    if user in ("", "0", "root", "0:0", "root:root"):
        # The runtime forces --user regardless (docs/11 §2.2); an image that
        # expects root is still worth a glance, because it usually means it
        # writes somewhere that --read-only will refuse.
        return Check("entrypoint", False,
                     f"USER is {user or 'unset'}; the image expects root")
    return Check("entrypoint", True,
                 f"USER {user}, entrypoint {list(entry or cmd)[:3]}")


def scan_archive(stream: BinaryIO, limits: ScanLimits | None = None) -> ScanResult:
    """Run the four §1.3 checks over one ``docker save`` archive stream."""
    limits = limits or ScanLimits()
    walk = _walk(stream, limits)
    cfg, fmt, layers = _resolve_config(walk)
    return ScanResult([
        _check_manifest(walk, cfg, fmt, layers, limits),
        _check_base(cfg, limits),
        _check_secrets(walk),
        _check_entrypoint(cfg),
    ])


# --------------------------------------------------------------------------
# Running a scan against a row
# --------------------------------------------------------------------------


class ScanUnavailable(Exception):
    """The archive could not be read at all -- a missing object, a store that
    is down. Distinct from a verdict: the row stays ``pending`` and the sweep
    tries again, because "we could not look" is not "we looked and it was
    bad"."""


def run_scan(conn: sqlite3.Connection, store, image_id: str,
             limits: ScanLimits | None = None) -> ScanResult:
    """Scan one finalized image and write the verdict onto its row.

    Out-of-band by design (docs/11 §1.3): never called from the claim path,
    never from ``/finalize``'s request cycle beyond enqueueing. The caller is
    ``drain_pending`` from the same cron that runs the ledger sweep.
    """
    row = conn.execute(
        "SELECT id, object_ref, finalized_at FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    if row is None:
        raise ScanUnavailable(f"no image {image_id}")
    if row["finalized_at"] is None:
        raise ScanUnavailable(f"image {image_id} is not finalized")
    try:
        payload = store.get_bytes(row["object_ref"])
    except Exception as exc:  # ObjectNotFound, StoreError, transport
        raise ScanUnavailable(f"cannot read {row['object_ref']}: {exc}") from exc

    result = scan_archive(io.BytesIO(payload), limits)
    with immediate(conn):
        conn.execute(
            "UPDATE images SET scan_status = ?, scanned_at = ?, "
            "scan_detail_json = ? WHERE id = ?",
            (result.status, _iso(utcnow()), json.dumps(result.detail()), image_id),
        )
    return result


def drain_pending(conn: sqlite3.Connection, store, limits: ScanLimits | None = None,
                  max_images: int = 10) -> list[tuple[str, str]]:
    """Scan up to ``max_images`` finalized-but-unscanned images.

    The queue is the ``images`` table itself -- ``scan_status='pending'`` with a
    ``finalized_at`` -- rather than a second table that could disagree with it.
    Returns ``(image_id, status)`` per image scanned; an image whose bytes could
    not be read is skipped, left ``pending``, and retried next sweep.
    """
    rows = conn.execute(
        "SELECT id FROM images WHERE scan_status = 'pending' "
        "AND finalized_at IS NOT NULL ORDER BY finalized_at LIMIT ?",
        (max_images,),
    ).fetchall()
    done: list[tuple[str, str]] = []
    for row in rows:
        try:
            result = run_scan(conn, store, row["id"], limits)
        except ScanUnavailable:
            continue
        done.append((row["id"], result.status))
    return done
