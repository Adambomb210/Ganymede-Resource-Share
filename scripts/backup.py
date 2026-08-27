"""Off-box backup: a consistent SQLite snapshot plus the live checkpoint chain
(docs/02-architecture-v2.md 6.6, "Backups -- the one thing this decision
genuinely breaks").

With the artifact store co-located on the coordinator VM, a backup that also
lives on that VM is worthless for the failure it exists to survive -- losing
the disk loses the database and every checkpoint together. So this script
refuses to run unless the destination is demonstrably a *different* store: a
different endpoint or a different bucket, backed by credentials the caller had
to supply separately from the source's. "Off-box" means a different machine,
not a different directory, and that distinction is the whole point of §6.6.

What it copies:
  1. A SQLite snapshot taken with ``sqlite3.Connection.backup()``, not a file
     copy. The coordinator's DB runs in WAL mode (db.py) and is written to by a
     live server; a plain ``cp`` can catch it mid-write and copy a torn file --
     a data file without its WAL frames replayed, or pages in an inconsistent
     half-written state. That tear is invisible at backup time and only
     surfaces as corruption when the copy is restored, potentially long after
     the good version has been overwritten by a later backup. SQLite's own
     backup API takes the locks it needs and copies page-by-page against the
     live source, so the destination is always a transactionally consistent
     snapshot -- however inconvenient a moment the backup lands on. This is the
     entire reason this script isn't a five-line ``cp`` + upload.
  2. The latest ``result_adapter_ref`` for each run, plus each run's
     ``outer_momentum_ref``. Per 6.6: "the database plus the newest adapter is
     enough to resume the run rather than restart it" if the primary store is
     lost entirely.
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

from ganymede.coordinator.config import Settings, StorageConfig
from ganymede.coordinator.db import connect
from ganymede.coordinator.store import ObjectNotFound, Store


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.backup",
        description="Copy a consistent DB snapshot and the checkpoint chain off-box.",
    )
    p.add_argument("--dest-endpoint", required=True)
    p.add_argument("--dest-bucket", required=True)
    p.add_argument("--dest-access-key", required=True)
    p.add_argument("--dest-secret-key", required=True)
    p.add_argument("--dest-region", default="auto")
    p.add_argument("--run-id", default=None, help="limit adapter backup to one run")
    return p


def snapshot_db_bytes(conn: sqlite3.Connection) -> bytes:
    """A consistent snapshot of `conn`'s database, as bytes.

    Backs up into a second, file-backed connection (not ``:memory:``) purely so
    the result can be read back as a single blob of bytes to upload -- the
    consistency guarantee comes from ``sqlite3.Connection.backup()`` itself, per
    the module docstring above.
    """
    with tempfile.TemporaryDirectory() as td:
        snap_path = str(pathlib.Path(td) / "snapshot.db")
        dest_conn = sqlite3.connect(snap_path)
        try:
            conn.backup(dest_conn)
        finally:
            dest_conn.close()
        return pathlib.Path(snap_path).read_bytes()


def _latest_result_ref(conn: sqlite3.Connection, run_id: str) -> str | None:
    row = conn.execute(
        """SELECT result_adapter_ref FROM rounds
           WHERE run_id = ? AND result_adapter_ref IS NOT NULL
           ORDER BY idx DESC LIMIT 1""",
        (run_id,),
    ).fetchone()
    return row["result_adapter_ref"] if row else None


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
    dest_store: Store | None = None,
) -> int:
    """CLI entrypoint. `settings`/`store`/`dest_store` are an injection seam for
    tests (see newrun.py) -- `dest_store` in particular lets tests exercise the
    off-box copy without a second real MinIO endpoint.
    """
    args = _build_arg_parser().parse_args(argv)
    settings = settings or Settings.from_env()

    # The refusal check, before any I/O: a backup that resolves to the same
    # store as the source is not a backup (module docstring). Comparing
    # (endpoint, bucket) rather than requiring different credentials too --
    # different creds against the *same* bucket would still not be off-box.
    if (
        args.dest_endpoint == settings.storage.endpoint_url
        and args.dest_bucket == settings.storage.bucket
    ):
        print(
            "error: destination endpoint+bucket match the source store -- a "
            "backup that lives on the machine it is backing up is not a "
            "backup (docs/02-architecture-v2.md 6.6)",
            file=sys.stderr,
        )
        return 1

    store = store or Store(settings.storage)
    dest_store = dest_store or Store(
        StorageConfig(
            endpoint_url=args.dest_endpoint, bucket=args.dest_bucket,
            region=args.dest_region, access_key=args.dest_access_key,
            secret_key=args.dest_secret_key,
        )
    )

    # The destination bucket may not exist yet, and the first backup is exactly
    # when it will not. `newrun` shipped with this same omission (roadmap M2
    # status): a script that writes to a bucket it never ensured turns a first
    # deployment into an unwritten prerequisite whose symptom is a boto stack
    # trace. A backup that fails the first time it is genuinely needed is worse
    # than most bugs, because nobody finds out until the disaster.
    dest_store.ensure_bucket()

    conn = connect(settings.db_path)
    try:
        now = datetime.now(timezone.utc)
        db_key = f"backups/{now.strftime('%Y%m%dT%H%M%SZ')}/coordinator.db"
        db_bytes = snapshot_db_bytes(conn)
        dest_store.put_bytes(db_key, db_bytes)

        manifest: list[tuple[str, str, int]] = [("sqlite_snapshot", db_key, len(db_bytes))]

        run_ids = (
            [args.run_id] if args.run_id
            else [r["id"] for r in conn.execute("SELECT id FROM runs").fetchall()]
        )
        for run_id in run_ids:
            run = conn.execute(
                "SELECT outer_momentum_ref FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                print(f"error: no such run {run_id!r}", file=sys.stderr)
                return 1

            refs: list[tuple[str, str]] = []
            result_ref = _latest_result_ref(conn, run_id)
            if result_ref:
                refs.append((f"{run_id}: latest result_adapter_ref", result_ref))
            if run["outer_momentum_ref"]:
                refs.append((f"{run_id}: outer_momentum_ref", run["outer_momentum_ref"]))

            for label, key in refs:
                try:
                    data = store.get_bytes(key)
                except ObjectNotFound:
                    print(f"warning: {label} points at {key!r}, which is missing "
                          f"from the source store -- skipped", file=sys.stderr)
                    continue
                # Mirror the source key verbatim: destination bucket/endpoint
                # differ by construction (the refusal check above), so there is
                # no collision risk, and an identical key makes matching a
                # restored object back to its source trivial.
                dest_store.put_bytes(key, data)
                manifest.append((label, key, len(data)))

        print("backup manifest:")
        for label, key, size in manifest:
            print(f"  {label:40} {key}  ({size} bytes)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
