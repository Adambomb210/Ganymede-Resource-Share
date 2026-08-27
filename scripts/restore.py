"""Rebuild a coordinator from an off-box backup (docs/02-architecture-v2.md 6.6).

§6.4 names the shared VM as the system's single point of failure, and §6.6
accepts that risk on the strength of one sentence: "the database plus the
newest adapter is enough to resume the run rather than restart it". This is the
script that makes that sentence true, and the roadmap's M5 criterion is
deliberately worded as **performed at least once, not merely configured** --
a restore path nobody has run is a belief, not a capability.

What comes back, and what does not
----------------------------------
`backup.py` copies a transactionally consistent SQLite snapshot plus, per run,
the newest round's `result_adapter_ref` and the run's `outer_momentum_ref`.
That is enough to resume, and it is deliberately not everything:

- **Individual worker submissions are not backed up.** They are already folded
  into the round result that *is* backed up, and they are the bulk of the bytes
  (§6.6's GC deletes them on the same reasoning). A restored coordinator
  therefore has a database that references submission artifacts which no longer
  exist anywhere.
- **The round in flight when the disk died is lost.** Its leases point at
  workers that will never call back, and its submissions point at objects the
  restore cannot produce.

So a restore is not a byte-for-byte resurrection, and pretending otherwise
would leave the run wedged in a way that looks like corruption. The reconcile
step below deliberately rewinds the in-flight round: leases are released and
partial submissions dropped, so the round reopens and the fleet re-does one
round's work. One round is the correct amount to lose -- it is exactly the work
whose artifacts went down with the disk.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ganymede.coordinator.config import Settings, StorageConfig
from ganymede.coordinator.db import connect, init_schema
from ganymede.coordinator.store import ObjectNotFound, Store

BACKUP_PREFIX = "backups/"
DB_BASENAME = "coordinator.db"


@dataclass
class RestoreReport:
    backup_key: str = ""
    db_path: str = ""
    adapters_copied: list[str] = field(default_factory=list)
    adapters_already_present: list[str] = field(default_factory=list)
    rounds_rewound: list[str] = field(default_factory=list)
    leases_released: int = 0
    submissions_dropped: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        lines = [
            f"restored from   {self.backup_key}",
            f"database        {self.db_path}",
            f"adapters        {len(self.adapters_copied)} copied, "
            f"{len(self.adapters_already_present)} already in the primary store",
        ]
        if self.rounds_rewound:
            lines.append(
                f"rewound         {', '.join(self.rounds_rewound)} "
                f"({self.leases_released} lease(s) released, "
                f"{self.submissions_dropped} partial submission(s) dropped)"
            )
        for p in self.problems:
            lines.append(f"PROBLEM         {p}")
        return "\n".join(lines)


def latest_backup_key(backup_store: Store) -> str | None:
    """The newest snapshot in the backup bucket.

    `backup.py` names each one ``backups/<UTC timestamp>/coordinator.db``, and
    that timestamp format sorts lexicographically in time order -- which is why
    it is written that way rather than as anything friendlier.
    """
    keys = [k for k in backup_store.list_prefix(BACKUP_PREFIX) if k.endswith(DB_BASENAME)]
    return max(keys) if keys else None


def referenced_adapters(conn: sqlite3.Connection) -> list[str]:
    """Every object a restored coordinator must be able to read to keep going.

    Round *base* adapters, not results: the base of the current open round is
    what the next worker will be handed, and it is the one object whose absence
    stops the run dead. It is also, for any round after the first, the previous
    round's result -- which is exactly what `backup.py` saved.
    """
    refs: list[str] = []
    for r in conn.execute(
        """SELECT rounds.base_adapter_ref AS ref
           FROM rounds JOIN runs ON runs.id = rounds.run_id
           WHERE rounds.idx = runs.current_round"""
    ).fetchall():
        if r["ref"]:
            refs.append(r["ref"])
    for r in conn.execute(
        "SELECT outer_momentum_ref AS ref FROM runs WHERE outer_momentum_ref IS NOT NULL"
    ).fetchall():
        refs.append(r["ref"])
    # Deduplicated but order-stable, so the report reads the same twice.
    return list(dict.fromkeys(refs))


def reconcile(conn: sqlite3.Connection, report: RestoreReport,
              now: datetime | None = None) -> None:
    """Rewind whatever was in flight when the disk went.

    A restored database describes a moment that no longer exists: workers hold
    leases they will never report against, and submissions point at artifacts
    that were never backed up because the aggregate already contained them.
    Leaving either in place means the round can neither close (its submissions
    cannot be read) nor progress (its shards look spoken for until every lease
    expires).

    Rewinding is not data loss beyond what already happened. The work being
    discarded is precisely the work whose artifacts died with the store.
    """
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat()

    for rnd in conn.execute(
        """SELECT rounds.run_id, rounds.idx FROM rounds JOIN runs ON runs.id = rounds.run_id
           WHERE rounds.idx = runs.current_round AND rounds.status IN ('open', 'closing')"""
    ).fetchall():
        run_id, idx = rnd["run_id"], rnd["idx"]

        dropped = conn.execute(
            """DELETE FROM submissions WHERE task_id IN
                 (SELECT id FROM tasks WHERE run_id = ? AND round_idx = ?)""",
            (run_id, idx),
        ).rowcount
        released = conn.execute(
            """UPDATE tasks SET status = 'expired', worker_id = NULL, lease_expires_at = NULL
               WHERE run_id = ? AND round_idx = ? AND status IN ('leased', 'submitted')""",
            (run_id, idx),
        ).rowcount

        # A round caught mid-close is the worst of the two: `closing` is held by
        # exactly one caller and released only when it finishes, so a restore
        # that left it there would wedge the run permanently -- the same
        # `stuck_close` state invariants.py reports.
        conn.execute(
            """UPDATE rounds SET status = 'open', opened_at = ?
               WHERE run_id = ? AND idx = ?""",
            (stamp, run_id, idx),
        )

        report.rounds_rewound.append(f"{run_id}#{idx}")
        report.leases_released += max(released, 0)
        report.submissions_dropped += max(dropped, 0)

    conn.commit()


def restore(
    *,
    backup_store: Store,
    primary_store: Store,
    db_path: str,
    backup_key: str | None = None,
    dry_run: bool = False,
) -> RestoreReport:
    report = RestoreReport(db_path=db_path)

    key = backup_key or latest_backup_key(backup_store)
    if key is None:
        report.problems.append(f"no {BACKUP_PREFIX}*/{DB_BASENAME} in the backup store")
        return report
    report.backup_key = key

    try:
        db_bytes = backup_store.get_bytes(key)
    except ObjectNotFound:
        report.problems.append(f"{key} is not in the backup store")
        return report

    if dry_run:
        # Restore into a scratch file rather than the destination. Half-doing
        # it in memory would exercise a different code path from the real
        # thing, and a rehearsal that does not rehearse the actual steps is
        # the same failure as a backup nobody has restored from.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            scratch = str(Path(tmp) / DB_BASENAME)
            with open(scratch, "wb") as fh:
                fh.write(db_bytes)
            conn = connect(scratch)
            try:
                init_schema(conn)
                _plan_adapters(conn, backup_store, primary_store, report, copy=False)
                reconcile(conn, report)
                _verify(conn, primary_store, report)
            finally:
                conn.close()
        report.db_path = f"{db_path} (dry run: nothing written)"
        return report

    with open(db_path, "wb") as fh:
        fh.write(db_bytes)

    conn = connect(db_path)
    try:
        # The snapshot came from a coordinator that may be older than this
        # binary. `init_schema` is additive and idempotent, so running it here
        # is what lets a restore double as an upgrade rather than failing on a
        # column the new code expects.
        init_schema(conn)
        _plan_adapters(conn, backup_store, primary_store, report, copy=True)
        reconcile(conn, report)
        _verify(conn, primary_store, report)
    finally:
        conn.close()
    return report


def _plan_adapters(conn, backup_store: Store, primary_store: Store,
                   report: RestoreReport, *, copy: bool) -> None:
    for ref in referenced_adapters(conn):
        if primary_store.head(ref) is not None:
            report.adapters_already_present.append(ref)
            continue
        try:
            data = backup_store.get_bytes(ref)
        except ObjectNotFound:
            report.problems.append(
                f"{ref} is referenced by the restored database but is in neither store"
            )
            continue
        if copy:
            primary_store.put_bytes(ref, data)
        report.adapters_copied.append(ref)


def _verify(conn: sqlite3.Connection, primary_store: Store, report: RestoreReport) -> None:
    """Prove the thing the drill exists to prove: the run can be resumed.

    Reported rather than assumed, because a restore that leaves a run
    unresumable is worse than one that fails loudly -- the operator would
    believe they were covered and find out at the next disaster.
    """
    for r in conn.execute(
        """SELECT runs.id AS run_id, runs.current_round AS idx, rounds.status AS status,
                  rounds.base_adapter_ref AS ref
           FROM runs LEFT JOIN rounds
             ON rounds.run_id = runs.id AND rounds.idx = runs.current_round
           WHERE runs.status = 'active'"""
    ).fetchall():
        where = f"{r['run_id']}#{r['idx']}"
        if r["ref"] is None:
            report.problems.append(f"{where}: the current round is missing from the snapshot")
            continue
        if primary_store.head(r["ref"]) is None:
            report.problems.append(f"{where}: base adapter {r['ref']} is not in the primary store")
        if r["status"] != "open":
            report.problems.append(f"{where}: round is {r['status']!r}, so no worker can claim it")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ganymede-restore",
        description="Rebuild a coordinator from an off-box backup (6.6).",
    )
    p.add_argument("--src-endpoint", required=True, help="the backup store")
    p.add_argument("--src-bucket", required=True)
    p.add_argument("--src-access-key", required=True)
    p.add_argument("--src-secret-key", required=True)
    p.add_argument("--src-region", default="auto")
    p.add_argument("--backup-key", default=None,
                   help=f"default: the newest {BACKUP_PREFIX}*/{DB_BASENAME}")
    p.add_argument("--db-path", default=None,
                   help="where to write the database; default is this coordinator's configured path")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing database at the destination")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None, *, settings: Settings | None = None,
         primary_store: Store | None = None, backup_store: Store | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    settings = settings or Settings.from_env()
    db_path = args.db_path or settings.db_path

    import os

    if os.path.exists(db_path) and not args.force and not args.dry_run:
        # The one destructive step in the whole system, and the moment it would
        # be taken is a bad one to be surprised in.
        print(
            f"error: {db_path} already exists. A restore overwrites it entirely.\n"
            f"       Move it aside, or pass --force if you mean to discard it.",
            file=sys.stderr,
        )
        return 2

    backup_store = backup_store or Store(StorageConfig(
        endpoint_url=args.src_endpoint, bucket=args.src_bucket, region=args.src_region,
        access_key=args.src_access_key, secret_key=args.src_secret_key,
    ))
    primary_store = primary_store or Store(settings.storage)
    primary_store.ensure_bucket()

    report = restore(
        backup_store=backup_store, primary_store=primary_store, db_path=db_path,
        backup_key=args.backup_key, dry_run=args.dry_run,
    )
    print(report)
    if report.ok:
        print("\nThe run can be resumed. Start the coordinator and let the fleet poll.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
