"""Delete stale worker submission artifacts (docs/02-architecture-v2.md 6.6,
"Retention and garbage collection").

Two classes of object live in the store, with different lifetimes: round
result adapters (the checkpoint chain a run is built on -- keep forever) and
individual worker submissions (safe to delete once their round has closed and
been aggregated, since the aggregate already carries their contribution). This
script deletes only the second class, and only once a round is old enough that
nobody would still want it for debugging a bad aggregate.

Deliberately a cron job / one-shot script rather than an S3 lifecycle rule --
lifecycle rules are exactly the part of the S3 API that MinIO, R2 and AWS
diverge on, and store.py stays inside the portable subset all three share
(store.py's module docstring). This is that "write the GC job instead".
"""

from __future__ import annotations

import argparse
import sys

from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, init_schema
from ganymede.coordinator.store import Store

# The prefix layout under a round mirrors adapter_key()/base_adapter_key() in
# store.py: "<round>/submissions/<task>.safetensors" for worker artifacts vs.
# "<round>/base.safetensors" for the round's own base adapter. Scoping every
# list_prefix() call to the "submissions/" sub-prefix means a round's
# base_adapter_ref -- a sibling key, not a descendant -- is never even a
# candidate, regardless of the protected-set check below.
_SUBMISSIONS_PREFIX = "runs/{run_id}/rounds/{idx:05d}/submissions/"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.gc",
        description="Delete worker submission artifacts from old, closed rounds.",
    )
    p.add_argument("--run-id", default=None, help="limit to one run; default is all runs")
    p.add_argument("--keep-rounds", type=int, default=None,
                    help="override settings.gc_keep_rounds")
    p.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted without deleting anything "
                         "(also the default whenever --yes is not given)")
    p.add_argument("--yes", action="store_true", help="actually delete")
    return p


def _gc_candidates(
    conn, store: Store, run_id: str, keep_rounds: int
) -> list[tuple[str, int]]:
    """(key, size) pairs eligible for deletion for one run."""
    run = conn.execute(
        "SELECT outer_momentum_ref FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if run is None:
        return []

    round_rows = conn.execute(
        """SELECT idx, status, base_adapter_ref, result_adapter_ref
           FROM rounds WHERE run_id = ? ORDER BY idx DESC""",
        (run_id,),
    ).fetchall()

    # Belt-and-suspenders: these three sets are exactly what must never be
    # deleted (the task's own "never" list), checked even though the prefix
    # scoping above already keeps them out of candidacy on their own.
    protected = {r["base_adapter_ref"] for r in round_rows}
    protected |= {r["result_adapter_ref"] for r in round_rows if r["result_adapter_ref"]}
    if run["outer_momentum_ref"]:
        protected.add(run["outer_momentum_ref"])

    closed = [r for r in round_rows if r["status"] == "closed"]  # already idx DESC
    # Keep the most recent `keep_rounds` closed rounds' submissions around --
    # that's the grace window for debugging a bad aggregate (6.6). Only rounds
    # older than that are eligible.
    stale = closed[keep_rounds:]

    candidates: list[tuple[str, int]] = []
    for r in stale:
        prefix = _SUBMISSIONS_PREFIX.format(run_id=run_id, idx=r["idx"])
        for key in store.list_prefix(prefix):
            if key in protected:
                continue  # unreachable given the prefix, but never trust it blindly
            info = store.head(key)
            candidates.append((key, info["size"] if info else 0))
    return candidates


def _print_report(candidates: list[tuple[str, int]], total_bytes: int, deleted: bool) -> None:
    verb = "Deleted" if deleted else "Would delete (dry run -- pass --yes to actually delete)"
    print(f"{verb} {len(candidates)} object(s), {total_bytes} bytes "
          f"({total_bytes / (1024 * 1024):.2f} MB)")
    for key, size in candidates:
        print(f"  {key}  ({size} bytes)")


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
) -> int:
    """CLI entrypoint. `settings`/`store` are an injection seam for tests (see newrun.py)."""
    args = _build_arg_parser().parse_args(argv)
    settings = settings or Settings.from_env()
    store = store or Store(settings.storage)
    conn = connect(settings.db_path)
    init_schema(conn)
    try:
        keep_rounds = args.keep_rounds if args.keep_rounds is not None else settings.gc_keep_rounds

        if args.run_id is not None:
            if conn.execute("SELECT 1 FROM runs WHERE id = ?", (args.run_id,)).fetchone() is None:
                print(f"error: no such run {args.run_id!r}", file=sys.stderr)
                return 1
            run_ids = [args.run_id]
        else:
            run_ids = [r["id"] for r in conn.execute("SELECT id FROM runs").fetchall()]

        candidates: list[tuple[str, int]] = []
        for run_id in run_ids:
            candidates.extend(_gc_candidates(conn, store, run_id, keep_rounds))
        total_bytes = sum(size for _, size in candidates)

        # Safety default: deletion requires an explicit --yes, and --dry-run
        # always wins even if --yes is also given.
        do_delete = args.yes and not args.dry_run

        _print_report(candidates, total_bytes, do_delete)
        if do_delete:
            for key, _ in candidates:
                store.delete(key)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
