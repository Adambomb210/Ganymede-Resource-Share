"""Print one enrollment token per orphaned worker (docs/08 "Migration 004").

Migration 004 keeps every existing ``workers`` row but mints no machine key for
it -- there is no channel to deliver one. Until a machine re-enrolls it leans on
auth's transitional legacy-worker rule (its owner's contributor key stands in).
This helper closes that gap: for every ``workers`` row with no ``machine_keys``,
it issues a fresh ``enroll_token`` bound to the machine's owner and prints it, so
the operator can paste one per box into its host config and be done in a minute.

Mirrors ``scripts/issue_key.py``: the token is generated here, printed once, and
never stored -- only its hash goes into ``enrollments``.
"""

from __future__ import annotations

import argparse
import uuid

from ganymede.coordinator import rounds
from ganymede.coordinator.auth import hash_key
from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, immediate, init_schema
from ganymede.coordinator.identity import new_enroll_token


def _orphans(conn):
    return conn.execute(
        """SELECT w.id, w.contributor_id, w.display_name, w.image_tag
             FROM workers w
            WHERE NOT EXISTS (SELECT 1 FROM machine_keys k WHERE k.machine_id = w.id)
            ORDER BY w.first_seen"""
    ).fetchall()


def main(argv: list[str] | None = None, *, settings: Settings | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.reenroll_fleet",
        description="Issue an enrollment token for every worker still without a machine key.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="list the orphaned workers without issuing tokens")
    args = p.parse_args(argv)

    settings = settings or Settings.from_env()
    conn = connect(settings.db_path)
    init_schema(conn)
    try:
        orphans = _orphans(conn)
        if not orphans:
            print("every worker already has a machine key -- nothing to re-enroll")
            return 0
        now = rounds._iso(rounds.utcnow())
        for w in orphans:
            name = w["display_name"] or w["image_tag"] or ("machine-" + w["id"][:8])
            if args.dry_run:
                print(f"{w['id']}  {name}")
                continue
            token = new_enroll_token()
            with immediate(conn):
                conn.execute(
                    """INSERT INTO enrollments
                         (id, user_id, token_hash, display_name, created_at,
                          consumed_at, machine_id)
                       VALUES (?, ?, ?, ?, ?, NULL, NULL)""",
                    (uuid.uuid4().hex, w["contributor_id"], hash_key(token), name, now),
                )
            print(f"{w['id']}  {name}\n    enroll_token: {token}")
        if not args.dry_run:
            print("\nEach token is shown once. Paste one per host into its config and "
                  "run claim-enrollment; the TTL is GANYMEDE_ENROLL_TTL_SEC.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
