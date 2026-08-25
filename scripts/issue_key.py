"""Mint, revoke, and list contributor bearer keys (docs/02-architecture-v2.md 6.3).

Mirrors auth.py's own contract: the plaintext key is generated here, printed
once, and never stored -- only its hash goes into the `contributors` table. If
this script's output is lost, the key is gone; there is no "look it up later".
"""

from __future__ import annotations

import argparse
import sys
import uuid

from ganymede.coordinator import rounds
from ganymede.coordinator.auth import generate_key, hash_key
from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, immediate, init_schema

_CLEARANCES = ("open", "internal", "restricted")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.issue_key",
        description="Mint, revoke, or list Ganymede contributor keys.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="mint a new key for this contributor name")
    group.add_argument("--revoke", metavar="CONTRIBUTOR_ID", help="disable a contributor's key")
    group.add_argument("--list", action="store_true", help="list contributors")
    p.add_argument("--clearance", choices=_CLEARANCES, default="open",
                    help="only used with --name")
    p.add_argument("--id", dest="contributor_id", default=None,
                    help="explicit contributor id; a uuid4 is generated if omitted "
                         "(only used with --name)")
    return p


def _issue(conn, name: str, clearance: str, contributor_id: str | None) -> int:
    cid = contributor_id or uuid.uuid4().hex
    key = generate_key()
    now = rounds._iso(rounds.utcnow())
    with immediate(conn):
        existing = conn.execute("SELECT id FROM contributors WHERE id = ?", (cid,)).fetchone()
        if existing is not None:
            print(f"error: contributor {cid!r} already exists", file=sys.stderr)
            return 1
        conn.execute(
            """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
               VALUES (?, ?, ?, 1, ?, ?)""",
            (cid, name, hash_key(key), clearance, now),
        )
    print(f"id: {cid}")
    print(f"name: {name}")
    print(f"clearance: {clearance}")
    print(f"key: {key}")
    print("This key is shown once. Only its hash is stored -- there is no way to "
          "recover it later, so save it now (e.g. into the worker's GANYMEDE_KEY).")
    return 0


def _revoke(conn, contributor_id: str) -> int:
    with immediate(conn):
        row = conn.execute(
            "SELECT id, name, enabled FROM contributors WHERE id = ?", (contributor_id,)
        ).fetchone()
        if row is None:
            print(f"error: no such contributor {contributor_id!r}", file=sys.stderr)
            return 1
        # enabled = 0, never DELETE: the audit table (audit.contributor_id) and the
        # workers/tasks history reference this row, and a revoked contributor's
        # past submissions and rejections are exactly the record Phase 2's
        # reputation scoring needs -- deleting the row would erase them too.
        conn.execute("UPDATE contributors SET enabled = 0 WHERE id = ?", (contributor_id,))
    print(f"revoked: {contributor_id} ({row['name']})")
    return 0


def _list(conn) -> int:
    rows = conn.execute(
        "SELECT id, name, clearance, enabled, created_at FROM contributors ORDER BY created_at"
    ).fetchall()
    if not rows:
        print("no contributors")
        return 0
    for r in rows:
        state = "enabled" if r["enabled"] else "revoked"
        print(f"{r['id']}  {r['name']!r:24}  {r['clearance']:10}  {state:8}  {r['created_at']}")
    return 0


def main(argv: list[str] | None = None, *, settings: Settings | None = None) -> int:
    """CLI entrypoint. `settings` is an injection seam for tests (see newrun.py)."""
    args = _build_arg_parser().parse_args(argv)
    settings = settings or Settings.from_env()
    conn = connect(settings.db_path)
    init_schema(conn)
    try:
        if args.name is not None:
            return _issue(conn, args.name, args.clearance, args.contributor_id)
        if args.revoke is not None:
            return _revoke(conn, args.revoke)
        return _list(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
