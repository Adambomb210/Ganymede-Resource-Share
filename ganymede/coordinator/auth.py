"""Bearer-token authentication (docs/02-architecture-v2.md 6.3;
docs/08-identity.md "authenticate() gains the machine path").

One key per contributor, 32 random bytes, base64url. The coordinator stores only
a hash: a database dump must not be a set of working credentials. Revocation is
``enabled = 0`` rather than deletion, so the audit trail survives the revocation.

``authenticate`` resolves three token kinds now (docs/06 "Auth: three token
kinds"): a **machine key** (``machine_keys.key_hash`` -> ``Machine``), the
existing **contributor key** (``contributors.key_hash`` -> ``Contributor``), and
a **session token** (``sessions.token_hash`` -> ``Contributor``, the web UI).
A token is exactly one kind; the fall-through order is a micro-optimisation for
the machine-key hot path, not a correctness property. Existing
``Authorization: Bearer <contributor-key>`` callers are unchanged -- they still
resolve to a ``Contributor``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass

KEY_BYTES = 32


def generate_key() -> str:
    """Mint a new contributor key. Returned once, at issue time, then never again."""
    return secrets.token_urlsafe(KEY_BYTES)


def hash_key(key: str) -> str:
    """Hash a key for storage.

    SHA-256 rather than a password KDF on purpose: these are 256-bit random
    tokens, not passwords. There is no dictionary to attack and no user-chosen
    entropy to stretch, so bcrypt/argon2 would only add per-request latency to
    every single API call.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Contributor:
    id: str
    name: str
    clearance: str
    # docs/05 adds the column; docs/08 puts it on this dataclass. ``require_admin``
    # is the only reader in this phase. Defaulted so the existing single
    # construction site and any test that builds one positionally keep working.
    is_admin: bool = False


@dataclass(frozen=True)
class Machine:
    """A worker process, authenticated by its per-machine key (docs/08). Scoped
    to its own task lifecycle and nothing else -- a leaked machine key acting on
    another machine's endpoint or on any job gets the same 404 as a missing
    resource."""

    id: str
    owner_id: str    # contributors.id
    standing: str    # good | probation | revoked


# docs/08: ``authenticate`` returns one or the other.
Principal = Contributor | Machine


class AuthError(Exception):
    """Raised when a credential does not resolve to an enabled principal."""


def parse_bearer(header: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def authenticate(
    conn: sqlite3.Connection,
    header: str | None,
    *,
    cookie: str | None = None,
) -> Principal:
    """Resolve a credential to an enabled principal, or raise ``AuthError``.

    The bearer header wins; ``cookie`` (the ``ganymede_session`` value) is the
    fallback for browser callers that cannot set headers. Three sha256 lookups
    worst case, all on ``UNIQUE`` indexes.
    """
    token = parse_bearer(header) or cookie
    if token is None:
        raise AuthError("missing or malformed Authorization header")

    digest = hash_key(token)

    # 1. machine key -- highest QPS (every heartbeat, every worker).
    row = conn.execute(
        """SELECT w.id, w.contributor_id, w.standing, k.enabled
             FROM machine_keys k JOIN workers w ON w.id = k.machine_id
            WHERE k.key_hash = ?""",
        (digest,),
    ).fetchone()
    if row is not None:
        if not row["enabled"]:
            raise AuthError("machine key revoked")
        return Machine(row["id"], row["contributor_id"], row["standing"])

    # 2. contributor key -- CLI, operators, admins.
    row = conn.execute(
        "SELECT id, name, clearance, is_admin, enabled, key_hash "
        "FROM contributors WHERE key_hash = ?",
        (digest,),
    ).fetchone()
    if row is not None:
        # The lookup is by hash, so it is already constant-time in the
        # interesting sense. The compare_digest guards the case where a future
        # change makes this a scan rather than an indexed lookup.
        if not hmac.compare_digest(row["key_hash"], digest):
            raise AuthError("unknown key")
        if not row["enabled"]:
            raise AuthError("key revoked")
        return Contributor(row["id"], row["name"], row["clearance"], bool(row["is_admin"]))

    # 3. session token -- the web UI, via the placeholder provider.
    from ganymede.coordinator.identity import resolve_session_row

    row = resolve_session_row(conn, token)
    if row is not None:
        return Contributor(
            row["id"], row["name"], row["clearance"], bool(row["is_admin"])
        )

    raise AuthError("unknown credential")
