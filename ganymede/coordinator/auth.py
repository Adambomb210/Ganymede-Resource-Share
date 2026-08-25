"""Bearer-token authentication (docs/02-architecture-v2.md 6.3).

One key per contributor, 32 random bytes, base64url. The coordinator stores only
a hash: a database dump must not be a set of working credentials. Revocation is
``enabled = 0`` rather than deletion, so the audit trail survives the revocation.
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


class AuthError(Exception):
    """Raised when a bearer token does not resolve to an enabled contributor."""


def parse_bearer(header: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def authenticate(conn: sqlite3.Connection, header: str | None) -> Contributor:
    """Resolve an Authorization header to an enabled contributor, or raise."""
    token = parse_bearer(header)
    if token is None:
        raise AuthError("missing or malformed Authorization header")

    digest = hash_key(token)
    row = conn.execute(
        "SELECT id, name, clearance, enabled, key_hash FROM contributors WHERE key_hash = ?",
        (digest,),
    ).fetchone()

    # The lookup is by hash, so it is already constant-time in the interesting
    # sense (an attacker learns nothing from timing a hash-table miss). The
    # compare_digest below guards the case where a future change makes this a
    # scan rather than an indexed lookup.
    if row is None or not hmac.compare_digest(row["key_hash"], digest):
        raise AuthError("unknown key")
    if not row["enabled"]:
        raise AuthError("key revoked")

    return Contributor(id=row["id"], name=row["name"], clearance=row["clearance"])
