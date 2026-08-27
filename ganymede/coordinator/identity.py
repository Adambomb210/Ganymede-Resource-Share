"""The identity provider seam, the session mechanism, and the bootstrap admin
(docs/08-identity.md).

`06` picks a *placeholder* provider (Decision 5) so accounts ship without an
OAuth dependency. The seam is one interface -- ``IdentityProvider`` -- with the
placeholder ``local`` provider as its only implementation here; a real GitHub /
OIDC provider is a later drop-in behind the same interface, changing nothing
about ``authenticate()``, ``sessions``, the ``require_*`` dependencies, or the
``contributors`` schema.

The coordinator owns the ``contributors`` row and the session. A provider only
turns a submitted credential into a verified ``ProviderIdentity``.
"""

from __future__ import annotations

import hmac
import json
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from ganymede.coordinator.auth import AuthError, generate_key, hash_key
from ganymede.coordinator.rounds import _iso, _parse, utcnow

# Token shapes are frozen in docs/08 "Frozen for downstream". All three are
# ``secrets.token_urlsafe(32)``; the machine and enrollment tokens carry a
# prefix so the spaces stay visibly disjoint, the session token has none
# (it travels as a cookie value too).
ENROLL_PREFIX = "gme_"
MACHINE_KEY_PREFIX = "gmk_"


@dataclass(frozen=True)
class ProviderIdentity:
    """What a provider vouches for. ``auth_subject`` is the provider's stable id
    (``sub`` / the GitHub user id); ``None`` for ``local``, whose rows are keyed
    by ``name`` (docs/08)."""

    auth_provider: str        # 'local' | 'github' | 'oidc'
    auth_subject: str | None
    name: str
    email: str | None


@dataclass(frozen=True)
class AuthChallenge:
    """A redirect provider's authorize URL plus the opaque state to round-trip.
    ``local`` has no challenge and returns ``None`` from ``begin``."""

    authorize_url: str
    state: str


class IdentityProvider(Protocol):
    name: str

    def begin(self, redirect_uri: str) -> AuthChallenge | None: ...

    def verify(self, credential: dict, conn: sqlite3.Connection) -> ProviderIdentity: ...


class LocalProvider:
    """The placeholder (docs/08 "The `local` placeholder").

    ``verify`` takes ``{"username", "secret"}`` and checks the secret against
    ``contributors.key_hash`` -- the secret *is* the contributor's issued key,
    the same 32-byte token ``auth.hash_key`` already stores. No password column,
    no KDF: these are random tokens, not passwords (auth.hash_key's own
    argument). When a real provider lands, the oddity leaves with it.
    """

    name = "local"

    def begin(self, redirect_uri: str) -> AuthChallenge | None:
        return None

    def verify(self, credential: dict, conn: sqlite3.Connection) -> ProviderIdentity:
        username = (credential or {}).get("username")
        secret = (credential or {}).get("secret")
        if not username or not secret:
            raise AuthError("username and secret required")
        row = conn.execute(
            "SELECT id, key_hash, enabled FROM contributors "
            "WHERE auth_provider = 'local' AND (name = ? OR email = ?)",
            (username, username),
        ).fetchone()
        if row is None:
            raise AuthError("unknown user")
        if not hmac.compare_digest(row["key_hash"], hash_key(secret)):
            raise AuthError("bad secret")
        if not row["enabled"]:
            raise AuthError("user disabled")
        return ProviderIdentity("local", None, username, None)


# env ``GANYMEDE_AUTH_PROVIDER`` selects one of these; default ``local``.
PROVIDERS: dict[str, IdentityProvider] = {
    LocalProvider.name: LocalProvider(),
}


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MintedSession:
    token: str          # shown once -- cookie value and bearer body
    expires_at: str     # ISO 8601, absolute; no sliding renewal in the placeholder


def mint_session(conn: sqlite3.Connection, user_id: str, ttl_sec: int) -> MintedSession:
    """Create one ``sessions`` row and return the plaintext token once. Only
    ``hash_key(token)`` is stored -- a DB dump is not a set of live sessions
    (docs/08 "Session token")."""
    token = secrets.token_urlsafe(32)
    now = utcnow()
    expires_at = _iso(now + timedelta(seconds=ttl_sec))
    conn.execute(
        """INSERT INTO sessions (token_hash, user_id, created_at, expires_at, last_used_at)
           VALUES (?, ?, ?, ?, ?)""",
        (hash_key(token), user_id, _iso(now), expires_at, _iso(now)),
    )
    return MintedSession(token=token, expires_at=expires_at)


def resolve_session_row(conn: sqlite3.Connection, token: str):
    """The ``sessions`` half of ``authenticate`` -- a joined row or ``None``,
    with the absolute-expiry check applied here so the caller never sees a dead
    session. Kept next to ``mint_session`` so the two halves of the mechanism
    read together."""
    row = conn.execute(
        """SELECT c.id, c.name, c.clearance, c.is_admin, c.enabled, s.expires_at
             FROM sessions s JOIN contributors c ON c.id = s.user_id
            WHERE s.token_hash = ?""",
        (hash_key(token),),
    ).fetchone()
    if row is None or not row["enabled"]:
        return None
    if utcnow() >= _parse(row["expires_at"]):
        return None
    return row


def gc_expired_sessions(conn: sqlite3.Connection) -> int:
    """Drop rows whose absolute expiry has passed. Runs on the ``audit`` /
    ``availability_ticks`` cron (docs/08); returns the row count for the caller
    to log."""
    cur = conn.execute(
        "DELETE FROM sessions WHERE expires_at <= ?", (_iso(utcnow()),)
    )
    return cur.rowcount


# --------------------------------------------------------------------------
# Machine enrollment support
# --------------------------------------------------------------------------


def new_enroll_token() -> str:
    return ENROLL_PREFIX + secrets.token_urlsafe(32)


def new_machine_key() -> str:
    return MACHINE_KEY_PREFIX + secrets.token_urlsafe(32)


def enroll_is_expired(created_at: str, ttl_sec: int) -> bool:
    return utcnow() > _parse(created_at) + timedelta(seconds=ttl_sec)


def fingerprint_from_profile(profile: dict) -> str:
    """The advisory hardware fingerprint, coordinator-derived from the submitted
    ``compute_profile`` (docs/08). **Advisory only** (invariant 1): it drives
    "this looks like a different computer" fraud review and nothing else -- never
    identity, never accrual, never ``credit()``. A best-effort projection of
    whatever the probe reported is all it needs, so a legitimate GPU upgrade
    changing it is the expected case, not the alarm."""
    probe = (profile or {}).get("probe") or {}
    return json.dumps(
        {
            "gpu_model": (profile or {}).get("device_name"),
            "gpu_uuid": probe.get("gpu_uuid"),
            "cpu_model": probe.get("cpu_model"),
            "cpu_count": probe.get("cpu_count"),
            "total_ram_mb": probe.get("total_ram_mb"),
            "board_serial": probe.get("board_serial"),
            "vram_mb": (profile or {}).get("vram_mb"),
            "backend": (profile or {}).get("backend"),
        },
        sort_keys=True,
    )


# --------------------------------------------------------------------------
# Bootstrap admin (docs/08 "GANYMEDE_BOOTSTRAP_ADMIN")
# --------------------------------------------------------------------------


def ensure_bootstrap_admin(conn: sqlite3.Connection, spec: str | None) -> None:
    """Set the first admin from an env var, not through the API (Decision 14 --
    `06`'s admin surface has no ``is_admin`` writer). ``spec`` is comma-separated
    ``name`` or ``name:secret``.

    Idempotent -- safe on every boot. Promotes an existing ``local`` row or
    creates one with ``clearance = 'restricted'`` (so an admin sees every run).
    **Never demotes**: removing the var does not strip ``is_admin``.
    """
    if not spec or not spec.strip():
        return
    from ganymede.coordinator.db import immediate

    now = _iso(utcnow())
    with immediate(conn):
        for entry in spec.split(","):
            name, _sep, secret = entry.strip().partition(":")
            if not name:
                continue
            row = conn.execute(
                "SELECT id FROM contributors WHERE auth_provider = 'local' AND name = ?",
                (name,),
            ).fetchone()
            if row is None:
                key = secret or generate_key()
                conn.execute(
                    "INSERT INTO contributors "
                    "(id, name, key_hash, enabled, clearance, is_admin, auth_provider, created_at) "
                    "VALUES (?, ?, ?, 1, 'restricted', 1, 'local', ?)",
                    (uuid.uuid4().hex, name, hash_key(key), now),
                )
                if not secret:
                    # Shown once, like scripts/issue_key.py -- there is no
                    # recovering it later.
                    print(f"[bootstrap] minted admin key for {name!r}: {key}")
            else:
                conn.execute(
                    "UPDATE contributors SET is_admin = 1 WHERE id = ?", (row["id"],)
                )
                if secret:
                    conn.execute(
                        "UPDATE contributors SET key_hash = ? WHERE id = ?",
                        (hash_key(secret), row["id"]),
                    )
