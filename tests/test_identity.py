"""Identity, enrollment, and admin (docs/08-identity.md).

Drives the real ``create_app`` through a ``TestClient`` against the file-backed
SQLite database, exactly like ``test_integration.py``. Covers the three token
kinds in ``authenticate``, the CSRF header rule, the placeholder-provider
session endpoints, the machine-enrollment flow, and the bootstrap admin.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from fastapi import FastAPI

from ganymede.coordinator import rounds
from ganymede.coordinator.app import (
    create_app,
    require_admin,
    require_machine,
    require_submitter,
)
from ganymede.coordinator.auth import Contributor, Machine, authenticate, hash_key
from ganymede.coordinator.identity import ensure_bootstrap_admin, mint_session


@pytest.fixture
def enrolled_machine(client, make_contributor):
    """Run the enroll -> claim-enrollment flow; return (owner_id, owner_key,
    machine_id, machine_key)."""
    owner_id, owner_key = make_contributor(name="fleet-owner")
    enroll = client.post(
        "/v1/machines/enroll",
        headers={"Authorization": f"Bearer {owner_key}"},
        json={"display_name": "lab-box"},
    )
    assert enroll.status_code == 200, enroll.text
    token = enroll.json()["enroll_token"]
    claim = client.post(
        "/v1/machines/claim-enrollment",
        json={"enroll_token": token,
              "compute_profile": {"backend": "cuda", "device_name": "RTX 4090",
                                  "vram_mb": 24564}},
    )
    assert claim.status_code == 200, claim.text
    body = claim.json()
    return owner_id, owner_key, body["machine_id"], body["machine_key"]


# --------------------------------------------------------------------------
# authenticate() -- the three token kinds
# --------------------------------------------------------------------------


def test_contributor_key_still_resolves_to_a_contributor(conn, make_contributor):
    cid, key = make_contributor(name="cli-user")
    principal = authenticate(conn, f"Bearer {key}")
    assert isinstance(principal, Contributor)
    assert principal.id == cid and principal.is_admin is False


def test_admin_flag_flows_through_authenticate(conn, make_contributor):
    cid, key = make_contributor(name="boss")
    conn.execute("UPDATE contributors SET is_admin = 1 WHERE id = ?", (cid,))
    principal = authenticate(conn, f"Bearer {key}")
    assert isinstance(principal, Contributor) and principal.is_admin is True


def test_machine_key_resolves_to_a_machine(client, conn, enrolled_machine):
    _owner_id, _owner_key, machine_id, machine_key = enrolled_machine
    principal = authenticate(conn, f"Bearer {machine_key}")
    assert isinstance(principal, Machine)
    assert principal.id == machine_id and principal.standing == "good"


def test_disabled_machine_key_is_rejected(client, conn, enrolled_machine):
    from ganymede.coordinator.auth import AuthError

    _o, _k, machine_id, machine_key = enrolled_machine
    conn.execute("UPDATE machine_keys SET enabled = 0 WHERE machine_id = ?", (machine_id,))
    with pytest.raises(AuthError):
        authenticate(conn, f"Bearer {machine_key}")


def test_session_token_resolves_and_expires(conn, make_contributor):
    from ganymede.coordinator.auth import AuthError

    cid, _key = make_contributor(name="web-user")
    minted = mint_session(conn, cid, ttl_sec=3600)
    conn.commit()
    principal = authenticate(conn, header=None, cookie=minted.token)
    assert isinstance(principal, Contributor) and principal.id == cid

    past = rounds._iso(rounds.utcnow() - timedelta(seconds=1))
    conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
                 (past, hash_key(minted.token)))
    with pytest.raises(AuthError):
        authenticate(conn, header=None, cookie=minted.token)


# --------------------------------------------------------------------------
# CSRF -- SameSite + X-Ganymede-UI (docs/06, docs/08)
# --------------------------------------------------------------------------


def test_cookie_post_without_ui_header_is_403(client, conn, make_contributor):
    cid, _key = make_contributor(name="csrf-user")
    minted = mint_session(conn, cid, ttl_sec=3600)
    conn.commit()
    client.cookies.set("ganymede_session", minted.token)
    r = client.post("/v1/machines/enroll", json={"display_name": "x"})
    assert r.status_code == 403
    r_ok = client.post("/v1/machines/enroll", json={"display_name": "x"},
                       headers={"X-Ganymede-UI": "1"})
    assert r_ok.status_code == 200
    client.cookies.clear()


def test_bearer_post_needs_no_ui_header(client, make_contributor):
    _cid, key = make_contributor(name="bearer-user")
    r = client.post("/v1/machines/enroll", headers={"Authorization": f"Bearer {key}"},
                    json={"display_name": "x"})
    assert r.status_code == 200


# --------------------------------------------------------------------------
# POST/DELETE /v1/auth/session  (the local placeholder provider)
# --------------------------------------------------------------------------


def test_session_login_sets_cookie_and_body(client, make_contributor):
    _cid, key = make_contributor(name="alice")
    r = client.post("/v1/auth/session", json={"username": "alice", "secret": key})
    assert r.status_code == 200
    body = r.json()
    assert body["session_token"] and body["expires_at"]
    assert "ganymede_session" in r.cookies
    client.cookies.clear()


def test_session_login_rejects_bad_secret_and_unknown_user(client, make_contributor):
    _cid, _key = make_contributor(name="bob")
    assert client.post("/v1/auth/session",
                       json={"username": "bob", "secret": "wrong"}).status_code == 401
    assert client.post("/v1/auth/session",
                       json={"username": "nobody", "secret": "x"}).status_code == 401


def test_session_delete_revokes_the_row(client, conn, make_contributor):
    cid, key = make_contributor(name="carol")
    login = client.post("/v1/auth/session", json={"username": "carol", "secret": key})
    token = login.json()["session_token"]
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?",
                        (cid,)).fetchone()[0] == 1
    client.cookies.set("ganymede_session", token)
    r = client.delete("/v1/auth/session", headers={"X-Ganymede-UI": "1"})
    assert r.status_code == 200
    assert conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?",
                        (cid,)).fetchone()[0] == 0
    client.cookies.clear()


# --------------------------------------------------------------------------
# Machine enrollment
# --------------------------------------------------------------------------


def test_claim_enrollment_writes_worker_key_and_consumes_token(client, conn,
                                                               enrolled_machine):
    owner_id, _owner_key, machine_id, machine_key = enrolled_machine

    w = conn.execute("SELECT * FROM workers WHERE id=?", (machine_id,)).fetchone()
    assert w["contributor_id"] == owner_id and w["standing"] == "good"
    assert w["display_name"] == "lab-box" and w["enrolled_at"] is not None
    assert w["reputation"] == 0.25

    k = conn.execute("SELECT * FROM machine_keys WHERE machine_id=?", (machine_id,)).fetchone()
    assert k["key_hash"] == hash_key(machine_key) and k["enabled"] == 1

    e = conn.execute("SELECT * FROM enrollments WHERE machine_id=?", (machine_id,)).fetchone()
    assert e["consumed_at"] is not None


def test_enrollment_token_is_single_use(client, make_contributor):
    _oid, okey = make_contributor(name="one-shot")
    token = client.post("/v1/machines/enroll", headers={"Authorization": f"Bearer {okey}"},
                        json={"display_name": "m"}).json()["enroll_token"]
    body = {"enroll_token": token,
            "compute_profile": {"backend": "cuda", "device_name": "g", "vram_mb": 1}}
    assert client.post("/v1/machines/claim-enrollment", json=body).status_code == 200
    # Second attempt: consumed -> 404, indistinguishable from unknown.
    assert client.post("/v1/machines/claim-enrollment", json=body).status_code == 404


def test_unknown_and_expired_tokens_are_404(client, conn, make_contributor):
    assert client.post("/v1/machines/claim-enrollment",
                       json={"enroll_token": "gme_nope",
                             "compute_profile": {"backend": "cpu"}}).status_code == 404

    _oid, okey = make_contributor(name="slowpoke")
    token = client.post("/v1/machines/enroll", headers={"Authorization": f"Bearer {okey}"},
                        json={"display_name": "m"}).json()["enroll_token"]
    stale = rounds._iso(rounds.utcnow() - timedelta(days=1))
    conn.execute("UPDATE enrollments SET created_at = ? WHERE token_hash = ?",
                 (stale, hash_key(token)))
    r = client.post("/v1/machines/claim-enrollment",
                    json={"enroll_token": token,
                          "compute_profile": {"backend": "cpu"}})
    assert r.status_code == 404


def test_machine_key_authenticates_end_to_end(client, enrolled_machine):
    _o, _k, machine_id, machine_key = enrolled_machine
    # GET /v1/machines is user-only; a machine key must not pass it.
    r = client.get("/v1/machines", headers={"Authorization": f"Bearer {machine_key}"})
    assert r.status_code == 401


def test_machines_list_is_scoped_to_the_owner(client, make_contributor, enrolled_machine):
    owner_id, owner_key, machine_id, _mk = enrolled_machine
    mine = client.get("/v1/machines", headers={"Authorization": f"Bearer {owner_key}"})
    assert mine.status_code == 200
    ids = {m["machine_id"] for m in mine.json()["machines"]}
    assert machine_id in ids

    _other_id, other_key = make_contributor(name="stranger")
    theirs = client.get("/v1/machines", headers={"Authorization": f"Bearer {other_key}"})
    assert theirs.json()["machines"] == []


def test_retire_revokes_standing_and_keys(client, conn, enrolled_machine):
    _o, owner_key, machine_id, machine_key = enrolled_machine
    r = client.post(f"/v1/machines/{machine_id}/retire",
                    headers={"Authorization": f"Bearer {owner_key}"})
    assert r.status_code == 200 and r.json()["standing"] == "revoked"
    w = conn.execute("SELECT standing FROM workers WHERE id=?", (machine_id,)).fetchone()
    assert w["standing"] == "revoked"
    k = conn.execute("SELECT enabled FROM machine_keys WHERE machine_id=?",
                     (machine_id,)).fetchone()
    assert k["enabled"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM audit WHERE event='owner_retire' AND worker_id=?",
        (machine_id,)).fetchone()[0] == 1


def test_rotate_key_swaps_the_credential(client, conn, enrolled_machine):
    _o, owner_key, machine_id, old_key = enrolled_machine
    r = client.post(f"/v1/machines/{machine_id}/rotate-key",
                    headers={"Authorization": f"Bearer {owner_key}"})
    assert r.status_code == 200
    new_key = r.json()["machine_key"]
    assert new_key != old_key
    assert isinstance(authenticate(conn, f"Bearer {new_key}"), Machine)
    from ganymede.coordinator.auth import AuthError
    with pytest.raises(AuthError):
        authenticate(conn, f"Bearer {old_key}")


def test_non_owner_machine_actions_are_404(client, make_contributor, enrolled_machine):
    _o, _ok, machine_id, _mk = enrolled_machine
    _sid, skey = make_contributor(name="intruder")
    h = {"Authorization": f"Bearer {skey}"}
    assert client.post(f"/v1/machines/{machine_id}/retire", headers=h).status_code == 404
    assert client.post(f"/v1/machines/{machine_id}/rotate-key", headers=h).status_code == 404


# --------------------------------------------------------------------------
# The legacy-worker transitional auth rule (Spine deviation 4)
# --------------------------------------------------------------------------


class _Req:
    """Minimal stand-in for a Starlette Request for a direct dependency call."""

    def __init__(self, app, method="GET", headers=None):
        self.app = app
        self.method = method
        self.headers = headers or {}
        self.cookies = {}
        self.url = type("U", (), {"scheme": "http"})()


def test_keyless_pre004_worker_authenticates_with_owner_contributor_key(
    settings, store, conn, make_contributor
):
    app = create_app(settings, store)
    cid, key = make_contributor(name="legacy-owner")
    wid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO workers (id, contributor_id, compute_profile_json, first_seen, "
        "last_seen, standing) VALUES (?, ?, '{}', 'now', 'now', 'good')",
        (wid, cid),
    )
    conn.commit()

    principal = require_machine(_Req(app), conn, f"Bearer {key}")
    assert isinstance(principal, Machine) and principal.id == wid
    assert conn.execute(
        "SELECT COUNT(*) FROM audit WHERE event='legacy_worker_auth' AND worker_id=?",
        (wid,)).fetchone()[0] == 1


def test_contributor_with_no_keyless_worker_gets_404_from_require_machine(
    settings, store, conn, make_contributor
):
    from fastapi import HTTPException

    app = create_app(settings, store)
    _cid, key = make_contributor(name="no-machines")
    with pytest.raises(HTTPException) as exc:
        require_machine(_Req(app), conn, f"Bearer {key}")
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------
# require_submitter / require_admin
# --------------------------------------------------------------------------


def test_require_submitter_needs_an_approved_row(settings, store, conn, make_contributor):
    from fastapi import HTTPException

    app = create_app(settings, store)
    cid, key = make_contributor(name="would-be-submitter")
    with pytest.raises(HTTPException) as exc:
        require_submitter(_Req(app), conn, f"Bearer {key}")
    assert exc.value.status_code == 404

    conn.execute("INSERT INTO submitters (user_id, status) VALUES (?, 'approved')", (cid,))
    conn.commit()
    assert require_submitter(_Req(app), conn, f"Bearer {key}").id == cid


def test_require_admin_is_404_for_non_admins(settings, store, conn, make_contributor):
    from fastapi import HTTPException

    app = create_app(settings, store)
    cid, key = make_contributor(name="plain-user")
    with pytest.raises(HTTPException) as exc:
        require_admin(_Req(app), conn, f"Bearer {key}")
    assert exc.value.status_code == 404

    conn.execute("UPDATE contributors SET is_admin = 1 WHERE id = ?", (cid,))
    conn.commit()
    assert require_admin(_Req(app), conn, f"Bearer {key}").is_admin is True


# --------------------------------------------------------------------------
# Bootstrap admin (docs/08 "GANYMEDE_BOOTSTRAP_ADMIN")
# --------------------------------------------------------------------------


def test_bootstrap_admin_creates_promotes_and_never_demotes(conn, make_contributor):
    # name:secret -> created with that key, restricted clearance, is_admin.
    ensure_bootstrap_admin(conn, "root:sekret")
    row = conn.execute(
        "SELECT id, is_admin, clearance, key_hash FROM contributors WHERE name='root'"
    ).fetchone()
    assert row["is_admin"] == 1 and row["clearance"] == "restricted"
    assert row["key_hash"] == hash_key("sekret")

    # An existing plain user named in the spec is promoted, not duplicated.
    uid, _key = make_contributor(name="promote-me")
    ensure_bootstrap_admin(conn, "promote-me")
    rows = conn.execute("SELECT is_admin FROM contributors WHERE name='promote-me'").fetchall()
    assert len(rows) == 1 and rows[0]["is_admin"] == 1

    # Idempotent, and dropping the var never demotes.
    ensure_bootstrap_admin(conn, None)
    assert conn.execute(
        "SELECT is_admin FROM contributors WHERE name='root'").fetchone()["is_admin"] == 1


# --------------------------------------------------------------------------
# scripts/reenroll_fleet.py (docs/08 "Migration 004")
# --------------------------------------------------------------------------


def test_reenroll_fleet_issues_a_token_per_keyless_worker(settings, conn, make_contributor,
                                                          capsys):
    from scripts import reenroll_fleet

    cid, _key = make_contributor(name="orphan-owner")
    wid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO workers (id, contributor_id, compute_profile_json, first_seen, "
        "last_seen, standing) VALUES (?, ?, '{}', 'now', 'now', 'good')",
        (wid, cid),
    )
    conn.commit()

    assert reenroll_fleet.main([], settings=settings) == 0
    out = capsys.readouterr().out
    assert wid in out and "gme_" in out

    row = conn.execute("SELECT * FROM enrollments WHERE machine_id IS NULL "
                       "AND user_id = ?", (cid,)).fetchone()
    assert row is not None and len(row["token_hash"]) == 64  # a real hash_key digest

    # Second run: the worker still has no key, so it is offered another token;
    # a worker that DID get a key is skipped.
    conn.execute("INSERT INTO machine_keys (machine_id, key_hash, enabled, created_at) "
                 "VALUES (?, ?, 1, 'now')", (wid, hash_key("gmk_whatever")))
    conn.commit()
    assert reenroll_fleet.main([], settings=settings) == 0
    assert "nothing to re-enroll" in capsys.readouterr().out
