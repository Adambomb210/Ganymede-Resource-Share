"""The web UI suite (docs/12-web-ui.md).

Covers the /ui/* page set, the login flow, the 303/404 failure modes, the
content-negotiated enroll token, the new /v1/admin/submitters endpoint, the
SSE hub's authorization semantics, and the CSP header. The hub's per-subscriber
authorization is tested directly against the module -- a live-stream E2E under
TestClient would test TestClient more than the hub.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from ganymede.coordinator import events
from ganymede.coordinator.auth import Contributor, generate_key, hash_key
from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, utcnow
from tests.conftest import FakeStore  # noqa: F401  (fixture imports)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _login(client, username: str, secret: str) -> str:
    """Login via the /ui form; return the session cookie pair."""
    resp = client.post(
        "/ui/login",
        data={"username": username, "secret": secret},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text
    return resp.headers["set-cookie"].split(";")[0]


def _as(client, cookie: str):
    """A headers dict carrying the session cookie + the CSRF header."""
    return {"Cookie": cookie, "X-Ganymede-UI": "1"}


def _make_local_user(conn, name: str, *, is_admin: bool = False,
                     secret: str | None = None, clearance: str = "open"):
    uid = uuid.uuid4().hex
    if secret is None:
        secret = generate_key()   # unique per user: key_hash is UNIQUE
    with immediate(conn):
        conn.execute(
            """INSERT INTO contributors
                 (id, name, key_hash, enabled, clearance, is_admin, auth_provider, created_at)
               VALUES (?, ?, ?, 1, ?, ?, 'local', ?)""",
            (uid, name, hash_key(secret), clearance, 1 if is_admin else 0,
             _iso(utcnow())),
        )
    return uid


# --------------------------------------------------------------------------
# Login / auth flow
# --------------------------------------------------------------------------


def test_login_page_renders_and_sets_csp(client):
    resp = client.get("/ui/login")
    assert resp.status_code == 200
    assert "Sign in" in resp.text
    csp = resp.headers.get("content-security-policy", "")
    assert "script-src 'self'" in csp


def test_bad_credentials_rerender_the_form(client, conn):
    _make_local_user(conn, "alice", secret="right")
    resp = client.post(
        "/ui/login", data={"username": "alice", "secret": "wrong"},
        follow_redirects=False,
    )
    assert resp.status_code == 401
    assert "Sign in" in resp.text


def test_login_sets_cookie_and_redirects(client, conn):
    _make_local_user(conn, "alice", secret="right")
    cookie = _login(client, "alice", "right")
    assert cookie.startswith("ganymede_session=")


def test_login_with_next_honours_ui_paths_only(client, conn):
    """The ``next`` param is an open-redirect seam: only /ui/* survives."""
    _make_local_user(conn, "alice", secret="right")
    resp = client.post(
        "/ui/login",
        data={"username": "alice", "secret": "right", "next": "https://evil.example"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/"
    resp = client.post(
        "/ui/login",
        data={"username": "alice", "secret": "right", "next": "/ui/jobs"},
        follow_redirects=False,
    )
    assert resp.headers["location"] == "/ui/jobs"


def test_unauthenticated_ui_redirects_to_login(client):
    for path in ("/ui/", "/ui/jobs", "/ui/machines", "/ui/leaderboard"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 303, path
        assert resp.headers["location"].startswith("/ui/login"), path


def test_login_page_does_not_connect_sse(client):
    """No sse-connect on the login page: the stream would 401-loop."""
    resp = client.get("/ui/login")
    assert "sse-connect" not in resp.text


# --------------------------------------------------------------------------
# Pages, roles, and 404s
# --------------------------------------------------------------------------


def test_every_page_renders_for_admin(client, conn):
    uid = _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "root", "rootpw")
    hdr = _as(client, cookie)
    for path in ("/ui/", "/ui/jobs", "/ui/queue", "/ui/submitters",
                 "/ui/machines", "/ui/leaderboard"):
        resp = client.get(path, headers={"Cookie": cookie})
        assert resp.status_code == 200, (path, resp.status_code)


def test_non_admin_gets_303_off_admin_pages(client, conn):
    _make_local_user(conn, "alice", secret="pw")
    cookie = _login(client, "alice", "pw")
    for path in ("/ui/queue", "/ui/submitters"):
        resp = client.get(path, headers={"Cookie": cookie},
                          follow_redirects=False)
        assert resp.status_code == 303, path
        assert resp.headers["location"] == "/ui/"


def test_admin_links_absent_for_non_admin(client, conn):
    _make_local_user(conn, "alice", secret="pw")
    cookie = _login(client, "alice", "pw")
    resp = client.get("/ui/", headers={"Cookie": cookie})
    assert 'href="/ui/queue"' not in resp.text
    assert 'href="/ui/submitters"' not in resp.text


def test_job_detail_cross_tenant_is_404(client, conn):
    owner = _make_local_user(conn, "owner", secret="ownpw")
    other = _make_local_user(conn, "mallory", secret="malpw")
    with immediate(conn):
        conn.execute(
            """INSERT INTO jobs (id, owner_id, job_type, spec_json, image_id, status,
                        priority_rank, constraints_json, cancel_mode, created_at)
               VALUES ('job1', ?, 'batch_inference', '{}', NULL, 'draft', 1, '{}', NULL, ?)""",
            (owner, _iso(utcnow())),
        )
    mallory_cookie = _login(client, "mallory", "malpw")
    resp = client.get("/ui/jobs/job1", headers={"Cookie": mallory_cookie})
    assert resp.status_code == 404

    owner_cookie = _login(client, "owner", "ownpw")
    resp = client.get("/ui/jobs/job1", headers={"Cookie": owner_cookie})
    assert resp.status_code == 200
    assert "job1" in resp.text


def test_dashboard_uses_status_py_facts(client, conn):
    """The fleet-health fragment renders scripts/status's stalls() output, not
    a forked derivation (docs/12 page note)."""
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "root", "rootpw")
    resp = client.get("/ui/frag/dashboard-fleet", headers={"Cookie": cookie})
    assert resp.status_code == 200
    assert 'id="fleet-health"' in resp.text
    assert "awake workers" in resp.text


# --------------------------------------------------------------------------
# Fragments
# --------------------------------------------------------------------------


def test_fragment_job_row_hidden_from_other_users(client, conn):
    owner = _make_local_user(conn, "owner", secret="ownpw")
    _make_local_user(conn, "mallory", secret="malpw")
    with immediate(conn):
        conn.execute(
            """INSERT INTO jobs (id, owner_id, job_type, spec_json, image_id, status,
                        priority_rank, constraints_json, cancel_mode, created_at)
               VALUES ('job1', ?, 'batch_inference', '{}', NULL, 'queued', 1, '{}', NULL, ?)""",
            (owner, _iso(utcnow())),
        )
    mallory = _login(client, "mallory", "malpw")
    resp = client.get("/ui/frag/jobs-row/job1", headers={"Cookie": mallory})
    assert resp.status_code == 404

    own = _login(client, "owner", "ownpw")
    resp = client.get("/ui/frag/jobs-row/job1", headers={"Cookie": own})
    assert resp.status_code == 200
    assert 'id="job-row-job1"' in resp.text


def test_fragment_queue_requires_admin(client, conn):
    _make_local_user(conn, "alice", secret="pw")
    cookie = _login(client, "alice", "pw")
    resp = client.get("/ui/frag/queue", headers={"Cookie": cookie})
    assert resp.status_code == 404


def test_fragment_machines_list_refetchable(client, conn):
    _make_local_user(conn, "alice", secret="pw")
    cookie = _login(client, "alice", "pw")
    resp = client.get("/ui/frag/machines-list", headers={"Cookie": cookie})
    assert resp.status_code == 200


# --------------------------------------------------------------------------
# Mutations through the UI: the CSRF gate and content negotiation
# --------------------------------------------------------------------------


def test_cookie_post_without_ui_header_is_403(client, conn):
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "root", "rootpw")
    resp = client.post("/v1/machines/enroll", json={"display_name": "x"},
                       headers={"Cookie": cookie})
    assert resp.status_code == 403
    assert "X-Ganymede-UI" in resp.text


def test_enroll_token_shown_once_via_html_fragment(client, conn):
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "root", "rootpw")
    hdr = _as(client, cookie)
    resp = client.post("/v1/machines/enroll",
                       headers={**hdr, "Accept": "text/html"},
                       json={"display_name": "box"})
    assert resp.status_code == 200
    assert "gme_" in resp.text          # the token, inline, once
    assert "shown once" in resp.text
    # No GET ever returns it: the machines page never echoes a token.
    resp = client.get("/ui/machines", headers={"Cookie": cookie})
    assert "gme_" not in resp.text


def test_enroll_json_caller_gets_frozen_shape(client, conn):
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "root", "rootpw")
    resp = client.post("/v1/machines/enroll", json={"display_name": "box"},
                       headers=_as(client, cookie))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"enroll_token", "enroll_id", "expires_at"}


# --------------------------------------------------------------------------
# POST /v1/admin/submitters/{user_id} (the docs/06 body, now built)
# --------------------------------------------------------------------------


def _seed_submitter(conn, uid, status="pending"):
    with immediate(conn):
        conn.execute(
            "INSERT INTO submitters (user_id, status) VALUES (?, ?)",
            (uid, status),
        )


def test_admin_submitters_decide_approves(client, conn):
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    bob = _make_local_user(conn, "bob")
    _seed_submitter(conn, bob)
    cookie = _login(client, "root", "rootpw")
    resp = client.post(f"/v1/admin/submitters/{bob}",
                       json={"status": "approved", "note": "ok"},
                       headers=_as(client, cookie))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    row = conn.execute(
        "SELECT status, decided_by, note FROM submitters WHERE user_id = ?",
        (bob,),
    ).fetchone()
    assert row["status"] == "approved"
    assert row["note"] == "ok"


def test_admin_submitters_rejects_bad_status(client, conn):
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    bob = _make_local_user(conn, "bob")
    _seed_submitter(conn, bob)
    cookie = _login(client, "root", "rootpw")
    resp = client.post(f"/v1/admin/submitters/{bob}",
                       json={"status": "bogus"}, headers=_as(client, cookie))
    assert resp.status_code == 422


def test_admin_submitters_unknown_user_404(client, conn):
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "root", "rootpw")
    resp = client.post("/v1/admin/submitters/nobody",
                       json={"status": "approved"}, headers=_as(client, cookie))
    assert resp.status_code == 404


def test_admin_submitters_requires_admin(client, conn):
    _make_local_user(conn, "alice", secret="pw")
    _make_local_user(conn, "root", is_admin=True, secret="rootpw")
    cookie = _login(client, "alice", "pw")
    resp = client.post("/v1/admin/submitters/x",
                       json={"status": "approved"}, headers=_as(client, cookie))
    assert resp.status_code == 404  # admin surface is not confirmed to exist


# --------------------------------------------------------------------------
# /v1/events -- the SSE endpoint and hub
# --------------------------------------------------------------------------


def test_events_endpoint_401_without_credential(client):
    resp = client.get("/v1/events")
    assert resp.status_code == 401


def test_events_endpoint_rejects_machine_principals():
    """A Machine authenticates but is not a user (docs/12: auth class user);
    the stream is for browsers. Exercise the endpoint's guard directly."""
    import asyncio
    from unittest.mock import MagicMock
    from ganymede.coordinator.auth import Machine
    from ganymede.coordinator import events as ev

    machine = Machine("m1", "u1", "good")
    with pytest.raises(HTTPException) as ei:
        asyncio.run(ev.events_endpoint(MagicMock(), machine))
    assert ei.value.status_code == 401


def test_hub_authorizes_per_subscriber():
    """The hub's audience table, tested directly (docs/12 envelope schema)."""
    h = events._Hub()
    alice = Contributor("u1", "alice", "open", False)
    admin = Contributor("a9", "root", "open", True)

    async def scenario():
        events._owned_machine_ids_sync = lambda uid: (
            frozenset({"m1"}) if uid == "u1" else frozenset()
        )
        sa = await h.subscribe(alice, None)
        sad = await h.subscribe(admin, None)

        h.publish("job.status", job_id="j1", owner_id="u1")  # alice+admin
        h.publish("queue.change", job_id="j2")                # admin only
        h.publish("fleet.delta")                              # everyone
        h.publish("job.status", job_id="j3", owner_id="uX")   # admin only
        h.publish("standing.change", machine_id="m1")        # alice+admin

        def drain(s):
            out = []
            while not s.queue.empty():
                out.append(s.queue.get_nowait())
            return out

        a = [getattr(i, "type", i) for i in drain(sa)]
        d = [getattr(i, "type", i) for i in drain(sad)]
        return a, d

    a, d = asyncio.run(scenario())
    assert a == ["sync", "job.status", "fleet.delta", "standing.change"]
    # Admin sees everything, in publish order (j3's job.status follows the
    # fleet.delta publish, not precedes it).
    assert d == ["sync", "job.status", "queue.change", "fleet.delta",
                 "job.status", "standing.change"]


def test_hub_replays_ring_after_last_event_id():
    h = events._Hub()
    alice = Contributor("u1", "alice", "open", False)
    admin = Contributor("a9", "root", "open", True)

    async def scenario():
        events._owned_machine_ids_sync = lambda uid: frozenset()
        h.publish("fleet.delta")                    # id 1
        h.publish("queue.change", job_id="j2")      # id 2
        h.publish("fleet.delta")                    # id 3
        sa = await h.subscribe(alice, "1")
        out = []
        while not sa.queue.empty():
            out.append(sa.queue.get_nowait())
        return [getattr(i, "type", i) for i in out]

    got = asyncio.run(scenario())
    # alice is not admin: job.status replay for others is filtered, fleet.delta
    # replays; sync arrives last.
    assert got == ["fleet.delta", "sync"]


def test_hub_publish_from_worker_thread_reaches_subscriber():
    """Sync endpoints run in FastAPI's threadpool; publish must still
    deliver onto the subscriber's event loop (call_soon_threadsafe)."""
    import threading

    h = events._Hub()
    alice = Contributor("u1", "alice", "open", False)

    async def scenario():
        events._owned_machine_ids_sync = lambda uid: frozenset()
        sa = await h.subscribe(alice, None)
        # drain the initial sync
        while not sa.queue.empty():
            sa.queue.get_nowait()

        done = asyncio.Event()

        async def waiter():
            item = await sa.queue.get()
            done.set()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)          # let the waiter start

        # Publish from a plain thread (no running loop).
        t = threading.Thread(
            target=lambda: h.publish("fleet.delta")
        )
        t.start()
        await asyncio.wait_for(done.wait(), timeout=2)
        await task
        t.join()
        return True

    assert asyncio.run(scenario())


def test_publish_unknown_type_is_a_programming_error():
    h = events._Hub()
    with pytest.raises(ValueError):
        h.publish("nonsense.type")


def test_sse_frame_matches_frozen_envelope():
    env = events.Envelope(42, "job.status", {"job_id": "j1"})
    frame = events.sse_frame(env)
    assert frame == (
        'id: 42\nevent: job.status\ndata: {"type": "job.status", "id": 42, "job_id": "j1"}\n\n'
    )
