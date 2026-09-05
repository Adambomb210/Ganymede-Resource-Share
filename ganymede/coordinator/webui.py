"""The server-rendered web UI under ``/ui/*`` (docs/12-web-ui.md).

Jinja2 + htmx, no build toolchain: ``htmx.min.js`` and the SSE extension are
vendored under ``static/`` and served from ``/ui/static/`` with
``script-src 'self'``. Every mutation is an explicit ``POST`` to a ``/v1/*``
endpoint (login is the C1 exception); every live update is an SSE-triggered
``hx-get`` against this module's fragment endpoints.

Roles (docs/12 page table): user / submitter / admin. A page a role may not
see is a ``303 -> /ui/`` (admin-only) or ``404`` (another user's resource),
never a naked 403.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ganymede.coordinator import events, identity, readmodel
from ganymede.coordinator.app import (
    ConnDep,
    _SESSION_COOKIE,
)
from ganymede.coordinator.auth import AuthError, Contributor, authenticate, hash_key
from ganymede.coordinator.config import Settings

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))

CSP = "default-src 'self'; script-src 'self'; style-src 'self'; " \
      "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"

USER_SAFE_METHODS = {"GET", "HEAD"}


def _ui_user(request: Request, conn: sqlite3.Connection) -> Contributor | None:
    """Resolve the session cookie to a Contributor, or None -> redirect to
    login. /ui/* never accepts bearer keys (docs/12: cookie-only)."""
    cookie = request.cookies.get(_SESSION_COOKIE)
    if not cookie:
        return None
    try:
        return authenticate(conn, None, cookie=cookie)
    except AuthError:
        return None


def _redirect_login(request: Request) -> RedirectResponse:
    nxt = request.url.path
    return RedirectResponse(f"/ui/login?next={nxt}", status_code=303)


def _csp(response: HTMLResponse) -> HTMLResponse:
    response.headers["Content-Security-Policy"] = CSP
    return response


def mount(app: FastAPI, settings: Settings) -> None:
    """Attach the /ui routes to the existing app. Called from create_app --
    same process, same per-request connection."""
    from starlette.staticfiles import StaticFiles

    app.mount(
        "/ui/static",
        StaticFiles(directory=str(_HERE / "static")),
        name="ui-static",
    )

    # ---------------------------------------------------------------- login

    @app.get("/ui/login", response_class=HTMLResponse)
    def ui_login(request: Request, conn: ConnDep) -> HTMLResponse:
        if _ui_user(request, conn) is not None:
            return RedirectResponse("/ui/", status_code=303)
        return _csp(templates.TemplateResponse(
            request, "login.html", {"error": None, "next": None}
        ))

    @app.post("/ui/login", response_class=HTMLResponse)
    def ui_login_post(
        request: Request,
        conn: ConnDep,
        username: Annotated[str, Form()],
        secret: Annotated[str, Form()],
        next: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        """The one POST /ui/* endpoint (docs/12 C1: \"No POST except login\").
        Reuses the placeholder provider's verify + identity.mint_session -- the
        same machinery ``POST /v1/auth/session`` uses, so a real OAuth drop-in
        changes both at once."""
        provider = identity.PROVIDERS.get(settings.auth_provider)
        if provider is None:
            raise HTTPException(status_code=500, detail="no such auth provider")
        error = None
        try:
            ident = provider.verify({"username": username, "secret": secret}, conn)
        except AuthError as exc:
            error = str(exc)
        if error is None:
            row = conn.execute(
                """SELECT id, enabled, is_admin FROM contributors
                    WHERE auth_provider = ? AND (auth_subject = ? OR name = ?)""",
                (ident.auth_provider, ident.auth_subject, ident.name),
            ).fetchone()
            if row is None or not row["enabled"]:
                error = "unknown credential"
        if error is not None:
            return _csp(templates.TemplateResponse(
                request, "login.html", {"error": error, "next": next or None},
                status_code=401,
            ))
        from ganymede.coordinator.db import immediate

        with immediate(conn):
            minted = identity.mint_session(conn, row["id"], settings.session_ttl_sec)
        target = next if next.startswith("/ui") else "/ui/"
        resp = RedirectResponse(target, status_code=303)
        resp.set_cookie(
            _SESSION_COOKIE, minted.token, httponly=True, secure=True,
            samesite="lax", path="/",
        )
        return resp

    @app.post("/ui/logout")
    def ui_logout(request: Request, conn: ConnDep) -> RedirectResponse:
        cookie = request.cookies.get(_SESSION_COOKIE)
        if cookie:
            from ganymede.coordinator.db import immediate

            with immediate(conn):
                conn.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (hash_key(cookie),)
                )
        resp = RedirectResponse("/ui/login", status_code=303)
        resp.delete_cookie(_SESSION_COOKIE, path="/")
        return resp

    # ------------------------------------------------- page helpers / auth

    def _page_user(request: Request, conn: sqlite3.Connection) -> tuple[Contributor | None, HTMLResponse | None]:
        """(user, None) or (None, redirect-to-login)."""
        user = _ui_user(request, conn)
        if user is None:
            return None, _redirect_login(request)
        return user, None

    def _need_admin(user: Contributor) -> HTMLResponse | None:
        if not user.is_admin:
            return RedirectResponse("/ui/", status_code=303)
        return None

    def _404() -> HTTPException:
        return HTTPException(status_code=404, detail="not found")

    # ---------------------------------------------------------------- pages

    @app.get("/ui/", response_class=HTMLResponse)
    def ui_dashboard(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        from ganymede.coordinator import invariants
        from scripts import status as status_mod

        data = readmodel.dashboard(conn)
        awake = status_mod.awake_workers(conn)
        stall_list = status_mod.stalls(conn)
        violations = invariants.check(conn)
        admin_fleet = readmodel_fleet = None
        if user.is_admin:
            from ganymede.coordinator import eligibility

            admin_fleet = eligibility.fleet_summary(conn)
        return _csp(templates.TemplateResponse(request, "dashboard.html", {
            "user": user, **data,
            "awake": awake,
            "stalls": stall_list,
            "violations": violations,
            "fleet_summary": admin_fleet,
            "is_admin": user.is_admin,
        }))

    @app.get("/ui/jobs", response_class=HTMLResponse)
    def ui_jobs(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        before = None
        raw = request.query_params.get("before")
        if raw and "," in raw:
            created, jid = raw.split(",", 1)
            before = (created, jid)
        rows = readmodel.jobs_page(conn, user.id, is_admin=user.is_admin, before=before)
        return _csp(templates.TemplateResponse(request, "jobs.html", {
            "user": user, "jobs": rows, "before": before, "is_admin": user.is_admin,
        }))

    @app.get("/ui/jobs/{job_id}", response_class=HTMLResponse)
    def ui_job_detail(request: Request, job_id: str, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        data = readmodel.job_detail(conn, job_id)
        if data is None:
            raise _404()
        if data["job"]["owner_id"] != user.id and not user.is_admin:
            raise _404()
        return _csp(templates.TemplateResponse(request, "job_detail.html", {
            "user": user, **data, "is_admin": user.is_admin,
        }))

    @app.get("/ui/queue", response_class=HTMLResponse)
    def ui_queue(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        denied = _need_admin(user)
        if denied:
            return denied
        return _csp(templates.TemplateResponse(request, "queue.html", {
            "user": user, "queue": readmodel.queue_page(conn), "is_admin": True,
        }))

    @app.get("/ui/submitters", response_class=HTMLResponse)
    def ui_submitters(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        denied = _need_admin(user)
        if denied:
            return denied
        return _csp(templates.TemplateResponse(request, "submitters.html", {
            "user": user, "submitters": readmodel.submitters_page(conn),
            "is_admin": True,
        }))

    @app.get("/ui/machines", response_class=HTMLResponse)
    def ui_machines(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        data = readmodel.machines_page(conn, user.id)
        return _csp(templates.TemplateResponse(request, "machines.html", {
            "user": user, **data, "is_admin": user.is_admin,
        }))

    @app.get("/ui/leaderboard", response_class=HTMLResponse)
    def ui_leaderboard(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        scope = request.query_params.get("scope", "machines")
        if scope not in ("machines", "users"):
            scope = "machines"
        data = readmodel.leaderboard_rows(conn, scope)
        return _csp(templates.TemplateResponse(request, "leaderboard.html", {
            "user": user, "rows": data["rows"], "scope": data["scope"],
            "is_admin": user.is_admin,
        }))

    # ------------------------------------------------------------ fragments

    # Every live region is an idempotent fragment endpoint: SSE fires
    # ``hx-trigger="sse:<type>"`` + ``hx-get`` against these, the swap
    # re-reads current truth through the read model, and applying the same
    # swap twice is harmless (docs/12 "htmx swap").

    @app.get("/ui/frag/dashboard-fleet", response_class=HTMLResponse)
    def frag_dashboard_fleet(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        from ganymede.coordinator import invariants
        from scripts import status as status_mod

        data = readmodel.dashboard(conn)
        return _csp(templates.TemplateResponse(request, "frags/fleet.html", {
            "user": user,
            "awake": status_mod.awake_workers(conn),
            "stalls": status_mod.stalls(conn),
            "violations": invariants.check(conn),
            "queued_jobs": data["queued_jobs"],
            "leased_tasks": data["leased_tasks"],
            "is_admin": user.is_admin,
        }))

    @app.get("/ui/frag/jobs-row/{job_id}", response_class=HTMLResponse)
    def frag_job_row(request: Request, job_id: str, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        rows = readmodel.jobs_page(conn, user.id, is_admin=user.is_admin, limit=100)
        row = next((r for r in rows if r["id"] == job_id), None)
        if row is None:
            # 404 on the fragment itself: outerHTML swap of nothing is a no-op.
            raise _404()
        return _csp(templates.TemplateResponse(request, "frags/job_row.html", {
            "job": row, "is_admin": user.is_admin,
        }))

    @app.get("/ui/frag/jobs-table", response_class=HTMLResponse)
    def ui_frag_jobs_table(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        rows = readmodel.jobs_page(conn, user.id, is_admin=user.is_admin, limit=100)
        return _csp(templates.TemplateResponse(request, "frags/jobs_table.html", {
            "jobs": rows, "is_admin": user.is_admin,
        }))

    @app.get("/ui/frag/job-header/{job_id}", response_class=HTMLResponse)
    def frag_job_header(job_id: str, request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        data = readmodel.job_detail(conn, job_id)
        if data is None or (data["job"]["owner_id"] != user.id and not user.is_admin):
            raise _404()
        return _csp(templates.TemplateResponse(request, "frags/job_header.html", {
            "job": data["job"], "is_admin": user.is_admin,
        }))

    @app.get("/ui/frag/rounds/{job_id}", response_class=HTMLResponse)
    def frag_rounds(job_id: str, request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        data = readmodel.job_detail(conn, job_id)
        if data is None or (data["job"]["owner_id"] != user.id and not user.is_admin):
            raise _404()
        return _csp(templates.TemplateResponse(request, "frags/rounds.html", {
            "job_id": job_id, "rounds": data["rounds"], "coverage": data["coverage"],
        }))

    @app.get("/ui/frag/queue", response_class=HTMLResponse)
    def frag_queue(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        denied = _need_admin(user)
        if denied:
            raise _404()
        return _csp(templates.TemplateResponse(request, "frags/queue_table.html", {
            "queue": readmodel.queue_page(conn), "is_admin": True,
        }))

    @app.get("/ui/frag/submitters", response_class=HTMLResponse)
    def frag_submitters(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        denied = _need_admin(user)
        if denied:
            raise _404()
        return _csp(templates.TemplateResponse(request, "frags/submitters_table.html", {
            "submitters": readmodel.submitters_page(conn), "is_admin": True,
        }))

    @app.get("/ui/frag/machines-row/{machine_id}", response_class=HTMLResponse)
    def frag_machines_row(machine_id: str, request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        data = readmodel.machines_page(conn, user.id)
        row = next(
            (m for m in data["machines"] if m["machine_id"] == machine_id), None
        )
        if row is None:
            raise _404()
        return _csp(templates.TemplateResponse(request, "frags/machine_row.html", {
            "m": row,
        }))

    @app.get("/ui/frag/machines-list", response_class=HTMLResponse)
    def frag_machines_list(request: Request, conn: ConnDep) -> HTMLResponse:
        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        data = readmodel.machines_page(conn, user.id)
        return _csp(templates.TemplateResponse(request, "frags/machines_list.html", {
            "machines": data["machines"],
        }))
