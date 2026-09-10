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

import json
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

    @app.post("/ui/jobs/new", response_class=HTMLResponse)
    def ui_jobs_new(request: Request, conn: ConnDep,
                    job_type: Annotated[str, Form()],
                    spec: Annotated[str, Form()] = "",
                    image_id: Annotated[str, Form()] = "") -> HTMLResponse:
        """Create a job from the browser (docs/12 C2).

        **Why this is a ``/ui`` POST when every other mutation goes straight to
        ``/v1``.** htmx's ``json-enc`` encodes a form as a *flat* object of
        strings, and ``POST /v1/jobs`` takes a nested ``spec`` object. With
        ``script-src 'self'`` and no build step there is no way for a browser
        form to produce that body -- so the spec arrives here as text, gets
        parsed, and goes on to the same :func:`~ganymede.coordinator.app.create_job`
        the API calls. Nothing about *which jobs are allowed* lives here.

        The other half of the reason is the error path: a 422 from ``/v1``
        swapped into the page by htmx would put a raw JSON error blob in front
        of a person. A malformed spec is the expected outcome of typing JSON
        into a textarea, so it is rendered as a message beside the field.
        """
        from ganymede.coordinator.app import JobCreateRequest, create_job

        user, redirect = _page_user(request, conn)
        if redirect:
            return redirect
        if not _is_submitter(conn, user):
            # Matches the API's answer for a non-allowlisted caller (docs/08):
            # the submitter surface is not confirmed to exist for them.
            raise _404()

        form = {"job_type": job_type, "image_id": image_id, "spec": spec}

        def fail(message: str) -> HTMLResponse:
            return _csp(templates.TemplateResponse(
                request, "frags/new_job.html",
                {"new_job": _new_job_context(conn, user, error=message, form=form),
                 "user": user},
                status_code=422,
            ))

        text = spec.strip() or "{}"
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            return fail(f"spec is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            return fail("spec must be a JSON object")

        try:
            result = create_job(conn, user, JobCreateRequest(
                job_type=job_type, spec=parsed,
                image_id=image_id.strip() or None,
            ))
        except HTTPException as exc:
            # create_job raises 422 with a field-named message; that is already
            # the sentence a submitter needs, so it is shown rather than mapped.
            return fail(str(exc.detail))

        return _csp(templates.TemplateResponse(
            request, "frags/new_job.html",
            {"new_job": _new_job_context(conn, user, created=result["job_id"]),
             "user": user},
        ))

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

    def _is_submitter(conn: sqlite3.Connection, user: Contributor) -> bool:
        """Approved on the vetted allowlist (docs/08). The API's
        ``require_submitter`` answers 404 for everyone else, because the
        submitter surface is not confirmed to exist for them; a *page* cannot
        do that to a logged-in user it is already rendering, so the UI asks the
        same question and simply omits the form."""
        row = conn.execute(
            "SELECT status FROM submitters WHERE user_id = ?", (user.id,)
        ).fetchone()
        return row is not None and row["status"] == "approved"

    def _new_job_context(conn: sqlite3.Connection, user: Contributor,
                         **over) -> dict:
        """What the submission form needs to render, empty or after a failure."""
        from ganymede.jobtypes import REGISTRY

        # `repo_tag` is accepted by `upload-url` and never stored (docs/11 §1.1
        # names it; the table does not have it), so the picker identifies an
        # image by id and digest. `object_ref IS NULL` means retention took the
        # archive -- `create_job` refuses those, so they are not offered.
        images = conn.execute(
            """SELECT id, digest, scan_status, uploaded_at FROM images
                WHERE submitter_id = ? AND finalized_at IS NOT NULL
                  AND object_ref IS NOT NULL
                ORDER BY uploaded_at DESC LIMIT 50""",
            (user.id,),
        ).fetchall()
        ctx = {
            "job_types": sorted(REGISTRY),
            # docs/11 §4: a contained type is unrunnable without one, so the
            # form says which types need an image rather than letting the
            # submitter find out from a 422.
            "image_required": sorted(
                name for name, cls in REGISTRY.items()
                if getattr(cls, "requires_image", False)
            ),
            "images": images,
            "error": None,
            "created": None,
            "form": {"job_type": "", "image_id": "", "spec": ""},
        }
        ctx.update(over)
        return ctx

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
        can_submit = _is_submitter(conn, user)
        return _csp(templates.TemplateResponse(request, "jobs.html", {
            "user": user, "jobs": rows, "before": before, "is_admin": user.is_admin,
            "can_submit": can_submit,
            "new_job": _new_job_context(conn, user) if can_submit else None,
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
            # `job_id` as well as `job`: frags/rounds.html renders from here and
            # from its own fragment route, and now carries the hx-get that
            # refetches it, so both contexts have to name the job the same way.
            "user": user, **data, "job_id": job_id, "is_admin": user.is_admin,
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
