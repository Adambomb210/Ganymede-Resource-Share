"""The coordinator HTTP API (docs/02-architecture-v2.md 6.2).

Weights never travel through this API's request bodies. The coordinator mints
presigned URLs and workers talk to object storage directly, which keeps a
~25 MB artifact off the Python process entirely (Finding on v1's design) and
means a slow uploader occupies a socket on MinIO rather than a worker thread
here.
"""

from __future__ import annotations

import hmac
import json
import sqlite3
import uuid
from datetime import timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ganymede.coordinator import budget as budget_mod
from ganymede.coordinator import constraints as constraints_mod
from ganymede.coordinator import eligibility, identity, ledger
from ganymede.coordinator import close, events, rounds
from ganymede.coordinator.migrations import SYSTEM_OWNER_ID
from ganymede.coordinator.auth import (
    AuthError,
    Contributor,
    Machine,
    Principal,
    authenticate,
    hash_key,
    parse_bearer,
)
from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, immediate, init_schema
from ganymede.coordinator.store import Store, adapter_key
from ganymede.jobtypes import REGISTRY, resolve
from ganymede.jobtypes.base import TaskSpec

API_VERSION = "v1"


# --------------------------------------------------------------------------
# Request/response models
# --------------------------------------------------------------------------


class ComputeProfile(BaseModel):
    backend: str
    device_name: str = "unknown"
    vram_mb: int = 0
    compute_capability: str | None = None
    driver: str | None = None
    torch_ver: str | None = None
    package_version: str | None = None
    supports: list[str] = Field(default_factory=list)
    probe: dict[str, Any] = Field(default_factory=dict)


class RegisterRequest(BaseModel):
    compute_profile: ComputeProfile
    image_tag: str | None = None


class ClaimRequest(BaseModel):
    # ``worker_id`` is the machine resolver (docs/07 §1 defers the machine-key
    # auth class to identity/sandbox). ``run_id`` is kept for a v1 worker and
    # mapped to its parent job; ``job_id`` is the new pin.
    worker_id: str | None = None
    capabilities: ComputeProfile | None = None
    cached_base_models: list[str] = Field(default_factory=list)
    run_id: str | None = None
    job_id: str | None = None


class JobCreateRequest(BaseModel):
    job_type: str
    spec: dict[str, Any] = Field(default_factory=dict)
    image_id: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class CancelRequest(BaseModel):
    mode: str = "soft"


class ReorderRequest(BaseModel):
    job_id: str
    before: str | None = None
    after: str | None = None
    rank: int | None = None


class HeartbeatRequest(BaseModel):
    steps_completed: int = 0
    loss_ewma: float | None = None


class SubmitRequest(BaseModel):
    artifact_key: str
    steps_completed: int
    tokens_seen: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)


class SessionRequest(BaseModel):
    username: str
    secret: str


class EnrollRequest(BaseModel):
    display_name: str | None = None


class ClaimEnrollmentRequest(BaseModel):
    enroll_token: str
    compute_profile: ComputeProfile


class SubmitterDecisionRequest(BaseModel):
    """The body docs/06 froze: ``{status, note}`` -- one of approved / denied /
    revoked, plus an optional note the contributor sees."""

    status: str
    note: str | None = None


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


# FastAPI resolves endpoint annotations with get_type_hints against the *module*
# globals, and `from __future__ import annotations` makes every annotation a
# string. So these dependency aliases have to live at module level: defined
# inside create_app they are invisible at resolution time, and every endpoint
# silently degrades into requiring `conn` and `contributor` as query parameters.
# Per-request configuration therefore comes off app.state rather than a closure.


def get_conn(request: Request):
    """One connection per request.

    SQLite connections are not safe to share across threads and FastAPI runs
    sync endpoints in a threadpool, so pooling one would be a race. Opening a
    connection is microseconds against a local file; the WAL pragmas in
    db.connect make concurrent readers free.
    """
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


ConnDep = Annotated[sqlite3.Connection, Depends(get_conn)]


_SESSION_COOKIE = "ganymede_session"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _principal(
    request: Request, conn: sqlite3.Connection, authorization: str | None
) -> tuple[Principal, bool]:
    """Shared front half of every auth dependency: TLS gate, then resolve the
    bearer header or the session cookie to a ``Principal``. Returns
    ``(principal, via_cookie)``.

    CSRF (docs/06 "CSRF", docs/08): a bearer caller is immune. A cookie caller
    on a state-changing method must additionally carry ``X-Ganymede-UI: 1`` --
    a static header htmx sets globally and a cross-origin form cannot forge.
    ``SameSite=Lax`` on the cookie already blocks the cross-site form post; this
    is the second lock, and there is no token store.
    """
    if request.app.state.settings.require_tls:
        # A bearer token over plain HTTP is a token in the clear on every hop.
        # X-Forwarded-Proto covers the reverse-proxy deployment (6.5).
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        if proto != "https":
            raise HTTPException(status_code=403, detail="TLS required")

    cookie = request.cookies.get(_SESSION_COOKIE)
    try:
        principal = authenticate(conn, authorization, cookie=cookie)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    via_cookie = parse_bearer(authorization) is None and cookie is not None
    if via_cookie and request.method not in _SAFE_METHODS:
        if request.headers.get("x-ganymede-ui") != "1":
            raise HTTPException(status_code=403, detail="missing X-Ganymede-UI header")
    return principal, via_cookie


def require_user(
    request: Request,
    conn: ConnDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Contributor:
    """Any authenticated contributor -- a contributor key or a web session
    (docs/08 auth-class table). A ``Machine`` principal is rejected here."""
    principal, _ = _principal(request, conn, authorization)
    if not isinstance(principal, Contributor):
        raise HTTPException(status_code=401, detail="user credential required")
    return principal


# The pre-08 name. Every existing endpoint depends on it; it is exactly
# ``require_user`` (docs/08: "today's require_contributor").
require_contributor = require_user


def require_machine(
    request: Request,
    conn: ConnDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Machine:
    """A worker process, by its machine key (docs/08). Transitional rule (Spine
    deviation 4): a pre-004 ``workers`` row with no ``machine_keys`` still
    authenticates with its owner's contributor key until it re-enrolls, logged
    ``audit(event='legacy_worker_auth')``. Anyone else resolving to a
    ``Contributor`` gets 404 -- existence is not confirmed."""
    principal, _ = _principal(request, conn, authorization)
    if isinstance(principal, Machine):
        return principal
    legacy = conn.execute(
        """SELECT w.id, w.contributor_id, w.standing
             FROM workers w
            WHERE w.contributor_id = ?
              AND NOT EXISTS (SELECT 1 FROM machine_keys k WHERE k.machine_id = w.id)
            ORDER BY w.first_seen LIMIT 1""",
        (principal.id,),
    ).fetchone()
    if legacy is None:
        raise HTTPException(status_code=404, detail="unknown machine")
    with immediate(conn):
        conn.execute(
            "INSERT INTO audit (at, contributor_id, worker_id, event, detail_json) "
            "VALUES (?, ?, ?, 'legacy_worker_auth', '{}')",
            (rounds._iso(rounds.utcnow()), principal.id, legacy["id"]),
        )
    return Machine(legacy["id"], legacy["contributor_id"], legacy["standing"])


def require_submitter(
    request: Request,
    conn: ConnDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Contributor:
    """A ``Contributor`` on the vetted allowlist (docs/08). A non-approved
    caller gets 404 -- the submitter surface is not confirmed to exist for
    them."""
    user = require_user(request, conn, authorization)
    row = conn.execute(
        "SELECT status FROM submitters WHERE user_id = ?", (user.id,)
    ).fetchone()
    if row is None or row["status"] != "approved":
        raise HTTPException(status_code=404, detail="not found")
    return user


def require_admin(
    request: Request,
    conn: ConnDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Contributor:
    """``Contributor.is_admin`` (docs/08). An authenticated non-admin gets 404
    on the whole ``/v1/admin/*`` tree -- the admin API is not confirmed to
    exist; the unauthenticated get 401 from ``_principal`` first."""
    user = require_user(request, conn, authorization)
    if not user.is_admin:
        raise HTTPException(status_code=404, detail="not found")
    return user


ContribDep = Annotated[Contributor, Depends(require_contributor)]
UserDep = Annotated[Contributor, Depends(require_user)]


def create_app(settings: Settings, store: Store) -> FastAPI:
    app = FastAPI(title="Ganymede coordinator", version="0.1.0")
    app.state.settings = settings
    app.state.store = store
    # webui imports this module (ConnDep, _SESSION_COOKIE), so it cannot be a
    # module-level import here -- circular. By create_app time this module is
    # fully initialized and the import resolves cleanly.
    from ganymede.coordinator import webui

    webui.mount(app, settings)
    events.db_path = settings.db_path

    # ------------------------------------------------- SSE stream (docs/12)

    @app.get(f"/{API_VERSION}/events")
    async def events_stream(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ):
        """Cookie-authenticated SSE (docs/12): EventSource cannot set a header,
        so the stream rides the session cookie. Auth class user -- a machine
        key is rejected inside the endpoint. An async endpoint must not touch
        a threadpool-created connection (Starlette runs sync generator
        dependencies in a worker thread and hands the object back across
        threads), so the principal resolves on a short-lived connection here
        -- and nothing DB is held for the stream's life (docs/12).
        """
        conn = connect(settings.db_path)
        try:
            cookie = request.cookies.get(_SESSION_COOKIE)
            try:
                principal = authenticate(conn, authorization, cookie=cookie)
            except AuthError as exc:
                raise HTTPException(status_code=401, detail=str(exc)) from exc
        finally:
            conn.close()
        return await events.events_endpoint(request, principal)

    # ---------------- discovery / health ----------------

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get(f"/{API_VERSION}/manifest")
    def manifest(conn: ConnDep, contributor: ContribDep) -> dict:
        runs = conn.execute(
            """SELECT id, base_model, base_precision, requires_json, required_image,
                      data_classification, status, current_round, target_rounds
               FROM runs WHERE status = 'active'"""
        ).fetchall()
        visible = [
            {
                "run_id": r["id"],
                "base_model": r["base_model"],
                "base_precision": r["base_precision"],
                "requires": json.loads(r["requires_json"]),
                # 7 step 3 exists to consume this: the host agent reconciles its
                # local image against it *before* the container ever starts, so
                # a worker never wastes a claim discovering the mismatch itself
                # (4.2 step 5 is the in-container backstop, not the primary path).
                "required_image": r["required_image"],
                "current_round": r["current_round"],
                "target_rounds": r["target_rounds"],
            }
            for r in runs
            if budget_mod.clearance_and_terms_permit(
                contributor.clearance, r["data_classification"], contributor.agreed_at
            )
        ]
        return {"api_version": API_VERSION, "runs": visible,
                "heartbeat_interval_sec": settings.heartbeat_interval_sec}

    # ---------------- worker lifecycle ----------------

    @app.post(f"/{API_VERSION}/workers/register")
    def register(body: RegisterRequest, conn: ConnDep, contributor: ContribDep) -> dict:
        now = rounds._iso(rounds.utcnow())
        profile = body.compute_profile.model_dump()
        # Identity is derived from (contributor, machine fingerprint) rather than
        # generated fresh, so a worker that restarts keeps its measured
        # throughput history instead of resetting to the cold-start default
        # every time its host reboots.
        fingerprint = json.dumps(
            [contributor.id, profile.get("device_name"), profile.get("backend"),
             profile.get("vram_mb")],
            sort_keys=True,
        )
        worker_id = uuid.uuid5(uuid.NAMESPACE_OID, fingerprint).hex

        with immediate(conn):
            existing = conn.execute(
                "SELECT id FROM workers WHERE id = ?", (worker_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE workers SET compute_profile_json = ?, image_tag = ?, last_seen = ?
                       WHERE id = ?""",
                    (json.dumps(profile), body.image_tag, now, worker_id),
                )
            else:
                conn.execute(
                    """INSERT INTO workers
                         (id, contributor_id, compute_profile_json, image_tag,
                          first_seen, last_seen)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (worker_id, contributor.id, json.dumps(profile),
                     body.image_tag, now, now),
                )
        # Re-probe (Decision 12, docs/09 3.4): a reweighting is a new
        # ``computed_at`` stamp applied to windows settled *after* it -- never
        # retroactive -- so correctness only requires the fresh row exists.
        ledger.recompute_machine_weight(conn, worker_id, profile)
        events.hub.publish("fleet.delta")
        return {"worker_id": worker_id,
                "heartbeat_interval_sec": settings.heartbeat_interval_sec}

    @app.post(f"/{API_VERSION}/contributors/agree")
    def agree_to_terms(conn: ConnDep, contributor: ContribDep) -> dict:
        """Record the contributor's acceptance of the data-handling terms
        (docs/03 open question 2). Stamps ``contributors.agreed_at`` on first
        call; re-agreeing is a no-op that returns the original timestamp -- the
        first acceptance is the one that counts, legally. Unlocks non-``open``
        run claims."""
        with immediate(conn):
            existing = conn.execute(
                "SELECT agreed_at FROM contributors WHERE id = ?", (contributor.id,)
            ).fetchone()
            if existing and existing["agreed_at"] is not None:
                return {"agreed_at": existing["agreed_at"]}
            now = rounds._iso(rounds.utcnow())
            conn.execute(
                "UPDATE contributors SET agreed_at = ? WHERE id = ?",
                (now, contributor.id),
            )
        return {"agreed_at": now}

    @app.post(f"/{API_VERSION}/tasks/claim")
    def claim(body: ClaimRequest, conn: ConnDep, contributor: ContribDep):
        from fastapi.responses import JSONResponse, Response

        if not body.worker_id:
            raise HTTPException(status_code=422, detail="worker_id required")
        worker = conn.execute(
            "SELECT * FROM workers WHERE id = ?", (body.worker_id,)
        ).fetchone()
        if worker is None or worker["contributor_id"] != contributor.id:
            raise HTTPException(status_code=404, detail="unknown worker")

        rounds.expire_leases(conn)
        profile = (body.capabilities.model_dump() if body.capabilities
                   else json.loads(worker["compute_profile_json"]))

        # One lease per machine, global (docs/07 §1, Decision 4). Cheap
        # pre-check before the walk: a machine already holding a leased task is
        # re-served that task -- resumed through its owning job type for a fresh
        # presign, never a replay of the expired URLs, and never a second task.
        held = conn.execute(
            "SELECT * FROM tasks WHERE worker_id = ? AND status = 'leased' LIMIT 1",
            (body.worker_id,),
        ).fetchone()
        # The poll is the availability signal (docs/09 1.1): every worker polls
        # whether or not it gets work, so this is where the ledger hears from
        # the fleet. ``leased`` is informational (the leased-vs-idle split) and
        # never changes the credited amount -- Decision 11 counts idle-available
        # time. The good-standing gate is evaluated on the tick itself.
        ledger.record_availability_tick(conn, body.worker_id, leased=held is not None)
        if held is not None:
            if held["run_id"] is not None:
                spec = _resume_held(conn, held, worker, contributor, profile, settings)
                if spec is not None:
                    # ``tasks.job_id`` is not backfilled by migration 005 (docs/05
                    # pins pre-005 task rows' job_id to NULL), so fall back to the
                    # spec's job_id -- which the type reads off ``runs.job_id``,
                    # always set post-005 -- rather than write a NULL into
                    # worker_eligibility and have record() swallow the FK error.
                    job_id = held["job_id"] or spec.job_id
                    eligibility.record(
                        conn, body.worker_id,
                        [eligibility.Verdict(job_id, eligibility.LEASED)],
                    )
                    return JSONResponse(_task_payload(spec, store, settings))
            else:
                # A held task from a static (no-``shape_claim``) type: rebuild
                # the payload from the ``tasks`` row and a fresh ``inputs_for``
                # presign. Returning here keeps the one-lease-per-machine
                # invariant -- a machine holding a batch task never reaches the
                # walk to lease a second.
                resumed = _resume_held_static(conn, store, held, settings)
                if resumed is not None:
                    spec, task_inputs, jt = resumed
                    eligibility.record(
                        conn, body.worker_id,
                        [eligibility.Verdict(held["job_id"], eligibility.LEASED)],
                    )
                    return JSONResponse(_task_payload(
                        spec, store, settings, jt=jt, task_inputs=task_inputs,
                        sdk={"job_type": jt.name, "version": jt.version},
                    ))

        # Map a v1 worker's run_id pin to its parent job (docs/07 §1); job_id is
        # the new pin. A pin still passes through the constraint gate below.
        pinned_job_id = body.job_id
        if pinned_job_id is None and body.run_id is not None:
            row = conn.execute(
                "SELECT job_id FROM runs WHERE id = ?", (body.run_id,)
            ).fetchone()
            if row is not None:
                pinned_job_id = row["job_id"]

        jobs = _selectable_jobs(conn, pinned_job_id, body.cached_base_models)
        # Every branch below records a verdict, including the ones that succeed.
        # A stale "refused" left behind by a worker that has since started
        # working would be worse than no record at all -- it is the answer a
        # contributor would act on, and it would send them looking for a fault
        # in a machine that is fine.
        verdicts: list[eligibility.Verdict] = []
        for job in jobs:
            try:
                jt = resolve(job["job_type"])
            except KeyError:
                # Not a registered type on this build -- nothing to hand out.
                verdicts.append(eligibility.Verdict(job["id"], eligibility.IDLE))
                continue

            # Version binding (docs/10 §2). ``spec_json.sdk.version`` was frozen
            # at POST /v1/jobs; a pinned version this build's REGISTRY is older
            # than is a refusal recorded verbatim in worker_eligibility --
            # exactly as the required_image mismatch is (claim.py).
            pinned = _pinned_sdk_version(job)
            if pinned is not None:
                try:
                    resolve(job["job_type"], pinned)
                except AssertionError:
                    verdicts.append(eligibility.Verdict(
                        job["id"], eligibility.REFUSED, "job_type_version_unsupported"
                    ))
                    continue

            # Evaluate completion here too, not only after a submit. A collab
            # round can become closeable through the passage of time alone, and
            # a parallel job's last accepted verdict may have landed on a submit
            # that could not see the whole task set yet. Advancing on the poll
            # moves the job on rather than handing out another empty 204.
            if job["run_id"] is not None:
                close.advance_job(conn, store, job["run_id"], settings=settings)
            else:
                close.advance_job(conn, store, job_id=job["id"], settings=settings)

            # The constraint gate (Decision 15, docs/07 §2). Pure -- reads only
            # (machine_id, profile), no DB, no write lock. A refusal is a
            # `continue`, never a `break`: the walk reaching the first lower-rank
            # job this machine fits *is* the capability backfill (Decision 10).
            ok, why = constraints_mod.check_constraints(
                job["constraints_json"], body.worker_id, profile
            )
            if not ok:
                verdicts.append(
                    eligibility.Verdict(job["id"], eligibility.REFUSED, why)
                )
                continue

            if hasattr(jt, "shape_claim"):
                # Dynamic type: per-machine task sizing at claim time
                # (docs/10 §3). ``collab_lora_finetune``.
                run_id = job["run_id"]
                if run_id is None:
                    verdicts.append(eligibility.Verdict(job["id"], eligibility.IDLE))
                    continue
                try:
                    spec = jt.shape_claim(
                        conn, run_id, body.worker_id, contributor.clearance, profile,
                        settings, worker_image_tag=worker["image_tag"],
                        agreed_at=contributor.agreed_at,
                    )
                except rounds.NotEligible as exc:
                    verdicts.append(
                        eligibility.Verdict(job["id"], eligibility.REFUSED, str(exc))
                    )
                    continue
                task_inputs = None
                # Carry the frozen pin only when the job actually has one
                # (docs/10 §2). A run seeded by scripts/newrun has spec '{}' --
                # no sdk block -- and the collab payload stays byte-for-byte.
                sdk = ({"job_type": jt.name, "version": jt.version}
                       if pinned is not None else None)
            else:
                # Static type: ``plan`` output claimed as-is -- take one unleased
                # ``tasks`` row for this job (docs/10 §3, §4).
                spec, task_inputs = _claim_static_task(
                    conn, jt, store, job, body.worker_id, settings
                )
                sdk = {"job_type": jt.name, "version": jt.version}

            if spec is not None:
                # First lease flips the job queued -> running (docs/07 §1). Its
                # own transaction; eligibility.record comes after, never between
                # (docs/07 §1 freezes that ordering).
                with immediate(conn):
                    flipped = conn.execute(
                        "UPDATE jobs SET status = 'running' "
                        "WHERE id = ? AND status = 'queued'",
                        (job["id"],),
                    ).rowcount
                if flipped:
                    owner = conn.execute(
                        "SELECT owner_id FROM jobs WHERE id = ?", (job["id"],)
                    ).fetchone()
                    events.hub.publish(
                        "job.status", job_id=job["id"], owner_id=owner["owner_id"]
                    )
                verdicts.append(eligibility.Verdict(job["id"], eligibility.LEASED))
                eligibility.record(conn, body.worker_id, verdicts)
                return JSONResponse(_task_payload(
                    spec, store, settings, jt=jt, task_inputs=task_inputs, sdk=sdk
                ))

            # Eligible, but nothing to hand out: no open round / no unleased
            # task left. Recorded separately from a refusal because a
            # uniformly-idle fleet is a different operator problem from a
            # uniformly-refused one.
            verdicts.append(eligibility.Verdict(job["id"], eligibility.IDLE))

        eligibility.record(conn, body.worker_id, verdicts)

        # 204 is a legitimate answer, not an error: nothing eligible, or too
        # little of the round left to be worth a 25 MB round trip.
        return Response(
            status_code=204,
            headers={"Retry-After": str(settings.poll_interval_sec)},
        )

    @app.get(f"/{API_VERSION}/workers/{{worker_id}}/eligibility")
    def worker_eligibility(worker_id: str, conn: ConnDep, contributor: ContribDep) -> dict:
        """Why this machine is or is not getting work (roadmap M5).

        Scoped to the contributor who registered the worker. A refusal reason
        names the run's requirements and the machine's measured profile, which
        is exactly what a contributor needs and exactly what nobody else should
        be able to enumerate across a fleet.
        """
        worker = conn.execute(
            "SELECT * FROM workers WHERE id = ?", (worker_id,)
        ).fetchone()
        if worker is None or worker["contributor_id"] != contributor.id:
            # 404 rather than 403: a worker id belonging to somebody else should
            # not be distinguishable from one that does not exist.
            raise HTTPException(status_code=404, detail="unknown worker")

        answer = eligibility.explain(conn, worker_id)
        return {
            "worker_id": worker_id,
            "last_polled": answer.checked_at,
            "eligible_for_something": answer.any_eligible,
            "runs": [
                {"job_id": v.job_id, "outcome": v.outcome, "reason": v.reason}
                for v in answer.verdicts
            ],
            # Echoed back because half of every refusal reason is a fact about
            # this machine, and a contributor comparing "vram_mb 6144 < 8000"
            # against what they think their card has needs to see what the
            # coordinator actually measured (6.9).
            "compute_profile": json.loads(worker["compute_profile_json"]),
        }

    @app.post(f"/{API_VERSION}/tasks/{{task_id}}/heartbeat")
    def heartbeat(task_id: str, body: HeartbeatRequest, conn: ConnDep,
                  contributor: ContribDep) -> dict:
        worker_id = _worker_for_task(conn, task_id, contributor)
        try:
            expires = rounds.heartbeat(
                conn, task_id, worker_id, body.steps_completed, settings
            )
        except rounds.RoundClosed as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except rounds.LeaseLost as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        # A heartbeat is an availability event too (docs/09 1.1): the machine is
        # awake and holding work, so it polls the ledger. ``leased=True`` here is
        # informational; it never changes the credited amount.
        ledger.record_availability_tick(conn, worker_id, leased=True)
        return {"lease_expires_at": expires.isoformat()}

    @app.post(f"/{API_VERSION}/tasks/{{task_id}}/upload-url")
    def upload_url(task_id: str, conn: ConnDep, contributor: ContribDep) -> dict:
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        _worker_for_task(conn, task_id, contributor)
        # The key is derived from the task, never taken from the request. For a
        # collab round it is the per-round submission key; for a static type it
        # is the output key ``plan`` fixed in the task descriptor.
        key = _task_artifact_key(conn, task)
        url, expires = store.presign_put(key)
        return {"url": url, "key": key, "expires_at": expires.isoformat()}

    @app.post(f"/{API_VERSION}/tasks/{{task_id}}/submit")
    def submit(task_id: str, body: SubmitRequest, conn: ConnDep,
               contributor: ContribDep) -> dict:
        worker_id = _worker_for_task(conn, task_id, contributor)
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        jt = _resolve_for_task(conn, task)

        # The key is derived, never taken from the request. Trusting a
        # worker-supplied key would let one contributor point a submission at
        # another's artifact -- or at the round's base adapter.
        expected_key = _task_artifact_key(conn, task)
        if body.artifact_key != expected_key:
            raise HTTPException(status_code=422, detail="artifact_key does not match task")

        try:
            rounds.record_submission(
                conn, task_id, worker_id, expected_key, body.steps_completed,
                body.tokens_seen, body.metrics,
            )
        except rounds.RoundClosed as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except rounds.LeaseLost as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc

        if hasattr(jt, "shape_claim"):
            # collab_lora_finetune: the ex-closer structural gates, unchanged.
            rnd = conn.execute(
                "SELECT base_adapter_ref FROM rounds WHERE run_id = ? AND idx = ?",
                (task["run_id"], task["round_idx"]),
            ).fetchone()
            expected = jt.expected_manifest(store, rnd["base_adapter_ref"])
            accepted, reason = jt.gate_submission(conn, store, task_id, expected)
            if accepted:
                _credit_work(conn, jt, task, None, worker_id)
            result = close.advance_job(conn, store, task["run_id"], settings=settings)
            round_closed = result is not None
        else:
            # A type with no reduce: run its per-submission validate(), record
            # the Verdict (compare_digest folded into metrics_json), then let
            # the dispatcher re-check completion. ``round_closed`` is not
            # meaningful for a type with no reduce (docs/06).
            result_obj = _infer_result_for(task, body, expected_key)
            verdict = jt.validate(task, result_obj, conn, store)
            _record_generic_verdict(conn, task_id, verdict, worker_id)
            accepted, reason = verdict.accepted, verdict.reason
            if accepted:
                _credit_work(conn, jt, task, result_obj, worker_id)
            close.advance_job(conn, store, job_id=task["job_id"], settings=settings)
            round_closed = False

        return {
            "accepted": accepted,
            "reject_reason": reason,
            "next_action": "claim" if accepted else "reclaim",
            "round_closed": round_closed,
        }

    @app.post(f"/{API_VERSION}/tasks/{{task_id}}/abandon")
    def abandon(task_id: str, conn: ConnDep, contributor: ContribDep) -> dict:
        worker_id = _worker_for_task(conn, task_id, contributor)
        rounds.abandon(conn, task_id, worker_id)
        return {"ok": True}

    # ---------------- observability ----------------

    @app.get(f"/{API_VERSION}/runs/{{run_id}}/rounds/current")
    def current(run_id: str, conn: ConnDep, contributor: ContribDep) -> dict:
        # A "round" is collab_lora_finetune-specific (docs/10 §5): this endpoint
        # reads that type's `rounds` row via the type. A non-training job has no
        # such row -> 404, indistinguishable from a missing run.
        jt = resolve("collab_lora_finetune")
        rnd = jt.current_round(conn, run_id)
        if rnd is None:
            raise HTTPException(status_code=404, detail="no current round")
        prog = jt.round_progress(conn, run_id, rnd["idx"])
        return {
            "run_id": run_id, "round_idx": rnd["idx"], "status": rnd["status"],
            "opened_at": rnd["opened_at"], "target_steps": rnd["target_steps"],
            **prog,
        }

    @app.get(f"/{API_VERSION}/fleet")
    def fleet(conn: ConnDep, contributor: ContribDep) -> dict:
        """The inventory, derived rather than maintained (6.11).

        Nobody keeps a roster. Capabilities were probed at registration,
        throughput was measured while working, availability is simply when a
        worker was last seen. This endpoint renders what the database already
        knows.
        """
        workers = conn.execute("SELECT * FROM workers ORDER BY last_seen DESC").fetchall()
        out = []
        for w in workers:
            profile = json.loads(w["compute_profile_json"])
            weight, _ver = ledger.current_weight(conn, w["id"])
            if weight == 0.0:
                weight, _comp, _ver = ledger.machine_weight_for(profile)
            out.append({
                "worker_id": w["id"],
                "backend": profile.get("backend"),
                "device_name": profile.get("device_name"),
                "vram_mb": profile.get("vram_mb"),
                "supports": profile.get("supports", []),
                "last_seen": w["last_seen"],
                "rounds_joined": w["rounds_joined"],
                "steps_total": w["steps_total"],
                # Ledger (docs/06 observability): per-machine standing and the
                # accrued Weighted System Hours (SUM kind='provisioned').
                "standing": w.get("standing") or "good",
                "weighted_hours_total": ledger.accrued(conn, machine_id=w["id"]),
                "system_weight": weight,
            })
        tp = conn.execute("SELECT * FROM throughput").fetchall()
        return {
            "workers": out,
            "measured_throughput": [
                {"run_id": t["run_id"], "gpu_model": t["gpu_model"],
                 "steps_per_min": t["steps_per_min"], "samples": t["samples"]}
                for t in tp
            ],
        }

    # ---------------- identity / enrollment (docs/08) ----------------

    @app.post(f"/{API_VERSION}/auth/session")
    def auth_session_create(body: SessionRequest, conn: ConnDep) -> Any:
        """Placeholder provider login (Decision 5). Verifies the credential
        through the configured ``IdentityProvider``, resolves the ``contributors``
        row, mints a session, and returns it both as a ``Set-Cookie`` (for
        ``/ui/*``) and in the body (for non-browser callers). Real OAuth/OIDC
        later swaps the body and the verification, not this endpoint."""
        from fastapi.responses import JSONResponse

        provider = identity.PROVIDERS.get(settings.auth_provider)
        if provider is None:
            raise HTTPException(status_code=500, detail="no such auth provider")
        try:
            ident = provider.verify(body.model_dump(), conn)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        row = conn.execute(
            """SELECT id, enabled, is_admin FROM contributors
                WHERE auth_provider = ?
                  AND (auth_subject = ? OR (? IS NULL AND name = ?))""",
            (ident.auth_provider, ident.auth_subject, ident.auth_subject, ident.name),
        ).fetchone()
        # External providers JIT-provision on first login; ``local`` never does
        # -- local rows come from the bootstrap var or an admin.
        if row is None or not row["enabled"]:
            raise HTTPException(status_code=401, detail="unknown credential")

        with immediate(conn):
            minted = identity.mint_session(conn, row["id"], settings.session_ttl_sec)

        resp = JSONResponse(
            {"session_token": minted.token, "expires_at": minted.expires_at}
        )
        resp.set_cookie(
            _SESSION_COOKIE, minted.token, httponly=True, secure=True,
            samesite="lax", path="/",
        )
        return resp

    @app.delete(f"/{API_VERSION}/auth/session")
    def auth_session_delete(request: Request, conn: ConnDep,
                            user: UserDep) -> Any:
        """Logout / revoke the calling session (docs/08 Spine deviation 1).
        Deletes the row for the cookie this request carried; a bearer caller has
        no session row and simply gets the clear-cookie response."""
        from fastapi.responses import JSONResponse

        cookie = request.cookies.get(_SESSION_COOKIE)
        if cookie:
            with immediate(conn):
                conn.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (hash_key(cookie),)
                )
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(_SESSION_COOKIE, path="/")
        return resp

    @app.post(f"/{API_VERSION}/machines/enroll")
    def machine_enroll(body: EnrollRequest, request: Request, conn: ConnDep, user: UserDep) -> dict:
        """Mint a one-time enrollment token bound to the calling user. The
        operator pastes it into the host config; the host then calls
        ``claim-enrollment`` with the token alone."""
        token = identity.new_enroll_token()
        enroll_id = uuid.uuid4().hex
        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            conn.execute(
                """INSERT INTO enrollments
                     (id, user_id, token_hash, display_name, created_at,
                      consumed_at, machine_id)
                   VALUES (?, ?, ?, ?, ?, NULL, NULL)""",
                (enroll_id, user.id, hash_key(token), body.display_name, now),
            )
        expires_at = rounds._iso(
            rounds.utcnow() + timedelta(seconds=settings.enroll_ttl_sec)
        )
        # Content negotiation (docs/12): the htmx form sends Accept: text/html
        # and gets the one-time token rendered once, inline, in this fragment;
        # no GET ever returns it, it never enters a URL. A JSON caller gets the
        # frozen JSON shape unchanged.
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            from ganymede.coordinator import webui as _webui

            return _webui.templates.TemplateResponse(
                request, "frags/enroll_token.html",
                {"enroll_token": token, "expires_at": expires_at},
            )
        # ``enroll_token`` is shown once -- only its sha256 is stored.
        return {"enroll_token": token, "enroll_id": enroll_id, "expires_at": expires_at}

    @app.post(f"/{API_VERSION}/machines/claim-enrollment")
    def machine_claim_enrollment(body: ClaimEnrollmentRequest, conn: ConnDep) -> dict:
        """Redeem an enrollment token (no auth -- the token is the credential).
        Mints the durable ``machine_id``, the first machine key, and consumes the
        token. Unknown / consumed / expired all return the same 404 so a probe
        cannot learn a token was ever valid (docs/08 404-not-403)."""
        profile = body.compute_profile.model_dump()
        digest = hash_key(body.enroll_token)
        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            row = conn.execute(
                "SELECT * FROM enrollments WHERE token_hash = ?", (digest,)
            ).fetchone()
            if (
                row is None
                or not hmac.compare_digest(row["token_hash"], digest)
                or row["consumed_at"] is not None
                or identity.enroll_is_expired(row["created_at"], settings.enroll_ttl_sec)
            ):
                raise HTTPException(status_code=404, detail="unknown enrollment")

            machine_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO workers
                     (id, contributor_id, compute_profile_json, display_name,
                      enrolled_at, first_seen, last_seen,
                      hardware_fingerprint_json, standing)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'good')""",
                (machine_id, row["user_id"], json.dumps(profile),
                 row["display_name"], now, now, now,
                 identity.fingerprint_from_profile(profile)),
            )
            machine_key = identity.new_machine_key()
            conn.execute(
                """INSERT INTO machine_keys (machine_id, key_hash, enabled, created_at)
                   VALUES (?, ?, 1, ?)""",
                (machine_id, hash_key(machine_key), now),
            )
            consumed = conn.execute(
                """UPDATE enrollments SET consumed_at = ?, machine_id = ?
                    WHERE id = ? AND consumed_at IS NULL""",
                (now, machine_id, row["id"]),
            )
            if consumed.rowcount == 0:
                # Lost the one-time race -- roll the whole block back.
                raise HTTPException(status_code=404, detail="unknown enrollment")
        events.hub.publish("fleet.delta")
        # ``machine_key`` shown once. Compute the provisional weight from the
        # enrollment probe (Decision 12, docs/09 3) -- the ``machine_weight``
        # row gates how much a settled window is worth. Recomputed on every
        # re-probe; never retroactive.
        ledger.recompute_machine_weight(conn, machine_id, profile)
        return {"machine_id": machine_id, "machine_key": machine_key}

    @app.get(f"/{API_VERSION}/machines")
    def machines_list(conn: ConnDep, user: UserDep) -> dict:
        """The caller's machines, their standing, and accrued Weighted System
        Hours (``SUM(weighted_hours) WHERE kind='provisioned'`` -- zero until the
        ledger workstream writes accrual rows)."""
        rows = conn.execute(
            """SELECT w.id, w.display_name, w.standing, w.reputation,
                      w.enrolled_at, w.last_seen,
                      COALESCE((SELECT SUM(weighted_hours) FROM credit_events c
                                 WHERE c.machine_id = w.id AND c.kind = 'provisioned'),
                               0.0) AS weighted_hours_total
                 FROM workers w
                WHERE w.contributor_id = ?
                ORDER BY w.enrolled_at, w.first_seen""",
            (user.id,),
        ).fetchall()
        return {"machines": [
            {"machine_id": r["id"], "display_name": r["display_name"],
             "standing": r["standing"], "reputation": r["reputation"],
             "enrolled_at": r["enrolled_at"], "last_seen": r["last_seen"],
             "weighted_hours_total": r["weighted_hours_total"]}
            for r in rows
        ]}

    def _owned_machine(conn: sqlite3.Connection, machine_id: str, user: Contributor):
        row = conn.execute(
            "SELECT id, contributor_id, standing FROM workers WHERE id = ?",
            (machine_id,),
        ).fetchone()
        if row is None or (row["contributor_id"] != user.id and not user.is_admin):
            # 404 not 403: a machine the caller may not see is indistinguishable
            # from one that does not exist (docs/08 404-not-403).
            raise HTTPException(status_code=404, detail="unknown machine")
        return row

    @app.post(f"/{API_VERSION}/machines/{{machine_id}}/retire")
    def machine_retire(machine_id: str, conn: ConnDep, user: UserDep) -> dict:
        """Owner removes a machine: ``standing = 'revoked'`` and every key
        disabled. ``credit_events`` rows stay (append-only). ``audit.event``
        separates ``owner_retire`` from a fraud ``revoke`` (Spine deviation 3)."""
        _owned_machine(conn, machine_id, user)
        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            conn.execute(
                "UPDATE workers SET standing = 'revoked' WHERE id = ?", (machine_id,)
            )
            conn.execute(
                "UPDATE machine_keys SET enabled = 0 WHERE machine_id = ?", (machine_id,)
            )
            conn.execute(
                "INSERT INTO audit (at, contributor_id, worker_id, event, detail_json) "
                "VALUES (?, ?, ?, 'owner_retire', '{}')",
                (now, user.id, machine_id),
            )
        events.hub.publish("standing.change", machine_id=machine_id)
        events.hub.publish("fleet.delta")
        return {"machine_id": machine_id, "standing": "revoked"}

    @app.post(f"/{API_VERSION}/machines/{{machine_id}}/rotate-key")
    def machine_rotate_key(machine_id: str, conn: ConnDep, user: UserDep) -> dict:
        """Issue a new machine key and disable the old ones -- for a leaked key
        with a live machine still behind it (Spine deviation 2). Standing is
        untouched; key-enabled and standing are separate axes (docs/08)."""
        _owned_machine(conn, machine_id, user)
        new_key = identity.new_machine_key()
        new_hash = hash_key(new_key)
        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            conn.execute(
                """INSERT INTO machine_keys (machine_id, key_hash, enabled, created_at)
                   VALUES (?, ?, 1, ?)""",
                (machine_id, new_hash, now),
            )
            conn.execute(
                """UPDATE machine_keys SET enabled = 0
                    WHERE machine_id = ? AND key_hash != ?""",
                (machine_id, new_hash),
            )
        return {"machine_id": machine_id, "machine_key": new_key}

    @app.get(f"/{API_VERSION}/me")
    def me(conn: ConnDep, user: UserDep) -> dict:
        """The contributor's machines, standings, accrued Weighted System Hours,
        and recent credit events (docs/09 6.1). Field list owned by the ledger
        doc. Retired machines still appear with frozen totals; a ``kind = 'work'``
        row shows in ``recent_events`` with ``weighted_hours = 0.0`` and never in
        any total."""
        submitter = conn.execute(
            "SELECT status FROM submitters WHERE user_id = ?", (user.id,)
        ).fetchone()
        machines = conn.execute(
            "SELECT * FROM workers WHERE contributor_id = ? ORDER BY enrolled_at, first_seen",
            (user.id,),
        ).fetchall()
        active = [m for m in machines if m["standing"] != "revoked"]
        all_ids = [m["id"] for m in machines]
        now = rounds.utcnow()
        rendered = []
        for m in machines:
            weight, ver = ledger.current_weight(conn, m["id"])
            if weight == 0.0:
                weight, _comp, ver = ledger.machine_weight_for(
                    json.loads(m["compute_profile_json"])
                )
            leased = conn.execute(
                "SELECT 1 FROM tasks WHERE worker_id = ? AND status = 'leased' LIMIT 1",
                (m["id"],),
            ).fetchone()
            rendered.append({
                "machine_id": m["id"],
                "display_name": m["display_name"],
                "standing": m["standing"],
                "reputation": m["reputation"],
                "enrolled_at": m["enrolled_at"],
                "last_available_at": m["last_available_at"],
                "system_weight": weight,
                "formula_version": ver,
                "weighted_hours_total": ledger.accrued(conn, machine_id=m["id"]),
                "accrued_current_window": ledger.accrued_current_window(
                    conn, m["id"], weight, now
                ),
                "leased_now": leased is not None,
                "in_good_standing_now": ledger.in_good_standing(conn, m["id"], now),
                "unverified_tasks": ledger.unverified_tasks(conn, m["id"]),
                "unverified_ceiling": ledger.unverified_ceiling(m["standing"]),
            })
        return {
            "user": {
                "id": user.id, "name": user.name,
                "auth_provider": getattr(user, "auth_provider", "local"),
                "is_admin": getattr(user, "is_admin", False),
                "submitter_status": submitter["status"] if submitter else None,
            },
            "totals": {
                "weighted_hours": ledger.accrued(conn, user_id=user.id),
                "machines": len(active),
            },
            "machines": rendered,
            "recent_events": ledger.recent_events(conn, all_ids),
        }

    @app.get(f"/{API_VERSION}/leaderboard")
    def leaderboard(conn: ConnDep, user: UserDep, scope: str = "machines",
                    limit: int = 50, offset: int = 0) -> dict:
        """Machines / users by ``SUM(weighted_hours) WHERE kind='provisioned'``
        (docs/09 6.2) -- Decision 7's "the leaderboard is the whole point", and
        the one place the 404-not-403 cross-tenant rule is deliberately relaxed.
        Exposed fields are display/user names and the sums only. Ranking spans
        ``formula_version``s by construction and is never re-priced. The sum
        filters ``kind = 'provisioned'``, so ``work`` signal can never inflate a
        rank (disc/09 4.1)."""
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        fv = conn.execute("SELECT MAX(formula_version) AS v FROM machine_weight").fetchone()
        out: dict = {
            "generated_at": rounds._iso(rounds.utcnow()),
            "formula_version_current": int(fv["v"] or 0),
        }
        if scope == "users":
            rows = conn.execute(
                """SELECT c.id AS user_id, c.name AS user_name,
                          SUM(e.weighted_hours) AS weighted_hours,
                          COUNT(DISTINCT e.machine_id) AS machines
                     FROM credit_events e JOIN contributors c ON c.id = e.user_id
                    WHERE e.kind = 'provisioned'
                    GROUP BY c.id
                    ORDER BY weighted_hours DESC, user_name
                    LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            out["by_user"] = [{
                "rank": offset + i + 1, "user_id": r["user_id"],
                "user_name": r["user_name"], "weighted_hours": r["weighted_hours"],
                "machines": r["machines"],
            } for i, r in enumerate(rows)]
        else:
            rows = conn.execute(
                """SELECT w.id AS machine_id, w.display_name,
                          w.contributor_id AS user_id, w.standing,
                          COALESCE(mw.weight, 0.0) AS system_weight,
                          COALESCE((SELECT SUM(e.weighted_hours) FROM credit_events e
                                     WHERE e.machine_id = w.id AND e.kind = 'provisioned'),
                                   0.0) AS weighted_hours
                     FROM workers w
                     LEFT JOIN machine_weight mw ON mw.machine_id = w.id
                    WHERE EXISTS (SELECT 1 FROM credit_events e
                                   WHERE e.machine_id = w.id AND e.kind = 'provisioned')
                    ORDER BY weighted_hours DESC, display_name
                    LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            out["by_machine"] = [{
                "rank": offset + i + 1, "machine_id": r["machine_id"],
                "display_name": r["display_name"], "user_id": r["user_id"],
                "weighted_hours": r["weighted_hours"],
                "system_weight": r["system_weight"], "standing": r["standing"],
            } for i, r in enumerate(rows)]
        # The optional ``work``-signal leaderboard (docs/09 4.2). ``job_type`` is
        # null: the frozen ``credit_events`` schema carries no task/job link, so
        # a ``work`` row's unit is not attributable from the row alone -- the sum
        # is still a truthful total-units figure per machine, kept separate from
        # Weighted System Hours.
        work_rows = conn.execute(
            """SELECT w.id AS machine_id, w.display_name,
                      COALESCE(SUM(e.raw_seconds), 0) AS work_units
                 FROM credit_events e JOIN workers w ON w.id = e.machine_id
                WHERE e.kind = 'work' GROUP BY w.id
                ORDER BY work_units DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        out["by_work"] = [{
            "rank": i + 1, "machine_id": r["machine_id"],
            "display_name": r["display_name"], "job_type": None,
            "work_units": r["work_units"],
        } for i, r in enumerate(work_rows)]
        return out

    @app.get("/status")
    def status(conn: ConnDep) -> dict:
        runs = conn.execute(
            "SELECT id, status, current_round, target_rounds FROM runs"
        ).fetchall()
        jobs = conn.execute(
            "SELECT id, job_type, status, priority_rank FROM jobs "
            "ORDER BY priority_rank, created_at"
        ).fetchall()
        return {"runs": [dict(r) for r in runs], "jobs": [dict(j) for j in jobs]}

    # ---------------- job submission & management (docs/06) ----------------

    def _job_for_caller(conn: sqlite3.Connection, job_id: str, user: Contributor):
        """A job the caller may act on, else 404 (not 403 -- docs/06)."""
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or (row["owner_id"] != user.id and not user.is_admin):
            raise HTTPException(status_code=404, detail="not found")
        return row

    @app.post(f"/{API_VERSION}/jobs")
    def jobs_create(body: JobCreateRequest, conn: ConnDep,
                    user: Annotated[Contributor, Depends(require_submitter)]) -> dict:
        """Create a job in ``draft`` (docs/06). The job type validates its own
        ``spec`` shape; the constraint grammar is validated here (Decision 15 --
        an unknown field or operator is a submit-time 422, not a silent
        never-place). ``priority_rank`` in the body is ignored, not a 422
        (docs/07 §5): priority is admin-write-only."""
        if body.job_type not in REGISTRY:
            raise HTTPException(status_code=422, detail=f"unknown job type: {body.job_type}")
        # Resolve at the version the body pins, if any -- a spec that names a
        # newer version than this build ships is a 422 here, not a silent
        # never-place (docs/10 §2).
        pinned = None
        sdk_in = body.spec.get("sdk") if isinstance(body.spec, dict) else None
        if isinstance(sdk_in, dict) and sdk_in.get("version") is not None:
            try:
                pinned = int(sdk_in["version"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail="spec.sdk.version must be an integer")
        try:
            jt = resolve(body.job_type, pinned)
        except AssertionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        validate_spec = getattr(jt, "validate_spec", None)
        if validate_spec is not None:
            try:
                validate_spec(body.spec)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"invalid spec: {exc}") from exc
        try:
            constraints_mod.validate(body.constraints)
        except constraints_mod.ConstraintError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Freeze the resolved pair into the spec and never mutate it again
        # (docs/10 §2). ``jobs`` gets no version column -- this is where the
        # binding lives.
        spec = dict(body.spec) if isinstance(body.spec, dict) else {}
        spec["sdk"] = {"job_type": jt.name, "version": jt.version}

        job_id = uuid.uuid4().hex
        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            conn.execute(
                """INSERT INTO jobs
                     (id, owner_id, job_type, spec_json, image_id, status,
                      priority_rank, constraints_json, cancel_mode, created_at)
                   VALUES (?, ?, ?, ?, ?, 'draft', 0, ?, NULL, ?)""",
                (job_id, user.id, body.job_type, json.dumps(spec),
                 body.image_id, json.dumps(body.constraints), now),
            )
        return {"job_id": job_id, "status": "draft"}

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/enqueue")
    def jobs_enqueue(job_id: str, conn: ConnDep,
                     user: Annotated[Contributor, Depends(require_submitter)]) -> dict:
        """``draft -> queued`` (docs/07 §5). ``priority_rank`` defaults to the
        tail (``MAX(priority_rank)+1``); only then is the job visible to
        ``_selectable_jobs``. The admin adjusts with ``reorder`` after."""
        row = _job_for_caller(conn, job_id, user)
        if row["status"] != "draft":
            raise HTTPException(status_code=409, detail=f"job is {row['status']}, not draft")
        with immediate(conn):
            tail = conn.execute(
                "SELECT COALESCE(MAX(priority_rank), 0) + 1 AS r FROM jobs"
            ).fetchone()["r"]
            conn.execute(
                "UPDATE jobs SET status = 'queued', priority_rank = ? WHERE id = ?",
                (tail, job_id),
            )
        events.hub.publish("job.status", job_id=job_id, owner_id=row["owner_id"])
        events.hub.publish("queue.change", job_id=job_id)
        # A static (no ``shape_claim``) type fans its task set out once, now
        # (docs/10 §3, §4). A dynamic type -- ``collab_lora_finetune`` -- is
        # sized per machine at claim time and seeds its own round elsewhere.
        jt = resolve(row["job_type"])
        if not hasattr(jt, "shape_claim"):
            job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            specs = jt.plan(job_row, conn)
            now = rounds._iso(rounds.utcnow())
            with immediate(conn):
                for s in specs:
                    conn.execute(
                        """INSERT INTO tasks
                             (id, run_id, round_idx, job_id, buckets_json,
                              input_ref_json, attempt_group, local_steps, status,
                              worker_id, lease_expires_at, attempts,
                              max_runtime_sec, created_at)
                           VALUES (?, NULL, NULL, ?, '[]', ?, ?, 0, 'planned',
                                   NULL, NULL, 0, ?, ?)""",
                        (s.id, s.job_id, s.input_ref, s.attempt_group,
                         s.max_runtime_sec, now),
                    )
        return {"job_id": job_id, "status": "queued", "priority_rank": tail}

    @app.get(f"/{API_VERSION}/jobs")
    def jobs_list(conn: ConnDep, user: UserDep) -> dict:
        if user.is_admin:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY priority_rank, created_at"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE owner_id = ? ORDER BY priority_rank, created_at",
                (user.id,),
            ).fetchall()
        return {"jobs": [_job_view(r) for r in rows]}

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}")
    def jobs_get(job_id: str, conn: ConnDep, user: UserDep) -> dict:
        return _job_view(_job_for_caller(conn, job_id, user))

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/cancel")
    def jobs_cancel(job_id: str, body: CancelRequest, conn: ConnDep,
                    user: UserDep) -> dict:
        """Owner or admin (docs/06). Sets ``jobs.cancel_mode`` and moves the job
        to ``cancelled``; propagation to leased tasks travels on the next
        heartbeat, which is the sandbox workstream's -- this only sets the
        state."""
        if body.mode not in ("soft", "hard"):
            raise HTTPException(status_code=422, detail="mode must be 'soft' or 'hard'")
        row = _job_for_caller(conn, job_id, user)
        with immediate(conn):
            conn.execute(
                "UPDATE jobs SET status = 'cancelled', cancel_mode = ? WHERE id = ?",
                (body.mode, job_id),
            )
        events.hub.publish("job.status", job_id=job_id, owner_id=row["owner_id"])
        events.hub.publish("queue.change", job_id=job_id)
        return {"job_id": job_id, "status": "cancelled", "cancel_mode": body.mode}

    # ---------------- admin queue surface (docs/06, auth: admin) ----------

    @app.get(f"/{API_VERSION}/admin/queue")
    def admin_queue(conn: ConnDep,
                    admin: Annotated[Contributor, Depends(require_admin)]) -> dict:
        """The admin-ordered queue with leased-task counts (docs/06, docs/07
        §4). A job stuck at zero leased tasks is the signal to ``reorder``."""
        rows = conn.execute(
            """SELECT j.id, j.job_type, j.status, j.priority_rank, j.owner_id,
                      j.created_at, j.constraints_json,
                      (SELECT COUNT(*) FROM tasks t
                        WHERE t.job_id = j.id AND t.status = 'leased') AS leased_tasks
               FROM jobs j
               WHERE j.status IN ('queued', 'running')
               ORDER BY j.priority_rank ASC, j.created_at ASC"""
        ).fetchall()
        return {"queue": [dict(r) for r in rows]}

    @app.post(f"/{API_VERSION}/admin/queue/reorder")
    def admin_queue_reorder(body: ReorderRequest, conn: ConnDep,
                            admin: Annotated[Contributor, Depends(require_admin)]) -> dict:
        """The only writer of ``jobs.priority_rank`` (docs/07 §5, Decision 13).
        ``{job_id, before | after | rank}``: ``rank`` sets it outright;
        ``before`` / ``after`` place it just outside the referenced job's rank.
        Ranks are integers, gaps allowed, no uniqueness constraint -- ties break
        on the walk's secondary sort."""
        given = [x for x in (body.before, body.after, body.rank) if x is not None]
        if len(given) != 1:
            raise HTTPException(
                status_code=422, detail="exactly one of before / after / rank"
            )
        target = conn.execute(
            "SELECT id FROM jobs WHERE id = ?", (body.job_id,)
        ).fetchone()
        if target is None:
            raise HTTPException(status_code=404, detail="not found")

        if body.rank is not None:
            new_rank = int(body.rank)
        else:
            ref_id = body.before or body.after
            ref = conn.execute(
                "SELECT priority_rank FROM jobs WHERE id = ?", (ref_id,)
            ).fetchone()
            if ref is None:
                raise HTTPException(status_code=404, detail="reference job not found")
            new_rank = ref["priority_rank"] + (-1 if body.before else 1)

        with immediate(conn):
            conn.execute(
                "UPDATE jobs SET priority_rank = ? WHERE id = ?",
                (new_rank, body.job_id),
            )
        events.hub.publish("queue.change", job_id=body.job_id)
        return {"job_id": body.job_id, "priority_rank": new_rank}

    @app.post(f"/{API_VERSION}/admin/submitters/{{user_id}}")
    def admin_submitters_decide(
        user_id: str, body: SubmitterDecisionRequest, conn: ConnDep,
        admin: Annotated[Contributor, Depends(require_admin)],
    ) -> dict:
        """The submitter allowlist's only writer (docs/06, Decisions 3, 9):
        ``{status, note}`` with status one of approved / denied / revoked.
        Revoking does **not** kill running jobs -- the admin cancels those
        explicitly, per job, with a chosen ``mode`` (Decision 18)."""
        if body.status not in ("approved", "denied", "revoked"):
            raise HTTPException(status_code=422, detail="status must be approved | denied | revoked")
        row = conn.execute(
            "SELECT user_id, status FROM submitters WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            conn.execute(
                """UPDATE submitters
                      SET status = ?, decided_by = ?, decided_at = ?, note = ?
                    WHERE user_id = ?""",
                (body.status, admin.id, now, body.note, user_id),
            )
        events.hub.publish("submitter.change", user_id=user_id)
        return {"user_id": user_id, "status": body.status, "decided_by": admin.id,
                "decided_at": now}

    return app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _worker_for_task(conn: sqlite3.Connection, task_id: str,
                     contributor: Contributor) -> str:
    """Resolve a task to its holder, refusing tasks another contributor holds."""
    row = conn.execute(
        """SELECT t.worker_id, w.contributor_id FROM tasks t
           JOIN workers w ON w.id = t.worker_id WHERE t.id = ?""",
        (task_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown task")
    if row["contributor_id"] != contributor.id:
        # 404 rather than 403: a wrong-contributor task should not be
        # distinguishable from a nonexistent one.
        raise HTTPException(status_code=404, detail="unknown task")
    return row["worker_id"]


def _job_view(row: sqlite3.Row) -> dict:
    """The public shape of a ``jobs`` row (docs/06 "Job submission & management")."""
    return {
        "job_id": row["id"],
        "owner_id": row["owner_id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "priority_rank": row["priority_rank"],
        "image_id": row["image_id"],
        "constraints": json.loads(row["constraints_json"] or "{}"),
        "cancel_mode": row["cancel_mode"],
        "spec": json.loads(row["spec_json"] or "{}"),
        "created_at": row["created_at"],
    }


def _selectable_jobs(conn: sqlite3.Connection, pinned_job_id: str | None,
                     cached_models: list[str]) -> list[sqlite3.Row]:
    """Walk the admin-ordered queue (docs/07 §1).

    Rows: jobs in status ``queued`` or ``running`` -- **both**, because a
    ``collab_lora_finetune`` job flips to ``running`` on its first lease but
    fans out tasks every round after; walking only ``queued`` would hand it one
    task ever. ``draft`` / ``paused`` / terminal states are skipped.

    Order: ``priority_rank`` ASC primary (lower = sooner), then the
    cache-affinity tiebreak **within a rank**, then ``created_at`` ASC. That
    sort key -- ``(priority_rank, affinity_miss, created_at)`` -- is the single
    seam a future weighted fair-share slots into (docs/07 §4); the walk itself
    stays head-first, backfilling, one task per machine.

    ``pinned_job_id`` short-circuits to that one row if it is selectable, else
    the caller 204s.
    """
    rows = conn.execute(
        """SELECT j.id, j.job_type, j.priority_rank, j.created_at,
                  j.constraints_json, j.image_id, j.spec_json,
                  r.id AS run_id, r.base_model
           FROM jobs j
           LEFT JOIN runs r ON r.job_id = j.id
           WHERE j.status IN ('queued', 'running')"""
    ).fetchall()
    cached = set(cached_models or [])

    def key(row: sqlite3.Row):
        # Affinity is advisory and never crosses a rank boundary. Two axes,
        # both advisory: base-model-on-disk (today) and image-digest-pulled
        # (Decision 18 -- stubbed while every first-party job's image_id is
        # NULL). The affinity function can be refined without touching the walk.
        base_warm = bool(row["base_model"]) and row["base_model"] in cached
        return (row["priority_rank"], 0 if base_warm else 1, row["created_at"] or "")

    ordered = sorted(rows, key=key)
    if pinned_job_id is not None:
        ordered = [r for r in ordered if r["id"] == pinned_job_id]
    return ordered


def _resume_held(conn: sqlite3.Connection, held: sqlite3.Row, worker: sqlite3.Row,
                 contributor: Contributor, profile: dict, settings: Settings):
    """Re-serve a collab task the machine already holds (docs/07 §1, "Re-serving
    is per-type"). The ``tasks`` row alone cannot rebuild the payload, so
    dispatch to the owning job type -- which returns a fresh spec, and
    ``_task_payload`` a fresh presign, never a replay of the expired URLs."""
    if held["run_id"] is None:
        return None
    try:
        return resolve("collab_lora_finetune").shape_claim(
            conn, held["run_id"], worker["id"], contributor.clearance,
            profile, settings, worker_image_tag=worker["image_tag"],
            agreed_at=contributor.agreed_at,
        )
    except rounds.NotEligible:
        return None


def _resume_held_static(conn: sqlite3.Connection, store: Store,
                        held: sqlite3.Row, settings: Settings):
    """Re-serve a held task from a static (no-``shape_claim``) type: rebuild the
    ``TaskSpec`` from the row and mint a fresh ``inputs_for`` presign. Returns
    ``(spec, inputs, jt)`` or ``None`` if the owning job is gone / not running."""
    jt = _resolve_for_task(conn, held)
    if jt is None or hasattr(jt, "shape_claim"):
        return None
    job = conn.execute(
        "SELECT status FROM jobs WHERE id = ?", (held["job_id"],)
    ).fetchone()
    if job is None or job["status"] not in ("queued", "running"):
        return None
    spec = _static_task_spec(held, settings)
    return spec, jt.inputs_for(held, store), jt


def _static_task_spec(row: sqlite3.Row, settings: Settings) -> TaskSpec:
    lease = row["lease_expires_at"]
    return TaskSpec(
        id=row["id"],
        job_id=row["job_id"],
        input_ref=row["input_ref_json"],
        attempt_group=row["attempt_group"],
        max_runtime_sec=int(row["max_runtime_sec"] or settings.lease_duration_sec),
        lease_expires_at=rounds._parse(lease) if lease else None,
    )


def _claim_static_task(conn: sqlite3.Connection, jt, store: Store,
                       job: sqlite3.Row, worker_id: str, settings: Settings):
    """Atomically take one unleased task for this job and lease it to the
    machine. Returns ``(spec, inputs)`` or ``(None, None)``.

    "Unleased" is ``planned`` (never claimed) plus ``expired`` / ``abandoned``
    (a machine went away) plus a ``submitted`` row whose verdict was a
    rejection -- a static type mints no fresh row per claim the way collab does,
    so without recycling these the shard would orphan and the job would never
    complete. Bounded by ``close.MAX_TASK_ATTEMPTS`` (a hard per-shard failure
    path is Phase D)."""
    now = rounds.utcnow()
    with immediate(conn):
        row = conn.execute(
            """SELECT * FROM tasks
                WHERE job_id = ? AND attempts < ?
                  AND ( status IN ('planned', 'expired', 'abandoned')
                        OR (status = 'submitted' AND EXISTS (
                              SELECT 1 FROM submissions s
                               WHERE s.task_id = tasks.id AND s.accepted = 0)) )
                ORDER BY created_at, id LIMIT 1""",
            (job["id"], close.MAX_TASK_ATTEMPTS),
        ).fetchone()
        if row is None:
            return None, None
        expires = now + timedelta(seconds=settings.lease_duration_sec)
        changed = conn.execute(
            "UPDATE tasks SET status = 'leased', worker_id = ?, "
            "lease_expires_at = ?, attempts = attempts + 1 "
            "WHERE id = ? AND status = ?",
            (worker_id, rounds._iso(expires), row["id"], row["status"]),
        ).rowcount
        if not changed:
            return None, None
        conn.execute(
            "UPDATE workers SET last_seen = ? WHERE id = ?",
            (rounds._iso(now), worker_id),
        )
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (row["id"],)).fetchone()
    return _static_task_spec(row, settings), jt.inputs_for(row, store)


def _pinned_sdk_version(job: sqlite3.Row) -> int | None:
    """``spec_json.sdk.version`` if the job froze one (docs/10 §2)."""
    try:
        sdk = (json.loads(job["spec_json"] or "{}") or {}).get("sdk") or {}
        v = sdk.get("version")
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _resolve_for_task(conn: sqlite3.Connection, task: sqlite3.Row):
    """The job type behind a task. Falls back to ``collab_lora_finetune`` for a
    pre-005 task row with no ``job_id`` (docs/05)."""
    job_type = "collab_lora_finetune"
    if task is not None and task["job_id"] is not None:
        row = conn.execute(
            "SELECT job_type FROM jobs WHERE id = ?", (task["job_id"],)
        ).fetchone()
        if row is not None:
            job_type = row["job_type"]
    try:
        return resolve(job_type)
    except KeyError:
        return None


def _task_artifact_key(conn: sqlite3.Connection, task: sqlite3.Row) -> str:
    """The one key a submission for this task may land at -- derived, never
    taken from the request (the cross-tenant / base-adapter guard)."""
    if task["run_id"] is not None:
        return adapter_key(task["run_id"], task["round_idx"], task["id"])
    desc = json.loads(task["input_ref_json"] or "{}")
    key = desc.get("output_key")
    if not key:
        raise HTTPException(status_code=422, detail="task has no derivable artifact key")
    return key


def _infer_result_for(task: sqlite3.Row, body: "SubmitRequest", key: str):
    """Rebuild the type's ``Result`` for ``validate()`` from what ``submit``
    carries. The coordinator sets ``output_ref`` to the *derived* key, never
    the worker's."""
    from ganymede.jobtypes.batch_inference.run import InferResult

    metrics = body.metrics or {}
    return InferResult(
        rows=int(body.steps_completed or metrics.get("rows", 0)),
        output_ref=key,
        digest=str(metrics.get("digest") or ""),
        seconds=float(metrics.get("seconds", 0.0) or 0.0),
    )


def _credit_work(conn: sqlite3.Connection, jt, task, result, worker_id: str) -> None:
    """Record the trusted ``credit()`` work signal on an accepted submission
    (docs/09 4, docs/10 "credit": coordinator-side, trusted, never banked).

    Only job types that implement ``credit`` contribute a ``kind = 'work'`` row
    (``batch_inference`` does; ``collab_lora_finetune`` does not). The row has
    ``weighted_hours = 0.0`` and carries the WorkUnits scalar in ``raw_seconds``;
    every banked total filters ``kind = 'provisioned'``, so this can never mint
    reputation. ``result`` is ``None`` for the collab path, whose artifact is not
    deserialised here; a future type with ``credit`` on the collab branch would
    pass its result through.
    """
    credit = getattr(jt, "credit", None)
    if credit is None:
        return
    units = credit(task, result)
    if units is None:
        return
    owner = conn.execute(
        "SELECT contributor_id FROM workers WHERE id = ?", (worker_id,)
    ).fetchone()
    ledger.record_work(
        conn, machine_id=worker_id,
        user_id=owner["contributor_id"] if owner else "system",
        units=units.count,
    )


def _record_generic_verdict(conn: sqlite3.Connection, task_id: str, verdict,
                            worker_id: str) -> None:
    """Persist a non-collab ``Verdict`` on the submission row: ``accepted`` /
    ``reject_reason`` as columns, ``compare_digest`` folded into
    ``metrics_json`` (``submissions`` has no such column, and docs/10 prefers no
    migration for it)."""
    sub = conn.execute(
        "SELECT metrics_json FROM submissions WHERE task_id = ?", (task_id,)
    ).fetchone()
    metrics = {}
    if sub is not None and sub["metrics_json"]:
        try:
            metrics = json.loads(sub["metrics_json"])
        except ValueError:
            metrics = {}
    if verdict.compare_digest is not None:
        metrics["compare_digest"] = verdict.compare_digest
    with immediate(conn):
        conn.execute(
            "UPDATE submissions SET accepted = ?, reject_reason = ?, metrics_json = ? "
            "WHERE task_id = ?",
            (1 if verdict.accepted else 0, verdict.reason,
             json.dumps(metrics), task_id),
        )
        if not verdict.accepted:
            row = conn.execute(
                "SELECT contributor_id FROM workers WHERE id = ?", (worker_id,)
            ).fetchone()
            conn.execute(
                "INSERT INTO audit (at, contributor_id, worker_id, event, detail_json) "
                "VALUES (?, ?, ?, 'submission_rejected', ?)",
                (rounds._iso(rounds.utcnow()),
                 row["contributor_id"] if row else None, worker_id,
                 json.dumps({"task": task_id, "reason": verdict.reason,
                             "detail": verdict.detail})),
            )


def _task_payload(spec: TaskSpec, store: Store, settings: Settings, *,
                  jt=None, task_inputs=None, sdk: dict | None = None) -> dict:
    if task_inputs is None:
        # collab_lora_finetune -- the payload assembled here since Phase A,
        # kept byte-for-byte (plus an optional additive ``sdk`` block, docs/06).
        url, expires = store.presign_get(spec.base_adapter_ref)
        payload = {
            "task_id": spec.id,
            "run_id": spec.run_id,
            "round_idx": spec.round_idx,
            # Generic handles (docs/06 "Claim path"). image_* is null for
            # first-party built-ins; input_ref is the type-agnostic input
            # handle, with buckets still present for collab_lora_finetune.
            "job_id": spec.job_id,
            "job_type": "collab_lora_finetune",
            "image_ref": None,
            "image_digest": None,
            "image_pull_url": None,
            "input_ref": spec.input_ref,
            "buckets": spec.buckets,
            "num_buckets": spec.num_buckets,
            "seed": resolve("collab_lora_finetune").task_seed(
                spec.run_id, spec.round_idx, spec.id
            ),
            "local_steps": spec.local_steps,
            "max_runtime_sec": spec.max_runtime_sec,
            "lease_expires_at": spec.lease_expires_at.isoformat(),
            "heartbeat_interval_sec": settings.heartbeat_interval_sec,
            # None when the run has no image requirement (the native-install
            # case). A worker running a different tag abandons here rather than
            # after downloading a base model (4.2 step 5).
            "required_image": spec.required_image,
            "base_model": spec.base_model,
            "base_precision": spec.base_precision,
            "lora_cfg": spec.lora_cfg,
            "hyperparams": spec.hyperparams,
            "dataset_ref": spec.dataset_ref,
            "base_adapter_url": url,
            "base_adapter_expires_at": expires.isoformat(),
        }
        if sdk is not None:
            payload["sdk"] = sdk
        return payload

    # A static type (batch_inference): the payload is what ``inputs_for``
    # named -- ``artifacts`` (a model GET) and ``params`` (the shard ref, an
    # output PUT, the decode settings). No base adapter, no round, no buckets.
    return {
        "task_id": spec.id,
        "job_id": spec.job_id,
        "job_type": jt.name,
        "run_id": None,
        "round_idx": None,
        "image_ref": None,
        "image_digest": None,
        "image_pull_url": None,
        "input_ref": spec.input_ref,
        "attempt_group": spec.attempt_group,
        "artifacts": task_inputs.artifacts,
        "params": task_inputs.params,
        "sdk": sdk,
        "max_runtime_sec": spec.max_runtime_sec or settings.lease_duration_sec,
        "lease_expires_at": spec.lease_expires_at.isoformat()
        if spec.lease_expires_at else None,
        "heartbeat_interval_sec": settings.heartbeat_interval_sec,
        "required_image": None,
    }


def bootstrap() -> FastAPI:
    """Application factory for ``uvicorn --factory ganymede.coordinator.app:bootstrap``.

    Takes no arguments on purpose: uvicorn calls a factory with none, and
    configuration is environment-driven anyway (6.5). Tests build their own app
    with ``create_app(settings, store)`` and never come through here.
    """
    settings = Settings.from_env()
    conn = connect(settings.db_path)
    init_schema(conn)
    # Decision 14: the first admin is named by env, after the schema and every
    # migration are in place. Idempotent -- safe on every boot.
    identity.ensure_bootstrap_admin(conn, settings.bootstrap_admin)
    conn.close()
    store = Store(settings.storage)
    store.ensure_bucket()
    return create_app(settings, store)
