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
from ganymede.coordinator import eligibility, identity
from ganymede.coordinator import close, rounds
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
from ganymede.jobtypes import resolve
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
    worker_id: str
    capabilities: ComputeProfile | None = None
    cached_base_models: list[str] = Field(default_factory=list)
    run_id: str | None = None


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
            if budget_mod.clearance_permits(contributor.clearance, r["data_classification"])
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
        return {"worker_id": worker_id,
                "heartbeat_interval_sec": settings.heartbeat_interval_sec}

    @app.post(f"/{API_VERSION}/tasks/claim")
    def claim(body: ClaimRequest, conn: ConnDep, contributor: ContribDep):
        from fastapi.responses import JSONResponse, Response

        worker = conn.execute(
            "SELECT * FROM workers WHERE id = ?", (body.worker_id,)
        ).fetchone()
        if worker is None or worker["contributor_id"] != contributor.id:
            raise HTTPException(status_code=404, detail="unknown worker")

        rounds.expire_leases(conn)
        profile = (body.capabilities.model_dump() if body.capabilities
                   else json.loads(worker["compute_profile_json"]))

        candidates = _selectable_runs(conn, body.run_id, body.cached_base_models)
        # Every branch below records a verdict, including the ones that succeed.
        # A stale "refused" left behind by a worker that has since started
        # working would be worse than no record at all -- it is the answer a
        # contributor would act on, and it would send them looking for a fault
        # in a machine that is fine.
        verdicts: list[eligibility.Verdict] = []
        for run_id in candidates:
            # Evaluate the close here too, not only after a submit. A round can
            # become closeable through the passage of time alone -- its backstop
            # arrives with work already in hand -- and on the submit path alone
            # nothing would ever notice: every worker has already submitted, and
            # none can claim, because there is too little of the round left to
            # be worth a budget. The round stays open, the run stops advancing,
            # and no request anywhere returns an error. Closing on the poll
            # makes any worker that is still awake enough to move the run on,
            # and hands this one the freshly opened round instead of another
            # empty 204.
            close.advance_job(conn, store, run_id, settings=settings)
            try:
                spec = resolve("collab_lora_finetune").shape_claim(
                    conn, run_id, body.worker_id, contributor.clearance,
                    profile, settings, worker_image_tag=worker["image_tag"],
                )
            except rounds.NotEligible as exc:
                verdicts.append(eligibility.Verdict(run_id, eligibility.REFUSED, str(exc)))
                continue
            if spec is not None:
                verdicts.append(eligibility.Verdict(run_id, eligibility.LEASED))
                eligibility.record(conn, body.worker_id, verdicts)
                return JSONResponse(_task_payload(spec, store, settings))
            # Eligible, but this run had nothing to hand out: no open round, or
            # too little of it left to be worth a budget. Not the worker's
            # problem, and recorded separately from a refusal because a fleet
            # that is uniformly idle is a different operator problem from a
            # fleet that is uniformly refused.
            verdicts.append(eligibility.Verdict(run_id, eligibility.IDLE))

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
                {"run_id": v.run_id, "outcome": v.outcome, "reason": v.reason}
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
        return {"lease_expires_at": expires.isoformat()}

    @app.post(f"/{API_VERSION}/tasks/{{task_id}}/upload-url")
    def upload_url(task_id: str, conn: ConnDep, contributor: ContribDep) -> dict:
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None:
            raise HTTPException(status_code=404, detail="unknown task")
        _worker_for_task(conn, task_id, contributor)
        key = adapter_key(task["run_id"], task["round_idx"], task_id)
        url, expires = store.presign_put(key)
        return {"url": url, "key": key, "expires_at": expires.isoformat()}

    @app.post(f"/{API_VERSION}/tasks/{{task_id}}/submit")
    def submit(task_id: str, body: SubmitRequest, conn: ConnDep,
               contributor: ContribDep) -> dict:
        worker_id = _worker_for_task(conn, task_id, contributor)
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()

        # The key is derived, never taken from the request. Trusting a
        # worker-supplied key would let one contributor point a submission at
        # another's artifact -- or at the round's base adapter.
        expected_key = adapter_key(task["run_id"], task["round_idx"], task_id)
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

        rnd = conn.execute(
            "SELECT base_adapter_ref FROM rounds WHERE run_id = ? AND idx = ?",
            (task["run_id"], task["round_idx"]),
        ).fetchone()
        jt = resolve("collab_lora_finetune")
        expected = jt.expected_manifest(store, rnd["base_adapter_ref"])
        accepted, reason = jt.gate_submission(conn, store, task_id, expected)

        result = close.advance_job(conn, store, task["run_id"], settings=settings)
        return {
            "accepted": accepted,
            "reject_reason": reason,
            "next_action": "claim" if accepted else "reclaim",
            "round_closed": result is not None,
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
            out.append({
                "worker_id": w["id"],
                "backend": profile.get("backend"),
                "device_name": profile.get("device_name"),
                "vram_mb": profile.get("vram_mb"),
                "supports": profile.get("supports", []),
                "last_seen": w["last_seen"],
                "rounds_joined": w["rounds_joined"],
                "steps_total": w["steps_total"],
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
    def machine_enroll(body: EnrollRequest, conn: ConnDep, user: UserDep) -> dict:
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
        # ``machine_key`` shown once. Weight recompute (Decision 12) is the
        # ledger workstream's hook and not wired here.
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

    @app.get("/status")
    def status(conn: ConnDep) -> dict:
        runs = conn.execute(
            "SELECT id, status, current_round, target_rounds FROM runs"
        ).fetchall()
        return {"runs": [dict(r) for r in runs]}

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


def _selectable_runs(conn: sqlite3.Connection, pinned: str | None,
                     cached_models: list[str]) -> list[str]:
    """Order active runs by cache affinity.

    v1 runs sequentially, so this is usually a list of one. The ordering is
    here because it costs nothing now and is the seam concurrent runs will
    need: pulling a 16 GB base model a volunteer already has on disk is the
    single most expensive thing a worker can be asked to do.
    """
    if pinned:
        return [pinned]
    rows = conn.execute(
        "SELECT id, base_model FROM runs WHERE status = 'active' ORDER BY created_at"
    ).fetchall()
    cached = set(cached_models)
    return [r["id"] for r in sorted(rows, key=lambda r: r["base_model"] not in cached)]


def _task_payload(spec: TaskSpec, store: Store, settings: Settings) -> dict:
    url, expires = store.presign_get(spec.base_adapter_ref)
    return {
        "task_id": spec.id,
        "run_id": spec.run_id,
        "round_idx": spec.round_idx,
        "buckets": spec.buckets,
        "num_buckets": spec.num_buckets,
        "seed": resolve("collab_lora_finetune").task_seed(spec.run_id, spec.round_idx, spec.id),
        "local_steps": spec.local_steps,
        "max_runtime_sec": spec.max_runtime_sec,
        "lease_expires_at": spec.lease_expires_at.isoformat(),
        "heartbeat_interval_sec": settings.heartbeat_interval_sec,
        # None when the run has no image requirement (the native-install case).
        # A worker running a different tag abandons here rather than after
        # downloading a base model (4.2 step 5).
        "required_image": spec.required_image,
        "base_model": spec.base_model,
        "base_precision": spec.base_precision,
        "lora_cfg": spec.lora_cfg,
        "hyperparams": spec.hyperparams,
        "dataset_ref": spec.dataset_ref,
        "base_adapter_url": url,
        "base_adapter_expires_at": expires.isoformat(),
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
