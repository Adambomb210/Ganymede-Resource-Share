"""The coordinator HTTP API (docs/02-architecture-v2.md 6.2).

Weights never travel through this API's request bodies. The coordinator mints
presigned URLs and workers talk to object storage directly, which keeps a
~25 MB artifact off the Python process entirely (Finding on v1's design) and
means a slow uploader occupies a socket on MinIO rather than a worker thread
here.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ganymede.coordinator import budget as budget_mod
from ganymede.coordinator import closer, rounds
from ganymede.coordinator.auth import AuthError, Contributor, authenticate
from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, immediate, init_schema
from ganymede.coordinator.store import Store, adapter_key

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


def require_contributor(
    request: Request,
    conn: ConnDep,
    authorization: Annotated[str | None, Header()] = None,
) -> Contributor:
    # TLS is mandatory: a bearer token over plain HTTP is a token in the clear
    # on every hop between a volunteer's home network and here.
    # X-Forwarded-Proto covers the reverse-proxy deployment (6.5).
    if request.app.state.settings.require_tls:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        if proto != "https":
            raise HTTPException(status_code=403, detail="TLS required")
    try:
        return authenticate(conn, authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


ContribDep = Annotated[Contributor, Depends(require_contributor)]


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
        reasons: list[str] = []
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
            closer.maybe_close(conn, store, run_id, settings=settings)
            try:
                spec = rounds.claim_task(
                    conn, run_id, body.worker_id, contributor.clearance,
                    profile, settings, worker_image_tag=worker["image_tag"],
                )
            except rounds.NotEligible as exc:
                reasons.append(f"{run_id}: {exc}")
                continue
            if spec is not None:
                return JSONResponse(_task_payload(spec, store, settings))

        # 204 is a legitimate answer, not an error: nothing eligible, or too
        # little of the round left to be worth a 25 MB round trip.
        return Response(
            status_code=204,
            headers={"Retry-After": str(settings.poll_interval_sec)},
        )

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
        expected = closer.expected_manifest(store, rnd["base_adapter_ref"])
        accepted, reason = closer.gate_submission(conn, store, task_id, expected)

        result = closer.maybe_close(conn, store, task["run_id"], settings=settings)
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
        rnd = rounds.current_round(conn, run_id)
        if rnd is None:
            raise HTTPException(status_code=404, detail="no current round")
        prog = rounds.round_progress(conn, run_id, rnd["idx"])
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


def _task_payload(spec: rounds.TaskSpec, store: Store, settings: Settings) -> dict:
    url, expires = store.presign_get(spec.base_adapter_ref)
    return {
        "task_id": spec.id,
        "run_id": spec.run_id,
        "round_idx": spec.round_idx,
        "buckets": spec.buckets,
        "num_buckets": spec.num_buckets,
        "seed": rounds.task_seed(spec.run_id, spec.round_idx, spec.id),
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
    conn.close()
    store = Store(settings.storage)
    store.ensure_bucket()
    return create_app(settings, store)
