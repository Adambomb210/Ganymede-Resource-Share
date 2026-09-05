"""The in-process SSE hub and ``GET /v1/events`` (docs/12-web-ui.md "SSE").

A tiny envelope, never rendered HTML: per-subscriber rendering inside the hub
does not scale, and the wire contract is frozen (``id`` / ``event`` / ``data``
with one entity-id field). Authorization is applied **at emit time, per
subscriber** -- the docs/06 404-not-403 rule extends to the stream: a non-admin
never receives a ``queue.change``, and nothing signals that the event exists.

Single-process assumption (docs/12): the ring buffer and the monotonic id
counter live in this module's state. Multi-worker uvicorn breaks both; scaling
out later means Redis pub/sub or a durable ``events`` table (a docs/05
migration), out of scope here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from collections import deque
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from ganymede.coordinator.auth import Contributor

# ``sync`` is not stored in the ring; it is emitted per-subscriber on connect /
# reconnect when Last-Event-ID cannot be honoured. Every live fragment on the
# page has ``hx-trigger="sse:sync"`` + ``hx-get`` -- an idempotent fragment
# refetch IS complete recovery (docs/12).
RING_SIZE = 256
HEARTBEAT_SEC = 15.0

# Event type -> the entity-id field name carried in ``data`` ("" = none).
ID_FIELDS = {
    "job.status": "job_id",
    "round.close": "job_id",
    "fleet.delta": "",
    "standing.change": "machine_id",
    "queue.change": "job_id",
    "submitter.change": "user_id",
}


@dataclass(frozen=True)
class Envelope:
    id: int
    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class _Sub:
    """One connected stream. Audience facts frozen at subscribe time."""
    queue: asyncio.Queue
    user_id: str
    is_admin: bool
    owned_machine_ids: frozenset[str]


class _Hub:
    """Module-level singleton. ``deque(maxlen)`` is the whole storage story."""

    def __init__(self) -> None:
        self.ring: deque[Envelope] = deque(maxlen=RING_SIZE)
        self.subs: set[_Sub] = set()
        self._next_id = 1
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- publish -----------------------------------------------------------

    def publish(
        self,
        type: str,
        *,
        owner_id: str | None = None,
        machine_id: str | None = None,
        job_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Synchronous and non-blocking. Called after the caller's
        ``immediate()`` transaction commits. Safe from any thread: sync
        endpoints run in FastAPI's threadpool, so when no loop is running in
        this thread, delivery is scheduled onto the hub's captured loop."""
        if type not in ID_FIELDS:
            raise ValueError(f"unknown event type: {type}")
        env = Envelope(
            id=self._next_id,
            type=type,
            data={
                k: v
                for k, v in (
                    ("job_id", job_id),
                    ("machine_id", machine_id),
                    ("user_id", user_id),
                )
                if v is not None
            },
        )
        self._next_id += 1
        self.ring.append(env)

        loop = self._loop
        if loop is None or loop.is_closed():
            # No async subscribers have ever connected (e.g. pure API tests or
            # the cron process) -- the ring alone is enough.
            return env.id

        sub_tasks: list[tuple[_Sub, Envelope | str]] = []
        for sub in list(self.subs):
            if not _authorized(sub, env, owner_id):
                continue
            sub_tasks.append((sub, env))
        if not sub_tasks:
            return env
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            for sub, e in sub_tasks:
                sub.queue.put_nowait(e)
        else:
            def _deliver():
                for sub, e in sub_tasks:
                    sub.queue.put_nowait(e)
            loop.call_soon_threadsafe(_deliver)
        return env.id

    # -- subscribe ---------------------------------------------------------

    async def subscribe(
        self,
        user: Contributor,
        last_event_id: str | None,
    ) -> _Sub:
        loop = asyncio.get_running_loop()
        # First (or only) subscriber wins the capture; the process runs one
        # event loop for its whole life, so this is a no-op after the first.
        if self._loop is None:
            self._loop = loop
        owned = await asyncio.to_thread(_owned_machine_ids_sync, user.id)
        sub = _Sub(queue=asyncio.Queue(), user_id=user.id,
                   is_admin=user.is_admin, owned_machine_ids=owned)
        self.subs.add(sub)
        if last_event_id is not None:
            try:
                since = int(last_event_id)
            except ValueError:
                since = None
            if since is not None:
                # Replay every ring entry after the id. If the id predates the
                # ring tail, the list is empty and the subscriber instead gets
                # the sync event below -- the refetch is complete recovery.
                for e in self.ring:
                    if e.id > since and _authorized(sub, e, None):
                        # owner_id is not replayable from the ring (it is not
                        # part of the frozen envelope), so job.status replay
                        # for non-admins falls back to the sync refetch; admin
                        # always sees everything.
                        if e.type == "job.status" and not sub.is_admin:
                            continue
                        sub.queue.put_nowait(e)
        sub.queue.put_nowait("sync")
        return sub

    def unsubscribe(self, sub: _Sub) -> None:
        self.subs.discard(sub)


def _authorized(sub: _Sub, env: Envelope, owner_id: str | None) -> bool:
    """Per-subscriber emit-time authorization (docs/12 audience table)."""
    if sub.is_admin:
        return True
    t = env.type
    if t in ("job.status", "round.close"):
        return owner_id is not None and owner_id == sub.user_id
    if t == "standing.change":
        return env.data.get("machine_id") in sub.owned_machine_ids
    if t in ("queue.change", "submitter.change"):
        # Non-admins never learn these exist (404-not-403 on the stream).
        return False
    return True  # fleet.delta -> all users


# Module-level singleton -- the frozen deployment is one process (§6.5).
hub = _Hub()

# Set once by ``create_app``; the hub's owned-machine lookup runs on its own
# short-lived connection so a stream never pins a per-request conn or a WAL
# snapshot (docs/12 read model). ``None`` (pure-unit contexts) makes the lookup
# return an empty set -- admin still sees standing.change, non-owners do not.
db_path: str | None = None


def _owned_machine_ids_sync(user_id: str) -> frozenset[str]:
    if db_path is None:
        return frozenset()
    from ganymede.coordinator.db import connect

    try:
        conn = connect(db_path)
        try:
            rows = conn.execute(
                "SELECT id FROM workers WHERE contributor_id = ?", (user_id,)
            ).fetchall()
            return frozenset(r["id"] for r in rows)
        finally:
            conn.close()
    except Exception:
        return frozenset()


def sse_frame(item: Envelope) -> str:
    """The wire format docs/12 froze: ``id:`` / ``event:`` / ``data:``."""
    payload = json.dumps({"type": item.type, "id": item.id, **item.data})
    return f"id: {item.id}\nevent: {item.type}\ndata: {payload}\n\n"


# --------------------------------------------------------------------------
# GET /v1/events -- the SSE endpoint
# --------------------------------------------------------------------------

async def events_endpoint(
    request: Request, user: Contributor
) -> StreamingResponse:
    """Cookie-authenticated SSE stream (auth class user, docs/06/12).

    A ``Machine`` principal is rejected here with 401 (a worker has no page
    to refresh); the /v1/events route resolves auth itself, so this is the
    guard. Holds no DB connection for the stream's life. On disconnect the
    subscriber is removed; EventSource reconnects, presents its
    Last-Event-ID, and either gets ring replay or one ``sync`` -- every
    fragment refetches.
    """
    if not isinstance(user, Contributor):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="user credential required")

    last_id = request.headers.get("last-event-id")
    sub = await hub.subscribe(user, last_id)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(
                        sub.queue.get(), timeout=HEARTBEAT_SEC
                    )
                except asyncio.TimeoutError:
                    yield ": hb\n\n"  # comment heartbeat; proxies stay open
                    continue
                if isinstance(item, str):
                    yield "event: sync\ndata: {}\n\n"
                    continue
                yield sse_frame(item)
        finally:
            hub.unsubscribe(sub)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
