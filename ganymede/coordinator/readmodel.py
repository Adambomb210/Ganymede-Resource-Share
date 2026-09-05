"""The web-UI read model (docs/12-web-ui.md "Read model").

A module of named read-only ``SELECT``s. It holds nothing, writes nothing, and
never runs inside ``immediate()`` -- so it never acquires the write lock and
adds zero contention on the claim path. Runs on the per-request connection
from ``get_conn``; WAL (db.connect pragmas) keeps readers and the writer from
blocking each other. Keyset pagination (``created_at, id``), never ``OFFSET``.

Pagination note (deliberate deviation, priced): ``/ui/jobs`` uses a keyset
cursor on ``(created_at, id)`` per docs/12, but the admin queue view and the
submitters / leaderboard lists are small (a volunteer fleet) and render whole;
the leaderboard keeps its existing ``LIMIT ? OFFSET ?`` shape because docs/09
owns that endpoint's envelope and its ``next_cursor`` is Stage-2. The
``WHERE (created_at, id) < (?, ?)`` keyset pattern is the only pagination the
UI itself invents.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def dashboard(conn: sqlite3.Connection) -> dict:
    """Everything the ``/ui/`` overview renders in one call's worth of rows.

    Healthy / stalled / awake are *not* re-derived here -- they stay owned by
    ``scripts.status`` / ``invariants`` (docs/12 page note: forking "stalled"
    into a template is the failure mode docs/03 M5 warns against), so the webui
    module imports and calls them with this module's rows.
    """
    runs = conn.execute(
        """SELECT id, status, current_round, target_rounds, base_model
             FROM runs ORDER BY created_at, id"""
    ).fetchall()
    jobs = conn.execute(
        """SELECT id, job_type, status, priority_rank FROM jobs
            ORDER BY priority_rank, created_at"""
    ).fetchall()
    queued = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'"
    ).fetchone()["n"]
    leased = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status = 'leased'"
    ).fetchone()["n"]
    return {
        "runs": [dict(r) for r in runs],
        "jobs": [dict(j) for j in jobs],
        "queued_jobs": queued,
        "leased_tasks": leased,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def jobs_page(
    conn: sqlite3.Connection, owner_id: str | None, *, is_admin: bool,
    before: tuple[str, str] | None = None, limit: int = 50,
) -> list[dict]:
    """The jobs list page: caller's jobs (admin: all), keyset-paginated on
    ``(created_at, id)`` descending -- never ``OFFSET`` (docs/12)."""
    where = ""
    params: list = []
    if not is_admin:
        where = "WHERE j.owner_id = ?"
        params.append(owner_id)
    if before is not None:
        conj = "AND" if where else "WHERE"
        where += (f" {conj} (j.created_at < ? OR "
                  f"(j.created_at = ? AND j.id < ?))")
        params.extend([before[0], before[0], before[1]])
    sql = f"""SELECT j.id, j.owner_id, j.job_type, j.status,
                      j.priority_rank, j.created_at,
                      (SELECT COUNT(*) FROM tasks t
                        WHERE t.job_id = j.id AND t.status = 'leased')
                          AS leased_tasks
               FROM jobs j {where}
              ORDER BY j.created_at DESC, j.id DESC LIMIT ?"""
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["owner_name"] = _owner_name(conn, d["owner_id"])
        out.append(d)
    return out


def _owner_name(conn: sqlite3.Connection, owner_id: str) -> str:
    if owner_id == "system":
        return "system"
    row = conn.execute(
        "SELECT name FROM contributors WHERE id = ?", (owner_id,)
    ).fetchone()
    return row["name"] if row is not None else owner_id


def job_detail(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """A job row plus its tasks and, for a ``collab_lora_finetune`` job, its
    rounds table and coverage (docs/12: the generic page delegates the round
    table to the type; types without ``reduce`` show tasks only)."""
    job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        return None
    tasks = conn.execute(
        """SELECT id, status, worker_id, attempt_group, attempts,
                  lease_expires_at, created_at
             FROM tasks WHERE job_id = ? ORDER BY created_at, id""",
        (job_id,),
    ).fetchall()
    rounds_rows: list[dict] = []
    coverage: dict = {}
    run = conn.execute(
        "SELECT id FROM runs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if run is not None:
        rounds_rows = [
            dict(r)
            for r in conn.execute(
                """SELECT idx, status, opened_at, closed_at, target_steps,
                          distinct_contributors, eval_loss, adapter_divergence
                     FROM rounds WHERE run_id = ? ORDER BY idx""",
                (run["id"],),
            ).fetchall()
        ]
        from ganymede.coordinator import invariants

        coverage = invariants.coverage(conn, run["id"])
    return {
        "job": dict(job),
        "tasks": [dict(t) for t in tasks],
        "rounds": rounds_rows,
        "coverage": coverage,
        "run_id": run["id"] if run is not None else None,
    }


def queue_page(conn: sqlite3.Connection) -> list[dict]:
    """The admin queue view, same shape as ``GET /v1/admin/queue``."""
    rows = conn.execute(
        """SELECT j.id, j.job_type, j.status, j.priority_rank, j.owner_id,
                  j.created_at,
                  (SELECT COUNT(*) FROM tasks t
                    WHERE t.job_id = j.id AND t.status = 'leased')
                      AS leased_tasks
           FROM jobs j
           WHERE j.status IN ('queued', 'running')
           ORDER BY j.priority_rank ASC, j.created_at ASC"""
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["owner_name"] = _owner_name(conn, d["owner_id"])
        out.append(d)
    return out


def submitters_page(conn: sqlite3.Connection) -> list[dict]:
    """``submitters`` ⋈ ``contributors`` with pinned images and running jobs
    (docs/12 page table)."""
    rows = conn.execute(
        """SELECT s.user_id, s.status, s.decided_by, s.decided_at, s.note,
                  c.name AS user_name, c.enabled
             FROM submitters s
             JOIN contributors c ON c.id = s.user_id
            ORDER BY c.name"""
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["decided_by_name"] = (
            _owner_name(conn, d["decided_by"]) if d["decided_by"] else ""
        )
        d["pinned_images"] = [
            dict(x)
            for x in conn.execute(
                """SELECT id, digest, scan_status, uploaded_at FROM images
                    WHERE submitter_id = ? AND finalized_at IS NOT NULL
                    ORDER BY uploaded_at DESC LIMIT 5""",
                (d["user_id"],),
            ).fetchall()
        ]
        d["running_jobs"] = conn.execute(
            """SELECT COUNT(*) AS n FROM jobs
                WHERE owner_id = ? AND status IN ('queued', 'running')""",
            (d["user_id"],),
        ).fetchone()["n"]
        out.append(d)
    return out


def machines_page(conn: sqlite3.Connection, user_id: str) -> dict:
    """The caller's machines in the ``/v1/me`` shape (docs/12: the page reads
    ``GET /v1/me``). Reuses the same queries by calling the ledger functions
    the endpoint uses -- the shape is owned by docs/09."""
    from ganymede.coordinator import ledger

    machines = conn.execute(
        "SELECT * FROM workers WHERE contributor_id = ? ORDER BY enrolled_at, first_seen",
        (user_id,),
    ).fetchall()
    now = datetime.now(timezone.utc)
    rendered = []
    for m in machines:
        weight, ver = ledger.current_weight(conn, m["id"])
        if weight == 0.0:
            weight, _comp, ver = ledger.machine_weight_for(
                __import__("json").loads(m["compute_profile_json"])
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
        "machines": rendered,
        "recent_events": ledger.recent_events(
            conn, [m["id"] for m in machines]
        ),
    }


def leaderboard_rows(conn: sqlite3.Connection, scope: str = "machines") -> dict:
    """The leaderboard in the ``/v1/leaderboard`` shape (docs/09 owns the
    fields; docs/12 owns the envelope). UI renders both scopes."""
    if scope == "users":
        rows = conn.execute(
            """SELECT c.id AS user_id, c.name AS user_name,
                      SUM(e.weighted_hours) AS weighted_hours,
                      COUNT(DISTINCT e.machine_id) AS machines
                 FROM credit_events e JOIN contributors c ON c.id = e.user_id
                WHERE e.kind = 'provisioned'
                GROUP BY c.id
                ORDER BY weighted_hours DESC, user_name
                LIMIT 50"""
        ).fetchall()
        return {"scope": "users", "rows": [dict(r) for r in rows]}
    rows = conn.execute(
        """SELECT w.id AS machine_id, w.display_name, w.contributor_id AS user_id,
                  COALESCE((SELECT SUM(e.weighted_hours) FROM credit_events e
                             WHERE e.machine_id = w.id AND e.kind = 'provisioned'),
                           0.0) AS weighted_hours
             FROM workers w
            WHERE EXISTS (SELECT 1 FROM credit_events e
                           WHERE e.machine_id = w.id AND e.kind = 'provisioned')
            ORDER BY weighted_hours DESC, display_name
            LIMIT 50"""
    ).fetchall()
    return {"scope": "machines", "rows": [dict(r) for r in rows]}
