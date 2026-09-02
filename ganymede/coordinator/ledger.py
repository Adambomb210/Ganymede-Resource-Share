"""The contribution ledger, the provisioned-accrual engine, and reputation
(docs/09-ledger.md).

This is Decision 11 made real: primary reputation accrues from *provisioned*
time -- enrolled, awake, available, in good standing -- not from time under
lease and not from anything job code reports. The engine is a periodic sweep
run from cron (like ``invariants.py`` / ``status.py --alert``), never a daemon
and never work on the claim path. ``record_availability_tick`` is the one
write on the hot path (one row per poll/heartbeat); everything else -- settle,
GC, reputation -- is the sweep's job.

``credit_events`` is append-only (``05`` freezes "never updated, never
deleted"). The running total is ``SUM(weighted_hours) WHERE kind='provisioned'``
derived at read time; there is no balance column and no debit row. A
``kind = 'work'`` row has ``weighted_hours = 0.0`` and carries the ``credit()``
WorkUnits scalar in ``raw_seconds``, so an unfiltered ``SUM`` stays inert. Every
banked query filters ``kind = 'provisioned'`` -- the invariant this module
enforces on its own writes and exposes as a documented rule for readers.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, _parse, utcnow

# --- constants (docs/09 "Constants"; tunable from observed data). ------------

# Hour-aligned accrual window. One ``credit_events`` row per (machine, window),
# kind 'provisioned'.
ACCRUAL_WINDOW_SEC = 3600
# A window is written only once its ``period_end`` is this far in the past, so
# straggler heartbeats have landed (2x the per-tick cap).
SETTLE_DELAY_SEC = 1800
# Ticks older than this are dropped by the sweep, matching ``audit`` /
# ``worker_eligibility`` GC.
TICK_RETENTION_DAYS = 3
# ``workers.standing == 'probation'`` scales a whole settled window by this.
PROBATION_FACTOR = 0.5
# Hard ceiling on accrued weighted hours while on probation ("probation-with-
# limits").
PROBATION_MONTHLY_CAP_HOURS = 40.0
# Reputation thresholds (docs/09 5.2). Earned slowly, lost fast.
REP_ENROLL = 0.25
REP_GOOD = 0.60
REP_REVOKE = 0.15
# A machine on probation is promoted back to good only after a clean window of
# this many days with no rejection.
PROBATION_RECOVERY_DAYS = 7
# Unverified-work ceilings (docs/09 5.4) -- how many accepted-but-unverified
# tasks a machine may hold before provisioned accrual pauses.
K_GOOD = 48
K_PROBATION = 3
# How far back the reputation score looks for rejections / accepted work.
REP_TRAILING_DAYS = 30
# The per-tick credit cap (docs/09 1.2) reuses scripts/status.py's judgment of
# "still awake since the last poll". Kept as a literal here rather than importing
# scripts/status.py into the coordinator package; the cross-package import would
# drag argparse CLI state in for one integer. The value is frozen at 900 by both
# docs/09 and docs/02 (3.2 of the scheduler doc reads it off the same source of
# truth); if one ever drifts the other is wrong. Do not "fix" one in isolation.
AWAKE_WINDOW_SEC = 900


# --- machine_weight (docs/09 3.2, Decision 12) -------------------------------


def machine_weight_for(profile: dict) -> tuple[float, str, int]:
    """``formula_version = 0`` -- the explicitly interim GPU-class lookup.

    ``weight = base * clamp(vram_mb / 12000, 0.5, 1.5)``, where ``base`` comes
    from a static lookup on fields ``compute_profile_json`` already carries
    (backend, vram_mb, compute_capability). Records the matched class and terms
    in ``components_json``. Deliberately under-provisions unknown hardware.

    ``formula_version >= 1`` (probe-derived, docs/09 3.3) is gated on a
    re-shaped ``bench_score`` with a demonstrated monotonic separation across
    >= 2 real GPU classes, on hardware we have not measured yet. Until then
    this is the formula in force.
    """
    profile = profile or {}
    backend = profile.get("backend") or "cpu"
    vram_mb = int(profile.get("vram_mb") or 0)
    cc = profile.get("compute_capability")
    gpu = backend != "cpu" and vram_mb > 0
    if not gpu:
        cls, base = "cpu", 0.10
    elif vram_mb < 8000:
        cls, base = "gpu_low", 0.40
    else:
        try:
            cap = float(cc) if cc else 0.0
        except (TypeError, ValueError):
            cap = 0.0
        if cap < 7.0:
            cls, base = "gpu_low", 0.40          # old / weak GPU
        elif vram_mb < 16000:
            cls, base = "gpu_3060", 1.00         # 8-14 GB, the 1.0 anchor
        elif vram_mb < 24576:
            cls, base = "gpu_3090", 2.00         # 16-24 GB (3090 / 4080)
        elif vram_mb >= 24576:
            cls, base = "gpu_high", 3.50         # >= 24 GB (4090 / A100 / H100)
        else:
            cls, base = "gpu_unknown", 0.75      # GPU present, unclassified
    clamp = min(max(vram_mb / 12000.0, 0.5), 1.5)
    weight = round(base * clamp, 6)
    components = {
        "class": cls, "base": base, "vram_mb": vram_mb,
        "compute_capability": cc, "clamp": clamp, "formula_version": 0,
    }
    return weight, json.dumps(components), 0


def recompute_machine_weight(conn: sqlite3.Connection, machine_id: str,
                             profile: dict, now: datetime | None = None) -> None:
    """Upsert the machine's ``machine_weight`` row from a probe profile.

    ``machine_weight`` is a *current-value* table (docs/09 3): one row per
    machine, recomputed on enrollment and on re-probe, never append-only. Its
    history lives in the ``credit_events`` rows it stamps. There is no
    admin-override column -- a correction is this recompute, effective forward
    under the ``formula_version`` roll-forward rule.
    """
    now = now or utcnow()
    weight, components, ver = machine_weight_for(profile)
    with immediate(conn):
        conn.execute(
            """INSERT INTO machine_weight
                 (machine_id, weight, components_json, formula_version, computed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(machine_id) DO UPDATE SET
                   weight = excluded.weight,
                   components_json = excluded.components_json,
                   formula_version = excluded.formula_version,
                   computed_at = excluded.computed_at""",
            (machine_id, weight, components, ver, _iso(now)),
        )


def current_weight(conn: sqlite3.Connection, machine_id: str) -> tuple[float, int]:
    """The ``machine_weight`` row in force now -- ``(weight, formula_version)``,
    or ``(0.0, 0)`` if the machine has never been weighted (pre-ledger row; the
    caller falls back to ``machine_weight_for`` on its profile)."""
    row = conn.execute(
        "SELECT weight, formula_version FROM machine_weight WHERE machine_id = ?",
        (machine_id,),
    ).fetchone()
    if row is None:
        return 0.0, 0
    return float(row["weight"]), int(row["formula_version"])


# --- availability ticks & the good-standing gate (docs/09 1.1, 1.3) ----------


def unverified_ceiling(standing: str) -> int:
    """How many accepted-but-unverified tasks a machine may hold (docs/09 5.4).
    ``revoked`` is 0 -- and already fails the gate via condition 1."""
    return {"good": K_GOOD, "probation": K_PROBATION, "revoked": 0}.get(standing, K_GOOD)


def unverified_tasks(conn: sqlite3.Connection, machine_id: str) -> int:
    """Accepted-by-``validate()`` submissions not yet corroborated.

    Corroboration here means the work passed through aggregation: an accepted
    submission on a round that is still ``open``/``closing`` is unverified;
    once the round closes (the coordinator combined + reduced it) it is
    substantiated. ``batch_inference`` has no round -- its ``validate()``
    carries the verification (row count + schema + sampled re-run) -- so its
    accepted rows never count as unverified. ``round_idx`` is NULL for those
    tasks, which the JOIN to ``rounds`` naturally excludes.
    """
    row = conn.execute(
        """SELECT COUNT(*) AS n
           FROM submissions s
           JOIN tasks t  ON t.id = s.task_id AND t.worker_id = ?
           JOIN rounds r ON r.run_id = t.run_id AND r.idx = t.round_idx
          WHERE s.accepted = 1 AND r.status IN ('open', 'closing')""",
        (machine_id,),
    ).fetchone()
    return int(row["n"])


def _infraction_since(conn: sqlite3.Connection, machine_id: str,
                      since: str) -> bool:
    """Condition 2 of the good-standing gate: since the given tick boundary the
    machine has not abandoned a claimed task, gone no-show on one (claimed, no
    heartbeat, lease expired), or had a submission rejected by ``validate()``."""
    abandoned = conn.execute(
        """SELECT 1 FROM tasks
            WHERE worker_id = ? AND status IN ('abandoned', 'expired')
              AND created_at >= ? LIMIT 1""",
        (machine_id, since),
    ).fetchone()
    if abandoned is not None:
        return True
    rejected = conn.execute(
        """SELECT 1 FROM submissions s
             JOIN tasks t ON t.id = s.task_id AND t.worker_id = ?
            WHERE s.accepted = 0 AND s.received_at >= ? LIMIT 1""",
        (machine_id, since),
    ).fetchone()
    return rejected is not None


def in_good_standing(conn: sqlite3.Connection, machine_id: str,
                     now: datetime | None = None) -> bool:
    """The three-condition gate (docs/09 1.3):
    1. ``workers.standing != 'revoked'``.
    2. No abandon / no-show / rejected submission since the previous tick.
    3. Outstanding unverified accepted tasks within the standing's ceiling.

    A capability ``REFUSED`` (predicate miss) is neutral -- it does not clear
    the gate. The machine cannot help the queue; it is not farming."""
    now = now or utcnow()
    row = conn.execute(
        "SELECT standing FROM workers WHERE id = ?", (machine_id,)
    ).fetchone()
    if row is None or row["standing"] == "revoked":
        return False
    # Boundary for condition 2 is the previous tick, or enrollment if this is
    # the first tick -- pre-enrollment history must not clear a fresh gate.
    since_row = conn.execute(
        "SELECT at FROM availability_ticks WHERE machine_id = ? ORDER BY at DESC LIMIT 1",
        (machine_id,),
    ).fetchone()
    enrolled = conn.execute(
        "SELECT enrolled_at FROM workers WHERE id = ?", (machine_id,)
    ).fetchone()
    since = since_row["at"] if since_row is not None else (
        enrolled["enrolled_at"] if enrolled and enrolled["enrolled_at"]
        else _iso(now - timedelta(days=365))
    )
    if _infraction_since(conn, machine_id, since):
        return False
    return unverified_tasks(conn, machine_id) <= unverified_ceiling(row["standing"])


def record_availability_tick(conn: sqlite3.Connection, machine_id: str, *,
                             leased: bool, now: datetime | None = None) -> None:
    """Append one ``availability_ticks`` row per poll / per heartbeat, carrying
    the good-standing gate. On a good-standing tick, ``last_available_at``
    advances (docs/09 1.2 -- a poll is the only event every worker generates
    whether or not it gets work)."""
    now = now or utcnow()
    good = 1 if in_good_standing(conn, machine_id, now) else 0
    with immediate(conn):
        conn.execute(
            "INSERT INTO availability_ticks (machine_id, at, leased, in_good_standing) "
            "VALUES (?, ?, ?, ?)",
            (machine_id, _iso(now), 1 if leased else 0, good),
        )
        if good:
            conn.execute(
                "UPDATE workers SET last_available_at = ? WHERE id = ?",
                (_iso(now), machine_id),
            )


# --- the provisioned-accrual engine (docs/09 1.2, 1.4) -----------------------


def _window_start(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _window_seconds(ticks: list[tuple[datetime, int]]) -> dict[datetime, float]:
    """Good-standing seconds per hour-aligned window, from a machine's ordered
    ticks. A tick's ``in_good_standing`` answers "nothing bad since the
    previous tick", so it certifies the interval that *ends* at it: the pair
    ``(t_{i-1}, t_i]`` banks ``min(t_i - t_{i-1}, AWAKE_WINDOW_SEC)`` when
    the *later* tick is good. The first tick certifies nothing (no prior
    interval), and a straddling pair splits at the window boundary."""
    accum: dict[datetime, float] = defaultdict(float)
    step = timedelta(seconds=ACCRUAL_WINDOW_SEC)
    for i in range(1, len(ticks)):
        t0, _ = ticks[i - 1]
        t1, good = ticks[i]
        if not good:
            continue
        remaining = min((t1 - t0).total_seconds(), AWAKE_WINDOW_SEC)
        cur = t0
        while remaining > 0:
            ws = _window_start(cur)
            room = (ws + step - cur).total_seconds()
            take = min(remaining, room)
            accum[ws] += take
            remaining -= take
            cur += timedelta(seconds=take)
    return accum


def _month_credited(conn: sqlite3.Connection, machine_id: str,
                    ws: datetime) -> float:
    """Provisioned weighted hours already accrued by this machine in the
    calendar month containing ``ws`` (the probation monthly-ceiling budget)."""
    month_start = ws.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    row = conn.execute(
        """SELECT COALESCE(SUM(weighted_hours), 0.0) AS s FROM credit_events
            WHERE machine_id = ? AND kind = 'provisioned'
              AND period_start >= ? AND period_start < ?""",
        (machine_id, _iso(month_start), _iso(next_month)),
    ).fetchone()
    return float(row["s"])


def settle_windows(conn: sqlite3.Connection, *,
                   probation_monthly_cap_hours: float = PROBATION_MONTHLY_CAP_HOURS,
                   now: datetime | None = None) -> int:
    """The accrual sweep. For every machine, integrate its good-standing ticks
    into settled hour windows and write one ``credit_events`` row per
    ``(machine, window)``. Returns the number of rows written.

    Idempotent by construction: a window whose ``(machine_id, kind,
    period_start)`` already has a ``credit_events`` row is skipped (docs/09
    freezes the schema, so the uniqueness is enforced by this check, not a
    unique index). A window is settled only once its ``period_end`` is more
    than ``SETTLE_DELAY_SEC`` in the past. ``PROBATION_FACTOR`` and the monthly
    ceiling are read at settle time, so a machine back to ``good`` before
    settle gets the full window.
    """
    now = now or utcnow()
    written = 0
    # All ticks for machines that have any, sliced per machine. Only machines
    # with a settled-eligible window need a write, so build the window map first
    # and decide below -- this keeps the (machine, window) grouping cheap.
    machines = conn.execute(
        "SELECT DISTINCT machine_id FROM availability_ticks"
    ).fetchall()
    with immediate(conn):
        for m in machines:
            mid = m["machine_id"]
            rows = conn.execute(
                "SELECT at, in_good_standing FROM availability_ticks "
                "WHERE machine_id = ? ORDER BY at",
                (mid,),
            ).fetchall()
            ticks = [
                (_parse(r["at"]), int(r["in_good_standing"])) for r in rows
            ]
            for ws, seconds in _window_seconds(ticks).items():
                if seconds <= 0:
                    continue
                period_end = ws + timedelta(seconds=ACCRUAL_WINDOW_SEC)
                if (now - period_end).total_seconds() <= SETTLE_DELAY_SEC:
                    continue  # not settled yet
                if conn.execute(
                    """SELECT 1 FROM credit_events
                        WHERE machine_id = ? AND kind = 'provisioned'
                          AND period_start = ? LIMIT 1""",
                    (mid, _iso(ws)),
                ).fetchone():
                    continue  # already settled
                machine = conn.execute(
                    "SELECT contributor_id, standing, compute_profile_json "
                    "FROM workers WHERE id = ?",
                    (mid,),
                ).fetchone()
                if machine is None:
                    continue
                standing = machine["standing"] or "good"
                if standing == "revoked":
                    continue  # revoked accrues nothing (docs/09 5.3)
                factor = PROBATION_FACTOR if standing == "probation" else 1.0
                raw = round(seconds * factor)
                if raw <= 0:
                    continue
                if standing == "probation":
                    already = _month_credited(conn, mid, ws)
                    # ponytail: whole-window drop when the monthly cap is
                    # reached -- not a pro-rata trim. A machine that crosses the
                    # ceiling mid-window keeps the leftovers; refactor to split
                    # the window level precisely if the coarse ceiling ever
                    # matters (it is a fraud brake, not a billing exactitude).
                    if (already + (raw / 3600.0) * _weight_for(conn, mid, machine)
                            > probation_monthly_cap_hours):
                        continue
                weight_row = conn.execute(
                    "SELECT weight, formula_version FROM machine_weight "
                    "WHERE machine_id = ?",
                    (mid,),
                ).fetchone()
                if weight_row is None:
                    weight, _comp, ver = machine_weight_for(
                        json.loads(machine["compute_profile_json"])
                    )
                else:
                    weight, ver = float(weight_row["weight"]), int(
                        weight_row["formula_version"]
                    )
                weighted = raw / 3600.0 * weight
                conn.execute(
                    """INSERT INTO credit_events
                         (machine_id, user_id, kind, weighted_hours, raw_seconds,
                          system_weight, formula_version, period_start, period_end,
                          created_at)
                       VALUES (?, ?, 'provisioned', ?, ?, ?, ?, ?, ?, ?)""",
                    (mid, machine["contributor_id"], weighted, raw, weight, ver,
                     _iso(ws), _iso(period_end), _iso(now)),
                )
                written += 1
        # Once a window is settled the ticks that fed it are GC-eligible; drop
        # ticks older than retention (docs/09 1.4), independent of settle.
        cutoff = _iso(now - timedelta(days=TICK_RETENTION_DAYS))
        conn.execute(
            "DELETE FROM availability_ticks WHERE at < ?", (cutoff,)
        )
    return written


def _weight_for(conn: sqlite3.Connection, machine_id: str,
                machine: sqlite3.Row) -> float:
    """Best-effort weight for the probation-cap pre-check without a second
    query where the settle path already holds rows. Falls back to the profile
    lookup so the cap check and the final write agree even for an unweighted
    pre-ledger machine."""
    row = conn.execute(
        "SELECT weight FROM machine_weight WHERE machine_id = ?", (machine_id,)
    ).fetchone()
    if row is not None:
        return float(row["weight"])
    return machine_weight_for(json.loads(machine["compute_profile_json"]))[0]


# --- the secondary ``work`` signal (docs/09 4) -------------------------------


def record_work(conn: sqlite3.Connection, *, machine_id: str, user_id: str,
                units: int, now: datetime | None = None) -> None:
    """Record a credited ``work`` row -- recorded, never banked.

    ``kind = 'work'`` gets ``weighted_hours = 0.0`` and ``system_weight = 0.0``
    so an unfiltered ``SUM`` stays inert; ``raw_seconds`` carries the
    ``credit()`` WorkUnits scalar (its unit is fixed by the row's job type,
    recoverable via ``tasks``/``jobs``). ``period_start = period_end =
    created_at``. Every banked query filters ``kind = 'provisioned'``; that
    filter is what makes the ``raw_seconds`` overload safe (docs/09 spine note).
    """
    now = now or utcnow()
    ts = _iso(now)
    with immediate(conn):
        conn.execute(
            """INSERT INTO credit_events
                 (machine_id, user_id, kind, weighted_hours, raw_seconds,
                  system_weight, formula_version, period_start, period_end,
                  created_at)
               VALUES (?, ?, 'work', 0.0, ?, 0.0, 0, ?, ?, ?)""",
            (machine_id, user_id, max(units, 0), ts, ts, ts),
        )


# --- reputation & standing transitions (docs/09 5) ---------------------------


def _trailing_outcomes(conn: sqlite3.Connection, machine_id: str,
                       now: datetime) -> tuple[int, int]:
    """``(rejections, accepted)`` submissions for this machine over the trailing
    reputation window -- the raw material ``audit`` / ``submissions`` have been
    gathering."""
    since = _iso(now - timedelta(days=REP_TRAILING_DAYS))
    row = conn.execute(
        """SELECT
             SUM(CASE WHEN s.accepted = 0 THEN 1 ELSE 0 END) AS rej,
             SUM(CASE WHEN s.accepted = 1 THEN 1 ELSE 0 END) AS acc
           FROM submissions s
           JOIN tasks t ON t.id = s.task_id AND t.worker_id = ?
          WHERE s.accepted IS NOT NULL AND s.received_at >= ?""",
        (machine_id, since),
    ).fetchone()
    return int(row["rej"] or 0), int(row["acc"] or 0)


def recompute_reputation(conn: sqlite3.Connection, machine_id: str,
                         now: datetime | None = None) -> None:
    """Update one machine's ``reputation`` scalar and drive its ``standing``
    transitions (docs/09 5.1-5.3). Earned slowly (asymptotic to 1.0 on clean
    accepted work), lost fast (each rejection knocks it down). A cached
    rollup, recomputed on the sweep.

    v0 inputs are the ``validate()`` rejection / acceptance stream only;
    spot-checks and redundant-execution disagreement are Phase D and enter here
    as the extra penalty terms the moment they exist.
    """
    now = now or utcnow()
    with immediate(conn):
        row = conn.execute(
            "SELECT reputation, standing FROM workers WHERE id = ?", (machine_id,)
        ).fetchone()
        if row is None:
            return
        score = float(row["reputation"])
        standing = row["standing"] or "good"
        rej, acc = _trailing_outcomes(conn, machine_id, now)
        # Increment toward 1.0 on clean work; each rejection multiplies down
        # and subtracts a floor (docs/09 5.2).
        score = min(1.0, score + min(acc, 8) * 0.02)
        for _ in range(rej):
            score = max(0.0, score * 0.5 - 0.10)
        # Transitions (docs/09 5.3). ``revoked`` is terminal for accrual;
        # reinstatement is admin-only and not this module's job.
        new_standing = standing
        if standing == "revoked":
            pass
        elif standing == "good":
            if score < REP_GOOD:
                new_standing = "probation"
        elif standing == "probation":
            if score < REP_REVOKE:
                new_standing = "revoked"
            elif score >= REP_GOOD:
                # clean PROBATION_RECOVERY_DAYS window: no rejection in it.
                since = _iso(now - timedelta(days=PROBATION_RECOVERY_DAYS))
                dirty = conn.execute(
                    """SELECT 1 FROM submissions s
                         JOIN tasks t ON t.id = s.task_id AND t.worker_id = ?
                        WHERE s.accepted = 0 AND s.received_at >= ? LIMIT 1""",
                    (machine_id, since),
                ).fetchone()
                if dirty is None:
                    new_standing = "good"
        conn.execute(
            "UPDATE workers SET reputation = ?, standing = ? WHERE id = ?",
            (round(score, 6), new_standing, machine_id),
        )


def evaluate_reputation(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Recompute reputation for every machine that has any recorded outcome.
    Returns the number of machines evaluated."""
    now = now or utcnow()
    ids = conn.execute(
        """SELECT DISTINCT t.worker_id FROM submissions s
           JOIN tasks t ON t.id = s.task_id WHERE t.worker_id IS NOT NULL"""
    ).fetchall()
    for r in ids:
        recompute_reputation(conn, r["worker_id"], now)
    return len(ids)


# --- read models: /v1/me and /v1/leaderboard (docs/09 6) ---------------------


def accrued_current_window(conn: sqlite3.Connection, machine_id: str,
                           system_weight: float,
                           now: datetime | None = None) -> float:
    """Advisory (docs/09 6.1): good-standing seconds so far this hour times
    the system weight over 3600. Not a settled row -- just what this machine is
    on track for -- so it is read from live ticks, never from ``credit_events``.
    """
    now = now or utcnow()
    ws = _window_start(now)
    rows = conn.execute(
        "SELECT at, in_good_standing FROM availability_ticks "
        "WHERE machine_id = ? AND at >= ? ORDER BY at",
        (machine_id, _iso(ws)),
    ).fetchall()
    ticks = [(_parse(r["at"]), int(r["in_good_standing"])) for r in rows]
    secs = _window_seconds(ticks).get(ws, 0.0)
    return round(secs / ACCRUAL_WINDOW_SEC * system_weight, 6)


def accrued(conn: sqlite3.Connection, *, machine_id: str | None = None,
            user_id: str | None = None) -> float:
    """``SUM(weighted_hours) WHERE kind = 'provisioned'`` -- the one filter that
    every banked total applies (docs/09 4.1 invariant)."""
    clauses, args = [], []
    if machine_id is not None:
        clauses.append("machine_id = ?")
        args.append(machine_id)
    if user_id is not None:
        clauses.append("user_id = ?")
        args.append(user_id)
    where = " AND ".join(clauses)
    sql = (
        "SELECT COALESCE(SUM(weighted_hours), 0.0) AS s FROM credit_events "
        "WHERE kind = 'provisioned'" + (f" AND {where}" if where else "")
    )
    return float(conn.execute(sql, args).fetchone()["s"])


def recent_events(conn: sqlite3.Connection, machine_ids: list[str],
                  limit: int = 50) -> list[dict]:
    if not machine_ids:
        return []
    placeholders = ",".join("?" * len(machine_ids))
    return [dict(r) for r in conn.execute(
        f"""SELECT id, machine_id, kind, weighted_hours, raw_seconds,
                   system_weight, formula_version, period_start, period_end,
                   created_at
              FROM credit_events WHERE machine_id IN ({placeholders})
             ORDER BY created_at DESC, id DESC LIMIT ?""",
        (*machine_ids, limit),
    )]