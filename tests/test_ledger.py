"""The contribution ledger: weight lookup, good-standing gate, the provisioned
accrual engine, the secondary ``work`` signal, reputation transitions, and the
``/v1/me`` / ``/v1/leaderboard`` read models (docs/09-ledger.md)."""

from __future__ import annotations

import json

from ganymede.coordinator import ledger, rounds
from ganymede.coordinator.ledger import (
    AWAKE_WINDOW_SEC,
    PROBATION_FACTOR,
    REP_GOOD,
    machine_weight_for,
)


def _iso(h, m=0, s=0, day=10):
    return rounds._iso(rounds.utcnow().replace(
        day=day, hour=h, minute=m, second=s, microsecond=0))


def _make_machine(conn, user_id, *, backend="cuda", device="RTX 3060",
                  vram_mb=12288, cc="8.6", enrolled="09:00", standing="good"):
    mid = "m1"
    conn.execute(
        "INSERT OR IGNORE INTO contributors (id, name, key_hash, enabled, "
        "clearance, created_at) VALUES (?, ?, ?, 1, 'open', ?)",
        (user_id, user_id, "key-" + user_id, _iso(9)))
    conn.execute(
        """INSERT INTO workers
             (id, contributor_id, compute_profile_json, display_name,
              enrolled_at, first_seen, last_seen, standing, reputation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mid, user_id, json.dumps({"backend": backend, "device_name": device,
                                    "vram_mb": vram_mb,
                                    "compute_capability": cc}),
         "worker-1", _iso(9), _iso(9), _iso(9), standing, ledger.REP_ENROLL),
    )
    return mid


def _add_tick(conn, mid, at, good=1, leased=0):
    conn.execute(
        "INSERT INTO availability_ticks (machine_id, at, leased, in_good_standing) "
        "VALUES (?, ?, ?, ?)", (mid, at, leased, good))


def _abandon(conn, mid, at):
    conn.execute(
        """INSERT INTO tasks (id, run_id, round_idx, job_id, buckets_json,
              input_ref_json, attempt_group, local_steps, status, worker_id,
              lease_expires_at, attempts, created_at)
           VALUES (?, NULL, NULL, NULL, '[]', NULL, NULL, 1, 'abandoned', ?, NULL, 1, ?)""",
        ("t-abandon-" + mid, mid, at))


# --- machine_weight (docs/09 3.2) --------------------------------------------


def test_weight_gpu_classes():
    cases = {
        "cpu": 0.1,
        "gpu_low": 0.4,      # vram < 8000 OR cc < 7.0
        "gpu_3060": 1.0,     # 8-14 GB, the 1.0 anchor
        "gpu_3090": 2.0,     # 16-24 GB
        "gpu_high": 3.5,     # >= 24 GB
    }
    profiles = [
        {"backend": "cpu", "vram_mb": 0},
        {"backend": "cuda", "vram_mb": 4096, "compute_capability": "6.1"},
        {"backend": "cuda", "vram_mb": 12288, "compute_capability": "8.6"},
        {"backend": "cuda", "vram_mb": 16384, "compute_capability": "8.9"},
        {"backend": "cuda", "vram_mb": 40960, "compute_capability": "8.0"},
    ]
    for p, expect in zip(profiles, cases.values(), strict=False):
        weight, comp_json, ver = machine_weight_for(p)
        comp = json.loads(comp_json)
        assert comp["class"] != "gpu_unknown"
        # weight = base * clamp(vram/12000, 0.5, 1.5)
        clamp = min(max(p["vram_mb"] / 12000.0, 0.5), 1.5)
        assert abs(weight - expect * clamp) < 1e-6, (p, weight)
    # The anchor: a 3060 (12 GB) lands within a few per cent of 1.0.
    w, _c, ver = machine_weight_for({"backend": "cuda", "vram_mb": 12288,
                                     "compute_capability": "8.6"})
    assert 0.9 < w < 1.2
    assert ver == 0  # formula_version is explicitly interim


# --- provisioned accrual engine (docs/09 1) ----------------------------------


def test_settle_window_and_idempotency(conn, settings):
    from ganymede.coordinator.db import immediate
    user = "u1"
    mid = _make_machine(conn, user)
    with immediate(conn):
        mw = machine_weight_for({"backend": "cuda", "vram_mb": 12288,
                                 "compute_capability": "8.6"})
        conn.execute(
            "INSERT INTO machine_weight (machine_id, weight, components_json, "
            "formula_version, computed_at) VALUES (?, ?, ?, 0, ?)",
            (mid, mw[0], mw[1], _iso(12)))

    # Ticks 10:00 -> 10:15 -> 10:30, all good. Intervals are 900s (the cap), so
    # the 10:00 window banks 1800s; the 10:30 tick is open-ended and banks 0.
    _add_tick(conn, mid, _iso(10, 0, 0), good=1)
    _add_tick(conn, mid, _iso(10, 15, 0), good=1)
    _add_tick(conn, mid, _iso(10, 30, 0), good=1)

    # Window [10:00, 11:00) settles only after 11:30 (period_end + 1800s).
    assert ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=11, minute=29)) == 0
    n = ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13))
    assert n == 1
    rows = conn.execute(
        "SELECT * FROM credit_events WHERE machine_id = ?", (mid,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == "provisioned"
    assert rows[0]["raw_seconds"] == 1800
    assert rows[0]["user_id"] == user
    assert abs(rows[0]["weighted_hours"] - 1800 / 3600.0 * mw[0]) < 1e-6

    # Idempotent: the same sweep writes nothing new.
    assert ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13)) == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM credit_events WHERE machine_id = ?", (mid,)
    ).fetchone()["n"] == 1


def test_revoked_accrues_nothing(conn, settings):
    user, _k = "u1", None
    mid = _make_machine(conn, user, standing="revoked")
    _add_tick(conn, mid, _iso(10, 0, 0), good=1)
    _add_tick(conn, mid, _iso(10, 15, 0), good=1)
    assert ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13)) == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM credit_events").fetchone()["n"] == 0


def test_abandon_clears_good_standing_gate(conn, settings):
    user = "u1"
    mid = _make_machine(conn, user)
    # First tick, clean: the boundary is enrolled_at (09:00), nothing happened
    # since, so the gate passes and the tick is good.
    ledger.record_availability_tick(
        conn, mid, leased=False,
        now=rounds.utcnow().replace(day=10, hour=9, minute=30))
    # The machine abandons a claimed task after that tick. The next poll's gate
    # checks back to the previous tick (09:30) and must clear it.
    _abandon(conn, mid, _iso(9, 40))
    ledger.record_availability_tick(
        conn, mid, leased=False,
        now=rounds.utcnow().replace(day=10, hour=10, minute=0))
    ticks = conn.execute(
        "SELECT at, in_good_standing FROM availability_ticks WHERE machine_id = ? "
        "ORDER BY at", (mid,)
    ).fetchall()
    assert ticks[0]["in_good_standing"] == 1
    assert ticks[1]["in_good_standing"] == 0
    # No good-standing time banks the abandoned window.
    assert ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13)) == 0


# --- good-standing gate, live ------------------------------------------------


def test_good_standing_gate_revoked_and_unverified(conn, settings):
    user = "u1"
    mid = _make_machine(conn, user)
    assert ledger.in_good_standing(conn, mid)  # clean, good standing
    assert ledger.unverified_ceiling("good") == ledger.K_GOOD
    assert ledger.unverified_ceiling("probation") == ledger.K_PROBATION
    assert ledger.unverified_ceiling("revoked") == 0
    conn.execute("UPDATE workers SET standing = 'revoked' WHERE id = ?", (mid,))
    assert not ledger.in_good_standing(conn, mid)


# --- probation factor and monthly cap ----------------------------------------


def test_probation_scales_window(conn, settings):
    user = "u1"
    mid = _make_machine(conn, user, standing="probation")
    mw = machine_weight_for({"backend": "cuda", "vram_mb": 12288,
                             "compute_capability": "8.6"})
    with __import__("ganymede.coordinator.db", fromlist=["immediate"]).immediate(conn):
        conn.execute(
            "INSERT INTO machine_weight (machine_id, weight, components_json, "
            "formula_version, computed_at) VALUES (?, ?, ?, 0, ?)",
            (mid, mw[0], mw[1], _iso(12)))
    _add_tick(conn, mid, _iso(10, 0, 0), good=1)
    _add_tick(conn, mid, _iso(10, 15, 0), good=1)
    _add_tick(conn, mid, _iso(10, 30, 0), good=1)
    assert ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13)) == 1
    row = conn.execute("SELECT * FROM credit_events").fetchone()
    # Whole window scaled by PROBATION_FACTOR: 1800 * 0.5 = 900 raw.
    assert row["raw_seconds"] == 1800 * PROBATION_FACTOR
    assert abs(row["weighted_hours"] - (900 / 3600.0) * mw[0]) < 1e-6


# --- the work signal (docs/09 4) ---------------------------------------------


def test_work_signal_recorded_and_inert(conn, settings):
    user = "u1"
    mid = _make_machine(conn, user)
    ledger.record_work(conn, machine_id=mid, user_id=user, units=1234)
    row = conn.execute(
        "SELECT * FROM credit_events WHERE machine_id = ?", (mid,)
    ).fetchone()
    assert row["kind"] == "work"
    assert row["weighted_hours"] == 0.0
    assert row["system_weight"] == 0.0
    assert row["raw_seconds"] == 1234
    # The banked total ignores it.
    assert ledger.accrued(conn, machine_id=mid) == 0.0
    assert ledger.accrued(conn, user_id=user) == 0.0


# --- reputation & standing transitions (docs/09 5) ----------------------------


def _reject(conn, mid, at):
    conn.execute(
        """INSERT INTO tasks (id, run_id, round_idx, job_id, buckets_json,
              input_ref_json, attempt_group, local_steps, status, worker_id,
              lease_expires_at, attempts, created_at)
           VALUES (?, NULL, NULL, NULL, '[]', NULL, NULL, 1, 'submitted', ?, NULL, 1, ?)""",
        ("t-" + at + mid, mid, at))
    conn.execute(
        """INSERT INTO submissions (task_id, artifact_ref, steps_completed,
              tokens_seen, metrics_json, accepted, reject_reason, received_at)
           VALUES (?, 'x', 1, 0, '{}', 0, 'bad', ?)""",
        ("t-" + at + mid, at))


def test_rejections_push_to_probation(conn, settings):
    user = "u1"
    mid = _make_machine(conn, user)  # starts good, reputatation REP_ENROLL
    _reject(conn, mid, _iso(11, 0))
    ledger.recompute_reputation(conn, mid, now=rounds.utcnow().replace(day=10, hour=12))
    w = conn.execute("SELECT reputation, standing FROM workers WHERE id = ?", (mid,)).fetchone()
    assert w["reputation"] < REP_GOOD
    assert w["standing"] == "probation"


# --- read models -------------------------------------------------------------


def test_me_route(client, conn, settings, make_contributor):
    user_id, key = make_contributor("alice")
    mid = _make_machine(conn, user_id)
    mw = machine_weight_for({"backend": "cuda", "vram_mb": 12288,
                             "compute_capability": "8.6"})
    from ganymede.coordinator.db import immediate
    with immediate(conn):
        conn.execute("INSERT INTO machine_weight (machine_id, weight, components_json, "
                     "formula_version, computed_at) VALUES (?, ?, ?, 0, ?)",
                     (mid, mw[0], mw[1], _iso(12)))
    _add_tick(conn, mid, _iso(10, 0, 0), good=1)
    _add_tick(conn, mid, _iso(10, 15, 0), good=1)
    _add_tick(conn, mid, _iso(10, 30, 0), good=1)
    ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13))
    ledger.record_work(conn, machine_id=mid, user_id=user_id, units=7)

    resp = client.get("/v1/me", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["name"] == "alice"
    assert body["totals"]["machines"] == 1
    assert body["totals"]["weighted_hours"] == 1800 / 3600.0 * mw[0]
    m = body["machines"][0]
    assert m["standing"] == "good"
    assert m["system_weight"] == mw[0]
    assert m["formula_version"] == 0
    kinds = {e["kind"] for e in body["recent_events"]}
    assert kinds == {"provisioned", "work"}
    work = next(e for e in body["recent_events"] if e["kind"] == "work")
    assert work["weighted_hours"] == 0.0
    assert work["raw_seconds"] == 7


def test_leaderboard_machines(client, conn, settings, make_contributor):
    uid, key = make_contributor("bob")
    other_uid, _other_key = make_contributor("carol")
    mid = _make_machine(conn, uid)
    mw = machine_weight_for({"backend": "cuda", "vram_mb": 12288,
                             "compute_capability": "8.6"})
    from ganymede.coordinator.db import immediate
    with immediate(conn):
        conn.execute("INSERT INTO machine_weight (machine_id, weight, components_json, "
                     "formula_version, computed_at) VALUES (?, ?, ?, 0, ?)",
                     (mid, mw[0], mw[1], _iso(12)))
        conn.execute(
            "INSERT INTO workers (id, contributor_id, compute_profile_json, "
            "display_name, enrolled_at, first_seen, last_seen, standing, reputation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'good', 0.25)",
            ("m2", other_uid, '{"backend":"cuda"}', "wm-2", _iso(9), _iso(9), _iso(9)))
    _add_tick(conn, mid, _iso(10, 0, 0), good=1)
    _add_tick(conn, mid, _iso(10, 15, 0), good=1)
    ledger.record_work(conn, machine_id="m2", user_id=other_uid, units=99)
    ledger.settle_windows(conn, now=rounds.utcnow().replace(day=10, hour=13))

    resp = client.get("/v1/leaderboard",
                      headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["formula_version_current"] == 0
    assert len(body["by_machine"]) == 1  # m2 has only 'work', never ranks
    assert body["by_machine"][0]["machine_id"] == mid
    assert body["by_machine"][0]["weighted_hours"] > 0
    # work units never leak into a machine rank, but the cosmetic list carries them
    assert {w["machine_id"] for w in body["by_work"]} == {"m2"}
    # users view
    u = client.get("/v1/leaderboard?scope=users",
                   headers={"Authorization": f"Bearer {key}"}).json()
    assert len(u["by_user"]) == 1
    assert u["by_user"][0]["user_name"] == "bob"