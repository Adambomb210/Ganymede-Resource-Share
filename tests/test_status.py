"""The operator view and the stall alert (docs/03-roadmap.md M5).

The assertions that matter here are the ones about *not* firing. An alert that
goes off on a healthy volunteer fleet is worse than no alert, because it trains
the operator to ignore the one that matters -- which is the same reason
``invariants.py`` deliberately says nothing about idleness. 3.2 is explicit
that contributors come and go and that a round sitting open overnight is the
design working.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ganymede.coordinator import eligibility
from scripts import status as status_mod


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _age_round(conn, run_id, *, minutes: int, max_round_sec: int = 600):
    """Backdate the open round so it looks that old."""
    opened = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    conn.execute(
        "UPDATE rounds SET opened_at = ?, max_round_sec = ? WHERE run_id = ?",
        (_iso(opened), max_round_sec, run_id),
    )
    conn.commit()


def _worker(conn, worker_id: str) -> str:
    """A real contributor and worker row.

    `worker_eligibility` has foreign keys onto both, and `record` swallows the
    error when they are missing -- correct in production, where the claim path
    always holds a real worker, and quietly wrong in a test that invents ids.
    """
    now = _iso(datetime.now(timezone.utc))
    conn.execute(
        """INSERT OR IGNORE INTO contributors (id, name, key_hash, enabled, clearance, created_at)
           VALUES (?, ?, ?, 1, 'open', ?)""",
        (f"c-{worker_id}", worker_id, f"hash-{worker_id}", now),
    )
    conn.execute(
        """INSERT OR IGNORE INTO workers
             (id, contributor_id, compute_profile_json, image_tag, first_seen, last_seen)
           VALUES (?, ?, '{}', NULL, ?, ?)""",
        (worker_id, f"c-{worker_id}", now, now),
    )
    conn.commit()
    return worker_id


def _poll_recorded(conn, worker_id, run_id, *, minutes_ago: float = 0.0):
    _worker(conn, worker_id)
    eligibility.record(
        conn, worker_id, [eligibility.Verdict(run_id, eligibility.IDLE)],
        now=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


# --------------------------------------------------------------------------
# The stall alert, and above all when it stays quiet
# --------------------------------------------------------------------------


def test_an_idle_fleet_overnight_is_not_a_stall(conn, seeded_run):
    """3.2's normal operation. Nobody polled, so nothing is wrong -- the round
    is simply waiting, which is what it is designed to do."""
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=600, max_round_sec=600)

    assert status_mod.stalls(conn) == []


def test_a_round_open_past_its_backstop_while_workers_poll_is_a_stall(conn, seeded_run):
    """The M4a wedge signature: every worker polling, every poll answered 204,
    the round open forever, nothing anywhere returning an error."""
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=600, max_round_sec=600)
    _poll_recorded(conn, "worker-a", run_id)
    _poll_recorded(conn, "worker-b", run_id)

    found = status_mod.stalls(conn)
    assert len(found) == 1
    assert found[0].run_id == run_id
    assert "2 worker(s) were polling" in found[0].detail


def test_a_round_within_its_grace_is_not_a_stall(conn, seeded_run):
    """max_round_sec is a target, not a deadline: the close fires on the next
    request after the backstop passes, and on a slow fleet that is minutes."""
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=15, max_round_sec=600)  # 1.5x, under the 3x grace
    _poll_recorded(conn, "worker-a", run_id)

    assert status_mod.stalls(conn) == []


def test_a_fleet_that_polled_long_ago_does_not_count_as_awake(conn, seeded_run):
    """One stale row must not keep a silent alert firing forever."""
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=600, max_round_sec=600)
    _poll_recorded(conn, "worker-a", run_id, minutes_ago=120)

    assert status_mod.stalls(conn) == []


def test_a_closed_round_is_never_a_stall(conn, seeded_run):
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=600, max_round_sec=600)
    conn.execute("UPDATE rounds SET status = 'closed' WHERE run_id = ?", (run_id,))
    conn.commit()
    _poll_recorded(conn, "worker-a", run_id)

    assert status_mod.stalls(conn) == []


def test_a_paused_run_is_not_stalled(conn, seeded_run):
    """Only active runs. An operator who paused a run does not want to be told
    about it every fifteen minutes."""
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=600, max_round_sec=600)
    conn.execute("UPDATE runs SET status = 'paused' WHERE id = ?", (run_id,))
    conn.commit()
    _poll_recorded(conn, "worker-a", run_id)

    assert status_mod.stalls(conn) == []


# --------------------------------------------------------------------------
# Who is contributing
# --------------------------------------------------------------------------


def test_awake_workers_counts_a_machine_that_is_always_refused(conn, seeded_run):
    """A poll is the only event every worker generates whether or not it gets
    work, and a machine refused every time is exactly the one an operator most
    wants counted as present."""
    run_id = seeded_run()
    _worker(conn, "always-refused")
    eligibility.record(conn, "always-refused",
                       [eligibility.Verdict(run_id, eligibility.REFUSED, "vram_mb 1 < 2")])
    assert status_mod.awake_workers(conn) == ["always-refused"]


def test_median_cohort_is_the_number_that_says_the_swarm_is_earning_its_overhead(
    conn, seeded_run
):
    """3.2. If most rounds close with one machine, a single-node job would have
    been simpler and faster, and that should be visible now rather than
    inferred months later."""
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET status='closed', distinct_contributors=1 WHERE run_id=?",
                 (run_id,))
    for idx, cohort in ((1, 1), (2, 1)):
        conn.execute(
            """INSERT INTO rounds (run_id, idx, base_adapter_ref, status, target_steps,
                                   min_round_sec, max_round_sec, opened_at, closed_at,
                                   distinct_contributors)
               VALUES (?, ?, 'ref', 'closed', 100, 0, 600, ?, ?, ?)""",
            (run_id, idx, _iso(datetime.now(timezone.utc)),
             _iso(datetime.now(timezone.utc)), cohort),
        )
    conn.commit()

    st = status_mod.run_status(conn, run_id)
    assert st.median_cohort == 1
    assert st.median_cohort < status_mod.COHORT_FLOOR


def test_the_loss_trend_needs_two_evaluated_rounds_to_say_anything(conn, seeded_run):
    """`eval_loss` is written by a separate process (5.2), so a healthy run
    routinely has rounds with no loss yet. Reporting a trend from one point
    would be inventing one."""
    run_id = seeded_run()
    st = status_mod.run_status(conn, run_id)
    assert "not enough" in st.loss_trend


def test_the_loss_trend_reports_direction_and_magnitude(conn, seeded_run):
    run_id = seeded_run()
    conn.execute("UPDATE rounds SET status='closed', eval_loss=5.0 WHERE run_id=? AND idx=0",
                 (run_id,))
    conn.execute(
        """INSERT INTO rounds (run_id, idx, base_adapter_ref, status, target_steps,
                               min_round_sec, max_round_sec, opened_at, eval_loss)
           VALUES (?, 1, 'ref', 'closed', 100, 0, 600, ?, 4.5)""",
        (run_id, _iso(datetime.now(timezone.utc))),
    )
    conn.commit()

    trend = status_mod.run_status(conn, run_id).loss_trend
    assert "5.000 -> 4.500" in trend
    assert "down" in trend


# --------------------------------------------------------------------------
# The CLI, which is the actual deliverable
# --------------------------------------------------------------------------


def test_alert_mode_is_silent_and_zero_when_nothing_is_wrong(conn, seeded_run, settings, capsys):
    seeded_run()
    rc = status_mod.main(["--db", settings.db_path, "--alert"])
    captured = capsys.readouterr()
    assert rc == 0
    # Silence is the success case: a cron job that mails a report every fifteen
    # minutes is one whose mail gets filtered, and then so does the real one.
    assert captured.out == ""
    assert captured.err == ""


def test_alert_mode_exits_non_zero_on_a_stall(conn, seeded_run, settings, capsys):
    run_id = seeded_run()
    _age_round(conn, run_id, minutes=600, max_round_sec=600)
    _poll_recorded(conn, "worker-a", run_id)

    rc = status_mod.main(["--db", settings.db_path, "--alert"])
    assert rc == 1
    assert "stalled" in capsys.readouterr().err


def test_the_human_view_answers_both_halves_of_the_question(
    conn, seeded_run, settings, capsys
):
    run_id = seeded_run()
    _poll_recorded(conn, "worker-a", run_id)
    status_mod.main(["--db", settings.db_path])

    out = capsys.readouterr().out
    assert "polled in the last" in out   # who is contributing
    assert "coverage" in out             # is it training
    assert "cohort" in out
    assert run_id in out


def test_the_json_view_carries_the_same_facts(conn, seeded_run, settings, capsys):
    run_id = seeded_run()
    _poll_recorded(conn, "worker-a", run_id)
    status_mod.main(["--db", settings.db_path, "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["awake_workers"] == ["worker-a"]
    assert payload["runs"][0]["run_id"] == run_id
    assert payload["stalls"] == []


def test_no_active_runs_is_reported_rather_than_crashing(conn, settings, capsys):
    rc = status_mod.main(["--db", settings.db_path])
    assert rc == 0
    assert "no active runs" in capsys.readouterr().out
