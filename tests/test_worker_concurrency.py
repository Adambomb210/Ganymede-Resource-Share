"""M4a: the protocol under real concurrency.

Three real worker *processes* against one real coordinator, one real MinIO and
one real run, for several rounds. Everything the rest of the suite fakes is
real here: separate OS processes with separate SQLite connections, separate
torch runtimes and separate contributors, racing each other through claim,
heartbeat, submit and close.

Processes rather than threads, deliberately. Threads would share a GIL, a torch
runtime and a process image, which serializes exactly the part under test -- and
you cannot SIGKILL a thread, so the blast-radius test below would have nothing
to kill.

The four things M4a says must hold (docs/03-roadmap.md, "M4a"):

1. Three concurrent workers complete several rounds with no double-leasing and
   no stuck tasks
2. Killing one mid-round costs that round's work for that worker and nothing
   else
3. Bucket coverage advances rather than re-training the same shards
4. Per-round loss descends across rounds

1 and 3 are checked against ``coordinator.invariants``, which is the same thing
an operator runs by hand -- so a green test and a healthy deployment mean the
same thing. 2 has a test of its own. 4 needs an evaluated loss curve, which is
what ``scripts.evalround`` produces.

Marked ``slow``: Docker for MinIO, uvicorn as a subprocess, and three torch
processes on whatever cores are going.

**Do not run a second pytest process against this repository while this one is
in flight.** The MinIO fixture uses a fixed container name and a fixed port, so
a concurrent run tears the container down underneath this one. The symptom is
every worker dying at the same instant on a refused connection to the object
store, which reads convincingly as a protocol failure and is not one.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from ganymede.coordinator import invariants
from tests.conftest import TINY_LORA_CFG
from tests.test_store import (  # noqa: F401 - imported for the fixture itself
    PUBLIC_ENDPOINT,
    ROOT_PASSWORD,
    ROOT_USER,
    minio_container,
)
from tests.test_worker_live import REPO_ROOT, _free_port, _wait_for

pytestmark = pytest.mark.slow

WORKERS = 3

NUM_BUCKETS = 32
TARGET_ROUNDS = 4
EVAL_SIZE = 40

# The run has to be big enough that three workers can each be given a *piece* of
# it. That is not a detail of the harness -- it is the condition under which
# sharding exists at all, and the first M4a run got it wrong in an instructive
# way.
#
# A worker's budget is `throughput x usable_time`, and the shard it is given is
# sized to hold one `target_passes` sweep of that budget. When the budget
# outgrows the whole dataset, `bucket_count` caps the shard at every bucket
# there is -- so every worker in the round gets the entire dataset, trains the
# identical rows, and aggregation weights the copies as independent
# contributions. On the first run that is exactly what happened from round 1
# onward, once measured throughput replaced the cold-start guess: 14,268 steps
# and all 32 buckets, each.
#
# Two things came out of it. `plan_budget` now clamps the budget to the data it
# actually handed out, so no worker silently does eighty passes on a run
# configured for four. And the run below is sized with room to spare: 20k rows
# over 32 buckets is ~620 samples each, which at `target_passes: 4` leaves the
# ceiling roughly seven times above what a CPU worker reaches in a 60-second
# round. `whole_dataset_tasks` in the coverage assertion is what fails if that
# headroom ever disappears.
DATASET_ROWS = 20_000
TARGET_PASSES = 4.0

# How many instruction categories the synthetic data has. Small enough that a
# 107k-parameter model can learn the mapping inside a few rounds, large enough
# that the mapping is not memorisable from one bucket -- so a worker holding a
# fraction of the data still learns something the others' shards agree with,
# which is the property aggregation is supposed to preserve.
TOPICS = 8

MIN_ROUND_SEC = 5
MAX_ROUND_SEC = 60

# High enough that no single worker's submission ends the round, so rounds are
# genuinely three-way rather than a race the first submitter wins. If the fleet
# comes in under it the round still closes on its backstop.
TARGET_STEPS = 15_000


def _rows(conn: sqlite3.Connection, sql: str, *params) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params)]


def _db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


class Fleet:
    """Three worker processes, started together and stopped together."""

    def __init__(self, stack, tmp_path, count: int = WORKERS):
        self.stack = stack
        self.tmp_path = tmp_path
        self.count = count
        self.procs: list[subprocess.Popen] = []
        self.logs: list[Path] = []
        self.handles: list = []

    def start(self, max_rounds: int = TARGET_ROUNDS * 4) -> "Fleet":
        for i in range(self.count):
            state = self.tmp_path / f"state-{i}"
            state.mkdir(parents=True, exist_ok=True)
            log_path = self.tmp_path / f"worker-{i}.log"
            self.logs.append(log_path)
            env = {
                **self.stack["env"],
                "GANYMEDE_COORDINATOR_URL": self.stack["url"],
                "GANYMEDE_KEY": self.stack["keys"][i],
                "GANYMEDE_RUN_ID": self.stack["run_id"],
                "GANYMEDE_STATE": str(state),
                # Three torch processes on a handful of cores will thrash if each
                # one opens a full thread pool. One thread each is also what a
                # contributor running a worker alongside their own work wants.
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            # A file, never subprocess.PIPE. A pipe nobody reads holds about
            # 64 KB, and a worker looping on short rounds fills that in under a
            # minute -- then blocks forever on its next log write, mid-round,
            # holding a lease. Two of three workers died exactly that way on the
            # first attempt at this test, and the symptom was indistinguishable
            # from a coordinator that had stopped handing out work.
            handle = log_path.open("w")
            self.handles.append(handle)
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "ganymede.worker.loop",
                 "--backend", "cpu", "--skip-bench",
                 "--max-rounds", str(max_rounds), "--log-level", "INFO"],
                cwd=str(REPO_ROOT), env=env,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            ))
        return self

    def kill(self, index: int) -> None:
        """SIGKILL, not terminate: a machine that loses power does not get to
        run its abandon handler, and that is the case the lease exists for."""
        self.procs[index].send_signal(signal.SIGKILL)
        self.procs[index].wait(timeout=30)

    def alive(self) -> int:
        return sum(1 for p in self.procs if p.poll() is None)

    def stop(self) -> list[str]:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        for handle in self.handles:
            handle.close()
        return [path.read_text() if path.exists() else "" for path in self.logs]


@pytest.fixture(scope="module")
def fleet_rows():
    """Synthetic Dolly-shaped rows with something in them that can be learned.

    Its own set rather than ``tiny_rows``: this run needs two orders of
    magnitude more of them, for the reason at DATASET_ROWS above, and it needs
    them to carry a signal.

    That second part is not decoration. The obvious synthetic dataset -- random
    words in, random words out -- has no learnable structure at all, so held-out
    loss on it cannot move for any reason except noise, and "loss descends
    across rounds" becomes a coin flip dressed up as an assertion. Measured on
    exactly that data: 5.185 -> 5.146 -> 5.151 -> 5.150 across four rounds, a
    total movement of 0.7% that was not even monotone. It satisfied
    ``last < first`` and meant nothing.

    So each row's response is a deterministic function of its instruction: the
    first token names one of ``TOPICS`` categories, and the response is that
    category's fixed phrase. Learning it means learning to read one token and
    emit five, which a 107k-parameter model gets in a few hundred steps -- and
    which no amount of aggregation noise will produce by accident.

    Every token here comes from ``tiny_model_dir``'s actual vocabulary, which is
    ``w0``..``w199`` and nothing else. Inventing readable names like ``topic_3``
    would send every one of them to ``<unk>``, and the "signal" would be a model
    learning to emit the unknown token five times -- which looks like learning
    in the loss curve and is not.
    """
    import random

    rng = random.Random(4)
    # Three disjoint slices of the vocabulary, so the signal cannot be confused
    # with the noise: markers, filler, and the answer phrases.
    topics = [f"w{k}" for k in range(TOPICS)]                       # w0..w7
    answers = {
        f"w{k}": " ".join(f"w{100 + k * 5 + j}" for j in range(5))  # w100..w139
        for k in range(TOPICS)
    }

    def filler(n):
        return " ".join(f"w{rng.randrange(10, 90)}" for _ in range(n))

    rows = []
    for i in range(DATASET_ROWS):
        topic = topics[i % TOPICS]
        rows.append({
            "instruction": f"{topic} {filler(5)}",
            "context": filler(4) if i % 3 == 0 else "",
            "response": answers[topic],
        })
    return rows


@contextmanager
def _stack(tmp_path, model_dir, rows, run_id: str, target_rounds: int):
    """A coordinator process, a multi-round run, and one key per worker.

    One contributor per worker rather than one shared key: ``distinct_contributors``
    is what a round records as its cohort size, and three workers under one key
    would report a cohort of one -- making a genuinely concurrent round
    indistinguishable in the database from a solo one.

    A context manager rather than a fixture, because the two things that need it
    want different lifetimes: the four assertions about one completed run share
    a module-scoped stack, and the kill test needs its own, since it cannot run
    against a run that has already finished.
    """
    from ganymede.coordinator.auth import generate_key, hash_key
    from ganymede.coordinator.config import Settings
    from ganymede.coordinator.db import connect, init_schema
    from ganymede.coordinator import rounds
    from ganymede.coordinator.store import Store
    from scripts import newrun

    port = _free_port()
    db_path = str(tmp_path / "fleet.db")
    bucket = f"ganymede-m4a-{uuid.uuid4().hex[:8]}"
    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps(rows))

    env = {
        **os.environ,
        "GANYMEDE_DB": db_path,
        "STORAGE_HOST": PUBLIC_ENDPOINT,
        "S3_BUCKET": bucket,
        "S3_ACCESS_KEY": ROOT_USER,
        "S3_SECRET_KEY": ROOT_PASSWORD,
        "COORDINATOR_HOST": f"http://127.0.0.1:{port}",
        "GANYMEDE_REQUIRE_TLS": "0",
        "GANYMEDE_MIN_USABLE_SEC": "5",
        "GANYMEDE_EST_DOWNLOAD_SEC": "1",
        "GANYMEDE_EST_UPLOAD_SEC": "1",
        "GANYMEDE_EST_SETUP_SEC": "5",
        "GANYMEDE_SAFETY_MARGIN_SEC": "1",
        # Short enough that a killed worker's shard comes back inside one round,
        # which is the whole point of the blast-radius test.
        "GANYMEDE_LEASE_SEC": "30",
        "GANYMEDE_HEARTBEAT_SEC": "5",
        # A 60-second poll would have a worker sit out an entire round after
        # each close. This is the knob that makes short rounds workable at all.
        "GANYMEDE_POLL_SEC": "2",
        "PYTHONPATH": str(REPO_ROOT),
    }

    n_train = len(rows) - EVAL_SIZE
    samples_per_bucket = n_train // NUM_BUCKETS
    argv = [
        "--run-id", run_id, "--base-model", model_dir, "--base-precision", "fp32",
        "--dataset", f"file://{dataset}", "--dataset-rows", str(len(rows)),
        "--eval-size", str(EVAL_SIZE), "--num-buckets", str(NUM_BUCKETS),
        "--target-rounds", str(target_rounds),
        "--target-steps", str(TARGET_STEPS),
        "--min-round-sec", str(MIN_ROUND_SEC), "--max-round-sec", str(MAX_ROUND_SEC),
        "--lora-r", str(TINY_LORA_CFG["rank"]), "--lora-alpha", str(TINY_LORA_CFG["alpha"]),
        "--lora-dropout", "0.0",
        "--target-modules", ",".join(TINY_LORA_CFG["target_modules"]),
        "--hyperparams", json.dumps({
            "seq_len": 32, "micro_batch": 2, "grad_accum": 1,
            "gradient_checkpointing": False,
            # Sized against what this model actually does on a CPU core, so
            # round 0 hands out a real budget instead of hundreds of seven-step
            # tasks. A cold start far below the truth is safe in production --
            # it costs one under-filled round and is corrected by measurement --
            # but here it turns round 0 into a claim/submit storm that tells us
            # nothing about concurrency.
            "cold_start_steps_per_min": 3000.0,
            "target_passes": TARGET_PASSES,
            "lr": 5e-3,
        }),
    ]

    saved = dict(os.environ)
    os.environ.update({k: v for k, v in env.items() if k != "PYTHONPATH"})
    try:
        settings = Settings.from_env()
        store = Store(settings.storage)
        assert newrun.main(argv, settings=settings, store=store) == 0

        conn = connect(db_path)
        init_schema(conn)
        keys = []
        for i in range(WORKERS):
            key = generate_key()
            keys.append(key)
            conn.execute(
                """INSERT INTO contributors (id, name, key_hash, enabled, clearance, created_at)
                   VALUES (?, ?, ?, 1, 'open', ?)""",
                (uuid.uuid4().hex, f"fleet-{i}", hash_key(key),
                 rounds._iso(rounds.utcnow())),
            )
        conn.commit()
        conn.close()
    finally:
        os.environ.clear()
        os.environ.update(saved)

    # Same reasoning as the worker logs above, and more acute: uvicorn logs a
    # line per request, and three workers polling every two seconds for several
    # minutes is thousands of them.
    server_log = tmp_path / "coordinator.log"
    server_handle = server_log.open("w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ganymede.coordinator.app:bootstrap",
         "--factory", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT), env=env,
        stdout=server_handle, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not _wait_for(f"http://127.0.0.1:{port}/healthz"):
            server.terminate()
            server.wait(timeout=5)
            server_handle.close()
            pytest.skip(f"coordinator did not start: {server_log.read_text()[-2000:]}")
        yield {
            "url": f"http://127.0.0.1:{port}", "keys": keys, "db": db_path,
            "run_id": run_id, "env": env, "samples_per_bucket": samples_per_bucket,
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        server_handle.close()


@pytest.fixture(scope="module")
def fleet_stack(minio_container, tmp_path_factory, tiny_model_dir, fleet_rows):
    with _stack(tmp_path_factory.mktemp("fleet"), tiny_model_dir, fleet_rows,
                "m4a", TARGET_ROUNDS) as stack:
        yield stack


@pytest.fixture
def kill_stack(minio_container, tmp_path, tiny_model_dir, fleet_rows):
    """Its own stack. The kill test needs a run with work still in it, and the
    module-scoped one has by then been driven to completion."""
    with _stack(tmp_path, tiny_model_dir, fleet_rows, "m4a-kill", TARGET_ROUNDS) as stack:
        yield stack


def _wait_for_rounds(db_path: str, closed: int, timeout: float) -> int:
    """Block until ``closed`` rounds have closed, or time out. Returns the count."""
    deadline = time.time() + timeout
    seen = 0
    while time.time() < deadline:
        conn = _db(db_path)
        try:
            seen = conn.execute(
                "SELECT COUNT(*) FROM rounds WHERE status = 'closed'"
            ).fetchone()[0]
        finally:
            conn.close()
        if seen >= closed:
            return seen
        time.sleep(1)
    return seen


@pytest.fixture(scope="module")
def completed_run(fleet_stack, tmp_path_factory):
    """The run, driven to completion by three concurrent worker processes.

    Module-scoped: the four assertions below are four questions about one run,
    not four runs. Rebuilding the fleet per test would quadruple the cost and,
    worse, quietly turn "these things were all true of one execution" into four
    independent executions that each happened to pass.
    """
    fleet = Fleet(fleet_stack, tmp_path_factory.mktemp("fleet-workers")).start()
    try:
        closed = _wait_for_rounds(fleet_stack["db"], TARGET_ROUNDS, timeout=900)
    finally:
        logs = fleet.stop()
    if closed < TARGET_ROUNDS:
        tail = "\n\n".join(log[-3000:] for log in logs)
        pytest.fail(f"only {closed}/{TARGET_ROUNDS} rounds closed in 900s\n{tail}")
    return {**fleet_stack, "logs": logs}


# --------------------------------------------------------------------------
# Exit criterion 1: several rounds, no double-leasing, no stuck tasks
# --------------------------------------------------------------------------


def test_the_run_completes_with_no_protocol_violations(completed_run):
    """The invariants, over the database three racing processes left behind.

    This is the assertion M4a's first criterion actually reduces to. Checking it
    from the database rather than inside the request path means it covers the
    interleavings no single request ever observes.
    """
    conn = _db(completed_run["db"])
    try:
        violations = invariants.check(conn)
    finally:
        conn.close()

    assert violations == [], "\n".join(str(v) for v in violations)


def test_every_round_closed_and_the_chain_of_adapters_is_unbroken(completed_run):
    """Each round's result is the next round's base. Break that link and every
    later round trains from the wrong weights while every gate still passes."""
    conn = _db(completed_run["db"])
    try:
        rounds_after = _rows(conn, "SELECT * FROM rounds ORDER BY idx")
        run = _rows(conn, "SELECT * FROM runs WHERE id = ?", completed_run["run_id"])[0]
    finally:
        conn.close()

    assert len(rounds_after) == TARGET_ROUNDS
    assert run["status"] == "done"
    for rnd in rounds_after:
        assert rnd["status"] == "closed", rnd
        assert rnd["result_adapter_ref"], rnd
    for earlier, later in zip(rounds_after, rounds_after[1:]):
        assert later["base_adapter_ref"] == earlier["result_adapter_ref"]


def test_the_rounds_were_genuinely_concurrent(completed_run):
    """Three workers finishing one after another would satisfy every other
    assertion here while proving nothing about concurrency. A round that
    aggregated more than one contributor is the evidence that they overlapped."""
    conn = _db(completed_run["db"])
    try:
        cohorts = [r["distinct_contributors"] for r in
                   _rows(conn, "SELECT distinct_contributors FROM rounds ORDER BY idx")]
        workers = _rows(conn, "SELECT id, rounds_joined, steps_total FROM workers")
    finally:
        conn.close()

    assert len(workers) == WORKERS, workers
    assert max(cohorts) > 1, f"no round had more than one contributor: {cohorts}"
    # Every worker did real work, not just registered and idled.
    assert all(w["steps_total"] > 0 for w in workers), workers


def test_late_submissions_were_refused_rather_than_folded_into_a_closed_round(
    completed_run
):
    """A worker still training when its round closes drops the work (3.2). The
    database record of that is a task expired by the close with no submission --
    and never a submission attached to a round that had already aggregated."""
    conn = _db(completed_run["db"])
    try:
        accepted_after_close = _rows(conn, """
            SELECT s.task_id FROM submissions s
            JOIN tasks t  ON t.id = s.task_id
            JOIN rounds r ON r.run_id = t.run_id AND r.idx = t.round_idx
            WHERE s.accepted = 1 AND r.closed_at IS NOT NULL
              AND s.received_at > r.closed_at""")
    finally:
        conn.close()

    assert accepted_after_close == []


# --------------------------------------------------------------------------
# Exit criterion 3: coverage advances
# --------------------------------------------------------------------------


def test_bucket_coverage_advances_rather_than_retraining_the_same_shards(completed_run):
    """The failure this rules out is a run that looks busy and learns nothing
    new: work accumulating on a handful of shards while the rest go untouched.

    ``spread`` is the diagnostic -- least-trained-first should keep every bucket
    within a round or so of every other, however unevenly the workers were
    budgeted.
    """
    conn = _db(completed_run["db"])
    try:
        cover = invariants.coverage(conn, completed_run["run_id"])
    finally:
        conn.close()

    assert cover["buckets"] == NUM_BUCKETS
    assert cover["distinct_trained"] == NUM_BUCKETS, cover
    assert cover["min"] >= 1, cover
    assert cover["spread"] <= 2, cover

    # And the pathology spread cannot see. The first M4a run put every worker on
    # every bucket from round 1 onward -- three copies of one piece of work,
    # weighted as three contributions -- and `spread` stayed at zero throughout,
    # because everyone trained everything equally. Budgets are now clamped to
    # the data they were assigned (budget.plan_budget), so this stays at zero.
    assert cover["whole_dataset_tasks"] == 0, cover


# --------------------------------------------------------------------------
# Exit criterion 4: the loss descends
# --------------------------------------------------------------------------


def test_the_loss_descends_across_the_run(completed_run):
    """Held-out loss for each round's *aggregated* adapter, computed after the
    fact by the same evaluator an operator would run.

    What this does and does not claim. It is a 107k-parameter model over forty
    held-out examples of synthetic data, so it says nothing about convergence at
    1.7B -- that is M4b's, and it needs hardware this milestone does not have.
    What it does establish is that **aggregation carries the training signal
    rather than destroying it**: three workers train disjoint shards, their
    adapters are averaged into one, and the average is better at the task than
    the round before. That is the property the whole design rests on, and it is
    testable at any scale where the data has something in it to learn.

    Which is why ``fleet_rows`` carries a signal. On random-words-in,
    random-words-out this assertion measured 5.185 -> 5.146 -> 5.151 -> 5.150 --
    0.7% of movement, not monotone, and satisfied by noise in either direction.

    The claim is about the trend rather than every adjacent pair: forty examples
    is a noisy measurement and a single tick upward between two rounds is not
    evidence of a broken aggregate. A last round no better than the first is.
    """
    from ganymede.coordinator.config import Settings
    from ganymede.coordinator.db import connect
    from ganymede.coordinator.store import Store
    from scripts import evalround

    saved = dict(os.environ)
    os.environ.update({k: v for k, v in completed_run["env"].items() if k != "PYTHONPATH"})
    try:
        settings = Settings.from_env()
        conn = connect(settings.db_path)
        try:
            evaluated = evalround.run_once(
                conn, evalround.Evaluator(Store(settings.storage)), completed_run["run_id"]
            )
            curve = invariants.loss_curve(conn, completed_run["run_id"])
        finally:
            conn.close()
    finally:
        os.environ.clear()
        os.environ.update(saved)

    assert evaluated == TARGET_ROUNDS, curve
    assert len(curve) == TARGET_ROUNDS
    losses = [loss for _, loss in curve]

    # A NaN here is the failure mode that matters most and shows up least: one
    # bad tensor in one submission poisons the mean, and every gate still passes.
    assert all(math.isfinite(loss) for loss in losses), f"non-finite loss: {curve}"

    # The signal in this data is worth roughly two nats to a model that learns
    # it. Asking for a tenth of that is well outside what noise on forty
    # examples produces, and well inside what learning the mapping delivers.
    assert losses[-1] < losses[0] - 0.2, (
        f"aggregation did not carry the training signal: {curve}"
    )


# --------------------------------------------------------------------------
# Exit criterion 2: the blast radius of one dead worker
# --------------------------------------------------------------------------


def test_killing_one_worker_mid_round_costs_only_that_worker_s_round(
    kill_stack, tmp_path
):
    """SIGKILL one of three workers while it holds a lease.

    What must happen: its lease is released rather than held forever, the other
    two finish their own work untouched, and rounds keep closing. What must not:
    the round wedging on a lease nobody will ever reclaim, or one dead machine
    taking the fleet's progress with it.

    SIGKILL rather than terminate, on purpose. A machine that loses power never
    gets to run its abandon handler, and lease expiry is the mechanism that
    exists for exactly that case -- so testing the polite path would test the
    wrong one.

    What this deliberately does *not* assert is that the dead worker's buckets
    are re-trained soon after. They are not, and that is the design: buckets are
    marked spoken for at claim rather than at submit, precisely so two live
    workers never train the same shard, and the documented cost is "one round of
    slightly uneven coverage" when a holder dies. Least-trained-first brings
    them round eventually. Asserting otherwise here would be asserting against
    the design rather than against a bug.
    """
    fleet = Fleet(kill_stack, tmp_path).start()
    victim = None
    try:
        # Wait until all three hold a lease. Killing before the lease exists
        # would test nothing at all.
        deadline = time.time() + 300
        while time.time() < deadline:
            conn = _db(kill_stack["db"])
            try:
                leased = _rows(
                    conn, "SELECT id, worker_id FROM tasks WHERE status = 'leased'"
                )
                # Identify the process by contributor, not by registration
                # order: the three start together and register in a race, so
                # first_seen order is not the order they were spawned in --
                # and killing the wrong process would quietly test nothing.
                owners = {
                    w["id"]: w["name"] for w in _rows(conn, """
                        SELECT w.id, c.name FROM workers w
                        JOIN contributors c ON c.id = w.contributor_id""")
                }
            finally:
                conn.close()
            if len(leased) >= WORKERS:
                victim = leased[0]
                break
            time.sleep(0.5)
        assert victim is not None, "workers never all held a lease"

        victim_worker = victim["worker_id"]
        index = int(owners[victim_worker].rsplit("-", 1)[1])   # "fleet-2" -> 2
        fleet.kill(index)

        closed = _wait_for_rounds(kill_stack["db"], 2, timeout=600)
    finally:
        logs = fleet.stop()

    conn = _db(kill_stack["db"])
    try:
        violations = invariants.check(conn)
        victim_task = _rows(conn, "SELECT * FROM tasks WHERE id = ?", victim["id"])[0]
        accepted_after = _rows(conn, """
            SELECT t.worker_id, COUNT(*) AS n FROM submissions s
            JOIN tasks t ON t.id = s.task_id
            WHERE s.accepted = 1 AND t.created_at > ?
            GROUP BY t.worker_id""", victim_task["created_at"])
        still_held = _rows(conn, """
            SELECT id FROM tasks
            WHERE id != ? AND status = 'leased' AND worker_id = ?""",
            victim["id"], victim_worker)
        victim_submission = _rows(
            conn, "SELECT * FROM submissions WHERE task_id = ?", victim["id"]
        )
        rounds_after = _rows(conn, "SELECT idx, status, distinct_contributors FROM rounds")
    finally:
        conn.close()

    tail = "\n\n".join(log[-2000:] for log in logs)
    assert closed >= 2, f"the fleet did not close two rounds after the kill\n{tail}"
    assert violations == [], "\n".join(str(v) for v in violations)

    # The dead worker's lease was reclaimed rather than left outstanding, and it
    # is not holding anything else either.
    assert victim_task["status"] in ("expired", "abandoned"), victim_task
    assert still_held == [], still_held

    # Its work for that round is gone -- that is the cost, and it is bounded to
    # the one task. A process that took SIGKILL cannot have uploaded anything.
    assert victim_submission == [], victim_submission

    # And the survivors carried on. This is the whole claim: one machine dying
    # costs its own round, not the fleet's progress.
    survivors = {row["worker_id"] for row in accepted_after} - {victim_worker}
    assert len(survivors) >= WORKERS - 1, (
        f"only {len(survivors)} of {WORKERS - 1} survivors submitted after the kill:"
        f" {accepted_after}\n{tail}"
    )
    closed_rounds = [r for r in rounds_after if r["status"] == "closed"]
    assert all(r["distinct_contributors"] >= 1 for r in closed_rounds), rounds_after
