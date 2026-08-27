"""Evaluate closed rounds: held-out loss for the adapter each round published.

docs/03-roadmap.md, "Where eval runs": *v1: on the coordinator, after
aggregation.* This is that, with one deliberate departure -- it is a separate
process rather than a call inside the submit handler.

Why not inline in ``submit``
----------------------------
``closer.maybe_close`` runs on the request thread of whichever worker's
submission happened to close the round. A forward pass over ~500 held-out
samples is minutes of CPU; putting it there would hold that worker's HTTP
response open for the whole of it, and the worker is holding nothing but a
finished round it wants to move on from. Worse, the cost lands on one
arbitrary contributor rather than on the operator.

Separating it also keeps ``pyproject``'s split honest: the coordinator's own
dependencies stay small (no torch, no transformers), and this script -- which
needs all of them -- runs wherever those already live. On a single-box
deployment that is the same machine; it does not have to be.

Running it
----------
    python3 -m scripts.evalround --run-id live --once     # everything pending
    python3 -m scripts.evalround --watch                  # poll forever

Idempotent by construction: it only picks up closed rounds whose ``eval_loss``
is still NULL, so re-running it after a crash costs nothing and finishes the
work. It never writes anything a worker or the coordinator reads back -- the
column is diagnostic, so an evaluator that is down slows nobody down.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time

from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, immediate
from ganymede.coordinator.store import Store

log = logging.getLogger("ganymede.evalround")

# How long to sleep between polls in --watch mode. Rounds are tens of minutes,
# so a minute of latency on a diagnostic number costs nothing.
WATCH_POLL_SEC = 60


def pending_rounds(
    conn: sqlite3.Connection, run_id: str | None = None, limit: int | None = None
) -> list[sqlite3.Row]:
    """Closed rounds that produced an adapter and have not been evaluated.

    Ordered oldest-first so a backlog is worked through in the order that makes
    the curve readable as it fills in, rather than newest-first which would
    leave a hole in the middle.
    """
    sql = """SELECT r.run_id, r.idx, r.result_adapter_ref
             FROM rounds r
             WHERE r.status = 'closed'
               AND r.result_adapter_ref IS NOT NULL
               AND r.eval_loss IS NULL"""
    params: list = []
    if run_id:
        sql += " AND r.run_id = ?"
        params.append(run_id)
    sql += " ORDER BY r.run_id, r.idx"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


class Evaluator:
    """Holds the loaded base model across rounds.

    The base model is frozen and identical for every round of a run, so loading
    it once and copying each round's adapter into the attached LoRA turns an
    N-round backlog from N model loads into one. At 1.7B that is the difference
    between minutes and tens of minutes of pure overhead.
    """

    def __init__(self, store: Store, device: str | None = None, eval_examples: int | None = None):
        self.store = store
        self.device_pref = device
        self.eval_examples = eval_examples
        self._run_id: str | None = None
        self._model = None
        self._tokenizer = None
        self._rows: list[dict] | None = None
        self._partition = None
        self._hp: dict | None = None
        self._device = None

    def _prepare(self, run: sqlite3.Row) -> None:
        """Load everything that is per-run rather than per-round."""
        from ganymede.trainer import data as data_mod
        from ganymede.trainer import model as model_mod
        from ganymede.trainer import train as train_mod

        if self._run_id == run["id"]:
            return

        hp = {**train_mod.DEFAULT_HYPERPARAMS, **json.loads(run["hyperparams_json"])}
        rows = data_mod.resolve_dataset(run["dataset_ref"])
        partition = data_mod.plan_partition(
            n_rows=len(rows),
            num_buckets=int(run["num_buckets"]),
            eval_size=int(hp["eval_size"]),
            data_seed=int(hp["data_seed"]),
        )
        device = model_mod.pick_device(self.device_pref)
        log.info("run %s: loading %s on %s", run["id"], run["base_model"], device)
        base = model_mod.load_base(run["base_model"], run["base_precision"], device=device)
        # No init_from: every round overwrites the whole adapter anyway, and
        # attach_lora's default zero-initialised lora_B is a no-op on the base.
        model = model_mod.attach_lora(base, json.loads(run["lora_cfg_json"]))

        self._run_id = run["id"]
        self._model = model
        self._tokenizer = model_mod.load_tokenizer(run["base_model"])
        self._rows, self._partition, self._hp, self._device = rows, partition, hp, device

    def evaluate(self, conn: sqlite3.Connection, rnd: sqlite3.Row) -> float:
        from ganymede.jobtypes.collab_lora_finetune import aggregate
        from ganymede.trainer import evaluate as eval_mod
        from ganymede.trainer import model as model_mod

        run = conn.execute("SELECT * FROM runs WHERE id = ?", (rnd["run_id"],)).fetchone()
        self._prepare(run)

        adapter = aggregate.load_adapter(self.store.get_bytes(rnd["result_adapter_ref"]))
        model_mod.load_lora_state(self._model, adapter)

        hp = self._hp
        result = eval_mod.held_out_loss(
            self._model, self._tokenizer, self._rows, self._partition.eval_indices(),
            seq_len=int(hp["seq_len"]), micro_batch=int(hp["micro_batch"]),
            prompt_format=hp["prompt_format"], completion_only=bool(hp["completion_only"]),
            device=self._device, limit=self.eval_examples,
        )
        with immediate(conn):
            conn.execute(
                "UPDATE rounds SET eval_loss = ? WHERE run_id = ? AND idx = ?",
                (result.loss, rnd["run_id"], rnd["idx"]),
            )
        log.info("round %s#%s: eval_loss %.4f over %d examples",
                 rnd["run_id"], rnd["idx"], result.loss, result.examples)
        return result.loss


def run_once(
    conn: sqlite3.Connection,
    evaluator: Evaluator,
    run_id: str | None = None,
    limit: int | None = None,
) -> int:
    """Evaluate every pending round. Returns how many were evaluated."""
    done = 0
    for rnd in pending_rounds(conn, run_id, limit):
        try:
            evaluator.evaluate(conn, rnd)
            done += 1
        except Exception:
            # One unreadable adapter must not stop the backlog. The round stays
            # pending, so a later pass retries it.
            log.exception("round %s#%s could not be evaluated", rnd["run_id"], rnd["idx"])
    return done


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.evalround",
        description="Compute held-out loss for closed rounds.",
    )
    p.add_argument("--run-id", default=None, help="limit to one run; default is all runs")
    p.add_argument("--once", action="store_true",
                   help="evaluate everything pending and exit (the default)")
    p.add_argument("--watch", action="store_true", help="keep polling for new closed rounds")
    p.add_argument("--poll-sec", type=int, default=WATCH_POLL_SEC)
    p.add_argument("--limit", type=int, default=None, help="evaluate at most N rounds per pass")
    p.add_argument("--device", default=None, help="cuda, cpu, ... default is auto")
    p.add_argument("--eval-examples", type=int, default=None,
                   help="subsample the held-out split; the full split is the default")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None, settings: Settings | None = None,
         store: Store | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    settings = settings or Settings.from_env()
    store = store or Store(settings.storage)
    evaluator = Evaluator(store, device=args.device, eval_examples=args.eval_examples)

    conn = connect(settings.db_path)
    try:
        if not args.watch:
            done = run_once(conn, evaluator, args.run_id, args.limit)
            print(f"evaluated {done} round(s)")
            return 0
        while True:
            run_once(conn, evaluator, args.run_id, args.limit)
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
