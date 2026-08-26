"""Create a run and mint round 0's seed adapter (docs/02-architecture-v2.md 5.1 #2, 3.1).

Gate 2 of the acceptance gates checks a submission's key set and shapes against
"the round's expected LoRA config" -- and the manifest for that check is simply
the round's own ``base_adapter_ref`` (5.1 #2): the coordinator hands out ``A_base``
at the start of every round, so "expected" already means "the keys and shapes of
``A_base``". That means round 0 needs a base adapter to exist before any worker
can be handed a task, and nothing else in the coordinator produces one -- this
script is that producer.

The seed adapter is a `peft` LoRA initialization with **no training**, built on a
meta device so building it never requires downloading the (possibly many-GB)
base model's weights -- only its `config.json`. This is the whole point: the
coordinator never needs to hold a base model, and neither does this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from ganymede.coordinator import config as config_mod
from ganymede.coordinator import rounds
from ganymede.coordinator.aggregate import save_adapter
from ganymede.coordinator.config import Settings
from ganymede.coordinator.db import connect, immediate, init_schema
from ganymede.coordinator.store import Store, base_adapter_key
from ganymede.trainer import data as data_mod

# The default seed for round 0's Kaiming-uniform init (see build_seed_adapter).
# Fixed rather than random: a re-run of this script against the same base model
# and lora_cfg after a crash before the DB write should reproduce byte-identical
# weights, not silently mint a different round 0 each time it's invoked.
_SEED_ADAPTER_SEED = 0


def build_seed_adapter(
    base_model: str,
    lora_cfg: dict[str, Any],
    seed: int = _SEED_ADAPTER_SEED,
    adapter_dtype: torch.dtype | None = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build round 0's LoRA adapter with no training and no weight download.

    ``AutoModelForCausalLM.from_config`` under ``torch.device("meta")`` builds the
    module graph -- and therefore every LoRA-wrapped layer's name and shape --
    from ``config.json`` alone, with no tensor storage allocated for the base
    model's own weights. That is enough to know exactly what `peft` would wrap
    and with what shapes; it is not enough to serialize, since meta tensors have
    no storage to read bytes from. Each LoRA tensor is therefore rebuilt with
    real storage below, matching the meta model's names, shapes and dtypes.

    Initialization matches what `peft` itself does for a freshly-created adapter
    (see ``peft.tuners.lora.layer.LoraLayer.reset_lora_parameters``): ``lora_A``
    gets the same Kaiming-uniform init `nn.Linear` uses for its own weights, and
    ``lora_B`` is exactly zero. That B == 0 is not an implementation detail --
    it is what makes a freshly-seeded adapter a no-op on the base model at step
    0: with B == 0, ``B @ A == 0`` for every tensor, so round 0 starts training
    from the unmodified base weights, not from noise.
    """
    cfg = AutoConfig.from_pretrained(base_model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    peft_cfg = LoraConfig(
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    peft_model = get_peft_model(model, peft_cfg)
    meta_params = {n: p for n, p in peft_model.named_parameters() if "lora_" in n}

    generator = torch.Generator().manual_seed(seed)
    adapter: dict[str, torch.Tensor] = {}
    # Adapters are stored in fp32 even when the base model is loaded in bf16 or
    # nf4, and that is a deliberate choice rather than torch's default falling
    # through. `base_precision` describes the frozen base; the adapter is the
    # only thing being refined, and it is refined *iteratively* -- each round's
    # output becomes the next round's input. Storing it in bf16 would round-trip
    # the accumulated result through three decimal digits once per round, for
    # twenty-odd rounds. aggregate.combine already does its arithmetic in fp32
    # for the same reason. The cost is that an artifact is ~25 MB rather than
    # ~13 MB; the bandwidth is worth the precision on the one tensor set that
    # actually carries the training.
    # Sorted iteration makes the sequence of draws from `generator` -- and so the
    # resulting weights -- independent of dict insertion order, which `peft`
    # does not otherwise guarantee across versions.
    for name in sorted(meta_params):
        meta_t = meta_params[name]
        shape, dtype = tuple(meta_t.shape), adapter_dtype or meta_t.dtype
        if "lora_B" in name:
            adapter[name] = torch.zeros(shape, dtype=dtype)
        else:
            t = torch.empty(shape, dtype=dtype)
            torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=generator)
            adapter[name] = t
    return adapter


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# Defaults for the values a run config can supply. Kept here rather than in
# argparse so that "the flag was not given" and "the flag was given its default"
# stay distinguishable -- without that distinction an explicit --eval-size 0
# would be indistinguishable from silence and the config would win over it.
_CONFIG_DEFAULTS = {
    "lora_dropout": 0.0,
    "eval_size": 750,
    "data_seed": 0,
    "prompt_format": "dolly-v1",
}

# CLI destination -> key in the run config file.
_CONFIG_KEYS = {
    "run_id": "run_id",
    "base_model": "base_model",
    "base_precision": "base_precision",
    "dataset_ref": "dataset_ref",
    "dataset_rows": "dataset_rows",
    "num_buckets": "num_buckets",
    "target_rounds": "target_rounds",
    "target_steps": "target_steps",
    "min_round_sec": "min_round_sec",
    "max_round_sec": "max_round_sec",
    "data_classification": "classification",
    "required_image": "required_image",
}

_REQUIRED = (
    "run_id", "base_model", "base_precision", "dataset_ref", "dataset_rows",
    "num_buckets", "target_rounds", "target_steps", "min_round_sec",
    "max_round_sec", "lora_r", "lora_alpha", "target_modules",
)


def _apply_run_config(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Fill unset arguments from a run config file, in place.

    The same file feeds ``ganymede-calibrate`` and ``ganymede-baseline``. That is
    the whole point: calibrating one configuration and then creating a run from a
    slightly different one is a mistake with no symptom -- the run trains
    normally, its budgets are simply sized for a model nobody measured.

    Explicit flags win over the file, so a config can be reused with one value
    nudged without editing it.
    """
    for dest, key in _CONFIG_KEYS.items():
        if getattr(args, dest, None) is None and key in cfg:
            setattr(args, dest, cfg[key])

    lora = cfg.get("lora_cfg") or {}
    if args.lora_r is None and "rank" in lora:
        args.lora_r = int(lora["rank"])
    if args.lora_alpha is None and "alpha" in lora:
        args.lora_alpha = int(lora["alpha"])
    if args.lora_dropout is None and "dropout" in lora:
        args.lora_dropout = float(lora["dropout"])
    if args.target_modules is None and "target_modules" in lora:
        args.target_modules = ",".join(lora["target_modules"])

    hp = cfg.get("hyperparams") or {}
    for dest, key in (("eval_size", "eval_size"), ("data_seed", "data_seed"),
                      ("prompt_format", "prompt_format")):
        if getattr(args, dest) is None and key in hp:
            setattr(args, dest, hp[key])

    # The rest of hyperparams travels through untouched. samples_per_bucket is
    # deliberately *not* taken from the file: it is recomputed from
    # plan_partition below, so a stale file cannot mis-size the budgets.
    merged = {k: v for k, v in hp.items() if k != "samples_per_bucket"}
    merged.update(json.loads(args.hyperparams))
    args.hyperparams = json.dumps(merged)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.newrun",
        description="Create a new Ganymede run and mint round 0's seed adapter.",
    )
    p.add_argument(
        "--from-config", default=None,
        help="run config JSON (see configs/README.md); explicit flags override it",
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--base-precision", default=None)
    p.add_argument("--dataset", default=None, dest="dataset_ref")
    p.add_argument("--num-buckets", type=int, default=None)
    p.add_argument("--target-rounds", type=int, default=None)
    p.add_argument("--target-steps", type=int, default=None)
    p.add_argument("--min-round-sec", type=int, default=None)
    p.add_argument("--max-round-sec", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument(
        "--target-modules", default=None,
        help="comma-separated, e.g. q_proj,k_proj,v_proj,o_proj",
    )
    p.add_argument(
        "--combine-mode", choices=["mean", "diloco"],
        default=config_mod.DEFAULT_COMBINE_MODE,
    )
    p.add_argument("--lr-outer", type=float, default=config_mod.DEFAULT_LR_OUTER)
    # Left unset (None) rather than defaulted here: the actual default depends
    # on --combine-mode (config.DEFAULT_OUTER_MOMENTUM applies only to diloco),
    # and argparse can't see one flag's value while computing another's default.
    p.add_argument("--outer-beta", type=float, default=None)
    p.add_argument(
        "--classification", choices=["open", "internal", "restricted"],
        default="open", dest="data_classification",
    )
    # --- the data contract (see ganymede/trainer/data.py) -------------------
    # The coordinator never sees the dataset, so these four values are how a
    # worker reconstructs the exact same partition. They are stored in
    # hyperparams_json and passed through to every task verbatim; changing any
    # of them mid-run repartitions the data under the workers, which is why
    # they belong to run creation and not to a later edit.
    p.add_argument(
        "--dataset-rows", type=int, default=None,
        help="total rows in the dataset (ganymede-calibrate reports this as run.dataset_rows)",
    )
    p.add_argument("--eval-size", type=int, default=None,
                   help="rows held out before bucketing; never trained on")
    p.add_argument("--data-seed", type=int, default=None,
                   help="seed for the permutation that defines eval split and buckets")
    p.add_argument("--prompt-format", default=None,
                   help="named format in ganymede.trainer.data.FORMATS")
    p.add_argument(
        "--required-image", default=None,
        help="image tag a worker must be running to claim this run (4.2 step 5). "
             "Omit for no requirement -- which is what allows native installs. "
             "Setting it is also how a restricted run (6.10) is held to the "
             "container path, since a native worker has no image tag to match",
    )
    p.add_argument("--requires", default="{}", help="JSON object, budget.is_eligible shape")
    p.add_argument("--hyperparams", default="{}", help="JSON object merged into hyperparams_json")
    p.add_argument("--dry-run", action="store_true",
                    help="build the adapter and print the summary; write nothing")
    return p


def _print_summary(summary: dict[str, Any]) -> None:
    tag = " (dry run -- nothing written)" if summary["dry_run"] else ""
    print(f"run: {summary['run_id']}{tag}")
    print(f"  tensors:      {summary['tensor_count']}")
    print(f"  adapter size: {summary['adapter_mb']:.2f} MB")
    print(f"  buckets:      {summary['num_buckets']} x {summary['samples_per_bucket']} samples"
          f" ({summary['dropped_rows']} rows dropped to keep them equal)")
    print(f"  round 0 key:  {summary['round0_key']}")


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    store: Store | None = None,
) -> int:
    """CLI entrypoint. `settings`/`store` are an injection seam for tests --
    the same pattern `create_app(settings, store)` uses -- so tests never need
    real MinIO or GANYMEDE_* environment variables to exercise this script.
    """
    args = _build_arg_parser().parse_args(argv)

    try:
        json.loads(args.hyperparams)
        requires = json.loads(args.requires)
    except json.JSONDecodeError as exc:
        print(f"error: --requires/--hyperparams must be valid JSON: {exc}", file=sys.stderr)
        return 1

    if args.from_config:
        try:
            with open(args.from_config) as fh:
                _apply_run_config(args, json.load(fh))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: --from-config: {exc}", file=sys.stderr)
            return 1

    for dest, fallback in _CONFIG_DEFAULTS.items():
        if getattr(args, dest) is None:
            setattr(args, dest, fallback)
    if args.data_classification is None:
        args.data_classification = "open"

    missing = [d for d in _REQUIRED if getattr(args, d) is None]
    if missing:
        print("error: missing required settings: "
              + ", ".join("--" + d.replace("_", "-") for d in missing)
              + " (supply them as flags or in --from-config)", file=sys.stderr)
        return 1

    hyperparams = json.loads(args.hyperparams)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        print("error: --target-modules must name at least one module", file=sys.stderr)
        return 1
    lora_cfg = {
        "rank": args.lora_r, "alpha": args.lora_alpha,
        "dropout": args.lora_dropout, "target_modules": target_modules,
    }

    # Derived by calling plan_partition itself rather than by reimplementing its
    # arithmetic. This number drives every step budget the coordinator issues
    # (budget.plan_budget), and the workers get their rows from plan_partition --
    # so a second copy of the formula that drifted by one would mis-size the
    # whole run while both halves looked individually correct.
    try:
        partition = data_mod.plan_partition(
            n_rows=args.dataset_rows, num_buckets=args.num_buckets,
            eval_size=args.eval_size, data_seed=args.data_seed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    hyperparams.setdefault("samples_per_bucket", partition.samples_per_bucket)
    hyperparams.setdefault("eval_size", args.eval_size)
    hyperparams.setdefault("data_seed", args.data_seed)
    hyperparams.setdefault("prompt_format", args.prompt_format)

    outer_beta = args.outer_beta
    if outer_beta is None:
        outer_beta = (
            config_mod.DEFAULT_OUTER_MOMENTUM if args.combine_mode == "diloco" else 0.0
        )

    # --dry-run validates a run config; making that require a configured
    # coordinator (GANYMEDE_STORAGE_HOST and friends) would defeat the point,
    # since the moment to check a config is before you have one deployed.
    if args.dry_run:
        adapter = build_seed_adapter(args.base_model, lora_cfg)
        _print_summary({
            "run_id": args.run_id,
            "tensor_count": len(adapter),
            "adapter_mb": len(save_adapter(adapter)) / (1024 * 1024),
            "num_buckets": args.num_buckets,
            "samples_per_bucket": partition.samples_per_bucket,
            "dropped_rows": partition.dropped,
            "round0_key": base_adapter_key(args.run_id, 0),
            "dry_run": True,
        })
        return 0

    settings = settings or Settings.from_env()
    conn = connect(settings.db_path)
    init_schema(conn)
    try:
        existing = conn.execute(
            "SELECT id FROM runs WHERE id = ?", (args.run_id,)
        ).fetchone()
        if existing is not None:
            print(f"error: run {args.run_id!r} already exists -- refusing to clobber it",
                  file=sys.stderr)
            return 1

        print(f"building seed adapter for {args.base_model} "
              f"(r={args.lora_r}, alpha={args.lora_alpha}, targets={target_modules})...",
              file=sys.stderr)
        adapter = build_seed_adapter(args.base_model, lora_cfg)
        adapter_bytes = save_adapter(adapter)
        round0_key = base_adapter_key(args.run_id, 0)
        summary = {
            "run_id": args.run_id,
            "tensor_count": len(adapter),
            "adapter_mb": len(adapter_bytes) / (1024 * 1024),
            "num_buckets": args.num_buckets,
            "samples_per_bucket": partition.samples_per_bucket,
            "dropped_rows": partition.dropped,
            "round0_key": round0_key,
            "dry_run": False,
        }

        store = store or Store(settings.storage)

        # Ordering matters and is the load-bearing part of this script: the
        # base adapter has to be durable in object storage *before* anything in
        # the database can reference it. If a crash landed between an
        # open_round() and this put_bytes(), a run would exist whose round 0
        # base_adapter_ref 404s for every worker that tries to fetch it -- and
        # unlike a DB row, there is no transaction spanning both stores to roll
        # that back automatically. Uploading first means the only way to end up
        # with a dangling reference is losing the object after this script
        # already reported success, which is an object-storage durability
        # question, not an ordering bug here.
        store.put_bytes(round0_key, adapter_bytes)

        now = rounds._iso(rounds.utcnow())
        with immediate(conn):
            conn.execute(
                """INSERT INTO runs
                     (id, status, base_model, base_precision, lora_cfg_json, dataset_ref,
                      hyperparams_json, current_round, target_rounds, combine_mode,
                      lr_outer, outer_beta, requires_json, data_classification,
                      num_buckets, required_image, created_at)
                   VALUES (?, 'active', ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (args.run_id, args.base_model, args.base_precision, json.dumps(lora_cfg),
                 args.dataset_ref, json.dumps(hyperparams), args.target_rounds,
                 args.combine_mode, args.lr_outer, outer_beta, json.dumps(requires),
                 args.data_classification, args.num_buckets, args.required_image, now),
            )
            for b in range(args.num_buckets):
                conn.execute(
                    "INSERT INTO buckets (run_id, bucket_idx, times_trained) VALUES (?, ?, 0)",
                    (args.run_id, b),
                )

        # open_round() is its own `immediate` transaction (rounds.py), run after
        # the block above commits rather than nested inside it -- SQLite doesn't
        # support nested BEGIN IMMEDIATE, and the runs/buckets rows are already
        # safe to have committed on their own: an active run with no open round
        # is a recoverable inconsistency an operator can repair, which a run
        # whose open round points at a missing object is not.
        rounds.open_round(
            conn, args.run_id, 0, round0_key,
            args.target_steps, args.min_round_sec, args.max_round_sec,
        )

        _print_summary(summary)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
