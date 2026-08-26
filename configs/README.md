# Run configs

One file per run. The same file is the input to all three tools, which is the
point of it existing at all:

```
ganymede-calibrate --run-config configs/bringup-1.7b.json --out calibration.json
ganymede-baseline  --run-config configs/bringup-1.7b.json --out baseline.json \
                   --smoke-out smoke-round0.json --seeds 1,2,3
python3 -m scripts.newrun --from-config configs/bringup-1.7b.json
```

Calibrating one configuration and then creating a run from a slightly different
one is a mistake with no symptom: the run trains, the budgets are simply sized
for a model that was never measured. Passing the same file to all three removes
the opportunity.

## The fields, and which ones you cannot change mid-run

| Field | Changing it mid-run | Why |
|---|---|---|
| `base_model`, `base_precision`, `lora_cfg` | **New run** | The adapter is a different shape; acceptance gate 2 rejects every submission |
| `dataset_ref`, `dataset_rows`, `num_buckets`, `hyperparams.eval_size`, `hyperparams.data_seed` | **New run** | Together these define the partition. Change one and every worker re-derives different rows for the same bucket index — including rows held out for eval |
| `hyperparams.prompt_format` | **New run** | The baseline was measured under the old format; the comparison stops meaning anything |
| `hyperparams.micro_batch`, `grad_accum`, `seq_len` | New run, in practice | They set `samples_per_step`, which the step budget is arithmetic over |
| `hyperparams.lr`, `weight_decay`, `max_grad_norm` | Safe | Per-round, and the inner optimizer is fresh every round anyway |
| `target_rounds`, `min_round_sec`, `max_round_sec` | Safe | Scheduling, not training |

`samples_per_bucket` is **derived**, not chosen: `(dataset_rows - eval_size) //
num_buckets`. It is written into the config so a reader can see it, and
`scripts/newrun.py` recomputes it from `plan_partition` rather than trusting the
file. If the two disagree, the file is stale.

## `bringup-1.7b.json`

The bring-up run (docs/03-roadmap.md). `Qwen3-1.7B-Base` in bf16 is ~3.4 GB of
weights, so it fits a 12 GB 3060 and a 16 GB Mac with room — which means no nf4
path is needed and **every contributor is eligible**, including the Macs. That
is deliberate: it exercises the heterogeneity paths early, where the interesting
bugs are.

`seq_len` is 1024 rather than the 2048 in the architecture's task-spec example.
Dolly's examples are short; the 8k-token ceiling that example was written for
belongs to a different corpus, and paying for padding to 2048 on a 15k-sample
instruction set buys nothing.

## `cpu-probe-0.6b.json`

Not a real run. `Qwen3-0.6B-Base` is small enough to train on CPU, which makes
it the right tool for protocol testing (M4a) on machines with no usable GPU, and
for anything where you want to exercise the full claim/train/submit path without
waiting on a card. Same architecture, so nothing about the code path differs.

Do not calibrate against it and use the numbers. A throughput figure from CPU
describes the CPU.
