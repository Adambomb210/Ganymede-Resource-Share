# Ganymede — Plan of Attack

Milestones for `llm_finetune` v1, per `02-architecture-v2.md`. This follows the
original spec's Appendix ordering, which was right — infra last, working training
first — with the milestones made concrete and given exit criteria.

Day estimates are focused-work days for one developer, and they assume the open
questions in §7 are answered. They are for sequencing, not for promising dates.

---

## M0 — Trainer + calibration harness

**~2–3 days, of which the trainer is most of it and is unskippable regardless.**

Because Ganymede will run **many** models and fine-tunes rather than one, this
milestone is not a one-time gate you pass and forget. Nearly everything in it is
per-run work that recurs on every new base model, dataset, or hardware mix. So it gets
built as **a tool you run per run**, not a script you run once.

That reframing is what makes it cheap. The alternative — measuring throughput and
baselines by hand each time — is the thing that doesn't scale to a lot of models.

### What's in it

**1. The trainer.** Not skippable under any plan: the worker has to call *something*,
and §4.3's `run_task` is that something. If you already have a working fine-tuning
setup, this is a **port into the `run_task` shape**, not a discovery exercise — pull
the loop in, make it stop cleanly at N steps, and emit safetensors.

**2. The calibration harness** — the part that pays for itself once there are many
runs. One command against a run config, on one GPU:

```
ganymede-calibrate --run-config run.json --gpu-class 4090
  → fits: {bf16: false, nf4: true}   max_seq_len at each
  → throughput: 4.1 steps/min, 6.2k tok/s
  → recommended local_steps: 143  (for a 35-min round)
  → baseline: eval_loss 1.84 after 2000 steps
```

Its output is a `calibration.json` the coordinator stores alongside the run. It feeds
three things directly:

- **Round sizing** (§3.3) — per-worker step budgets, so a 3090 and a 4090 in the same
  round both finish near the deadline instead of the 3090 straggling every time
- **Capability filtering** (§6.2) — which GPUs are eligible for this run at all
- **The M4 comparison** — the single-node number the distributed run must beat

**3. The baseline — multi-seed.** A held-out eval split carved out before bucketing,
and single-node results across **2–3 seeds**, so `baseline.json` carries a mean and a
spread rather than one number. The spread is what makes M4's "matches the baseline"
a measurement instead of a judgement call. See *Eval metric* below.

### Exit criteria

- Windows environment set up per *Developing on Windows* below — long paths, `HF_HOME`,
  developer mode — before anything else, since all three bite during the first model
  download
- Trainer runs, loss descends, adapter round-trips through safetensors
- `ganymede-calibrate` produces a valid `calibration.json` for the first run config
- The run fits the target card at the chosen `base_precision`, with headroom
- `baseline.json` exists with **multiple seeds**, giving a mean and a variance band
- The ~20-prompt greedy generation smoke set exists and its round-0 output is recorded

### Why the baseline can't be dropped

This is the one part with no substitute, and *many models* is what makes it
non-negotiable rather than optional.

A distributed run that is quietly **worse than one GPU** looks exactly like one that's
working. The loss curve descends in both cases. Nothing errors. You find out months
later, or never. The §5.2 research risk — DiLoCo's outer step is characterized over
full-parameter training, not LoRA adapters — means this is a live possibility, not a
hypothetical.

One un-baselined run is a gamble you might get away with. Twenty of them is a
platform whose output nobody can trust and nobody can debug, because there is no
reference point to debug against. The habit is cheap; acquiring it after the fact
means re-running everything.

---

## M0 status — built, minus what needs a GPU

**Done, with one hard boundary.** Everything that can be written and mechanically
validated without a GPU is written and validated. What is left is not code — it is
*measurement*, and measurement needs the card.

```
ganymede/trainer/
  data.py       dataset resolution, bucketing, prompt format, loss masking
  model.py      base loading, LoRA attach, adapter serialization
  train.py      run_task and the single optimizer loop
  evaluate.py   held-out loss, the 20-prompt greedy smoke set
  calibrate.py  ganymede-calibrate -> calibration.json
  baseline.py   ganymede-baseline  -> baseline.json
configs/
  bringup-1.7b.json     the run: Qwen3-1.7B-Base, bf16, Dolly 15k, 64 buckets
  cpu-probe-0.6b.json   Qwen3-0.6B-Base, for protocol testing without a GPU
```

| Exit criterion | State |
|---|---|
| Trainer runs, loss descends, adapter round-trips through safetensors | met on CPU — held-out loss falls on `Qwen3-0.6B-Base` over real Dolly, and the artifact passes the coordinator's own `check_structural` |
| `ganymede-calibrate` produces a valid `calibration.json` | harness met and run end to end — a complete `calibration.json` with fit, throughput and a recommended `local_steps`; **the numbers need a GPU** |
| The run fits the target card at the chosen `base_precision`, with headroom | **needs the card** — the fit probe is written and walks the ladder, but "does 1.7B bf16 fit a 12 GB 3060" is a question only the 3060 answers |
| `baseline.json` exists with multiple seeds, mean and variance band | harness met and run end to end over two seeds — identical start, divergent paths, a band, and a printed M4 pass threshold; **the real run needs a GPU** (~3.5 h/seed on CPU, and a CPU baseline is the wrong comparison anyway) |
| ~20-prompt greedy smoke set exists, round-0 output recorded | **met** — `configs/smoke-round0-qwen3-1.7b.json`, generated on the real bring-up model. Round 0 is base + seed adapter, and the seed adapter is a no-op, so this is a hardware-independent-in-principle reference (re-record on the machine you compare against; greedy decoding is deterministic but not bit-portable) |
| Windows environment set up | **your machine** — long paths, `HF_HOME`, developer mode |

Verified here rather than assumed: `Qwen/Qwen3-0.6B-Base` and
`databricks/databricks-dolly-15k` (15,011 rows, CC BY-SA 3.0) are both current and
fetchable; the bring-up config derives 64 × 222 samples with 53 rows dropped, and
mints a 224-tensor, 24.53 MB adapter with **no weight download**, matching the sizes
this document already claimed.

### Three contract gaps, found by building M0 against M1's real output

Each was invisible until the trainer was written against what the coordinator
actually emits, and each would have failed quietly rather than loudly:

1. **`num_buckets` and `seed` were missing from the claim payload.** The worker maps
   bucket indices to rows itself — the coordinator never sees the data — so indices
   without a total are not an assignment, and §4.3's loop referenced a `task.seed`
   that was never sent. The M0 trainer could not have been written against M1's
   output as it stood.
2. **`claim_task` read `batch_size` for the budget; the task spec and the trainer
   both say `micro_batch`.** A run config written for the trainer would have had the
   coordinator sizing budgets for one effective batch while the worker trained with
   another — wrong by exactly the ratio between them, in a number nothing
   cross-checks.
3. **`samples_per_bucket` defaulted to a hardcoded 234.** It is a fact about the
   dataset. `newrun.py` now derives it by calling `plan_partition` itself rather than
   reimplementing the formula, and a claim without it fails loudly instead of
   silently mis-sizing every budget in the run.

### The one design decision M0 had to settle

**How a bucket index becomes rows** — now §4.7. Permute with `random.Random(data_seed)`,
carve the eval split off the front, slice the rest into exactly-equal buckets. The
alternative (`hash(row) % num_buckets`) gives approximate bucket sizes, and every step
budget in the system is arithmetic over that size.

### What genuinely needs your Windows + NVIDIA box

| Blocked | Why it cannot be faked here |
|---|---|
| Real throughput numbers | A steps/min figure from CPU describes the CPU. Nothing in the system should read one |
| The multi-seed baseline | M4 compares single-**GPU** against distributed-GPU; a CPU baseline answers a different question |
| VRAM fit and `max_seq_len` | The probe is written; the answer is a property of the card |
| `nf4` / `bitsandbytes` | CUDA-only. The trainer refuses the precision with a clear error rather than silently training at another one |

The split is clean: this box supplies the harness and the proof that it runs; your
machine supplies the numbers. Nothing in M1 or M2 waits on those numbers — the
coordinator's cold-start default exists precisely so a run can start uncalibrated
and converge on measured throughput within a round or two.


---

## Bring-up model and dataset

**Separate the model you're proving the system with from the model you actually want
to train.** Using your real target data for M0–M4 conflates two failure modes that
look identical from the outside: "the infrastructure is broken" and "this fine-tune
doesn't work." You want to eliminate the first before investigating the second.

### Model: start much smaller than the scale target

**Decided: `Qwen/Qwen3-1.7B-Base`, dense, text-only, bf16.** Scale to
`Qwen3-8B-Base` after M4 passes, as a run-config change.

**This reverses an earlier pick of `Qwen3.5-2B-Base`, and the reason matters.** That
choice was made by checking the HuggingFace *listing* — generation, parameter count,
licence, download counts — which is enough to tell you a model is current and is not
an obvious MoE. It is not enough to tell you what the architecture actually is. Loading
`config.json` says otherwise on two counts:

- **The whole Qwen3.5 dense line is multimodal**, `-Base` included. Every size — 0.8B,
  2B, 4B, 9B — reports `architectures: ["Qwen3_5ForConditionalGeneration"]` with a
  24-layer vision tower in `vision_config`. `AutoModelForCausalLM` is the wrong class
  for it; the text stack has to be addressed explicitly. The vision tower is weight,
  memory, and LoRA-target surface that a text fine-tune has no use for.
- **Attention is hybrid, 3:1.** `layer_types` alternates `linear_attention` and
  `full_attention` — 18/6 at 2B, 24/8 at 4B and 9B. Only the `full_attention` layers
  carry `self_attn.{q,k,v,o}_proj`. The `linear_attention` layers carry a different
  module set entirely: `linear_attn.{in_proj_qkv, in_proj_a, in_proj_b, in_proj_z,
  out_proj}` plus a `conv1d`.

The second point is a live trap rather than a curiosity. **The standard LoRA recipe —
`target_modules=["q_proj","k_proj","v_proj","o_proj"]` — attaches to 6 of 24 layers on
Qwen3.5-2B and reports no error.** Training runs. Loss descends. The adapter is a
quarter of the model it looks like. That failure would surface at M4 as "the
distributed run underperforms the baseline," and the investigation would go straight to
DiLoCo's outer step — §5.2's known research risk, and an entirely innocent one here.
Bring-up exists to eliminate exactly this class of confusion, so the bring-up model
should not be the one introducing it.

`Qwen3-1.7B-Base` has neither problem: `Qwen3ForCausalLM`, no vision tower, and all 28
layers `full_attention`. One module-name set, uniform coverage, `AutoModelForCausalLM`
loads it directly.

Verified here rather than assumed:

- **The adapter manifest is 224 tensors, 6.42 M parameters** at `r=16` over
  `{q,k,v,o}_proj` — 28 layers × 4 modules × 2 (A and B). Generated from
  `config.json` on a meta device, with no weight download, which is also how the
  coordinator can hold an expected-key manifest without ever holding a base model
  (see *Blocking check for M1* below).
- **The stored artifact is ~25 MB, not the ~13 MB the parameter count implies.**
  Adapters are kept in **fp32 even when the base model is bf16 or nf4**, and that is
  deliberate. `base_precision` describes the frozen base; the adapter is the only
  thing being refined, and it is refined *iteratively* — each round's output is the
  next round's input. Storing it in bf16 would round-trip the accumulated result
  through three decimal digits once per round, for twenty-odd rounds. Budget the
  bandwidth at 25 MB per worker per round in each direction.
- **`transformers` must know the architecture.** `qwen3` is long-settled, so this is
  no longer the first-hour hazard it was for `qwen3_5` — but pin the version and record
  the pin regardless.
- **`-Base`, not the instruct variant, and that's deliberate**: a base model has no
  chat template, so you define the format yourself and the hybrid thinking/non-thinking
  template hazard below doesn't apply during bring-up. One less way for the baseline to
  be silently wrong. The hazard returns when you move to an instruct model — which is
  why the check stays documented.
- **`Qwen3-0.6B-Base`** is the second option worth knowing about: small enough to train
  on CPU, which makes it useful for protocol testing (M4a) on machines with no usable
  GPU at all.

**Qwen3.5 is not ruled out — it is deferred until the platform is proven.** It is the
newer line and the better model, and both of its complications are tractable once
there's a working baseline to compare against. Adopting it needs three things, none of
which belong in bring-up: scoping the LoRA to the text stack so the vision tower is
untouched; a target-module set that covers **both** attention types (or a deliberate
decision to use `mlp.{gate,up,down}_proj`, which is present and identically shaped in
every layer of both types and so sidesteps the hybrid problem entirely); and a
re-measured baseline, because the adapter is a different shape. Revisit at the scale
step, not before.

Two constraints that apply at **every** size, not just bring-up:

**Dense, not MoE — for now.** Recent Qwen generations include MoE variants, and
training the *experts* interacts badly with averaging: which experts a worker trains
depends on how its own shard routes, so sparsely-activated experts get diluted by a
uniform mean. Router state adds a second divergence path.

This is deferred, **not foreclosed** — §5.4 records exactly what MoE would need, and
the one design accommodation for it (per-tensor aggregation weights) is already in
§5.2. Worth knowing now: **an MoE base with attention-only LoRA works on the current
design unchanged**, since frozen experts and a frozen router reduce it to the dense
case. Re-check at the scale step, where MoE options get more tempting.

**Verify the chat template before generating any SFT data** — this applies the moment
you use an *instruct* model, and is sidestepped during bring-up only because the pick
above is a base model. Recent Qwen instruct models use a hybrid thinking/non-thinking
template. Fine-tuning against the wrong template degrades
behavior in ways that **do not show up in training loss** — you see it only in
generation, which means it would silently corrupt M0's baseline and everything
compared against it. Confirm the template round-trips before the baseline run. This is
also what the 20-prompt greedy smoke set (below) is there to catch.

This is worth more than it sounds:

- **No `nf4` needed.** A 1.7B model in bf16 is ~3.4 GB of weights — it fits a 12 GB
  3060 and a 16 GB Mac comfortably. That removes the entire nf4/MPS exclusion problem
  (§6.8 Tier 3) during bring-up, so **every contributor is eligible for the bring-up
  run**, including the Macs.
- **You exercise the heterogeneity paths early**, which is where the interesting bugs
  are. On an 8B nf4 run the Macs sit idle and you'd find their bugs months later.
- **Rounds are minutes, not hours.** You can iterate the whole pipeline several times
  a day instead of once. Use ~10-minute rounds for bring-up rather than §3.4's 15–20.
- **Adapters are ~25 MB** (measured, not estimated), so the storage and bandwidth plumbing gets tested
  without waiting on transfers.

Then scale: same code, new run, `base_model` and `base_precision` changed in the run
config. That's the whole point of config-over-code, and scaling the model is a good
first proof that the property actually holds.

### Dataset: Dolly 15k

**Decided: Databricks Dolly 15k, `open` class (§6.10).** Human-written, permissively
licensed, and small enough that a full run takes minutes — which is what you want
while the thing under test is the plumbing rather than the model. You want a dataset
with a known-good outcome, so that a bad result means your pipeline is wrong rather
than your data. Verify the current licence on the dataset card before use; these
change.

**Buckets: 64** — exactly 222 samples each once the 750-row eval split is carved
out first, with 53 rows dropped to keep every bucket equal (§4.7). Per the sizing
rule below.

A larger corpus will be wanted eventually, but that's an M4 question and not worth
settling now. One caveat to carry forward when you get there: at 15k samples a round
of any real size covers a meaningful fraction of the set, so workers repeat data
across rounds sooner than they would on a larger corpus. That's fine for proving the
*machinery* — Dolly measures whether aggregation works, not how far the model can
get. If M4's convergence result looks suspiciously flat, **dataset exhaustion is the
first thing to check**, and moving to a larger corpus is the test.

### Eval metric

The M4 question is **"did distributed training preserve the training signal?"**, not
"is the model good." Those need different instruments, and reaching for a benchmark
suite here measures the wrong thing.

#### Primary: held-out loss on a fixed Dolly split

Hold out **500–1000 samples before bucketing** — the split must be carved out first,
or eval samples leak into training buckets and the number becomes meaningless.

Held-out loss is right here because it's cheap enough to run every round,
deterministic, and *sensitive to small degradation* — which is exactly the failure
mode to catch, since a distributed run that is quietly slightly worse than single-node
looks identical to a healthy one on a training curve.

**What not to use, and why:** MMLU, GSM8K, IFEval and friends are high-variance at
this scale, expensive, and insensitive to the deltas that matter. A 1.7B model
fine-tuned on 15k general instruction samples will barely move them, so you'd be
reading noise. They're the right tool for a real run's quality much later; they are
the wrong tool for validating infrastructure.

#### The comparison is two curves, answering two different questions

This is the part that's easy to get wrong. Plot held-out loss against two different
x-axes:

| Axis | Question | What failure looks like |
|---|---|---|
| **Cumulative steps** summed across all workers | Does aggregation preserve the training signal? | Distributed curve sits **above** single-node at equal total compute → averaging is lossy |
| **Wall-clock** | Is this worth doing at all? | Distributed reaches a given loss no faster → overhead ate the parallelism |

**Both matter, and a system can pass one while failing the other.** Aggregation can be
perfectly sound while round-trip overhead eats the entire speedup; or it can be faster
in wall-clock while producing a measurably worse model, which is a bad trade you would
not notice with only one curve. The first axis is the correctness test; the second is
the value test.

#### Define "matches" by measuring your noise floor first

"Matches or beats the baseline" needs a tolerance, and you can't pick one honestly
without knowing run-to-run variance.

**Run the single-node baseline 2–3 times with different seeds during M0** and record
the spread. That spread *is* your tolerance:

- Distributed lands **inside** the seed-variance band at equal total steps → aggregation
  is working
- Consistently **above** the band → aggregation is lossy; investigate before scaling

At 1.7B on Dolly this costs perhaps an hour of compute and it converts M4's exit
criterion from a judgement call into a measurement. `baseline.json` should carry all
seeds plus the mean and spread, not a single number.

#### Diagnostics worth logging from the start

Cheap to compute at aggregation time, and each one localizes a different failure:

- **Per-round loss delta** — `loss(round N+1 base) − loss(round N base)`. Negative most
  rounds. A positive round is a signal: a bad contribution, or `lr_outer` too high.
- **Inter-worker adapter divergence** — pairwise distance between submitted adapters
  before averaging. This directly measures the thing DiLoCo trades away: more local
  steps means more drift means a less meaningful average. **If drift grows across
  rounds, `local_steps` is too high.** Probably the single most actionable number in
  the system.
- **Gate rejection rate per contributor** (§5.1) — already recorded; watch it.

#### One qualitative check loss won't give you

Held-out loss can look healthy while generation quality degrades — chat-template
corruption is the specific risk already flagged for Qwen, and it barely moves loss.

Keep **~20 fixed prompts, generated greedily (temperature 0) at each checkpoint**, and
diff the outputs against the previous round. Not a metric; a smoke test. It catches
template breakage and mode collapse, which are exactly the failures that a loss number
happily reports as fine.

#### Where eval runs

**v1: on the coordinator, after aggregation.** At 1.7B, a forward-only pass over ~500
held-out samples on CPU is a few minutes — acceptable inside a 15–20 minute round, and
it keeps eval out of the worker protocol entirely.

This stops being comfortable at 8B. The natural extension is to make eval **a task
type** dispatched like any other, which reuses the whole claim/submit mechanism — and
has a pleasant side effect worth noting: a machine below the *training* throughput
floor (§3.5) may be perfectly capable of running evals. That gives the Macs and
low-end cards a real job on runs they can't otherwise join. Worth doing when the
coordinator-side path starts to hurt, not before.

### Bucket count should scale with dataset size

§6.10 and the task spec assume ~1000 buckets. That was a placeholder, and it's wrong
for small datasets: 15k samples across 1000 buckets is 15 samples each, so a worker's
"shard" is statistical noise.

**Target ~100–500 samples per bucket**, so bucket count is roughly
`dataset_size / 200`, clamped to a sane range:

| Dataset size | Buckets | Samples/bucket |
|---|---|---|
| 15k | 64 | 222 (measured) |
| 200k | 1000 | 200 |
| 940k | 2048 | ~460 |

Bucket count is fixed **per run** at prep time and never changes mid-run — worker
count varies freely against it, which is the property that matters (§6.10).

---

## Developing on Windows

Primary development is Windows + NVIDIA, which is the least-trodden of the three
supported platforms. Nothing here is a blocker, but each item costs an afternoon if
you meet it by surprise instead of on purpose.

**Good news first:** Docker Desktop's WSL2 backend *can* reach the GPU, unlike macOS.
So the container path is testable locally and **M2 needs no rented hardware** — only
M4b does.

| Issue | What to do |
|---|---|
| `MAX_PATH` 260-char limit vs. HF cache paths | Enable Win32 long paths, and set `HF_HOME` to something short like `C:\hf` |
| HF cache uses symlinks; Windows needs developer mode or admin | Enable developer mode, or accept that the cache silently doubles in size from copies |
| `multiprocessing` uses spawn, not fork | Guard entry points with `if __name__ == "__main__"`. The trainer sidesteps this entirely — it uses a plain generator rather than a `DataLoader`, so there are no worker processes to spawn |
| No cgroups for resource limits | Job Objects, or accept no limits on native installs. Container path is unaffected |
| `bitsandbytes` is least reliable here | Irrelevant during bring-up — bf16 at 1.7B needs no quantization. Re-check before any `nf4` run |
| Git line endings can corrupt scripts inside containers | `.gitattributes` pinning shell scripts to LF |
| Signals don't work like Unix | Already handled — the stop path is a sentinel file, not `SIGTERM` (§4.4) |

**One risk worth naming.** Developing on Windows means the Linux container path — the
primary deployment target for any third-party host — gets less day-to-day exercise.
Linux-specific bugs will otherwise surface late, at exactly the moment you're asking
someone else to install something. Mitigation: **CI builds and smoke-tests the
container on Linux from M2 onward**, so the path you use least is still exercised on
every commit.

---

## Blocking check for M1

Run against a real Linux box (Aug 2026) rather than reasoned about. **Nothing blocks
building or testing M1.** What follows is what was actually verified, what still needs
a value, and what is genuinely deferred to deploy time.

### Verified working

| Check | Result |
|---|---|
| `fastapi`, `uvicorn`, `boto3`, `httpx`, `pytest` from PyPI | Install clean |
| `torch` (CPU), `transformers`, `peft`, `safetensors` | Install clean; enough to build adapters and exercise the gates without a GPU |
| MinIO container | Pulls and runs |
| Presigned `PUT` then `GET`, path-style, **signed against a non-localhost hostname** | **200 / 200, bytes round-trip** |
| LoRA key manifest from `config.json`, meta device, no weights | 224 tensors, 25 MB fp32 |

The presigned-URL result is the one worth calling out. §6.6's footgun is that MinIO
signs against whatever `MINIO_SERVER_URL` says, so a coordinator that signs
`localhost` mints URLs no worker can use. Reproducing that *class* of bug needs a
hostname that isn't `localhost` — not a real domain. Point one at the loopback in
`/etc/hosts`, set `MINIO_SERVER_URL` to it, sign with it, and fetch as a separate
client. That test belongs in M1's suite, and it works with no DNS and no certificate.

### Two environment notes for CI

- **Start `dockerd` explicitly.** `docker --version` reports the client and says
  nothing about the daemon; on a fresh container there is no socket until something
  starts one. Worth an explicit check in CI rather than a confusing first failure.
- **Docker Hub rate-limits unauthenticated pulls** — `429 Too Many Requests` from a
  shared cloud IP, which is what CI runs on. `quay.io/minio/minio` pulled without
  complaint. Prefer non-Hub registries, or authenticate, and pin by digest either way.

### Three values M1 must choose (none blocking)

1. **Where gate 2's expected key set comes from.** §5.1 says a submission's keys and
   shapes must match "the round's expected LoRA config" without saying who holds that.
   **The round's `base_adapter_ref` is the manifest** — the coordinator hands out
   `A_base` every round, so the expected key set is simply `A_base`'s. Self-referential,
   no dependency on M0, and it makes gate 2 a set comparison against a file the
   coordinator already has. The only new requirement is that **something must produce
   round 0's seed adapter**: a `peft` init, no training, verified above at 224 tensors.
   Do it as an admin script (`scripts/newrun.py`) and M1's dependency on M0 disappears
   entirely.
2. **The cold-start throughput constant.** §3.5's third tier is a "conservative
   default" for the first worker of an unseen GPU class on an uncalibrated run, with no
   number attached. It self-corrects after one round, so pick something deliberately
   low, name it as a constant, and let the measurement fix it.
3. **`lr_outer`'s default.** §5.2 sets `β = 0.9` but leaves `lr_outer` unset except by
   implication — the conservative arm of the A/B is `lr_outer = 1, β = 0`, the plain
   weighted mean. Both modes ship (M1 deliverables), so this is choosing which one a
   fresh run gets by default. Default to the conservative arm until M4 says otherwise.

### Deferred to deploy, not to build

- **Real hostnames and a TLS certificate.** §6.5 keeps every hostname in an environment
  variable, so the code is unaffected; the fake-worker suite runs over plain HTTP on
  loopback. Needed the day M1 leaves the dev box.
- **An off-box backup destination.** M1's exit criteria require backup to land off-box,
  which by definition can't be tested against a destination that doesn't exist yet.
  Build `scripts/backup.py` against a second local MinIO — the code path is identical
  and only the endpoint changes — and re-run the check against the real destination at
  deploy.

---

## M1 — Coordinator

**~5–6 days** (was 4–5; self-hosted storage adds MinIO standup, TLS, GC, and backup).

FastAPI + SQLite: schema (§6.1), API (§6.2), auth (§6.3), round state machine (§3.1),
lease manager, aggregator with acceptance gates (§5), and self-hosted MinIO with
presigned URL minting (§6.6).

Storage is self-hosted on the coordinator VM, on a **separate volume from the OS
disk**, with S3 config kept in environment variables so the later R2 swap is a config
change rather than a code change.

Deliverables:
- `ganymede/coordinator/` — app, models, rounds, leases, aggregate, auth
- `ganymede/coordinator/aggregate.py` with **both** combine modes (plain weighted mean
  and DiLoCo outer momentum), selectable per run — M4 needs to A/B them
- `ganymede/coordinator/store.py` — thin S3 wrapper, `boto3` with `endpoint_url` from
  env. Restricted to the portable API subset in §6.6
- `ganymede/coordinator/budget.py` — per-worker step budgets from observed and
  calibrated throughput (§3.4), plus cache-affinity run selection (§6.7)
- `deploy/` — MinIO service unit, reverse proxy + TLS config, separate volume mount
- `scripts/backup.py` — off-box SQLite dump + latest-adapter copy, on round close
- `scripts/gc.py` — drop worker submissions older than 3 closed rounds
- `scripts/newrun.py`, `scripts/issue-key.py` — admin CLI
- Test suite with a **fake worker**: concurrent claims, lease expiry, late submission,
  gate rejection, work-target-vs-backstop close, zero-submission reopen

Exit criteria:
- N fake workers concurrently claim, train (with stub adapters), and submit; rounds
  advance correctly
- No double-leasing under concurrent claim (the `BEGIN IMMEDIATE` path is explicitly
  tested)
- A NaN adapter, a wrong-shape adapter, and a pickle file are each rejected with the
  right reason
- Killing the coordinator mid-round loses nothing already submitted
- A round closes on accumulated steps with 1, 2, and 8 fake workers — same run config,
  no quorum retuning (§3.2). With zero workers it reopens quietly and does not alert
- A worker claiming with 3 minutes left in a round gets `204`, not unfinishable work
- Two fake workers with 3× different throughput both finish a round near the deadline,
  receive bucket counts scaled to their budgets, and are weighted by actual steps (§3.4)
- A run requiring `nf4` is never offered to a worker whose profile lacks it, and a
  worker below the throughput floor is not leased work for that run (§6.8)
- **A presigned URL minted by the coordinator is fetchable by an external client**
  over the public hostname. This is the §6.6 signing footgun; catch it here, with one
  integration test, rather than in M2 against a real GPU
- Backup lands **off-box** and GC actually deletes. An untested backup script is
  indistinguishable from no backup script

Do this before any Docker work. Testing the state machine against fake workers is
vastly faster than testing it against real GPUs.

---

## M1 status — built

**Done.** Now 234 tests with M0 alongside it, nothing skipped, including the real-MinIO storage tests.

| Exit criterion | State |
|---|---|
| N workers concurrently claim, train, submit; rounds advance | met |
| No double-leasing under concurrent claim (`BEGIN IMMEDIATE` tested directly) | met |
| NaN, wrong-shape and pickle adapters each rejected with the right reason | met — plus Inf, missing key, extra key, wrong dtype |
| Killing the coordinator mid-round loses nothing submitted | met |
| Round closes with 1, 2 and 8 workers, one unchanged config | met |
| Zero workers reopens quietly, no alert | met |
| Worker with ~3 minutes left gets `204` | met |
| 3× throughput → budgets and bucket counts both scale; weight follows steps | met |
| `nf4` requirement and the throughput floor gate eligibility | met |
| Presigned URL fetchable by an external client over the public hostname | met — verified live, `urllib` only, no boto3 |
| GC deletes; backup lands off-box | GC met. Backup is tested against a second store and refuses a destination equal to its source; pointing it at a **real** off-box destination is deploy work |

Also verified live rather than only in tests: the `uvicorn` factory boots against
real MinIO, `issue_key` and `newrun` produce a real run from `Qwen3-1.7B-Base` in
about six seconds with **no weight download**, and a worker speaking only `urllib`
runs a full round through presigned URLs. Measured throughput then replaces the
cold-start default in the next round's budget — 108 steps to 1172, with bucket
counts scaling alongside so the faster worker gets more *data*, not more epochs.

### Three bugs the integration suite found

Worth recording, because each was invisible to the module-level tests and two were
in code that looked obviously correct:

1. **The round-close race.** `close_round` checked "is this round open" with a plain
   `SELECT` outside any transaction. Two submissions landing together both passed
   the check, both ran the whole aggregate-and-advance path, and the loser died on
   a `rounds` UNIQUE violation — a 500 to a worker that did nothing wrong. Closing
   is now claimed with a conditional `UPDATE`; the loser gets a clean no-op. The
   claim path had this right from the start, which is exactly why the omission
   survived review.
2. **Two inert tunables.** `GANYMEDE_NORM_REJECT_K` and `GANYMEDE_DOMINANCE_CAP`
   were read from the environment, stored on `Settings`, and documented — and never
   passed to the gates, which hardcoded the same values. The deployment knob did
   nothing. A tunable nobody reads is worse than no tunable, because the operator
   believes it works.
3. **A cache keyed too loosely.** The expected-key manifest was cached by adapter
   ref alone, so the same key path against different storage returned a manifest
   for bytes it had never read.

### Still deploy-time, not build-time

Unchanged from the pre-M1 check: real hostnames and a TLS certificate, and a real
off-box backup destination. Neither blocks M2.

---

## M2 — Worker package + container

**~4–5 days** (was 3–4; the native path and the probe are new scope).

The worker **package** first, then the image around it (§4.1). Entrypoint loop (§4.2),
SIGTERM/abandon (§4.4), capability probe (§6.9), isolation flags (§4.6). The trainer is
M0's, unchanged.

Broad hardware compatibility is a goal (§0), so this milestone delivers `pip install
ganymede-worker` working standalone, with the container as a wrapper — not the other
way round.

Deliverables:
- `ganymede/worker/` — loop, client, signals; installable as `ganymede-worker`
- `ganymede/worker/probe.py` — the §6.9 self-test: allocation ceiling, precision
  support, benchmark score. Backend-dispatched (cuda / mps / rocm / cpu)
- `docker/torch-base.Dockerfile`, `worker-core.Dockerfile`, `worker-llm.Dockerfile`
- CI that builds and pushes all three on **Linux**, pinning `torch-base` by digest,
  and runs a container smoke test — the primary deployment path must not depend on
  someone remembering to test it manually (see *Developing on Windows*)

Exit criteria:
- One real GPU claims from a real coordinator, trains, submits, loops
- **The same package runs natively on Linux/CUDA, macOS/MPS, and Windows/CUDA**,
  registering with a correct probe on each. The Mac need not be fast; it needs to be a
  real participant. Windows is where `nf4` availability is least predictable — the
  §6.9 probe is what keeps that from becoming a support burden
- The probe correctly reports a machine as ineligible (too little memory, no nf4)
  without failing registration (§6.9)
- Read-only rootfs works with the declared writable mounts (expect to discover one or
  two more cache paths here — that's normal, add them explicitly rather than
  reverting to a writable rootfs)
- `docker stop` mid-training → lease released → another worker picks the shard up
- Egress allowlist enforced; worker still functions with exactly three destinations
  reachable
- HF cache volume: base model pulled once on first run, reused on every subsequent
  container start

---

## M2 status — built, minus the hardware

**Done on this box; three exit criteria need machines it doesn't have.**

```
ganymede/worker/
  probe.py    6.9 self-test, backend registry (cuda, rocm, xpu, mps, cpu)
  client.py   the 6.2 surface over stdlib urllib
  control.py  the 4.4 stop/pause sentinels
  loop.py     the 4.2 loop, plus the ganymede-worker CLI
docker/
  torch-base.Dockerfile        CUDA + pinned torch, the layer that must not move
  torch-base-cpu.Dockerfile    CPU stand-in, so the container path is CI-testable
  worker-core.Dockerfile       ~30 MB: loop, client, probe. torch + stdlib only
  worker-llm.Dockerfile        ~2 GB: transformers, peft, datasets, bitsandbytes
.github/workflows/ci.yml       tests + a real container build and smoke test
```

| Exit criterion | State |
|---|---|
| One real GPU claims from a real coordinator, trains, submits, loops | **met on CPU.** A containerized `worker-llm` ran 207 steps against a live uvicorn coordinator and real MinIO under `--read-only --cap-drop=ALL --security-opt=no-new-privileges`, non-root, and the coordinator closed the round and opened the next. Repeatable as `tests/test_worker_live.py`. **The GPU itself is yours** |
| The same package runs natively on Linux/CUDA, macOS/MPS, Windows/CUDA | **needs three machines.** The backend registry (§4.5) makes each one an entry rather than a port, and the probe reports honestly on any of them — but "it registers correctly on a Mac" is a claim only a Mac can settle |
| The probe reports a machine ineligible without failing registration | met — nothing in the probe raises; a machine where every measurement failed still produces a valid profile and is simply never eligible |
| Read-only rootfs works with the declared writable mounts | met — verified by hand on this box, and asserted in CI. The CI half of that was aspirational until M4a: the workflow's container job had never actually executed, and its first run failed on the image chain (see "M4a status") |
| `docker stop` mid-training → lease released → another worker picks the shard up | partially: abandon-on-stop is covered by tests, and the sentinel path works. The two-worker handoff needs a second machine to be worth calling proven |
| Egress allowlist enforced; worker functions with three destinations reachable | **not done.** Needs a firewall to enforce, which this container does not have |
| HF cache volume: base model pulled once, reused on every container start | met — observed directly. Setup was 110 s on a cold cache and 19 s on a warm one, same image, same model |

### What the live rounds found

Running it rather than reasoning about it turned up four things, three of which
no test would have caught:

1. **`required_image` was checked only by the worker.** The coordinator offered
   the task anyway, so an ineligible worker was handed a lease it had to abandon
   — once per poll interval, marking a shard spoken for and churning bucket
   counters each time. The database recorded **seven leases in under three
   minutes, all abandoned**, before the check moved into the coordinator's
   eligibility filter where it belonged. The worker-side check stays as defense
   in depth for the case where registration and the run disagree.
2. **Setup time was not reserved in the budget.** A round measured **110 s of
   setup against 46 s of training** — loading a 0.6B model with a warm cache cost
   2.4× the work it enabled. `usable_seconds` reserved download, upload and
   margin but not that, so every worker was handed more steps than it could
   reach and delivered short every round with nothing reporting a problem. It is
   a fixed per-task cost, so it now sits beside download and upload as
   `est_setup_sec` rather than being folded into a rate, where the error would
   have scaled with round length.
3. **`worker-core` shipped without numpy.** `--no-deps` kept torch from being
   reinstalled and dropped numpy with it — and `safetensors.torch.save` converts
   through numpy. The image probed fine, registered fine, would have trained a
   full round, and died on the one line that serializes the result. CI now
   round-trips a tensor through safetensors inside the built image, because
   "does it start" was never the right question.
4. **`newrun` wrote to a bucket it never ensured**, so a first deployment
   depended on the coordinator having been started at least once beforehand —
   an unwritten prerequisite whose symptom is a boto stack trace.

### Still needs your hardware

| Blocked | Why |
|---|---|
| The GPU round | A CPU round proves the protocol, not the card |
| macOS/MPS and Windows/CUDA registration | Three platforms, three machines. The code paths exist; the evidence does not |
| AMD (ROCm) and Intel (XPU) | Wired into the registry and labelled `untested` (§4.5). Deferred by your call, and structured so adopting them is filling in an entry rather than a refactor |
| Egress allowlist | Needs a firewall to enforce it against |
| Pushing images to a registry | Needs credentials and a decision about where they live |


---

## M3 — Host agent

**~4 days** (was 2; three OSes means three schedulers, three install paths, and three
idle probes).

`IdleBackend` protocol with the `local` implementation (§7.1), manifest reconciliation,
systemd timer + unit, the `/etc/ganymede/pause` kill switch.

Deliverables:
- `ganymede/host/` — agent, idle backends
- `packaging/ganymede-host.service` + `.timer` (Linux systemd)
- `packaging/com.ganymede.host.plist` (macOS launchd)
- `packaging/ganymede-host-task.xml` (Windows Scheduled Task)
- HF cache size cap with LRU eviction (§6.7) — many runs means many ~16 GB base
  models, and a disk that fills silently is a bad first experience for a volunteer
- `INSTALL.md` — the contributor-facing document. This is the actual product surface
  for everyone who isn't you; treat it as a deliverable, not an afterthought

Exit criteria:
- Machine idles → worker starts within one timer interval
- `touch /etc/ganymede/pause` → running container stops, no new ones start
- Manifest tag bump → agent pulls and restarts on the new tag with no manual step
- **The install path is validated on a fresh machine**, not just yours. With no second
  contributor yet, a clean VM or a fresh container is a good proxy — it catches the
  undocumented dependency and the "works because your shell already had it" bug, which
  is most of the value. Re-run the test with a real second person when one exists;
  until then, treat a from-scratch install on clean OS images as the bar
- `INSTALL.md` states a minimum free-disk figure, and cache eviction demonstrably
  holds the cache under its cap across two runs with different base models

---

## M3 status — built, minus the three schedulers

**The agent, the idle probe, the cache cap, the packaging and the contributor
document.** Everything §7 describes now exists and is tested; what is *not*
established is that any of the three schedulers actually fires, because none of
them exists on a Linux CI container.

```
ganymede/host/config.py     settings, file-first because a timer has no environment
ganymede/host/idle.py       7.1's IdleBackend across three platforms
ganymede/host/cache.py      6.7's cache cap with LRU eviction
ganymede/host/manifest.py   7 step 3: what image should this machine run
ganymede/host/runtime.py    docker and native, behind one protocol
ganymede/host/agent.py      the tick, and the CLI a timer invokes
packaging/                  systemd, launchd, Task Scheduler, three installers
INSTALL.md                  the contributor-facing document
```

**`ganymede/host` imports nothing outside the standard library.** No `requests`,
no `huggingface_hub`, no Docker SDK. That constraint is the delivery story: the
host agent runs *outside* the container on a machine where the only guaranteed
thing is a Python interpreter, so it installs by copying a directory. It costs
about thirty lines of hand-built `docker` argv over `client.containers.run(...)`
and it is worth every one of them.

| Exit criterion | State |
|---|---|
| Machine idles → worker starts within one timer interval | **met in the agent, unproven in the scheduler.** The tick is asserted; that a timer fires it on a real machine is not something a container can show |
| `pause` → running container stops, no new ones start | **met**, and the spec's step order had to be inverted to get it — see below |
| Manifest tag bump → agent pulls and restarts, no manual step | **met** |
| From-scratch install on a clean OS image | **met for the package, not for the scheduler.** A `python:3.11-slim` container with no git, no docker and no systemd, which found three real bugs |
| `INSTALL.md` states a minimum free disk, and the cap holds across two runs with different base models | **met.** 35 GB, derived from the image stack plus one base model, named as a constant so the document and the `--check` warning cannot drift |

### Where the spec was wrong

Three places, all found by building against it.

1. **§7's step order disables the kill switch.** The block lists "already
   running a Ganymede container?" first and `is_idle()` second, which reads as
   an optimization — skip the idle probe when there is nothing to decide. Taken
   literally a running worker is never re-examined, so the pause sentinel would
   prevent *new* workers and never stop the one currently holding the GPU.
   Someone who wants their machine back is not helped by "after this round".
   Idleness is evaluated first now, and a running worker on a no-longer-idle
   machine is stopped.

2. **Named volumes made the kill switch a no-op.** §7 mounts
   `-v ganymede-hf:/cache/hf` and `docker/README.md` adds
   `-v ganymede-state:/var/lib/ganymede`. A named volume lives under
   `/var/lib/docker` and is Docker's to manage — so the contributor's
   `/var/lib/ganymede/pause` and the file the worker polls inside the container
   were **two different files with the same name**. The sentinel appeared to
   work and did nothing; only the agent-level stop was ever real, which is
   precisely the mechanism §4.4 says correctness must not rest on. The same
   argument applies to the cache: §6.7's cap was pruning a host directory
   nothing was filling. Both are bind mounts now, and the state mount is
   read-only, since the worker only reads sentinels and one that cannot write
   there cannot clear its own kill switch.

3. **`--user` as written runs the worker as root.** §7 and the README both pass
   it unconditionally. Under systemd the agent holds the Docker socket by being
   root, so `--user 0:0` would quietly undo the `USER ganymede` the image sets
   for itself. Passing nothing leaves the image's own unprivileged user in
   place; a rootless-Docker contributor still gets their own uid.

### One gap in the coordinator

`/v1/manifest` returned everything about an active run except `required_image`
— the single field §7 step 3 exists to consume. The column has existed since
M1 and `newrun` has always written it; nothing ever read it back out. Without
it there is literally nothing for the host agent to reconcile against, and the
image check would have fallen through to §4.2 step 5, where a worker that has
already claimed can only abandon.

### What the clean container found

The roadmap's guess was right: a fresh image "catches the undocumented
dependency and the 'works because your shell already had it' bug, which is most
of the value." Three, none visible on the development machine.

1. **`pip install .` dragged torch onto the host** — about 2 GB the docker path
   already has inside the worker image and will never load outside it. The
   stdlib-only rule made the host agent installable with `--no-deps`; nothing
   had ever checked that the *installer* took advantage of it.
2. **`--check` printed its failure above the report explaining it.** stdout is
   block-buffered when it is a pipe — which it is whenever an install script
   captures it — and stderr is not. Invisible on a terminal, guaranteed in the
   place it matters.
3. **The stated minimum free disk was a number nothing enforced.** A minimum
   people can ignore silently is a minimum they stop believing.

### And two XML files that never parsed

Both scheduler documents were malformed on their first draft, in the two ways
this class of file fails *silently*.

The plist had `--once` written into a rationale comment, and **XML forbids a
doubled hyphen inside a comment** — which is a delightful trap, because the flag
being documented is the thing that breaks the file. launchd's response to a
plist it cannot parse is to decline to load it and say nothing. The Task
Scheduler document declared UTF-16 while being ASCII on disk, which every parser
rejects before reading a single element.

In both cases the contributor's install "succeeds", nothing ever ticks, and
there is no error anywhere to find. `tests/test_packaging.py` now parses all
four scheduler files and asserts the things that fail quietly — comment syntax,
declared encoding, the element order the Task Scheduler importer demands, and
that the three platforms still agree on a cadence, since the exit criterion is
stated in timer intervals and a platform that has drifted has a different
criterion.

### Still needs a real machine

| Blocked | Why |
|---|---|
| The systemd timer firing | No systemd in a Linux CI container. `systemd-analyze verify` is the closest available check and needs the package installed |
| The launchd job firing | Needs a Mac. So does the `ioreg` idle path, and so does the LaunchAgent-vs-LaunchDaemon call, which rests on Metal being available in a session |
| The Scheduled Task importing | Needs Windows. The XML parses and its element order is right; that `schtasks` accepts it is a separate claim |
| `install-windows.ps1` running at all | No PowerShell on this box — it has never been executed, only read |
| Idle detection against a real desktop | `xprintidle`, `ioreg` and `GetLastInputInfo` are all faked in tests. The predicate is tested; the three platform probes are not |

---

## M4 — Multi-node convergence — *the milestone that proves the thesis*

**Splits in two, because you currently have only your own hardware.** That's less of a
blocker than it looks: the two things M4 proves need very different setups, and only
one of them needs other people.

**M4a is done** — see "M4a status" below. What follows is the original plan; the
status section records what running it actually turned up.

### M4a — Protocol under real concurrency (solo, ~2 days)

Run **several worker processes against your own machine**. On one GPU they time-slice
and run slowly; that's fine, because nothing here is measuring speed. If you have more
than one GPU, better. Even CPU workers on a tiny model validate the protocol.

Proves: concurrent claims don't double-lease, rounds close on accumulated work with
varying worker counts, late submissions get `409`, leases expire and re-lease cleanly,
aggregation combines several real adapters, bucket coverage advances.

Exit criteria:
- Three concurrent workers complete several rounds with no double-leasing and no
  stuck tasks
- Killing one mid-round costs that round's work for that worker and nothing else
- Bucket coverage advances rather than re-training the same shards
- Per-round loss descends across rounds

**This is reachable today with one machine**, and it's where most protocol bugs live.

### M4b — Convergence on genuinely parallel hardware (~1 day + training time)

Proving that aggregation *helps* needs real parallelism, which time-slicing can't give
you. You don't need contributors for this: **rent three cheap instances for an
afternoon.** At Qwen3-1.7B almost anything with a GPU qualifies, and a few hours of
three small rentals costs about as much as lunch. That is a very cheap way to
de-risk the thesis before asking anyone to install anything.

Exit criteria:
- Held-out loss vs. **cumulative steps** lands inside the M0 seed-variance band —
  aggregation preserves the training signal
- Held-out loss vs. **wall-clock** beats single-node — otherwise the system is an
  expensive way to train slower
- Greedy generations on the fixed prompt set show no template corruption or collapse
- Inter-worker adapter divergence is stable across rounds, not growing
- Both combine modes A/B'd (§5.2's research risk). If DiLoCo outer momentum doesn't
  beat plain weighted mean on LoRA adapters, **ship the plain mean** and record the
  negative result
- Mixed-speed workers both land near the round deadline, confirming §3.5's budgets
- No single worker exceeds the §3.5 dominance cap

Everything before this is plumbing. This is where you find out whether the idea works.
Budget for it to fail the first time and need a tuning pass — that's the normal
outcome, not a signal to abandon the approach.

---

## M4a status — built and run

**Three real worker processes, one real coordinator, one real object store, a
multi-round run driven to completion — repeatedly.** Everything the rest of the
suite fakes is real here: separate OS processes with separate SQLite
connections, separate torch runtimes and separate contributors, racing each
other through claim, heartbeat, submit and close.

```
ganymede/coordinator/invariants.py   what "broken" means, checked over the DB
scripts/evalround.py                 held-out loss per closed round (§5.2)
tests/test_worker_concurrency.py     the fleet, the kill, the four criteria
```

Processes rather than threads, deliberately. Threads share a GIL, a torch
runtime and a process image, which serializes exactly the thing under test — and
you cannot `SIGKILL` a thread, so the second exit criterion would have had
nothing to kill.

| Exit criterion | State |
|---|---|
| Three concurrent workers complete several rounds with no double-leasing and no stuck tasks | **met.** Four rounds, every one closed with a cohort of three, `invariants.check` clean over the database the three racing processes left behind |
| Killing one mid-round costs that round's work for that worker and nothing else | **met.** `SIGKILL`, not `terminate` — a machine that loses power never gets to abandon its lease, so the polite path is the wrong one to test. The lease is reclaimed, the survivors keep submitting, rounds keep closing |
| Bucket coverage advances rather than re-training the same shards | **met**, after a fix. All 32 buckets trained, spread of one round between the most- and least-trained |
| Per-round loss descends across rounds | **met, in the only sense this hardware can settle** — see below |

### The numbers

Held-out loss for each round's *aggregated* adapter, and inter-worker divergence
at the moment of aggregation:

```
round      0        1        2        3
loss     4.878    4.589    4.491    4.472     monotone, −0.41 nats
diverge  0.657    0.312    0.030    0.006
```

The loss curve says aggregation **carries** the training signal: three workers
train disjoint shards, their adapters are averaged into one, and the average is
better at the task than the round before. That is the property the whole design
rests on, and it is testable at any scale where the data has something in it to
learn.

The divergence curve is the more interesting one. §5.2 records divergence
because it measures precisely what DiLoCo trades away — more local steps means
more drift means a less meaningful mean — and *growing* drift across rounds is
the signal that `local_steps` is too high. Here it collapses by two orders of
magnitude, which is what agreement looks like: workers on different shards
finding the same solution because there is one to find.

**What none of this claims.** It is a 107k-parameter model over forty held-out
examples of synthetic data. It says nothing about convergence at 1.7B, nothing
about whether aggregation *beats* single-node on wall-clock, and nothing about
mixed-speed fleets. All of that is M4b's, and it needs genuinely parallel
hardware.

### What running it found

Seven things. Six are protocol bugs, and not one of them was reachable from a
unit test: each needs several machines, real elapsed time, or a real outage
before it exists at all.

1. **The close rule was never evaluated on the passage of time.** `maybe_close`
   ran only after a submission — and the second branch of the close rule
   (`elapsed ≥ max_round_sec`) is triggered by *time*, which does not arrive on
   any request. So: every worker submits, none reaches `target_steps`, none can
   claim because too little of the round is left to be worth a budget, and the
   round stays open **forever**. The run stops advancing, every worker gets
   `204` on every poll, and nothing anywhere returns an error. The rule is now
   checked on claim too, so any worker still awake moves the run on.

2. **That same omission had silently disabled the empty-round reopen.** §3.2
   says a round that hits its backstop with nothing submitted reopens with a
   fresh deadline, because an idle fleet overnight is normal operation. That
   path is reached only through the close check — which only ran *after a
   submission*, at which point the round has one by definition and the reopen
   condition is false. Specified, implemented, and unreachable. It works now for
   the same reason the wedge is gone: the check runs where an idle fleet
   actually shows up, which is the poll.

3. **The step budget could outgrow the dataset.** `bucket_count` caps a worker's
   shard at the run's total, and nothing fed that cap back into the budget
   sitting beside it. Once measured throughput replaced the cold-start guess, a
   worker was budgeted **14,268 steps over 352 training samples — about 81
   passes on a run configured for one** — and every worker in the round was
   handed the identical whole dataset, so aggregation weighted three copies of
   one piece of work as three contributions. Exactly the overfitting §3.5's
   bucket scaling exists to prevent, arriving by the one path that scaling
   cannot see. Budgets are now clamped to the data they were actually given, and
   a claim that hits the clamp logs it: the run's *shape* is wrong, and only an
   operator can fix that.

4. **A storage outage killed the workers outright.** MinIO went away mid-round
   and all three exited on the spot, on an unhandled exception out of the upload.
   §6.4 says a worker rides this out and retries with backoff; on an unscheduled
   volunteer fleet, exiting is the worst available behaviour, because the
   machines that were contributing stop permanently and nobody is watching to
   notice. Infrastructure failures now cost the round in progress, not the
   worker. Everything else still stops it — a bug is not something a worker can
   retry its way out of, and a machine failing every round forever while taking
   leases is worse than one the host agent restarts.

5. **`--max-rounds` counted tasks, not rounds.** A worker that finishes its step
   budget while the round is still open claims again, routinely several times
   over. So `--max-rounds 20` stopped all three workers *inside round 0*, having
   taken twenty tiny cold-start tasks each, before the round they were all
   working had closed. The run stalled and nothing reported a problem.

6. **A close that dies partway wedged the round permanently.** `closing` is
   claimed by exactly one caller and released only when that caller finishes,
   and everything in between is storage I/O and tensor arithmetic. A failed
   close now gives the round back, and the next request retries it. The case
   with no exception to catch — a coordinator killed between the two writes —
   is detection rather than recovery, and is what `invariants.check` reports as
   `stuck_close`.

7. **The seventh was in the harness**, and is worth writing down because it cost
   an hour and looked exactly like a protocol failure: **a subprocess whose
   stdout is an unread `subprocess.PIPE` blocks forever once the pipe fills** —
   about 64 KB, which a worker logging at INFO reaches in under a minute. Two of
   three workers froze mid-round holding leases, and the symptom was
   indistinguishable from a coordinator that had stopped handing out work.

### And one bug in the test itself, which passed

The first version of the loss assertion was worthless, and worth recording
*because* it was green.

The synthetic dataset was random words in, random words out. There is no
learnable structure in that at all, so held-out loss cannot move for any reason
except noise — and the measurement bore that out: **5.185 → 5.146 → 5.151 →
5.150**. Total movement 0.7%, not even monotone. It satisfied
`losses[-1] < losses[0]` and demonstrated nothing whatsoever about aggregation.

A test that cannot fail for the reason it claims to test is worse than no test,
because it reads as evidence. Each response is now a deterministic function of
one token in its instruction, so the assertion has something real to detect, and
the threshold sits well outside what noise on forty examples produces. That one
change is the difference between the two curves above.

### And the CI job that had never run

The push that closed M4a triggered **run #1** of the workflow — the first time
GitHub Actions had ever executed it. The fast suite passed. The container job
failed in one second, on its second `docker build`:

```
ERROR: failed to solve: ganymede/torch-base:cpu:
  pull access denied, repository does not exist or may require authorization
```

The images are a chain: `worker-core`'s `FROM` is the `torch-base` built by the
step above it. `docker/setup-buildx-action`'s default `docker-container` driver
runs the build inside its own container with its own image store, so it resolved
that `FROM` against Docker Hub and asked for an image that only exists locally.
`load: true` does not help — it puts the base in the *host* daemon's store,
which is not where the builder looks.

Two things worth taking from it, neither of them about Docker:

**A CI job that has never run is not a check, it is a plan.** The M2 exit
criteria table said read-only rootfs was "asserted in CI on every push". The
underlying claim was true — the images were built and run by hand, under
`--read-only --cap-drop=ALL`, and a containerized worker completed 207 steps —
but the sentence described a mechanism that had never once executed. That entry
now says which half was verified how.

**The failure mode that worries me more is the one that did not happen.** Had
some unrelated `ganymede/torch-base:cpu` existed on Docker Hub, the pull would
have *succeeded*, and every check in the job would have passed against an image
nobody in this repository built. The job now builds with the host daemon — which
is also what `docker/README.md` tells a contributor to run — and asserts that
`worker-core`'s base layer is the `torch-base` from the step above it.

### Two things worth knowing before M4b

- **`invariants.py` is the M5 alerting hook, early.** It exits non-zero, so it
  is already usable as a cron check, and it deliberately says nothing about an
  idle fleet — §3.2 is explicit that idleness is normal operation, and alerting
  on it would train the operator to ignore the alert that matters.
- **A worker re-loads the base model for every task**, and takes several tasks
  per round when its budget is small relative to the round. At the measured M2
  setup cost of 110 s on a 1.7B model, three tasks in a round is 330 s of setup
  against one round of work. Not a correctness problem and not M4a's to fix, but
  it will show up as soon as rounds run on real models.

---

## M5 — Operations

**~2 days.**

Per-round loss and per-contributor throughput surfaced on the read-only `/status`
page, coordinator alerting (gate rejection rate spiking, no submissions accepted in
N rounds despite workers being active), a runbook, and verified restore-from-backup.

Alert on *broken*, not on *idle* — with an unscheduled fleet, "no workers right now"
is the normal overnight state and paging on it trains you to ignore the alerts (§3.2).

Scope note: v1 targets **your group now, opening later**. So this builds the operator
view, not a contributor-facing product — no leaderboards, no public stats. The two
things kept honest from the start, because they're what opening up later depends on,
are the **install path** (M3's "second person installs unaided" test) and **eligibility
diagnostics** — a contributor who never gets work must be able to find out why.

Exit criteria:
- You can answer "is it training, and who is contributing?" in one glance
- You get alerted about a stalled run without having to look
- A registered worker that is never leased can be told *why*: insufficient memory,
  missing `nf4`, below the throughput floor, or clearance (§6.9, §6.10)
- **Distinct contributors per round is visible** (§3.2). If most rounds are closing
  with one machine, the swarm isn't earning its overhead — that should be a number you
  can see, not something you infer months later
- **A full restore has been performed at least once**, not merely configured: rebuild
  the coordinator from the off-box SQLite dump plus the latest round adapter, and
  resume the run. §6.4 names the shared VM as the system's single point of failure,
  and self-hosted storage is what makes this drill non-optional
- Artifact volume has headroom, and GC is demonstrably keeping it that way

---

## M5 status — the operator view, built early

Three of M5's exit criteria landed alongside M3, because each of them turned
out to be a report over facts the database already held rather than new
instrumentation. §6.11's rule — the fleet inventory is derived, not maintained —
is what makes that true: an inventory written down separately is one that goes
stale and then lies.

```
ganymede/coordinator/invariants.py   what "broken" means (built during M4a)
ganymede/coordinator/eligibility.py  why a worker is never leased
scripts/status.py                    is it training, and who is contributing
```

| Exit criterion | State |
|---|---|
| Answer "is it training, and who is contributing?" in one glance | **met.** `ganymede-status` |
| Alerted about a stalled run without having to look | **met.** `--alert` is silent on success and exits 1, so cron is the whole scheduler |
| A worker that is never leased can be told *why* | **met.** `GET /v1/workers/{id}/eligibility`, and `--fleet` for the operator's side |
| Distinct contributors per round is visible (§3.2) | **met.** Reported per round, with a median and a flag when most rounds close with one machine |
| A full restore performed at least once | **met.** Performed against real MinIO with the database and the primary bucket both destroyed — see below |
| Artifact volume has headroom, GC demonstrably keeping it | **not started** |

### The restore drill, performed

§6.4 names the shared VM as the system's single point of failure and §6.6
accepts that risk on one sentence: "the database plus the newest adapter is
enough to resume the run rather than restart it". That sentence had never been
tested, and `backup.py` had never been restored from — which is why the
criterion is worded as *performed at least once, not merely configured*.

`scripts/restore.py` is the missing half. The drill, run against a real MinIO:
seed a run at round 2 with two closed rounds and a worker mid-round holding a
lease and a submission; back up off-box; **delete the database and destroy the
primary bucket**; restore; start a coordinator on the result; register a real
worker and claim.

```
restored from   backups/20260827T011617Z/coordinator.db
adapters        1 copied, 0 already in the primary store
rewound         drill-run#2 (2 lease(s) released, 1 partial submission(s) dropped)

claim -> HTTP 200
  CLAIMED ecb715df... | round 2 | buckets [4, 5, 6, 7, 0, 1, 2, 3] | steps 100
  base adapter downloaded from the restored store: 240 bytes
```

Bucket ordering is the detail worth noticing: least-trained first, so §4.7's
coverage state came through the restore intact. `invariants.check` is clean on
the rebuilt database, and coverage still reads 8/8 with a spread of one.

**A restore is not a resurrection, and the difference is the whole design.**
Individual worker submissions are deliberately not backed up — they are already
folded into the round result that *is* backed up, and they are the bulk of the
bytes. So a restored database necessarily references artifacts that exist
nowhere, and holds leases for workers talking to a machine that no longer
exists. Left alone the round can neither close (its submissions cannot be read)
nor progress (its shards look spoken for until every lease expires). The
reconcile step rewinds exactly the in-flight round: one round of work re-done,
which is precisely the work whose artifacts went down with the disk.

A round caught mid-`closing` is the dangerous case, since `closing` is held by
one caller and released only when it finishes — a restore that left it there
would wedge the run permanently, in the same `stuck_close` state M4a's sixth
bug produced.

### And one bug in the backup, found by needing it

`backup.py` never called `ensure_bucket` on the destination, so **the first
backup to a fresh off-box bucket died with a boto stack trace**. This is
verbatim the bug already recorded against `newrun` in M2 status — "wrote to a
bucket it never ensured, so a first deployment depended on the coordinator
having been started at least once beforehand" — in a second script, found only
because someone finally ran the thing end to end.

A backup that fails the first time it is genuinely needed is a worse class of
bug than most, because nobody finds out until the disaster.

### The alert had to be defined against idleness, not around it

The obvious stall check fires on a healthy volunteer fleet every night. §3.2 is
explicit that contributors come and go and that a round sitting open overnight
is the design working — so an alert that cannot tell "nobody asked for work"
from "workers asked and the round did not move" would go off constantly and
train the operator to ignore the one that matters. That is the same reason
`invariants.py` deliberately says nothing about idleness at all.

A stall here therefore requires **both** halves: a round well past its own
backstop, *and* evidence that workers were awake while it sat there. The second
half only became available with the eligibility table, which records every
poll — including the polls that were refused, which is exactly the machine an
operator most wants counted as present. Together they are the M4a wedge
signature: every worker polling, every poll answered 204, the round open
forever, and nothing anywhere returning an error.

### Recorded, not recomputed

Every refusal reason in `eligibility.py` was produced by `rounds.claim_task` on
a real poll and written down verbatim. Nothing re-derives eligibility, and that
is the design rather than an implementation detail. A diagnostic that
reimplements the decision it explains will eventually disagree with it, and a
contributor told they are eligible for a run that keeps turning them away is
worse off than one told nothing — the suspicion moves from their machine to the
operator's competence. The claim path had been computing a reason for every run
it declined and dropping the list on the floor since M1.

Three outcomes are kept apart, because they are three different problems that
read identically one worker at a time: `refused` carries the reason, `idle`
means eligible but the run had nothing to hand out, and a worker with **no rows
at all** has never completed a poll — so the fault is upstream of eligibility
entirely, and the agent is not running or cannot reach the coordinator.

Recording is best-effort by construction and tested that way: dropping the
table must not turn a successful claim into a 500. A diagnostic that can break
what it diagnoses is a worse trade than not knowing.

---

## Sequencing

```
M0 ─────────► M2 ──────► M4a ──► M4b ──► M5
 │  training              solo    rented
 └──► M1 ─────► M3
     coordinator
```

M1 depends on M0 only for the LoRA config shape (needed by the acceptance gates), so
the two tracks can overlap if there's a second person.

**M4a turned out not to need M3** — the diagram used to route it through both tracks
and no longer does. Three worker processes started by hand exercise the protocol
exactly as well as three started by a host agent, and doing M4a first was the better
order: M3's whole job is starting a worker, and it is cheaper to find a double-lease
bug before there is a supervisor wrapped around it. Six protocol bugs came out of M4a,
and every one of them would still have been there, harder to see, underneath M3.

**Everything through M4a is reachable with one machine and no contributors** — and
that is now demonstrated rather than predicted. Only M4b needs genuinely parallel
hardware, and renting it for an afternoon is cheaper and faster than waiting for
volunteers — and it means the first person you *do* ask to install something is
joining a system already known to work.

If you already have a fine-tuning pipeline to port, M0's trainer is a day rather than
two, and the calibration harness is the only genuinely new build.

Solo and sequential: roughly **16–20 focused days** to M4, plus training wall-clock.

---

## Phase 2 (not scheduled)

In rough order of when they'll start to hurt:

1. **Spot-check redundancy and reputation scoring** — the first thing needed when the
   contributor pool outgrows people you personally vouch for. The audit data
   accumulates from M1 (§6.3), so this is analysis, not instrumentation.
2. **Concurrent runs.** Deferred from v1 by decision, but wide hardware diversity
   raises its value (§6.8): with sequential runs, an nf4 run idles every Mac and a
   bf16 8B run idles every 3060. Needs run selection, priority, and starvation
   handling in claim. The eligibility model is already written so this is a
   scheduling change, not a redesign.
3. **MLX trainer for Apple Silicon.** Macs participate from v1 via PyTorch MPS
   (§6.8) — slow but real. MLX is the fast path and a second trainer implementation
   interoperating via safetensors. Only worth it if Mac *throughput* becomes the
   point rather than Mac participation.
4. **`vast` / `tensordock` `IdleBackend`s** — needed when you move past own hardware.
   Small, given M3's interface. Verify the platform-policy question first (§7).
5. **Postgres migration** — when workers exceed a few hundred (§6.1).
6. **Hivemind `SyncBackend`** — when central bandwidth genuinely binds. At ~3.4 GB/hr
   for ten workers, that is a long way off. Do not do this early; it buys nothing at
   current scale and costs NAT traversal work.
7. **MoE expert training.** Deferred, not blocked (§5.4). Needs per-expert routed-token
   counts in submit metrics and the per-tensor weighting already provided for in §5.2;
   no schema change. Keep the router frozen longest — averaging routers trained on
   different shard distributions is the genuinely hard part. Note the cheap
   intermediate step: an MoE base with attention-only LoRA needs nothing new at all.
8. **`rl_rollout`** — needs its own spec review. Isaac Lab's image size and PyTorch
   pinning break the shared-layer story (Review Finding B), and shipping raw
   trajectories over WAN is a heavier data path than shipping weight deltas. Worth
   evaluating whether workers can send gradients or advantages instead of trajectories.
9. **gVisor/Kata, TOPLOC verification, Byzantine-robust aggregation** — when the pool
   includes people you don't know. §5.2's weighted mean is the drop-in point for
   trimmed-mean/median.

---

## Open questions

Two remain, and neither blocks M0.

1. **Your own dataset**, when bring-up is done: what it is, how large, where it lives,
   and which classification it warrants (§6.10). Dolly carries M0–M4; this is the
   first *real* run's input. *Not needed until after M4.*

2. **Contributor agreement**, before the first non-`open` run. §6.10 gates `internal`
   and `restricted` runs on clearance, which implies contributors accept something
   before receiving non-public data. It needn't be elaborate, but it's also the
   natural place to settle who owns the resulting adapters. Dormant while you're the
   only contributor. *Needed before the first non-`open` run.*

### Closed

- ~~**Base model**~~ → `Qwen/Qwen3-1.7B-Base`, dense, text-only, bf16, for bring-up;
  `Qwen3-8B-Base` later. Settled by reading `config.json`, not the model listing: the
  newer Qwen3.5 dense line is multimodal with hybrid attention at every size, which
  makes the standard LoRA target set silently cover a quarter of the layers. Deferred
  to the scale step with its requirements written down.
- ~~**Bring-up dataset**~~ → Dolly 15k, 64 buckets.
- ~~**Object store**~~ → self-hosted MinIO, R2 by config swap (§6.6).
- ~~**Coordinator hosting**~~ → deployment config, not a design question. Everything
  refers to `COORDINATOR_HOST` / `STORAGE_HOST` and friends (§6.5); no hostname
  appears in the codebase. Still need real values before M1 *deploys*, but nothing is
  blocked on choosing them.
- ~~**Hardware inventory**~~ → **derived, not maintained** (§6.11). Capabilities are
  probed at registration, throughput and bandwidth are measured while working, and
  availability is observed. `GET /v1/fleet` renders the current picture. There is no
  roster to collect or keep current.
- ~~**Platform policy check**~~ → deferred with the rented-host phase; nothing depends
  on it while you're on your own hardware.
- ~~**Eval metric**~~ → held-out loss on a fixed Dolly split, compared against the
  single-node baseline on two axes (total steps for correctness, wall-clock for
  value), with tolerance set by measured seed variance. See *Eval metric* above.

### A note on being the only contributor

Nothing in the design assumes a fleet, so solo development runs into no walls:

- Rounds close on **accumulated work**, not worker count (§3.2), so a one-machine run
  is a legitimate run rather than a degenerate case.
- `distinct_contributors = 1` every round is **expected** right now, not a warning.
  M5's health metric only becomes meaningful once others join — don't tune against it
  yet.
- §6.10's classification machinery is dormant and costs nothing. It's there so that
  the first sensitive run doesn't require retrofitting.
- Everything through **M4a** is reachable today. Only M4b needs parallel hardware, and
  three cheap rentals for an afternoon beats waiting for volunteers.
