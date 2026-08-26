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

**Buckets: 64** (~230 samples each), per the sizing rule below.

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
| 15k | 64 | ~230 |
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
| `multiprocessing` uses spawn, not fork | Guard entry points with `if __name__ == "__main__"`; start with `num_workers=0` in the DataLoader |
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

**Done.** 136 tests, nothing skipped, including the real-MinIO storage tests.

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
SIGTERM/abandon (§4.4), capability probe (§6.9), isolation flags (§4.5). The trainer is
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

## M4 — Multi-node convergence — *the milestone that proves the thesis*

**Splits in two, because you currently have only your own hardware.** That's less of a
blocker than it looks: the two things M4 proves need very different setups, and only
one of them needs other people.

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

## Sequencing

```
M0 ─────────► M2 ─┐
 │  training       ├──► M4a ──► M4b ──► M5
 └──► M1 ─────► M3 ┘   solo    rented
     coordinator
```

M1 depends on M0 only for the LoRA config shape (needed by the acceptance gates), so
the two tracks can overlap if there's a second person. M4a needs all four.

**Everything through M4a is reachable with one machine and no contributors.** Only M4b
needs genuinely parallel hardware, and renting it for an afternoon is cheaper and
faster than waiting for volunteers — and it means the first person you *do* ask to
install something is joining a system already known to work.

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
