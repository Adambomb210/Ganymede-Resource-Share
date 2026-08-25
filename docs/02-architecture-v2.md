# Ganymede — Architecture v2

**Status:** Draft v2, incorporating `01-spec-review.md`
**Scope:** `llm_finetune` only (LoRA on a 7–8B base). `rl_rollout` deferred to Phase 2.
**Sync:** Central aggregation, behind a `SyncBackend` seam for a later Hivemind swap.
**Trust:** Trusted circle, with two items pulled forward from v1 §10 (safetensors, sanity gates).

Supersedes `00-original-spec-v1.md`. Sections that are unchanged in substance are
summarized rather than restated.

---

## 0. Goals

Carried from v1 §1, with one addition made explicit:

- One deployment story for contributors, regardless of what's being trained
- Job configuration is data, not code — most changes require zero rebuilds
- Container footprint stays small even as supported job types grow
- Reuse existing open-source infrastructure rather than rebuilding distributed-systems
  primitives
- **Run on almost any hardware.** A contributor's machine should be able to join the
  platform even when it can't serve every run. Platform compatibility and run
  eligibility are separate concerns (§6.8, §6.9), and only the second is allowed to
  exclude anyone.

That last goal is why the worker is a package rather than a container (§4.1), and why
capabilities are probed rather than enumerated (§6.9).

---

## 1. What changed from v1

| v1 | v2 | Why |
|---|---|---|
| Job queue | **Run → round → task** | Review Finding A |
| `worker-core` as shared base | **`torch-base` as shared base**, core above it | Finding B |
| Egress: coordinator + bootstrap | Egress: coordinator + artifact store + HuggingFace | Finding C |
| (unspecified `s3://`) | **Self-hosted MinIO**, S3-API-compatible, R2-swappable by config | Decision |
| Weights in POST body | Presigned PUT; coordinator sees only refs | Finding H |
| Hivemind DHT | Central aggregator behind `SyncBackend` | Decision |
| Axolotl / torchtune | Owned ~250-line `peft` trainer | Finding K |
| `.pt` checkpoints | **safetensors everywhere** | Finding G |
| Worker checks image version | Host agent checks; worker declares capabilities | Finding E |
| SIGTERM → save partial, submit | SIGTERM → **abandon lease**, exit; partial submit best-effort | Finding D |
| (absent) | Dataset bucket sharding | Finding I |
| (absent) | Pinned `base_precision` | Finding J |
| (absent) | Bearer auth | Finding F |
| Worker-count quorum | **Work-based round close**, robust to unscheduled fleets | §3.2 |
| Uniform `local_steps_per_sync` | **Per-worker step budgets** sized to a common deadline | §3.4 |
| (single run assumed) | Multi-run cache affinity + per-run calibration | §6.7 |
| Uniform hardware assumed | **Three-tier heterogeneity model**; `compute_profile` eligibility | §6.8 |
| Container = the worker | **Package = the worker**; container is one delivery path | §4.1 |
| (absent) | Capability **probing** rather than hardware enumeration | §6.9 |
| (absent) | Per-run **data classification** as a second eligibility axis | §6.10 |

---

## 2. System overview

```
                         ┌──────────────────────────────┐
                         │  Coordinator  (1 small VM)   │
                         │                              │
   claim / heartbeat     │  FastAPI + SQLite            │
   submit(ref) / abandon │   · run + round state machine│
  ┌─────────────────────►│   · lease manager            │
  │                      │   · aggregator (weighted     │
  │                      │     mean + DiLoCo outer step)│
  │                      │   · acceptance gates         │
  │                      │   · presigned URL minting    │
  │                      └───────────────┬──────────────┘
  │                                      │ mints presigned URLs;
  │                                      │ app never proxies bytes
  │                                      ▼
  │                      ┌──────────────────────────────┐
  │                      │  MinIO  (self-hosted, §6.6)  │
  │                      │   · round base adapters      │
  │                      │   · worker submissions       │
  │                      │   · sharded dataset buckets  │
  │                      │                              │
  │                      │  same VM in v1 — so this VM  │
  │                      │  DOES serve artifact bytes   │
  │                      └───────────────┬──────────────┘
  │                                      │ presigned GET/PUT
  ├──────────────┬───────────────────────┤
  │              │                       │
┌─┴──────────┐ ┌─┴──────────┐  ┌─────────┴──┐
│ Worker A   │ │ Worker B   │  │ Worker C   │      ┌──────────────┐
│ own 4090   │ │ Vast 3090  │  │ TD 4090    │─────►│ HuggingFace  │
└────────────┘ └────────────┘  └────────────┘      │ (base model, │
                                                    │  cached once)│
                                                    └──────────────┘
```

The coordinator is the only always-on component and remains GPU-free. The FastAPI
application never proxies artifact bytes — it mints presigned URLs and handles only
references, metrics, and control flow.

In v1 the artifact store is **self-hosted MinIO on the same VM**, so that VM does
serve the bytes at the network level even though the application doesn't touch them.
That is a deliberate, and cheap, trade — see §6.6 for the sizing that justifies it and
the one consequence that genuinely bites (backups).

---

## 3. The round model

This is the core structural change. Three nested concepts:

**Run** — one training campaign. Pins base model, base precision, LoRA config,
dataset, and hyperparameters. Changing any of these means a new run, never a mutation
of the current one.

**Round** — one synchronization cycle within a run. Pins exactly one base adapter and
one deadline. Every worker in round N starts from byte-identical weights.

**Task** — one worker's unit of work within a round: a set of dataset buckets plus a
local step budget.

### 3.1 Round lifecycle

```
      ┌─────────────────────────────────────────────────────┐
      │                                                     │
      ▼                                                     │
 ┌─────────┐  Σsteps ≥ target  ┌─────────────┐  gates+   ┌──┴────────┐
 │  OPEN   │  OR backstop+≥1   │ AGGREGATING │  outer    │  CLOSED   │
 │         │──────────────────►│             │  step     │  round N  │
 │ round N │                   │  round N    │──────────►│  result   │
 └─────────┘                   └─────────────┘           └───────────┘
      │                                                        │
      │ tasks: pending → leased → submitted                     │
      │                    │                                    │ becomes base
      │        lease expiry│ / abandon                          │ for round N+1
      │                    ▼                                    │
      │              back to pending (attempts++)               │
      └─────────────────────────────────────────────────────────┘
```

1. Coordinator opens round N with `base_adapter_ref` = round N−1's result (round 0:
   freshly initialized LoRA, seeded from `hyperparams.seed`). It creates K tasks with
   bucket assignments drawn from the least-covered buckets.
2. Workers claim tasks. The lease is sized from the worker's self-reported throughput,
   with a floor and a ceiling, and is renewed by heartbeat.
3. A worker fetches the base adapter, trains its buckets for `local_steps` (or until
   `max_runtime_sec`, whichever first), uploads a safetensors adapter via presigned
   PUT, and submits the reference plus metrics.
4. The round closes on **accumulated work**, not on a worker count — see §3.2. A
   submission arriving after close gets `409 Conflict` carrying the current round
   number; the worker re-claims into it and moves on.
5. Aggregation runs (§5). The result becomes round N+1's base.

### 3.2 Closing a round when availability is unpredictable

Contributor machines come and go without a schedule. Nobody can say in advance whether
two or eight will be up. That rules out the obvious close condition — a fixed worker
quorum — because any value you pick is wrong most of the time: too high and the round
never closes, too low and it fires on the first submission, which defeats the point.

**So the round closes on accumulated work, not on how many machines showed up:**

```
close round N when:
     Σ steps_completed ≥ round_target_steps   AND   elapsed ≥ min_round_sec
  OR elapsed ≥ max_round_sec                  AND   submissions ≥ 1
```

- **`round_target_steps`** is the real trigger. What matters for aggregation quality
  is total work across distinct shards, not the number of distinct machines: two
  workers doing 500 steps each is roughly two doing 250 plus two more doing 250. This
  degrades smoothly at *any* fleet size — eight machines close the round quickly, two
  take longer and still get there.
- **`min_round_sec`** stops a large fleet from churning through rounds faster than
  aggregation overhead can justify.
- **`max_round_sec`** is a backstop, not the primary mechanism. Set it generously.

Zero submissions at the backstop simply reopens the round with a fresh deadline.
Nothing is lost and nothing degrades — an idle fleet means the run pauses, which is
the correct behavior, not an error. **Do not alert on it**; with a volatile fleet it
is the normal overnight state.

#### Single-contributor rounds: accept, but surface

Nothing above prevents one machine from supplying all of `round_target_steps` while
everyone else is offline. That round is not *wrong* — with outer momentum it behaves
like an ordinary training step — but it is single-node training paying distributed
overhead.

Blocking it would stall the run for no benefit, so accept it and **record the distinct
contributor count per round**. If most rounds are closing with one contributor, the
system isn't earning its complexity, and that should be visible in `/status` rather
than inferred months later. It's a health metric, not an error condition.

#### Claim against time remaining, not a full round

A machine that is up for twenty minutes should still contribute. If step budgets
(§3.5) are always sized to a *full* round, a worker joining mid-round gets work it
cannot finish, and the round closes underneath it — its effort wasted entirely.

So budget from **time remaining in the current round**:

```
usable   = minutes_remaining − est_download − est_upload − safety_margin
local_steps_i = throughput_i × usable
```

`est_download` and `est_upload` come from that worker's **measured** transfer rates,
recorded on its previous rounds (§6.9). A contributor on a slow uplink automatically
gets a smaller step budget so the transfer still fits inside the round, instead of
becoming an unexplained straggler. Nobody has to know or declare their bandwidth.

With a cutoff: if `usable` falls below a few minutes, return `204` with a
`Retry-After` pointing past the round boundary rather than handing out work that
can't land. This
makes opportunistic, short-lived participation productive, which matters much more when
machines appear at random than when they're on a schedule.

#### Volatility argues for shorter rounds

§3.4's 30–45 minute target assumes reasonably stable hosts. A volatile fleet inverts
the trade: shorter rounds mean more chances to contribute and less work lost when a
machine disappears mid-round.

Start nearer **15–20 minutes**. The cost of shorter rounds is aggregation overhead and
more artifact round-trips — and at bring-up scale a 1.7B LoRA adapter is only ~10–15 MB,
so that cost is negligible. Revisit when scaling to 8B, where adapters are ~85 MB and
round-trip overhead starts to matter.

### 3.3 Why this solves several problems at once

- **Averaging is well-defined.** Every contribution to round N provably started from
  the same weights.
- **Version skew is structural.** A worker holding a stale base can only submit to a
  closed round, which is rejected by construction. v1 §8's "wind down the swarm before
  an architecture change" becomes: start a new run. No coordination ritual.
- **Stragglers degrade gracefully.** Weighting by steps actually completed (§5) means
  a slow worker contributes proportionally less rather than blocking or corrupting.
- **Fleet size is irrelevant to correctness.** Because closing is work-based (§3.2),
  the same run behaves sensibly with two machines or twenty, which is required when
  nobody can say which will be up.
- **Preemption costs exactly one round.** Which is why round sizing is a safety
  parameter, not just a tuning knob (§3.3).

### 3.4 Round sizing

Target **15–20 minutes** for a volatile fleet (§3.2), extending toward 30–45 only if
hosts turn out to be stable and adapters get large.

- Too long: a flaky host may be preempted before it ever completes a round, so it
  contributes nothing, ever.
- Too short: aggregation overhead and the ~170 MB per-worker round trip dominate the
  useful compute.

### 3.5 Per-worker step budgets

Uniform `local_steps` across heterogeneous hardware means the slowest card sets the
deadline for everyone. A 3090 paired with a 4090 straggles every round, and either
blocks the close or gets truncated — wasting a fraction of its work each time,
forever.

Instead, **fix the round deadline and vary the step budget per worker**:

```
local_steps_i = throughput_i × target_round_minutes × safety_factor
```

Everyone finishes near the same deadline, and §5.2's weighting by steps actually
completed then reflects real contribution rather than hardware luck. Note the input is
*minutes remaining in the round*, not full round duration — §3.2.

`throughput_i` comes from three places, in order of preference:

1. **Measured** — the coordinator stores observed steps/min per `(run, gpu_model)`
   from submitted metrics, and uses it for subsequent assignments
2. **Calibrated** — `calibration.json` for the run, produced in M0 (see
   `03-roadmap.md`), keyed by GPU class
3. **Conservative default** — first ever worker of an unseen GPU class on an
   uncalibrated run; deliberately low, corrected after one round

`max_runtime_sec` becomes a safety net against a wedged worker rather than the primary
control.

This matters more as the number of runs grows: a 3B and a 70B fine-tune need step
budgets an order of magnitude apart, and hand-setting `local_steps_per_sync` per run
is exactly the kind of manual step that stops scaling once there are many runs.

#### Scale buckets with the budget, not just steps

A subtlety that bites specifically on wide hardware spreads. If every worker gets the
same number of dataset buckets but a different step budget, the fast worker simply
does **more epochs over the same small shard** — overfitting it. That is worse than
useless: it earns a large aggregation weight while contributing a narrower view of the
data than the slow worker beside it.

So bucket count scales with the budget too. A worker receives roughly enough buckets
that its step budget amounts to a comparable number of passes over its assigned data,
whatever its speed. An A100 gets more steps *and* proportionally more of the dataset.

This converts hardware heterogeneity into **data coverage** rather than weight
concentration, which is the outcome you want: extra capacity buys diversity, not a
louder vote on the same shard.

#### Guard against one worker dominating

Even with bucket scaling, an A100 among 3060s may carry a large share of a round's
weight. Proportional weighting is *correct* — it reflects work actually done — but
past a point the round becomes "one worker's trajectory plus a small correction",
which has different convergence behavior from balanced averaging, and craters when
that worker drops out.

Cap any single worker's share at roughly **2× the median** contribution. The cost is a
little efficiency; the benefit is a run that degrades gracefully when its best machine
is preempted. Revisit the cap once §5.2's A/B has real data — this is a knob with a
defensible default, not a settled number.

#### Minimum viable throughput

Below some speed, a worker costs more than it contributes: it consumes a lease slot
and a full ~170 MB round trip to add noise-level weight.

Set a floor — a worker must be able to complete some minimum fraction (start with
~10%) of the median budget within the round window, or it isn't offered work **for
that run**. It stays eligible for others. This is the mechanism that lets a very wide
hardware spread coexist without silently wasting bandwidth on machines that can't
keep up with a given model.

---

## 4. Worker

### 4.1 Image stack

```
nvidia/cuda:12.4.1-runtime-ubuntu22.04
  └─ ganymede/torch-base:cu124-t2.5    ~8 GB    torch + CUDA runtime libs
       └─ ganymede/worker-core:vN       ~30 MB  poll loop, HTTP client, GPU probe,
            │                                   safetensors, SIGTERM handling
            ├─ ganymede/worker-llm:vN    ~2 GB  transformers, peft, bitsandbytes,
            │                                   datasets, accelerate
            └─ ganymede/worker-rl:vN     Phase 2 — likely cannot share torch-base
                                         (Isaac Lab pins its own build)
```

`torch-base` is pinned by digest, not tag, so every worker in a run has a bit-identical
PyTorch. Rebuild it deliberately and rarely; a silent PyTorch bump across a running
swarm is a genuinely nasty class of bug.

#### The worker is a package; the image is one way to ship it

**Broad hardware compatibility is a project goal** (§1), so the container is *a*
packaging option rather than *the* architecture. `worker-core` is a pip-installable
package, and every delivery path wraps the same code:

| Host | Path | Isolation |
|---|---|---|
| Linux + NVIDIA, third-party host | Container | Full §4.5 hardening |
| Linux + NVIDIA, own hardware | Container or `pip` + systemd | Contributor's choice |
| macOS + Apple Silicon | `pip` + launchd (**no container** — §6.8) | OS-level only |
| Windows + NVIDIA | `pip` + Scheduled Task, or Docker Desktop/WSL2 | Weakest natively |
| Anything else | `pip` | OS-level only |

Windows is supported natively, not only through WSL2 — it's where most consumer GPUs
actually live, and requiring WSL2 is a real hurdle for a volunteer. Note that unlike
macOS, Docker Desktop on Windows *can* reach the GPU through WSL2, so the container
path stays available for anyone who wants it (and is required for restricted runs —
§6.10).

The pinned-digest guarantee (above) applies *within* a backend. A CUDA container fleet
shares one bit-identical PyTorch build; native installs necessarily won't. That is
acceptable because §6.8 Tier 1 establishes that cross-hardware numerical differences
don't meaningfully affect adapter averaging — but it does mean native installs should
pin a **version range** in the package metadata and report the resolved version in
their `compute_profile`, so a genuinely incompatible build is visible rather than
silently averaged in.

### 4.2 Entrypoint loop

```
 1. Read env: GANYMEDE_KEY, COORDINATOR_URL, GANYMEDE_CACHE_DIR
 2. Probe GPU (name, VRAM, driver, compute capability); exit 0 cleanly if none
 3. POST /v1/workers/register → worker_id
    (image tag reconciliation already happened host-side — see §7)
 4. POST /v1/tasks/claim {capabilities}
      → 204 : sleep with jittered backoff, goto 4
      → 200 : task spec (§8)
 5. Verify we can honor the task: base_precision supported, VRAM sufficient,
    required image tag matches ours. If not → POST abandon, exit 0.
 6. Ensure base model in host-persistent HF cache (pull once, reuse forever)
 7. GET base adapter via presigned URL (safetensors)
 8. Train local_steps on assigned buckets, or until max_runtime_sec
      · heartbeat every 60 s with step progress (renews lease)
      · 409 from heartbeat → round closed, drop work, goto 4
      · SIGTERM → POST abandon, exit 0  (see §4.4)
 9. Save adapter as safetensors → presigned PUT → POST submit {ref, metrics}
10. goto 4
```

Note step 3 has no version check. That is deliberate — see Finding E and §7.

### 4.3 The trainer

`transformers` + `peft` + a hand-written loop, ~250 lines. Owns the sync boundary
explicitly:

```python
def run_task(task, base_adapter_bytes, on_step, should_stop):
    model = load_base(task.base_model, precision=task.base_precision)  # cached
    model = attach_lora(model, task.lora_cfg, init_from=base_adapter_bytes)
    opt   = AdamW(lora_params(model), lr=task.hyperparams.lr)  # fresh each round
    data  = load_buckets(task.dataset_ref, task.buckets, seed=task.seed)

    for step, batch in enumerate(data):
        if step >= task.local_steps or should_stop():
            break
        loss = model(**batch).loss
        loss.backward(); opt.step(); opt.zero_grad()
        on_step(step, loss.item())

    return save_lora_safetensors(model), step, metrics
```

Inner optimizer state is **not** carried across rounds — that's standard DiLoCo: fresh
inner optimizer each round, momentum lives only in the outer step on the coordinator.
It also means nothing optimizer-shaped needs to survive preemption.

### 4.4 Preemption handling

On SIGTERM the worker **abandons its lease and exits**. It does not try to checkpoint
and upload under time pressure.

Rationale (Finding D): Docker's default stop grace is 10 s, and platform preemption can
be harder still. A save-and-upload race is likely to lose and, worse, may half-upload.
Abandoning releases the shard for immediate re-lease by another worker — strictly
better for the swarm than a partial artifact of uncertain quality.

`docker run --stop-timeout 120` gives room for a best-effort partial submit *when the
signal arrives early enough*, which is an optimization, not a requirement. The
correctness path is abandon-and-exit.

### 4.5 Isolation (baseline, unchanged in spirit from v1 §3.4)

- `--user` non-root, `--cap-drop=ALL`, `--security-opt=no-new-privileges`
- Read-only rootfs, with explicit writable mounts:
  `/cache/hf` (persistent volume), `/tmp` (tmpfs), `/run/ganymede` (tmpfs)
  — these are required; `transformers`, Triton, and `torch.compile` all write caches
- cgroup limits: `--memory`, `--cpus`, `--pids-limit`
- **Egress allowlist**: coordinator host, artifact-store host, `huggingface.co` +
  `cdn-lfs*.hf.co`. Three destinations, all named, all HTTPS. (Finding C)
  In v1 the first two are the same machine on different subdomains, so the effective
  allowlist is two hosts — but keep them as separate config entries, or moving storage
  to R2 later becomes an edit to every contributor's firewall rules instead of one
  manifest field.

**Isolation posture follows the packaging path**, and that's a deliberate trade rather
than an oversight. A native `pip` install has none of the above — no container, no
cap-drop, no read-only rootfs. That is acceptable when the person installing it owns
the machine and has chosen to run your code; it is *not* acceptable on a third-party
rented host, where the container path stays mandatory.

So: **own hardware may install natively; donated and rented third-party hosts run the
container.** Say this plainly in `INSTALL.md` rather than leaving contributors to
infer it.

Stronger sandboxing (gVisor/Kata) stays deferred to Phase 2 — but note it's the host
that's being protected here, and every v1 host is either yours or a person you vouch
for.

---

## 5. Aggregation

Runs on the coordinator when a round closes. CPU-only; a LoRA adapter is small enough
that this takes seconds.

### 5.1 Acceptance gates (pulled forward from v1 §10)

Every submission passes all of these or is rejected with a recorded reason:

1. Loads as **safetensors** — never `torch.load` (Finding G)
2. Key set and tensor shapes exactly match the round's expected LoRA config
3. dtype matches; no NaN, no Inf
4. Per-tensor Frobenius norm within `k×` the cohort median for this round
   (`k = 5` initially; tune from observed spread)
5. `steps_completed > 0` and consistent with the last heartbeat

Gates 1–3 catch bugs, which during bring-up is the overwhelmingly common failure. Gate
4 catches divergence — a bad LR, a broken shard. Rejections are logged per contributor,
which is also the raw material for Phase 2 reputation scoring.

### 5.2 Combine

```
w_i[t]   = weight of worker i for tensor t,  Σ_i w_i[t] = 1     # see below
A_mean[t]= Σ_i w_i[t] · A_i[t]                                  # per-tensor mean
outer_g  = A_base − A_mean                                      # pseudo-gradient
m        ← β·m + outer_g                                        # per run, β = 0.9
A_next   = A_base − lr_outer · (outer_g + β·m)                  # Nesterov
```

**Weights are per-tensor, not per-worker — write it this way from the start.** For
dense models every tensor gets the same scalar:

```
w_i[t] = steps_completed_i / Σ steps_completed      # identical for all t
```

which is exactly the straggler-tolerant weighting you want, and behaves identically to
a scalar implementation. The reason for the extra index is §5.4: it is the one thing
that would otherwise have to be rewritten to support MoE, and it costs nothing today.

Guard: if `Σ_i w_i[t] == 0` — no worker contributed to tensor `t` — pass `A_base[t]`
through **unchanged**. Never divide by zero, and never let an untouched tensor decay
toward zero.

Two things worth stating plainly:

- **Outer momentum `m` is durable coordinator state**, persisted alongside the run
  (Finding L7). Losing it isn't fatal but does cost convergence progress.
- **Record inter-worker divergence at this point** — pairwise distance across the
  `A_i` before they're averaged. It measures precisely what DiLoCo trades away: more
  local steps means more drift means a less meaningful mean. Growing drift across
  rounds says `local_steps` is too high, which is otherwise very hard to see.
- **Averaging LoRA adapters is better-behaved than averaging full weights**, because
  every adapter is a delta against the *same frozen base*. This is exactly why
  `base_precision` must be pinned (Finding J) — heterogeneous quantization silently
  breaks the shared-base assumption while leaving shapes compatible.

**Research risk, stated honestly:** DiLoCo is characterized in the literature over
full-parameter training. Applying the outer-momentum step to LoRA adapters is a
variant. The plain weighted mean (`lr_outer = 1`, `β = 0`) is the conservative
fallback and reduces to straightforward federated averaging. **M4 must A/B these two
against a single-node baseline** rather than assuming the momentum variant helps.

### 5.3 The `SyncBackend` seam

The entire sync layer reduces to two methods:

```python
class SyncBackend(Protocol):
    def get_base(self, round_id: str) -> bytes: ...
    def publish(self, round_id: str, adapter: bytes, weight: float) -> None: ...
```

`CentralBackend` implements these with presigned GET/PUT plus a submit call. A future
`HivemindBackend` implements the same two methods over a DHT and averaging group. The
worker's training code never learns which is in use — that is the whole point of the
seam, and it's cheap to maintain precisely because the interface is this narrow.

---

### 5.4 Forward compatibility: what MoE would need

Nothing in this design forecloses MoE. It's worth recording exactly what would and
wouldn't have to change, because the answer determines one decision worth taking now
(§5.2's per-tensor weights) and leaves the rest genuinely deferrable.

#### Works today, unchanged: MoE base + attention-only LoRA

If LoRA targets only the attention projections (`q/k/v/o`) and leaves the experts and
the router frozen, **there is no MoE interaction at all**. Every worker computes deltas
against the same frozen function, routing is identical everywhere, and every adapter
tensor receives gradient from every token. That is precisely the dense case, and it
runs on the current design with a `lora_cfg` change and nothing else.

So if MoE becomes attractive before the expert-training questions are settled, this
path is available immediately. Note only that MoE bases are large in *total* params
even when active params are small — a 30B-A3B needs ~60 GB in bf16 — so §6.8's VRAM
eligibility will correctly restrict such a run to datacenter-class hardware.

#### The real problem: LoRA on the experts themselves

Token routing is data-dependent, and that is what breaks uniform averaging:

- Worker A's shard routes mostly to experts {3, 7, 12}; worker B's to {1, 3, 19}.
- Worker A's adapter for expert 7 receives real gradient. Worker B's stays at
  initialization — a zero delta.
- A uniform mean divides expert 7's genuine learning by the worker count. **Sparsely
  activated experts get systematically diluted**, and the more experts relative to
  workers, the worse it gets.

This isn't invalid, it's *mis-weighted* — which is why the fix is a weighting change
rather than a redesign:

```
w_i[t] ∝ tokens routed by worker i to the expert owning tensor t
```

That requires per-tensor weights (§5.2, already provided for) and per-expert token
counts from workers. The submit body's `metrics` is free-form and `submissions.
metrics_json` is unstructured, so **carrying that data needs no schema change either**.

#### Three softer things, none of them blockers

- **Acceptance gate 2** (§5.1) requires an exact key set. Keep it strict and adopt the
  convention that untrained experts submit explicit zero tensors rather than omitting
  keys — zeros compress to almost nothing, and a strict gate is worth more than the
  bytes.
- **Momentum on untouched experts.** `m` is already per-tensor, so nothing breaks, but
  momentum will keep pushing an expert that received no update this round. Whether to
  decay or freeze it is a tuning question to answer with data, not a design constraint.
- **Router training** is the genuinely hard part and should stay out of scope longest.
  Averaging routers trained on different shard distributions can produce a router worse
  than any input, and it breaks consistency: the expert deltas being averaged were
  computed under *different* routing than the averaged router will produce. Freeze the
  router; revisit only with evidence.

#### One thing already working in MoE's favour

Bucket sharding uses a stable hash (§3.2), so shards are effectively random rather than
topically clustered. That means every worker sees a broadly similar routing
distribution, which *minimizes* the dilution problem above. Topical sharding would
specialize workers to experts and make averaging much worse — so the existing default
is the MoE-friendly one, and worth not changing casually.

---

## 6. Coordinator

FastAPI + SQLite on one small VM. No GPU.

### 6.1 Schema

```
contributors  id, name, key_hash, enabled, clearance, created_at  -- §6.10
workers       id, contributor_id, compute_profile_json, image_tag,
              first_seen, last_seen, rounds_joined, steps_total       -- §6.8
              -- availability is measured here, never declared up-front
runs          id, status, base_model, base_precision, lora_cfg_json,
              dataset_ref, hyperparams_json, current_round, target_rounds,
              outer_momentum_ref, requires_json, data_classification,
              created_at
              -- requires_json: { backend, min_vram_mb, needs: [...] }  §6.7
              -- data_classification: open | internal | restricted     §6.9
rounds        run_id, idx, base_adapter_ref, status, target_steps,
              min_round_sec, max_round_sec, opened_at, closed_at,
              result_adapter_ref, distinct_contributors,             -- §3.2
              eval_loss, adapter_divergence                          -- §5.2
tasks         id, run_id, round_idx, buckets_json, local_steps, status,
              worker_id, lease_expires_at, attempts, created_at
submissions   task_id, artifact_ref, steps_completed, tokens_seen,
              metrics_json, accepted, reject_reason, received_at
buckets       run_id, bucket_idx, times_trained, last_round
throughput    run_id, gpu_model, steps_per_min, samples, updated_at   -- §3.4
calibration   run_id, calibration_json, created_at                    -- §6.7, M0
```

**Concurrency (Finding L2):** WAL mode, and every claim wrapped in `BEGIN IMMEDIATE`
so two simultaneous claims can't lease the same task. Good to a few hundred workers;
past that, move to Postgres. Write the DB access behind a thin module so that swap
stays a day of work rather than a rewrite.

### 6.2 API

```
GET  /v1/manifest
     → { image_tags: {llm_finetune: "ganymede/worker-llm:v3"},
         min_worker_core: "v2", object_store_host: "...", active_runs: [...] }

POST /v1/workers/register
     body { compute_profile, image_tag }
     → { worker_id, heartbeat_interval_sec }

     compute_profile = { backend: "cuda" | "mps" | "rocm" | "cpu",
                         device_name, vram_mb, compute_capability,
                         driver, torch_ver, package_version,
                         supports: ["bf16", "fp16", "nf4", "flash_attn"],
                         probe: { alloc_max_mb, bench_score, probed_at } }

POST /v1/tasks/claim
     body { worker_id, capabilities, cached_base_models: [...] }
     → 200 { task spec — §8 }
     | 204 No Content + Retry-After   (no eligible work)

POST /v1/tasks/{id}/heartbeat
     body { steps_completed, loss_ewma }
     → 200 { lease_expires_at }
     | 409 Conflict  (round closed — drop work, re-claim)
     | 410 Gone      (lease lost to expiry — drop work, re-claim)

POST /v1/tasks/{id}/upload-url   → { url, key, expires_at }   presigned PUT
POST /v1/tasks/{id}/submit
     body { artifact_key, steps_completed, tokens_seen, metrics }
     → 200 { accepted, next_action } | 409 (round closed) | 422 (gate failure)
POST /v1/tasks/{id}/abandon      → 200   releases lease immediately

GET  /v1/runs/{id}/rounds/current → { idx, status, base_adapter_url, deadline_at }
GET  /v1/fleet                    → derived inventory (§6.11)
GET  /healthz    GET /metrics     GET /status   (read-only HTML)
```

Differences from v1 §4.3: capability-aware claim (L3), explicit `204` for no work
(L4), `409` vs `410` distinguished, `upload-url` and `abandon` added, artifact bytes
gone from every request body.

`cached_base_models` exists because base models are ~16 GB each and Ganymede will run
many of them (§6.7). Given two eligible runs, the coordinator should prefer the one
whose base model the worker already holds — that's the difference between starting
work in seconds and starting it after a 16 GB download.

`compute_profile` replaces v1's loose capability fields. `backend` and `supports` are
what make §6.8's eligibility rules expressible: a run requiring `nf4` must not be
offered to a worker that can't provide it, and silently falling back to a different
precision would break §5.2's shared-frozen-base assumption without erroring.

**`probe` is measured, not declared** — see §6.9. Given the goal of running on almost
any hardware, the coordinator cannot hold a table of known devices; it has to accept
what the worker demonstrates.

### 6.3 Auth (Finding F)

- `Authorization: Bearer <GANYMEDE_KEY>`, TLS mandatory, plain HTTP refused outright
- Key = 32 random bytes, base64url. Stored as a hash; the plaintext exists once, at
  issue time
- One key per contributor. Revoke by setting `enabled = false`
- Rate limit per key. A wedged worker in a hot loop shouldn't be able to take the
  coordinator down
- Every submission records its contributor — the audit trail that Phase 2 reputation
  scoring will need, gathered from day one at no cost

### 6.4 Failure behavior (replacing v1 §9's over-claim)

| Failure | Actual consequence |
|---|---|
| Worker preempted mid-round | Lease expires or is abandoned; shard re-leased. Lost: that worker's steps this round (≤ ~45 min by §3.3) |
| Worker submits garbage | Rejected by §5.1 gates, recorded, round proceeds with remaining submissions |
| Round misses its step target | Closes at `max_round_sec` with ≥1 submission. Zero submissions → reopens quietly with a fresh deadline; an idle fleet is normal, not an error (§3.2) |
| Coordinator down | No claims, no submissions, no aggregation. In-flight workers finish, fail to submit, retry with backoff. **Nothing is lost that was already submitted** |
| Artifact store down | Same as coordinator down, from the worker's side. In v1 they share a VM, so in practice these fail together |
| Coordinator VM lost | **The failure that matters.** In v1 the database and every checkpoint are on one machine, so this loses both unless backups are genuinely off-box. See §6.6 — separate volume, off-box SQLite dump each round close, off-box copy of the latest round adapter |
| Artifact volume lost, VM survives | Recoverable from the off-box dump + latest adapter: resume the run rather than restart it. Loses older checkpoint history |

### 6.5 Deployment

The coordinator is **containerized from day one**, because it moves through three
environments and should be the same artifact in all of them:

| Stage | Shape |
|---|---|
| Development | `docker compose up` — coordinator + MinIO + N fake workers, all local |
| Bring-up | Same compose file on your server, reachable on the LAN |
| Production | Same image, own container on your server, behind two subdomains |

"Emulated for now" is therefore just the first column, and M1's fake-worker harness is
the emulation. Nothing needs rewriting when it moves to the server — the compose file
gains a reverse-proxy front end and stops binding to localhost.

**Two subdomains, not one** — path-routing both services under one host fights the S3
client, which expects to own its URL space. Concrete names are deployment config, not
architecture; everything below refers to them as variables:

| Variable | Meaning |
|---|---|
| `COORDINATOR_HOST` | Public host for the FastAPI app |
| `STORAGE_HOST` | Public host for MinIO |
| `S3_BUCKET`, `S3_REGION` | Bucket and region (`auto` on R2) |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | Store credentials |
| `BACKUP_ENDPOINT` | Off-box backup target (§6.6) — must not resolve to the same machine |
| `GANYMEDE_KEY` | Per-contributor bearer token, worker side (§6.3) |

Nothing in the codebase should contain a hostname. The same compose file runs on
localhost, on the LAN, and in production with only these values changing.

**Use path-style S3 addressing** (`STORAGE_HOST/bucket/key`), not virtual-host style
(`bucket.STORAGE_HOST/key`). Virtual-host style needs wildcard DNS and a wildcard
certificate for what is one bucket. Path-style is one A record and one cert:

```python
boto3.client("s3", endpoint_url=..., config=Config(s3={"addressing_style": "path"}))
```

R2 supports path-style too, so this costs nothing in portability (§6.6).

**TLS terminates at your existing reverse proxy**, which makes the presigned-URL host
footgun in §6.6 a certainty rather than a risk — the coordinator will be signing for
`STORAGE_HOST` while talking to MinIO over the internal network. Set MinIO's
`MINIO_SERVER_URL` to `STORAGE_HOST` and sign with that exact value.

### 6.6 Artifact storage — self-hosted now, portable later

**Decision: self-hosted MinIO on the coordinator VM for v1, with a documented path to
Cloudflare R2 or plain S3.**

#### Why this is a good default, not a concession

The instinct is to read self-hosting as the cheap option you tolerate until you can
afford managed storage. For *this* access pattern it's the other way round. Every
worker pulls a base adapter at the start of every round and pushes one at the end, so
cost scales with egress, which is exactly where object storage is priced worst:

| | 10 workers | 50 workers |
|---|---|---|
| Egress volume | ~2.5 TB/mo | ~12.4 TB/mo |
| Self-hosted (VM with 20 TB included) | included | included |
| Cloudflare R2 (no egress fee) | $0 egress | $0 egress |
| S3-class storage at ~$0.09/GB egress | ~$225/mo | ~$1,100/mo |

*List-price illustrations, not quotes. Assumes ~85 MB per adapter — the conservative
all-linear-modules figure; the q/k/v/o config in §8 is closer to 27 MB — and 30-minute
rounds.*

So the real ordering is: self-hosted and R2 are both fine, S3-class egress pricing is
the one to avoid. Self-hosting first is the right call, and R2 stays the escape hatch
if you'd rather not operate storage than not pay for it.

#### Can one small VM actually serve this?

Yes, comfortably, at v1 scale. At 10 workers on 30-minute rounds: ~3.4 GB/hour
sustained, about **7.6 Mbit/s average**. The load is bursty rather than steady — when
a round opens, every worker wants the base adapter at once, a ~850 MB thundering herd.
On a 1 Gbit/s link that's ~7 seconds; on 100 Mbit/s, ~70 seconds.

Both are acceptable, but **jitter the claim response** so workers don't all start
their fetch on the same tick. §4.2's backoff already jitters the `204` path; apply the
same to round-open.

This scales linearly, so it's worth naming where it stops being free: past roughly
**50 workers**, or if the VM's monthly egress allowance is tight, revisit. That's the
R2 trigger — not an arbitrary future date.

#### The portability contract

The point of S3-compatibility is that migration should be **configuration, not code**:

```
GANYMEDE_S3_ENDPOINT=https://${STORAGE_HOST}    # → R2's endpoint
GANYMEDE_S3_BUCKET=${S3_BUCKET}
GANYMEDE_S3_REGION=${S3_REGION}                 # → "auto" for R2
GANYMEDE_S3_ACCESS_KEY / _SECRET_KEY
```

`boto3` against MinIO and `boto3` against R2 are the same client with a different
`endpoint_url`. To keep it that way, **stay inside the S3 subset every implementation
supports**: PUT / GET / HEAD / DELETE object, `list-objects-v2`, presigned URLs,
multipart upload.

Avoid depending on: object versioning, lifecycle rules, object lock, storage classes,
and server-side-copy-heavy patterns. Implementations diverge there, and each one you
use is a migration blocker you won't notice until you try to move. Where you'd reach
for a lifecycle rule, write the GC job instead (below) — it's a cron entry, and it
works identically everywhere.

Verify against the target provider's current S3-compatibility documentation before any
swap; the compatible surface shifts over time.

#### The presigned-URL host footgun

Presigned signatures cover the **host**. If MinIO sits behind the same reverse proxy
that terminates TLS for the coordinator, the coordinator must sign for the *public*
endpoint (`https://${STORAGE_HOST}`), not for MinIO's internal address.
Get this wrong and every worker fetch fails signature validation with an error that
does not obviously say "you signed the wrong hostname."

Set MinIO's `MINIO_SERVER_URL` to the public endpoint and sign with the same value.
Worth an explicit integration test in M1 — it's a five-minute fix and a two-hour
debugging session.

#### Retention and garbage collection

Self-hosting means the disk is finite and nobody else is managing it. Two classes of
object with different lifetimes:

- **Round result adapters** — keep all. This is the checkpoint history, and it's what
  makes §6.4's "nothing submitted is lost" true. ~85 MB × rounds.
- **Individual worker submissions** — delete once the round has closed and aggregated
  successfully, keeping ~3 rounds of grace for debugging a bad aggregate.

Steady state at 10 workers over 100 rounds: ~8.5 GB of results + ~3.4 GB of in-flight
submissions + the dataset. **A 200 GB volume is generous.** Note the 16 GB base model
is deliberately *not* stored here — workers pull it from HuggingFace into a
host-persistent cache (§4.1). Keep it that way; putting it in the artifact store would
multiply egress by an order of magnitude for no benefit.

The GC job runs on round close, alongside the backup.

#### Backups — the one thing this decision genuinely breaks

§6.4 previously said: back up SQLite to object storage after every round close. **With
storage co-located on the coordinator VM, that backup is worthless for the failure it
was written to survive.** Losing the disk loses the database and every checkpoint
together.

So the rule for v1 is stricter than it was:

1. **The artifact store lives on a separate volume from the OS disk.** Cheap, and it
   turns "VM died" into a recoverable event.
2. **SQLite dumps go off-box, every round close.** The file is kilobytes-to-megabytes;
   any external target covers it indefinitely, and a free object-storage tier is a
   perfectly good destination. This is also the natural place to start an R2 account
   long before you migrate primary storage to it.
3. **The latest round result adapter replicates off-box too.** ~85 MB per round. If
   everything else burns down, the database plus the newest adapter is enough to
   resume the run rather than restart it.

Off-box means a different machine, not a different directory. That distinction is the
entire value of the item.

### 6.7 Running many models

Ganymede is intended to carry a lot of runs over time, not one. Three consequences
that don't exist with a single run:

**Base-model cache pressure on contributors.** Each base model is ~16 GB in the
host-persistent HF cache (§4.1). Ten runs across five base models means a contributor's
disk fills quietly until something breaks at an unhelpful moment. Required:

- A cache size cap with LRU eviction, configured host-side and defaulting to something
  conservative
- A stated **minimum free disk** in `INSTALL.md` — the contributor-facing number, and
  the one people will actually check before volunteering a machine
- Workers report `cached_base_models` on claim (§6.2) so the coordinator can prefer
  work they can start immediately

**Cache affinity beats naive scheduling.** Handing a worker a run whose base model it
lacks costs a 16 GB pull before any useful work happens — potentially longer than the
round itself. Prefer cached matches; fall back to a cold pull only when there's no
cached work available, or when a run would otherwise starve.

**Calibration is per-run, not global.** Round sizing, GPU eligibility, and the M4
comparison all depend on the specific base model, precision, sequence length, and LoRA
config. §3.4's step budgets read from a per-run `calibration.json`, which is why M0
builds calibration as a repeatable command rather than a one-time measurement.

### 6.8 Hardware heterogeneity

The expected fleet spans Apple Silicon laptops, 12 GB consumer cards, and datacenter
A100s. Speed differences are handled by §3.4 and are not a problem. **Capability
differences are**, and they fall into three tiers.

#### Tier 1 — different speed, same capability (3060 ↔ 3090 ↔ 4090 ↔ A100)

Not a problem. All CUDA, all Ampere-or-later, all bf16-capable. Per-worker step
budgets absorb a 20× spread without stragglers.

Worth knowing but not worth acting on: different GPUs produce slightly different
numerics (kernel selection, reduction order, TF32 accumulation). This does **not**
meaningfully affect adapter averaging — the variance between workers from training on
*different data shards* dwarfs it, which is the same reason DiLoCo works at all.

One cheap precaution: **pin TF32 matmul settings in the run config**. TF32 changes the
forward pass through the frozen base, so leaving it to each worker's defaults is a
mild version of the precision-mismatch problem below. One line to fix, annoying to
diagnose.

#### Tier 2 — VRAM decides which runs a worker can serve

A 12 GB 3060 and an 80 GB A100 aren't the same class of participant. For an 8B model:

| | bf16 base (~16 GB weights) | nf4 base (~5.5 GB weights) |
|---|---|---|
| 3060 12 GB | no | yes, short sequences |
| 3090 / 4090 24 GB | tight | comfortable |
| A100 40/80 GB | comfortable | comfortable |

This is ordinary capability filtering (§6.2) — a run declares required precision and
minimum VRAM, workers declare what they have, the coordinator matches. It's already in
the design.

The consequence to internalize: **not every worker serves every run, and that's
normal, not a failure.** A 3060 fleet can carry small models and quantized larger
ones; an A100 can carry anything.

#### Tier 3 — Apple Silicon is a different platform, not a slower GPU

This is the structural one, and it's worth being blunt: **the container-based worker
does not work on a Mac at all.**

- **No GPU in containers.** Docker on macOS runs a Linux VM with no MPS passthrough. A
  containerized worker on a Mac gets CPU only. This is a platform property, not a
  configuration problem — there is no flag that fixes it.
- **No nf4.** `bitsandbytes` is CUDA-oriented; quantized bases should not be assumed
  available on MPS. So a Mac **cannot join any nf4 run** — and the Tier-2 table
  above shows nf4 is exactly what makes larger models reachable for small cards. The 3060
  needs nf4; the Mac can't do nf4. They are mutually exclusive on the same run.
- **No flash-attn**, and MPS operator coverage and bf16 behavior lag CUDA.
- **Unified memory cuts both ways.** An M2 Air at 8–16 GB is not doing 8B. An M2/M3
  Max at 64 GB+ genuinely can hold it — but MPS training throughput is far below a
  comparable discrete GPU, so §3.4's floor may exclude it from larger runs anyway.

**Apple Silicon is supported in v1, via the native package path** (§4.1). A Mac joins
the platform, registers, probes, and appears in the fleet like anything else. It is
simply ineligible for nf4 runs — which is the intended shape: *the platform is
compatible; a given run may not be.*

Two deliberate limits, worth stating so nobody is surprised:

- **PyTorch MPS, not MLX, for v1.** MLX is Apple's own framework and is the genuinely
  good path for training on this hardware, but it's a second trainer implementation
  interoperating through safetensors — a real build. PyTorch MPS is slow and works,
  which is what compatibility requires. Revisit MLX only if Mac *throughput* becomes
  the point rather than Mac participation.
- **Expect modest contribution.** An M2 Air won't clear §3.4's floor on an 8B run. An
  M-series Max with 64 GB+ will participate in bf16 runs at a fraction of a discrete
  GPU's rate. That's fine — it's a contributor who can participate.

#### Why this raises the value of concurrent runs

With **sequential** runs — the v1 decision — a run that requires nf4 excludes every
Mac, and a bf16 8B run excludes every 3060. Those contributors idle until the next run
happens to suit them.

Wide hardware diversity therefore makes concurrency worth more than it would be
otherwise: two active runs with different requirements keep the whole fleet busy where
one cannot. Sequential is still right for v1 — it avoids scheduling, priority, and
starvation entirely — but the eligibility model above is deliberately written so that
turning on concurrency later is a scheduling change, not a redesign.

---

### 6.9 Capability probing, not hardware enumeration

Supporting almost any hardware rules out the obvious implementation. The coordinator
cannot keep a table mapping device names to capabilities — the table would be wrong
the first time someone shows up with a card nobody anticipated, and the failure would
be a silently-excluded contributor rather than a loud error.

**So the worker measures and reports, and the coordinator believes it.** On
registration, and whenever the package or driver changes, the worker runs a short
self-test (target: under a minute):

1. **Allocation ceiling** — how much device memory can actually be reserved. This is
   the number that matters, not the spec-sheet figure: unified memory is shared with
   the OS, and a desktop card may already be driving displays.
2. **Precision support** — attempt a small bf16 matmul, an fp16 matmul, and an nf4
   load. Record what genuinely worked rather than what the hardware theoretically
   supports.
3. **Benchmark score** — a fixed tiny transformer forward+backward at a set shape,
   producing a normalized throughput number.

That last item does real work for §3.4. It replaces "conservative default for an
unseen GPU class" with an actual measurement, so a brand-new device gets a sensible
step budget on its *first* round rather than after one wasted one.

Consequences worth naming:

- **Registration always succeeds.** A CPU-only machine, or a 6 GB card, registers
  fine and simply never matches a run's requirements. Turning someone away at the door
  is the wrong behavior for a platform whose goal is broad compatibility — and it
  costs nothing to let them sit in the fleet until a run suits them.
- **New hardware needs no coordinator change.** Someone appears with an Arc card or a
  ROCm box; if the probe passes and the numbers are good, they're eligible. No
  allowlist to update, no deploy.
- **The probe is also the diagnostic.** When a contributor asks why they're never
  getting work, the answer is in their stored profile: not enough memory, no nf4, or
  below the floor. That is a far better support experience than silence.

**Transfer rates are measured too, not probed.** The worker times its own artifact
download and upload each round and reports both. After one round the coordinator knows
that machine's real-world bandwidth, which feeds the step budget (§3.2). This is
strictly better than asking a contributor for a number they'd have to look up and
would often get wrong.

Together, the probe plus these observations mean **the fleet inventory is something
the system produces, not something anyone maintains** — see §6.11.

### 6.10 Per-run data classification

Data sensitivity varies by run, and **every eligible contributor receives the dataset
in plaintext** — now including personal laptops, not only machines you administer. So
sensitivity becomes a second eligibility dimension alongside capability (§6.8):

```
eligible(worker, run) = capability_match(worker, run)
                      AND contributor.clearance >= run.data_classification
```

Three classifications, with different rules:

| Class | Who may serve it | Packaging | Dataset cache |
|---|---|---|---|
| `open` | any registered contributor | any path | may persist |
| `internal` | contributors who have accepted an agreement | any path | may persist |
| `restricted` | named contributors on hosts you administer | **container only** | **wiped on task exit** |

Three consequences worth building in rather than bolting on:

- **`restricted` runs forbid the native install path.** Not because native is
  insecure per se, but because the container is what makes "wipe the dataset when the
  task ends" enforceable rather than aspirational.
- **Cache retention is per-classification.** `open` and `internal` datasets may sit in
  the host cache between rounds — that's a real speed win. `restricted` ones are
  removed when the task exits, accepting the re-download cost.
- **Clearance is recorded per contributor, not per machine.** §6.3 already stores a
  contributor row per key; this is one more column on it.

This is cheap now and awkward later: retrofitting classification means auditing which
datasets already reached which machines, which is a question with no good answer once
it's been asked.

### 6.11 The fleet inventory is derived, not maintained

There is no roster to keep current. Every fact worth knowing about a contributor
machine is either probed on registration (§6.9) or observed while it works, so the
inventory is a **view the coordinator renders**, never a document that drifts out of
date:

| Fact | Source |
|---|---|
| OS, backend, device, VRAM | Probe, at registration (§6.9) |
| Real allocation ceiling, precision support | Probe — measured, not spec-sheet |
| Compute throughput | Probe benchmark, then refined from each round's metrics |
| Upload / download bandwidth | Timed during the worker's own transfers (§6.9) |
| Availability | Observed `last_seen` and `rounds_joined`. Never declared (§3.2) |
| Contributor, clearance | The key it authenticates with (§6.3, §6.10) |

`GET /v1/fleet` renders it: who is registered, what they can do, when they were last
seen, how much they've contributed, and — for anyone not currently eligible — why.

Two properties this buys:

- **Hardware nobody anticipated just works.** No allowlist to update, no deploy.
- **Nothing to keep in sync.** A contributor who upgrades a GPU re-probes and the
  inventory is correct; nobody has to remember to edit anything.

---

## 7. Host agent

Runs on the contributor's machine, outside the container. Owns everything about
*whether and what* to run; the worker owns only *doing the work* (Finding E).

```
systemd timer, every N minutes:
  1. Already running a Ganymede container for this GPU?     → exit
  2. IdleBackend.is_idle()?                                 → if no, exit
  3. GET /v1/manifest; if required image tag != local, pull it
  4. docker run --gpus all --stop-timeout 120 \
       --user ganymede --cap-drop=ALL --security-opt=no-new-privileges \
       --read-only --tmpfs /tmp --tmpfs /run/ganymede \
       -v ganymede-hf-cache:/cache/hf \
       --memory=... --cpus=... --pids-limit=... \
       -e GANYMEDE_KEY -e COORDINATOR_URL \
       ganymede/worker-llm:<tag>
```

### 7.1 `IdleBackend`

```python
class IdleBackend(Protocol):
    def is_idle(self) -> bool: ...
```

- **`local`** (v1, own hardware): no non-Ganymede process in
  `nvidia-smi --query-compute-apps`, AND no `/etc/ganymede/pause` file, AND (optional)
  inside a configured time window.
- **`vast`**, **`tensordock`** (later): query the platform API for active rentals on
  this GPU.

`/etc/ganymede/pause` is the contributor's kill switch. It must work with no network,
no coordinator, and no explanation required — `touch` the file and Ganymede stops
taking the GPU. Document it prominently; it's what makes the ask reasonable.

**Verify before the rented-host phase:** whether running your own workload on a listed
GPU affects platform reliability scoring (Review Finding M).

---

## 8. Task spec (replaces v1 §5)

```json
{
  "task_id": "uuid",
  "run_id": "uuid",
  "round_idx": 42,
  "job_type": "llm_finetune",
  "required_image": "ganymede/worker-llm:v3",

  "base_model": "meta-llama/Meta-Llama-3.1-8B",
  "base_precision": "nf4",
  "lora_cfg": {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "base_adapter_url": "https://…presigned-GET…",

  "dataset_ref": "s3://ganymede/data/combat-robot-sim-v2",
  "buckets": [17, 143, 288, 401, 655],   // count scales with the budget — §3.4
                                         // bucket total scales with dataset size

  "hyperparams": {
    "lr": 2e-4,
    "seq_len": 2048,
    "micro_batch": 2,
    "grad_accum": 8,
    "seed": 20260825
  },

  "local_steps": 250,          // derived per-worker from throughput — §3.4
  "max_runtime_sec": 2700,     // safety net, not the primary control
  "heartbeat_interval_sec": 60,
  "lease_seconds": 3600
}
```

Changes from v1 §5: round binding, bucket sharding (I), pinned `base_precision` (J),
explicit `lora_cfg` (needed by the §5.1 shape gate), presigned base adapter URL, seed
(L6), lease and heartbeat intervals (L1), and a per-worker `local_steps` budget rather
than a run-wide constant (§3.4).

**Still true, and still the point:** everything here except `required_image` is
mutable without a rebuild. v1 §8's change-management table holds unchanged — with the
improvement that architecture changes now mean "start a new run" rather than a manual
swarm wind-down.

---

## 9. Deferred to Phase 2

Unchanged from v1 §10, minus two items promoted into v1:

- ~~Bounds/sanity checks on submitted weights~~ → **now §5.1**
- ~~(implicit) safe deserialization~~ → **now safetensors-only, §5.1 gate 1**

Still deferred: spot-check redundancy, reputation scoring (the data for which
accumulates from day one via §6.3), gVisor/Kata sandboxing, TOPLOC-style verification,
Byzantine-robust aggregation (trimmed-mean/median — note §5.2's weighted mean is a
drop-in replacement point).

Also Phase 2: `rl_rollout`, and the Hivemind `SyncBackend`.
