# Ganymede — Architecture v2

**Status:** Draft v2, incorporating `01-spec-review.md`
**Scope:** `llm_finetune` only (LoRA on a 7–8B base). `rl_rollout` deferred to Phase 2.
**Sync:** Central aggregation, behind a `SyncBackend` seam for a later Hivemind swap.
**Trust:** Trusted circle, with two items pulled forward from v1 §10 (safetensors, sanity gates).

Supersedes `00-original-spec-v1.md`. Sections that are unchanged in substance are
summarized rather than restated.

---

## 1. What changed from v1

| v1 | v2 | Why |
|---|---|---|
| Job queue | **Run → round → task** | Review Finding A |
| `worker-core` as shared base | **`torch-base` as shared base**, core above it | Finding B |
| Egress: coordinator + bootstrap | Egress: coordinator + object store + HuggingFace | Finding C |
| Weights in POST body | Presigned PUT; coordinator sees only refs | Finding H |
| Hivemind DHT | Central aggregator behind `SyncBackend` | Decision |
| Axolotl / torchtune | Owned ~250-line `peft` trainer | Finding K |
| `.pt` checkpoints | **safetensors everywhere** | Finding G |
| Worker checks image version | Host agent checks; worker declares capabilities | Finding E |
| SIGTERM → save partial, submit | SIGTERM → **abandon lease**, exit; partial submit best-effort | Finding D |
| (absent) | Dataset bucket sharding | Finding I |
| (absent) | Pinned `base_precision` | Finding J |
| (absent) | Bearer auth | Finding F |

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
  │                                      │ refs only, never bytes
  │                                      ▼
  │                      ┌──────────────────────────────┐
  │                      │  Object store (S3-compatible)│
  │                      │   · round base adapters      │
  │                      │   · worker submissions       │
  │                      │   · sharded dataset buckets  │
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

The coordinator is the only always-on component and remains GPU-free. It never
handles artifact bytes — only references, metrics, and control flow — which is what
keeps it on a small VM.

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
 ┌─────────┐  quorum met OR    ┌─────────────┐  gates+   ┌──┴────────┐
 │  OPEN   │  deadline+≥1 sub  │ AGGREGATING │  outer    │  CLOSED   │
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
4. The round closes when `submissions ≥ quorum_min`, or when the deadline passes with
   at least one submission. A submission arriving after close gets `409 Conflict`
   carrying the current round number; the worker re-claims into it and moves on.
5. Aggregation runs (§5). The result becomes round N+1's base.

### 3.2 Why this solves several problems at once

- **Averaging is well-defined.** Every contribution to round N provably started from
  the same weights.
- **Version skew is structural.** A worker holding a stale base can only submit to a
  closed round, which is rejected by construction. v1 §8's "wind down the swarm before
  an architecture change" becomes: start a new run. No coordination ritual.
- **Stragglers degrade gracefully.** Weighting by steps actually completed (§5) means
  a slow worker contributes proportionally less rather than blocking or corrupting.
- **Preemption costs exactly one round.** Which is why round sizing is a safety
  parameter, not just a tuning knob (§3.3).

### 3.3 Round sizing

Target **30–45 minutes of work per round** on the slowest supported GPU.

- Too long: a flaky host may be preempted before it ever completes a round, so it
  contributes nothing, ever.
- Too short: aggregation overhead and the ~170 MB per-worker round trip dominate the
  useful compute.

`local_steps_per_sync` should be derived from measured throughput during M0 (see
`03-roadmap.md`), not guessed. Recalibrate when the slowest supported card changes.

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
- **Egress allowlist**: coordinator host, object-store host, `huggingface.co` +
  `cdn-lfs*.hf.co`. Three destinations, all named, all HTTPS. (Finding C)

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
w_i      = steps_completed_i / Σ steps_completed        # straggler-tolerant weighting
A_mean   = Σ w_i · A_i                                  # weighted mean of adapters
outer_g  = A_base − A_mean                              # pseudo-gradient
m        ← β·m + outer_g                                # persisted per run, β = 0.9
A_next   = A_base − lr_outer · (outer_g + β·m)          # Nesterov
```

Two things worth stating plainly:

- **Outer momentum `m` is durable coordinator state**, persisted alongside the run
  (Finding L7). Losing it isn't fatal but does cost convergence progress.
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

## 6. Coordinator

FastAPI + SQLite on one small VM. No GPU.

### 6.1 Schema

```
contributors  id, name, key_hash, enabled, created_at
workers       id, contributor_id, gpu_name, vram_mb, driver, torch_ver,
              image_tag, first_seen, last_seen
runs          id, status, base_model, base_precision, lora_cfg_json,
              dataset_ref, hyperparams_json, current_round, target_rounds,
              outer_momentum_ref, created_at
rounds        run_id, idx, base_adapter_ref, status, quorum_min, deadline_at,
              opened_at, closed_at, result_adapter_ref, mean_loss
tasks         id, run_id, round_idx, buckets_json, local_steps, status,
              worker_id, lease_expires_at, attempts, created_at
submissions   task_id, artifact_ref, steps_completed, tokens_seen,
              metrics_json, accepted, reject_reason, received_at
buckets       run_id, bucket_idx, times_trained, last_round
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
     body { gpu_name, vram_mb, driver, torch_ver, image_tag }
     → { worker_id, heartbeat_interval_sec }

POST /v1/tasks/claim
     body { worker_id, capabilities }
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
GET  /healthz    GET /metrics     GET /status   (read-only HTML)
```

Differences from v1 §4.3: capability-aware claim (L3), explicit `204` for no work
(L4), `409` vs `410` distinguished, `upload-url` and `abandon` added, artifact bytes
gone from every request body.

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
| Round misses quorum | Closes on deadline with ≥1 submission. Zero submissions → round reopens with a fresh deadline |
| Coordinator down | No claims, no submissions, no aggregation. In-flight workers finish, fail to submit, retry with backoff. **Nothing is lost that was already submitted** |
| Object store down | Same as coordinator down, from the worker's side |
| Coordinator disk lost | Everything is lost that isn't in object storage. **Therefore: back up the SQLite file to object storage after every round close.** It is small, and this is the single point of failure in the system |

That last row is the one genuinely fragile spot in the design. The backup is a
five-line job and should exist from M1, not be added later.

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
  "buckets": [17, 143, 288, 401, 655],

  "hyperparams": {
    "lr": 2e-4,
    "seq_len": 2048,
    "micro_batch": 2,
    "grad_accum": 8,
    "seed": 20260825
  },

  "local_steps": 250,
  "max_runtime_sec": 2700,
  "heartbeat_interval_sec": 60,
  "lease_seconds": 3600
}
```

Changes from v1 §5: round binding, bucket sharding (I), pinned `base_precision` (J),
explicit `lora_cfg` (needed by the §5.1 shape gate), presigned base adapter URL, seed
(L6), lease and heartbeat intervals (L1).

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
