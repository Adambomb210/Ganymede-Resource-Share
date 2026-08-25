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
| Uniform `local_steps_per_sync` | **Per-worker step budgets** sized to a common deadline | §3.4 |
| (single run assumed) | Multi-run cache affinity + per-run calibration | §6.6 |

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
  │                      │  MinIO  (self-hosted, §6.5)  │
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
That is a deliberate, and cheap, trade — see §6.5 for the sizing that justifies it and
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

### 3.4 Per-worker step budgets

Uniform `local_steps` across heterogeneous hardware means the slowest card sets the
deadline for everyone. A 3090 paired with a 4090 straggles every round, and either
blocks the close or gets truncated — wasting a fraction of its work each time,
forever.

Instead, **fix the round deadline and vary the step budget per worker**:

```
local_steps_i = throughput_i × target_round_minutes × safety_factor
```

Everyone finishes near the same deadline, and §5.2's weighting by steps actually
completed then reflects real contribution rather than hardware luck.

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
- **Egress allowlist**: coordinator host, artifact-store host, `huggingface.co` +
  `cdn-lfs*.hf.co`. Three destinations, all named, all HTTPS. (Finding C)
  In v1 the first two are the same machine on different subdomains, so the effective
  allowlist is two hosts — but keep them as separate config entries, or moving storage
  to R2 later becomes an edit to every contributor's firewall rules instead of one
  manifest field.

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
throughput    run_id, gpu_model, steps_per_min, samples, updated_at   -- §3.4
calibration   run_id, calibration_json, created_at                    -- §6.6, M0
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
GET  /healthz    GET /metrics     GET /status   (read-only HTML)
```

Differences from v1 §4.3: capability-aware claim (L3), explicit `204` for no work
(L4), `409` vs `410` distinguished, `upload-url` and `abandon` added, artifact bytes
gone from every request body.

`cached_base_models` exists because base models are ~16 GB each and Ganymede will run
many of them (§6.6). Given two eligible runs, the coordinator should prefer the one
whose base model the worker already holds — that's the difference between starting
work in seconds and starting it after a 16 GB download.

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
| Artifact store down | Same as coordinator down, from the worker's side. In v1 they share a VM, so in practice these fail together |
| Coordinator VM lost | **The failure that matters.** In v1 the database and every checkpoint are on one machine, so this loses both unless backups are genuinely off-box. See §6.5 — separate volume, off-box SQLite dump each round close, off-box copy of the latest round adapter |
| Artifact volume lost, VM survives | Recoverable from the off-box dump + latest adapter: resume the run rather than restart it. Loses older checkpoint history |

### 6.5 Artifact storage — self-hosted now, portable later

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
GANYMEDE_S3_ENDPOINT=https://storage.ganymede.example   # → R2's endpoint
GANYMEDE_S3_BUCKET=ganymede
GANYMEDE_S3_REGION=us-east-1                            # → "auto" for R2
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
endpoint (`https://storage.ganymede.example`), not for MinIO's internal address.
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

### 6.6 Running many models

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
