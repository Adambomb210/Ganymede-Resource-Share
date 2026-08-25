# Project Ganymede — Distributed Compute Worker Spec

**Status:** Draft v1
**Scope:** Donated idle-GPU compute (TensorDock/Vast.ai hosts + Patchwork members) for LLM fine-tuning and robotic sim RL training.
**Explicitly deferred:** hardened trust/verification (Section 10). Trusted-circle model for v1.

---

## 1. Goals

- One deployment story for contributors, regardless of what's being trained
- Job configuration is data, not code — most changes require zero rebuilds
- Container footprint stays small even as supported job types grow
- Reuse existing open-source infrastructure instead of building distributed-systems primitives from scratch

## 2. System Overview

```
┌─────────────┐        poll/claim         ┌──────────────────┐
│ Coordinator │◄──────────────────────────│  Worker (host A)  │
│  (1 small   │       heartbeat/submit     │  TensorDock 4090  │
│  always-on  │──────────────────────────►│                    │
│    VM)      │                            └──────────────────┘
│             │        poll/claim         ┌──────────────────┐
│ - job queue │◄──────────────────────────│  Worker (host B)  │
│ - manifest  │       heartbeat/submit     │  Vast.ai 3090     │
│ - Hivemind  │──────────────────────────►│                    │
│   bootstrap │                            └──────────────────┘
│ - checkpoint│
│   storage   │        (peers also sync directly with
└─────────────┘         each other via Hivemind DHT
                         once discovered through bootstrap)
```

The coordinator is the only always-on, non-GPU component. Workers are ephemeral — they run only when a host is idle, claim a job, do the work, and exit or loop for the next one.

## 3. Worker Container

### 3.1 Design principles

- **Generic executor, not job-specific.** The image contains a polling loop and a runtime, not a hardcoded training script.
- **Config over code.** Dataset, hyperparameters, base checkpoint, and sync interval are all job-spec fields (Section 5), never baked into the image.
- **Modular images on a shared minimal base**, to resolve the small-footprint vs. broad-support tension (see 3.3).

### 3.2 Entrypoint loop

```
1. Read env: GANYMEDE_KEY, COORDINATOR_URL
2. Detect GPU (nvidia-smi); abort cleanly if none found
3. GET /manifest → current job-type image tags + Hivemind bootstrap address
4. If running image tag != manifest's required tag for available job types:
     exit (host's cron/systemd layer re-pulls correct image next cycle)
5. POST /jobs/claim → receive job spec (Section 5)
6. Dispatch by job_type:
     - llm_finetune  → Axolotl/torchtune + Hivemind averaging
     - rl_rollout     → Isaac Lab + Ray actor-learner
7. Run job:
     - checkpoint locally on a fixed interval (independent of network)
     - heartbeat to coordinator on a fixed interval
     - on SIGTERM (host preempted by a paying renter): checkpoint
       immediately, submit partial progress, exit 0
8. On natural completion: submit final weights/deltas, loop to step 3
```

### 3.3 Image strategy — small footprint, broad support

A single monolithic image with every ML framework installed defeats "as small as possible." Instead:

| Image | Contents | Approx role |
|---|---|---|
| `ganymede/worker-core` | polling loop, GPU detection, heartbeat/claim/submit client, no ML frameworks | shared parent layer, pulled once |
| `ganymede/worker-llm:vN` | `worker-core` + Axolotl/torchtune + Hivemind | pulled only by hosts running LLM jobs |
| `ganymede/worker-rl:vN` | `worker-core` + Isaac Lab + Ray | pulled only by hosts running RL jobs |

Because `worker-llm` and `worker-rl` both extend `worker-core`, Docker's layer caching means a host that's run one variant only downloads the *delta* when switching to the other — not the whole image again. This is the mechanism that lets the system support more job types over time without every contributor's disk footprint growing.

### 3.4 Baseline isolation (lightweight, not Section 10)

- Run as non-root, `--cap-drop=ALL`
- Read-only root filesystem where the framework allows it
- Egress restricted to `COORDINATOR_URL` and the Hivemind bootstrap address only
- cgroup resource limits so a runaway job can't starve the host

## 4. Coordinator

### 4.1 Responsibilities

- Job queue (claim / heartbeat / submit)
- Version manifest (current image tag per job type, current Hivemind bootstrap address)
- Hivemind DHT bootstrap node (always-on so new peers always have somewhere to connect)
- Durable checkpoint storage, independent of peer network state

### 4.2 Minimal implementation

FastAPI + SQLite on a single small VM. No GPU required. This is deliberately the *only* piece of infrastructure that must be always-on — everything else is contributor hardware coming and going.

### 4.3 API surface

```
GET  /manifest
     → { job_types: { llm_finetune: "worker-llm:v3", rl_rollout: "worker-rl:v1" },
         bootstrap_addr: "..." }

POST /jobs/claim
     → { job_id, job_type, config }   (Section 5)

POST /jobs/{id}/heartbeat
     → 200 OK | 410 Gone (job reassigned due to timeout)

POST /jobs/{id}/submit
     body: { weights_delta | trajectory_batch, metrics }
     → 200 OK
```

## 5. Job Spec Schema

```json
{
  "job_id": "uuid",
  "job_type": "llm_finetune | rl_rollout",
  "worker_image": "ganymede/worker-llm:v3",
  "base_checkpoint_ref": "s3://ganymede/checkpoints/latest.pt",
  "dataset_ref": "s3://ganymede/data/combat-robot-sim-v2",
  "hyperparams": {
    "lr": 2e-4,
    "lora_rank": 16,
    "local_steps_per_sync": 500
  },
  "max_runtime_sec": 3600
}
```

Everything here is mutable without a rebuild. Only `worker_image` changes when the underlying architecture or algorithm changes (Section 8).

## 6. Distributed Sync Layer

**LLM fine-tuning:** Hivemind-based parameter averaging (DiLoCo-style — many local steps, infrequent averaging). OpenDiLoCo is a ready-made reference implementation built on Hivemind; the training loop itself runs through Axolotl or torchtune.

**Sim RL:** Ray's actor-learner pattern. Rollout workers run Isaac Lab + current policy, push trajectories to a central buffer; a learner samples the buffer, updates the policy, and Ray handles redistribution of updated weights.

Both approaches tolerate nodes joining, leaving, or lagging mid-run — neither requires uniform hardware or constant connectivity.

## 7. Idle-Detection & Host Automation

On each contributor's TensorDock/Vast.ai host:

```
# cron, every N minutes
1. Query platform API for active rentals on this GPU
2. If idle: check GET /manifest, pull correct worker image if tag changed
3. docker run --gpus all -e GANYMEDE_KEY=... ganymede/worker-<type>:<tag>
4. Container runs until preempted (rental starts) or job completes
```

This mirrors the existing daemon pattern both platforms already use for rental provisioning — no new paradigm for a host who already runs one of these platforms.

## 8. Versioning & Change Management

| Change | Requires new image? | Mechanism |
|---|---|---|
| Dataset | No | New `dataset_ref` in job spec |
| Hyperparameters | No | New `hyperparams` in job spec |
| Resume checkpoint | No | New `base_checkpoint_ref` |
| New RL reward shaping logic | Yes | New `worker-rl` tag, manifest bump |
| Switch base model architecture | Yes | New `worker-llm` tag, manifest bump |
| Switch algorithm (e.g. GRPO → PPO) | Yes | New tag, manifest bump |

Architecture-level changes require winding down the current swarm before starting a new one — active peers are averaging weights together, which only holds if everyone shares the same shapes.

## 9. Fault Tolerance

- **Single node drops:** LLM — that node's local steps since last sync are simply lost, others unaffected. RL — in-flight rollout discarded, no training state lost since rollout workers are stateless.
- **Total swarm outage:** No data is corrupted; weights live in local worker state and periodic durable checkpoints, not in the DHT itself. A fresh peer just needs the coordinator's bootstrap address to rejoin.
- **Coordinator outage:** Existing peers who've already discovered each other keep training; new peers can't join until the coordinator (bootstrap node) is back.

## 10. Deferred to Phase 2 — Trust & Verification

Not required while the contributor pool is people you personally vouch for. Build when the pool grows past that:

- Bounds/sanity checks on submitted weights (reject NaN/Inf, out-of-range norms) — cheapest to add first
- Spot-check redundancy (duplicate jobs across two workers, compare results)
- Reputation scoring per contributor
- gVisor/Kata sandboxing (stronger isolation than 3.4's baseline)
- TOPLOC-style cryptographic verification of submitted computation
- Byzantine-robust aggregation (trimmed-mean/median instead of plain averaging)

## Appendix: Minimal Path to First Deployment

1. Run the training job single-node, by hand, end to end — confirm it works before any infra
2. 2-3 trusted contributors run the worker manually, no automation yet
3. Stand up the minimal coordinator (Section 4.2)
4. Add host-side cron automation (Section 7)
5. Add Phase 2 trust items only as the contributor pool outgrows personal trust
