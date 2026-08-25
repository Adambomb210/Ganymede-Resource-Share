# Spec Review — Ganymede Worker Spec v1

Review of `00-original-spec-v1.md`. Verdict first, then findings ordered by how much
they change the design.

---

## Verdict

**The architecture is sound and the scoping instinct is right.** Ephemeral workers,
an always-on coordinator as the only fixed infrastructure, config-as-data job specs,
and an explicitly deferred trust model are all correct calls for a trusted-circle v1.
Most drafts at this stage get the deferral wrong in one direction or the other; this
one doesn't.

Four things need to change before it's buildable:

1. **There is no round/generation concept.** This is the structural gap. A job queue
   alone cannot express "everyone trains from checkpoint N, then we combine." Adding
   rounds subsumes most of §8's versioning ceremony as a side effect. (Finding A)
2. **The image footprint argument is technically true but economically wrong.**
   `worker-core` is ~30 MB; the shared cost that actually matters is the ~8 GB CUDA +
   PyTorch layer, which the spec's layer stack doesn't share. (Finding B)
3. **§3.4's egress allowlist contradicts §5's artifact refs.** Datasets, checkpoints,
   and the base model all live somewhere that isn't the coordinator or the bootstrap
   node. (Finding C)
4. **§9 over-claims durability.** "Weights live in local worker state" is not a
   fault-tolerance property when workers are preemptible by design. (Finding D)

Plus one recommendation that diverges from the spec: **drop Axolotl/torchtune for the
LLM path** and own a ~250-line trainer. Reasons in Finding K.

Everything else below is a gap to fill, not a mistake.

---

## Decisions taken (from clarification)

| Question | Decision |
|---|---|
| v1 scope | `llm_finetune` only. `rl_rollout` deferred to Phase 2. |
| Sync layer | Central aggregation now, behind a `SyncBackend` seam so Hivemind can be swapped in without touching the worker protocol. |
| First job | LoRA on a 7–8B base model. |
| Host model | All three (own hardware, donated rented hosts, paid instances) — **starting with own hardware**, so idle detection ships as a pluggable backend with `local` first. |
| Storage | **Self-hosted, S3-compatible (MinIO)** on the coordinator VM, with R2 or S3 reachable by config change later. |

These are assumed throughout the revised architecture (`02-architecture-v2.md`).

---

## Findings

### A. No round/generation concept — the structural gap

§4.3 gives a job queue: claim, heartbeat, submit. But collaborative training isn't a
queue workload. The actual semantics are:

> All participating workers train **from the same base checkpoint**, for a bounded
> amount of local work, then their results are **combined into one new checkpoint**,
> which becomes the base for the next cycle.

A queue can't express that. Two workers claiming at different times get different
base checkpoints, and averaging results computed from different starting points is
not a well-defined operation — it silently degrades the model rather than failing
loudly.

**Fix:** introduce `run → round → task` (see `02-architecture-v2.md` §3). A *round*
pins one base checkpoint and one deadline; *tasks* within a round are shard
assignments. A worker submitting against a closed round gets `409` and re-claims into
the current one.

**Side benefit:** this makes §8's "architecture-level changes require winding down the
current swarm" automatic rather than procedural. A round carries its base checkpoint
ref; if the architecture changed, that's a new run with a new round 0, and stale
submissions are rejected by construction. No manual swarm wind-down needed.

### B. The layer-caching footprint claim doesn't hold up

§3.3 says `worker-llm` and `worker-rl` both extend `worker-core`, so switching costs
only the delta. Docker layer sharing does work that way — but only for layers below
the divergence point, and `worker-core` as specced (polling loop, GPU detection, HTTP
client, *no ML frameworks*) is roughly 30 MB. The expensive content sits **above** the
shared layer:

```
Spec's stack                          Actual sizes
  worker-core          ~30 MB   ← the only shared part
    ├─ worker-llm      +10 GB   (torch, CUDA runtime, transformers, peft…)
    └─ worker-rl       +18 GB   (torch, CUDA runtime, Isaac Lab, Ray…)
```

A host switching variants re-downloads ~10 GB. The 30 MB it saved is noise.

**Fix:** put the expensive shared thing at the bottom.

```
nvidia/cuda:12.4.1-runtime-ubuntu22.04
  └─ ganymede/torch-base:cu124-t2.5      ~8 GB   ← the real shared layer
       └─ ganymede/worker-core:vN        ~30 MB
            ├─ ganymede/worker-llm:vN    ~2 GB
            └─ ganymede/worker-rl:vN     (Phase 2)
```

Now a variant switch costs ~2 GB instead of ~10 GB.

**Honest caveat:** Isaac Lab pins its own PyTorch and CUDA build, so `worker-rl` may
not be able to sit on `torch-base` at all. The delta story is real for the LLM path
and weak-to-nonexistent for RL. Worth knowing now rather than discovering it in
Phase 2.

**Separate and larger:** the 16 GB base model itself. Baking it into the image
contradicts the footprint goal; shipping it through the coordinator every round is
absurd. It belongs in a **host-persistent HF cache volume**, pulled once from
HuggingFace and reused across every container start. This is what forces Finding C.

### C. §3.4's egress allowlist contradicts §5's artifact refs

§3.4 restricts egress to `COORDINATOR_URL` and the Hivemind bootstrap address. But
§5's job spec has `base_checkpoint_ref: s3://…` and `dataset_ref: s3://…`, and
Finding B adds a HuggingFace pull. That's three destinations the allowlist forbids.

There are two coherent resolutions, and the spec needs to pick one:

- **Proxy everything through the coordinator.** Keeps the allowlist to one host.
  Fine for LoRA adapters (tens of MB), untenable for a 16 GB base model or a large
  dataset, and it puts all artifact bandwidth on the small always-on VM.
- **Widen the allowlist to a named object store + HuggingFace.** Costs two more
  entries, keeps the coordinator small, and lets you use presigned URLs so the
  coordinator never touches artifact bytes.

**Recommendation: the second.** Allowlist = coordinator, artifact store, HuggingFace
(`huggingface.co` + `cdn-lfs*.hf.co`). Artifacts move worker↔store directly via
presigned PUT/GET; the coordinator application only ever handles *references*, which
makes the bandwidth math work (Finding H).

**Per the storage decision**, the store is self-hosted MinIO on the coordinator VM in
v1, so two of those three destinations are the same machine. Keep them as separate
config entries anyway — otherwise moving storage to R2 later means editing every
contributor's firewall rules instead of one manifest field. Architecture v2 §6.5
covers the sizing, the retention policy, and the backup consequence, which is the one
thing co-location genuinely breaks.

### D. §9 over-claims durability

> "weights live in local worker state and periodic durable checkpoints, not in the DHT"

Local worker state is not durable. The entire premise of §7 is that workers are
preempted without warning when a paying renter arrives. A host that disappears takes
its local state with it.

The honest invariant is narrower and still sufficient:

> **Anything a worker has successfully submitted is durable. Everything since its last
> submission is lost on preemption, and losing it costs at most one round of that
> worker's local steps.**

That's a fine property — it just needs to be stated as what it is. It also implies a
design rule: **size `local_steps_per_sync` so that losing one round is cheap.** If a
round is 30–45 minutes of work, preemption costs 30–45 minutes. If it's 6 hours, a
flaky host may never contribute anything at all.

**Consequence for §3.2 step 7:** the SIGTERM handler shouldn't fight to checkpoint and
submit partial progress under time pressure — Docker's default stop grace is 10 s and
platform preemption can be harder than that. Far better: **release the lease and
exit**, so another worker picks the shard up immediately. Partial submission becomes a
best-effort optimization, not a correctness requirement. See `02-architecture-v2.md`
§4.4.

### E. §3.2 step 4's version check can't work as written

Step 4 says exit if the running image tag doesn't match "the manifest's required tag
for available job types" — but the worker doesn't learn its job type until step 5,
after it claims. The check runs before it has the information it needs.

**Fix:** move version reconciliation entirely to the host agent, which checks the
manifest *before* starting a container. The worker then declares what it can run when
it claims, and the coordinator refuses to hand it work it can't do. Cleaner
separation: the host agent owns "which image", the worker owns "do the work".

### F. No auth model anywhere

`GANYMEDE_KEY` appears once in §3.2 step 1 and never again. Nothing in §4.3 mentions
authentication. Since `POST /jobs/{id}/submit` accepts weights that get merged into
the shared model, that endpoint *is* the entire trust surface — even inside a trusted
circle, an accidentally-leaked key is an accidentally-poisoned model.

Minimum viable for v1: bearer token over mandatory TLS, hashed at rest, one key per
contributor, revocable by deleting a row, per-key rate limits. ~40 lines. Detailed in
`02-architecture-v2.md` §6.

### G. `torch.load` on submitted artifacts is remote code execution

Not listed in §10, and it shouldn't be deferred there — it's not a *trust* problem,
it's a "don't build the vulnerability in the first place" problem. PyTorch's `.pt`
format is pickle; loading one from a worker executes arbitrary code on the
coordinator. Trusted circle or not, this is free to avoid and expensive to retrofit.

**Fix, v1, non-negotiable:** all worker→coordinator artifacts are **safetensors**.
Never call `torch.load` on anything a worker produced. Note that §5's example
`base_checkpoint_ref: ".../latest.pt"` should become `.safetensors`.

While there: §10's first bullet (NaN/Inf and norm bounds checks) is ~15 lines and runs
in milliseconds. Pull it forward into v1 as an acceptance gate. It catches honest
bugs — a diverged run, a bad LR — far more often than malice, which is exactly what
you want during bring-up.

### H. Submitting weights in an HTTP body is the wrong shape

§4.3 has `POST /jobs/{id}/submit` with body `{ weights_delta | trajectory_batch }`.
A LoRA adapter for an 8B model at rank 16 is roughly **15–85 MB in bf16** depending on
which modules you target (~14 MB for q/v only, ~84 MB for all linear layers). Pushing
that through FastAPI request handling and into SQLite works right up until it doesn't.

**Fix:** worker asks for a presigned PUT, uploads to the store directly, then submits
the *reference* plus metrics. The coordinator handles kilobytes. This is the same
change Finding C recommends, from the other direction.

**Bandwidth sanity check, which is the real justification for the central design:**
per worker per round, ~85 MB down + ~85 MB up. Ten workers on 30-minute rounds is
~3.4 GB/hour, about 7.6 Mbit/s average — comfortably served by one small VM.
Peer-to-peer averaging buys nothing at this scale.

This is also what makes self-hosted storage a sensible default rather than a
compromise: the access pattern is egress-dominated, which is precisely where metered
object storage prices worst. A VM with a bundled traffic allowance and R2's
zero-egress model are both fine; S3-class egress pricing is the option to avoid.
Architecture v2 §6.5 has the numbers and names the worker count at which to revisit.

### I. Missing: dataset sharding

Nowhere in §5 or §6 does a worker learn *which part* of the dataset to train on. If
five workers pull the same `dataset_ref` and iterate it the same way, they compute
five highly correlated updates and averaging them gains almost nothing over running
one worker. This is a correctness gap, not a tuning detail — it determines whether
the whole system produces value.

**Fix:** pre-shard the dataset into a fixed number of buckets (say 1000) at prep time.
Each task carries a bucket list. Bucket count stays fixed while worker count varies
freely, and the coordinator tracks coverage so rounds don't re-train the same data.

### J. Missing: heterogeneous quantization breaks aggregation

Since v1 targets 24 GB cards, QLoRA (4-bit base) is the comfortable option — bf16 base
weights alone are ~16 GB on a 24 GB card, leaving little room for activations.

The subtlety: LoRA adapters are deltas against a **frozen** base, which is precisely
why averaging them across workers is well-behaved (better-behaved than averaging full
weights). But that argument only holds if every worker froze the *same* base. If one
worker runs bf16 and another runs nf4, their adapters are shape-compatible but were
trained against measurably different functions, and averaging them injects noise
that's invisible in the logs.

**Fix:** `base_precision` becomes a required, pinned job-spec field. The worker
asserts it can honor it or declines the task.

### K. Recommendation: don't use Axolotl or torchtune

§6 names Axolotl or torchtune for the training loop. Both are good tools; neither fits
what this system actually needs.

DiLoCo-style training requires **owning the outer loop** — stop cleanly at exactly N
steps, extract the adapter, upload, pull a new base, resume with fresh optimizer
state. Axolotl owns the training loop and is configured by YAML; bending it to yield
control at a step boundary means fighting the framework. torchtune's recipes are
meant to be forked, which is better, but then you're maintaining a fork.

For LoRA on a 7–8B model, `transformers` + `peft` + a hand-written loop is about
**250 lines** and gives exact control over the sync boundary, the SIGTERM path, and
the safetensors I/O. It also drops a large dependency tree, which serves the footprint
goal directly.

Revisit this if the job types diversify enough that a framework earns its keep. For
one job type, it doesn't.

### L. Smaller gaps

| # | Gap | Fix |
|---|---|---|
| L1 | No lease semantics — §4.3's `410 Gone` implies them, but no duration, renewal, or max-attempt policy | Explicit lease, sized from the worker's reported throughput; heartbeat renews; expiry returns the task to pending |
| L2 | SQLite concurrent claims will double-assign | WAL mode + `BEGIN IMMEDIATE` around claim. Good to a few hundred workers; note the ceiling |
| L3 | `POST /jobs/claim` has no request body, so the coordinator can't route by capability | Claim takes GPU name, VRAM, driver, torch version, image tag |
| L4 | No "no work available" response defined | `204 No Content` + a `Retry-After` hint; worker backs off with jitter |
| L5 | `max_runtime_sec: 3600` and `local_steps_per_sync: 500` can conflict — 500 steps on a 3090 at 8B/seq-2048 exceeds an hour | Submit on whichever fires first; aggregator weights by steps actually completed |
| L6 | No seed in `hyperparams` | Add `seed`; derive per-task seeds from it for reproducibility |
| L7 | No DiLoCo outer-optimizer state anywhere | Outer Nesterov momentum lives on the coordinator, persisted alongside each round |
| L8 | No metrics/observability | Per-round loss and per-contributor throughput in SQLite; one read-only status page. Cheap, and you cannot debug divergence without it |
| L9 | §5's `.pt` checkpoint extension | `.safetensors` (Finding G) |

### M. Host-model note (own hardware first)

With own hardware there is no platform API to query for active rentals, so §7 step 1
has no implementation on the primary path. Local idle detection needs different
primitives:

- No non-Ganymede CUDA process holds the GPU (`nvidia-smi --query-compute-apps`)
- No manual pause file present (`/etc/ganymede/pause` — the contributor's kill switch)
- Optionally: no active graphical session, or within a configured time window

Ship this as `IdleBackend` with a `local` implementation, then add `vast` and
`tensordock` backends against the same interface when you move to donated rented
hosts.

**Flag for the rented-host phase, worth verifying before you rely on it:** running
your own workload on a GPU you have listed for rent may interact with the platform's
reliability or availability scoring if a rental request can't be filled promptly.
Confirm against current provider policy — it affects whether the donated-host model
works as designed.

---

## "Is there a better way to do it?"

Three alternatives worth naming, and why the spec's approach still wins.

**Use an off-the-shelf queue (Celery/RQ + Redis, or Ray) instead of a bespoke
coordinator.** Goal #4 says reuse open-source infrastructure rather than building
distributed-systems primitives, and §4.2 then builds a job queue — an apparent
contradiction. It isn't one. A claim/lease/heartbeat queue over HTTP is ~300 lines and
is *genuinely simpler* than operating Redis or a Ray head node across WAN-separated,
untrusted, preemptible hosts. More importantly, HTTP-only is what makes the §3.4
egress allowlist tractable at all; a Redis client on every contributor's box is a much
worse security posture than an HTTPS client. The primitives you're correctly not
rebuilding are the training loop, the optimizer, and the gradient math.

**Keep Hivemind for v1.** Right at large scale, wrong now. Peer-to-peer averaging
across consumer hosts behind NAT means real time spent on hole-punching and relay
fallback, it conflicts directly with the egress allowlist, and at ten workers moving
85 MB adapters it saves bandwidth that was never scarce (Finding H). The chosen path —
central now, `SyncBackend` seam for later — is correct. Concretely, that seam is:
`get_base(round) → bytes` and `publish(round, adapter, weight) → None`. A Hivemind
backend implements the same two methods; the worker never learns which is in use.

**Reframe the coordinator's identity.** The most useful change isn't a technology
swap — it's conceptual. The coordinator is **a round-based aggregation server that
happens to expose a queue**, not a queue that happens to aggregate. Once framed that
way, Finding A's rounds are the natural primitive, §8's manual swarm wind-down
disappears, and stale-submission rejection falls out for free instead of needing
process discipline.

---

## What this review does not cover

- `rl_rollout` beyond noting that Isaac Lab's image size and PyTorch pinning weaken
  §3.3's shared-layer argument, and that shipping raw trajectories over WAN is a
  heavier data path than shipping weight deltas. Phase 2 will need its own review.
- §10's trust items, other than pulling the sanity-check bullet and the safetensors
  requirement forward into v1.
