# Ganymede — Platform Expansion

*Draft. Eighteen scoping decisions are recorded near the end; no open question of
consequence remains. The build sequencing below is the current plan of attack.*

This is the track that turns Ganymede from *a collaborative LoRA trainer* into
*a compute exchange*: many jobs at once, placed on the machines that suit them,
run for accounts that earn something for the compute they lend, watched through a
web UI, and — the deep one — not necessarily fine-tuning jobs at all.

It sits on top of the v1 milestones in `03-roadmap.md`, and it promotes and
widens most of that document's **Phase 2 (not scheduled)** list. Read this as an
expansion of that section, not a replacement for the roadmap.

---

## What's being added

1. **Concurrent jobs.** More than one job runnable across the fleet at the same
   time, with placement, priority, and starvation handling.
2. **Targeted placement.** Jobs that name the machine or the class of machine they
   run on — by GPU model, VRAM, OS, region, owner, or a specific host.
3. **Accounts.** People sign in, attach their machines, and accrue points for the
   compute those machines contribute. An external auth provider slots in later.
4. **A web UI.** Operator view, contributor view, and job-submitter view — not
   just the read-only JSON the coordinator serves now.
5. **Job types beyond fine-tuning.** A contract that a new kind of compute job can
   implement, proven by carrying at least one non-training type end to end.

Each of these removes an assumption the current build leans on. The value of this
document is naming which assumption, because that is where the work actually is.

---

## M4b runs in parallel — with a guardrail

**M4b is not done.** It is the milestone that proves the thesis — that a run split
across parallel, heterogeneous hardware converges to the same place as the same
run on one GPU. The multi-seed single-node baseline that makes "the same place"
falsifiable is *being measured right now* and does not exist yet.

**Decision: interleave.** Phase A design starts now; M4b runs alongside it on
rented hardware. This is faster in calendar time and accepted with eyes open —
the risk is real. Everything in this document builds new layers between a job and
the hardware that runs it, and the roadmap's own argument for why the baseline
can't be dropped is that **a distributed run that is quietly worse than one GPU
looks identical to a healthy one.**

Phase A's structural churn is large — invariant 4 below moves the round lifecycle
behind a plugin boundary. The churn being large is exactly why the guardrail has
to be **mechanical**, not a promise to be careful:

- **Golden trace, captured first.** Before Phase A touches anything, record a
  fixed-seed short-run loss trace (a few hundred steps, plus one `reduce` step
  with two synthetic submissions) on today's code, checked in as a golden file.
  This is a half-hour job on the 3060 already in hand.
- **Reproduction is a Phase A entry criterion.** The refactored
  `collab_lora_finetune` path must reproduce that trace bit-for-bit where the
  maths is deterministic, and inside a stated tolerance band where it isn't
  (reduction order, non-deterministic kernels). Fail that and Phase A stops until
  M4b is closed and can be used as the reference instead.
- **M4b runs against today's code**, not the refactor, so the convergence result
  is established on the proven shape. Once both exist, the M4b baseline and
  tolerance band become a **standing regression test** every later phase re-runs.

---

## Four invariants this breaks

### 1. "Inventory is derived, not maintained" (§6.11) does not survive accounts

Today a worker is disposable. It registers, it is probed, it works, and if it
vanishes nothing is lost — `GET /v1/fleet` just stops rendering it. There is no
roster to keep current because there is nothing about a worker worth persisting.

An account-owned machine with an accrued reputation total is the opposite: it is
**authoritative durable state**. It has to survive re-enrollment, a GPU swap, an
OS reinstall, and the machine being offline for a month. "Which machine is this,
and whose is it?" becomes a question with a permanent answer, and the credit
ledger is only correct if that answer is stable.

So this track adds the one thing §6.11 explicitly refused: a maintained record.

- **Machine identity** is an enrollment-time `machine_id`, minted when a user
  attaches a machine and bound to that user. It is not derived from hardware.
- A **hardware fingerprint** (GPU model, CPU, board) is recorded as *advisory* —
  it detects "this looks like a different computer" for fraud review, but it is
  not the identity and a legitimate GPU upgrade must not orphan the accrued
  total.
- Capability probing (§6.9) stays exactly as it is. What changes is that the
  probe result now hangs off a durable `machine_id` instead of a throwaway
  `worker_id`.

### 2. The acceptance gates (§5.1) flip from sanity checks to fraud checks

The §5.1 gates — safetensors-only, key/shape match against the round's
`base_adapter_ref`, dtype/NaN/Inf, Frobenius norm within k× the cohort median,
`steps_completed > 0` — were designed against **broken** workers. A crashed
trainer, a truncated upload, a config mismatch.

The moment a task earns points, they face **workers optimising for points**, and
they do not hold. A worker that returns the round's base adapter plus a small
amount of noise passes every one of those checks — keys and shapes match by
construction, no NaNs, Frobenius norm sits comfortably near the median, and
`steps_completed` is whatever integer it cares to report — and collects full
credit for doing nothing.

This is the single hardest problem in the whole request, and it is not solved by
tightening a threshold. It needs some combination of:

- **Redundant execution** on a sampled fraction of tasks — the same task to two
  or three machines, credit contingent on agreement.
- **Spot-check tasks** with a known answer, issued indistinguishably from real
  work, with a wrong answer costing reputation.
- **Reputation** per `machine_id`, earned slowly and lost fast, gating how much
  unverified work a machine may hold and how its submissions are weighted.
- Eventually, verification that is cryptographic rather than statistical (the
  roadmap names TOPLOC; §5.2's weighted mean is the drop-in point for
  trimmed-mean / median aggregation).

The audit data for all of this already accumulates from M1 (§6.3). This is
analysis and enforcement built on existing instrumentation, not new logging — but
it is a subsystem, not a patch.

### 3. Cohort-synchronous rounds assume a dedicated worker

`min_round_sec` / `max_round_sec`, `COHORT_FLOOR = 2`, the DiLoCo round cadence,
and the stall detector ("a round past its backstop **and** evidence workers were
awake") all assume that a machine claiming a task is *giving that task the
machine*. `calibration.json`'s `recommended.local_steps: 56` was computed from
**14.006 steps/min on an otherwise-idle 3060.** Under contention that number is
fiction, and a round pinned to wall-clock will close on far fewer steps than it
budgeted for.

"Parallel tasks on one machine" and "cohort-synchronous rounds" pull in opposite
directions. **Decision: one Ganymede task per machine at a time.** Concurrency is
across the fleet, not within a host, so this invariant survives untouched — the
calibrated throughput number stays meaningful and the round closer keeps
assuming the rate it measured. Fractional workers (throughput re-estimated under
load, `local_steps` as a moving target, MPS/MIG GPU partitioning) are explicitly
out of scope for this track and revisited only if per-host utilisation becomes
the bottleneck.

### 4. "Round" is training-specific state, and most of the coordinator is built around it

The `rounds` table, `closer.py`, `aggregate.py`, and `GET
/v1/runs/{id}/rounds/current` all encode one job type's control flow: train
locally, submit a delta, average deltas, repeat N times. A batch-inference job
has no round, no base adapter, and no combine step.

Generalising job types means **the round becomes private to the
`collab_lora_finetune` plugin.** That is not a bullet — it is moving the
round lifecycle, the aggregation step, and a public API endpoint behind a plugin
boundary, and leaving the generic layer with only *jobs* and *tasks* and an
optional per-job "reduce epoch" counter. It is the sharpest structural change in
this document and the one most likely to churn existing code.

---

## New subsystems

Beyond the four breakages, this track stands up components that don't exist in
any form today:

- **A placement scheduler.** Not "pick the one active run" — a matcher from
  pending tasks to available machines. **An admin-ordered queue with capability
  backfill (Decisions 10, 13):** the admin arranges jobs in a queue and can
  reorder it; there are no fixed tiers. For each free machine the scheduler walks
  the queue from the head and runs the first job whose constraints that machine
  *satisfies* — so a job deep in the queue still runs on a machine that nothing
  ahead of it can use, rather than the machine idling. Job constraints
  (Decision 15) are either a **pin to named `machine_id`s** or a **spec
  predicate** (`vram_gb >= 24`, `gpu_model in {...}`, OS, region). Fair-share
  between submitters and preemption are later, if ever.
- **Identity.** An `IdentityProvider` seam. First implementation is a **placeholder**
  (Decision 5): a local user table, admin flag included, real OAuth/OIDC behind
  the same interface later. Machine enrollment binds a host to a user. Per-machine
  keys replace the hand-issued bearer keys, still hashed at rest.
- **A contribution ledger.** Reputation-only (Decision 7 — no redemption). An
  append-only log, a versioned formula, a derived running total; no debit path,
  no balance invariants, no expiry. **The unit is provisioned Weighted System
  Hours (Decisions 6, 11):** the time a machine is *enrolled, awake, and
  available* — not only the time it is leased — multiplied by a per-machine
  weight for the whole system (GPU, CPU, RAM, bandwidth), not GPU-seconds. The
  accrual engine integrates over the availability signal that already exists
  (`worker_eligibility`, `AWAKE_WINDOW_SEC`). **Anti-farming:** provisioned time
  only counts while the machine is in good standing — it accepts the leases it is
  offered and its submissions pass `validate`. A machine that idles "available"
  while refusing or failing work accrues nothing. The per-task `credit()` figure
  becomes a secondary signal (work actually done, for the reputation score and an
  optional output leaderboard), not the primary accrual. The log is kept clean
  enough that a spendable-points system could be built on it later without a
  migration.
- **A web UI.** Server-rendered templates + htmx over the existing FastAPI app
  (Decision 17) — no JS build, one language. A read model and an SSE channel for
  live updates. Read-only dashboard first; then job submission and the admin
  surface (queue reordering, submitter approve/deny per Decision 9); then account
  and machine management.
- **A job-type SDK, and an image path.** The contract below, plus image handling
  (Decision 18): an approved submitter uploads a Docker image to the coordinator,
  which stores it (object store) and serves it to workers — replacing v1's
  pre-agreed `required_image` tag for third-party jobs. This is a new
  distribution load (images are hundreds of MB to GBs) but it is job *code*, not
  job *data*, so Decision 16 still stands for the latter.
- **A sandboxing story — in scope, at the "vetted allowlist" level.** Job code
  comes from a small set of manually-approved submitters, not the open internet,
  so the threat model is *a trusted author's honest mistake* rather than an
  adversary. That justifies resource caps, a default-deny egress policy with a
  per-job allowlist, and read-only mounts — but not (yet) gVisor / Kata isolation
  or the assumption that job code is actively hostile. **Cancellation carries a
  soft/hard flag** (Decision 18): the admin or submitter ending a job — including
  by allowlist revocation — chooses `soft` (SIGTERM, grace period, finish or
  checkpoint the current unit) or `hard` (SIGKILL now). It is delivered on the
  next heartbeat, since workers are pull-only (Decision 8), so worst-case
  shutdown latency is one heartbeat interval. Heavier isolation is the upgrade
  path if the allowlist ever opens up.

---

## The job-type contract

The generalisation is only real if the contract admits a **concrete second
type**. The one to design against is **batch inference over dataset shards**,
because it has no base adapter and no combine step — it stresses exactly the seam
where the LoRA assumptions are currently welded in. Two working types prove the
seam; "supports all job types" is not a plan.

Minimum contract a job type implements:

| Method | Runs on | Today's equivalent |
| --- | --- | --- |
| `plan(job_spec) -> [TaskSpec]` | coordinator | round → per-worker task fan-out |
| `inputs_for(task) -> InputRefs` | coordinator | base adapter ref + data bucket |
| `run(task, ctx) -> Result` | worker | `run_task(...)` in §4.3 |
| `validate(task, result) -> Verdict` | coordinator | the §5.1 acceptance gates |
| `reduce(job, results) -> State \| None` | coordinator | the §5.2 combine; `None` = embarrassingly parallel |
| `is_complete(job, state) -> bool` | coordinator | `round_idx >= target_rounds` |
| `credit(task, result) -> Units` | coordinator, **trusted** | *(new — feeds the ledger)* |

**`credit()` is a secondary, trusted signal — not the primary accrual.** Primary
reputation accrues from *provisioned* Weighted System Hours (Decisions 6, 11),
which the coordinator computes from the availability signal and never from job
code. `credit()` returns only a *work-done* figure (tokens trained, rows
inferred) used for the per-machine reputation score and an optional output
leaderboard. Even in that reduced role it is **trusted surface**: with
allowlisted submitters (Decision 3) a job type is third-party code, so the
coordinator normalises whatever `credit()` reports against the machine's
provisioned hours rather than banking it directly. Same split applies to
`validate()`: a job type supplies the check, the coordinator owns
redundant-execution comparison and spot-check scoring.

How the two types differ under that contract:

| | `collab_lora_finetune` | `batch_inference` |
| --- | --- | --- |
| `plan` | one round's worth of tasks, repeatedly | every shard, once, up front |
| `inputs_for` | current base adapter + data bucket | model ref + one shard |
| `reduce` | per-tensor weighted mean → new base adapter | `None` |
| `is_complete` | `target_rounds` reached | all shards validated |
| `validate` | key/shape/dtype/norm gates | row count matches shard + schema + sampled re-run agreement |

A "round" is then just a `reduce` checkpoint that some job types have and others
don't. The `rounds` table, `closer.py`, and `/v1/runs/{id}/rounds/current` move
inside `collab_lora_finetune`.

---

## Phasing

M4b runs alongside Phase A on rented hardware (see above). Phases are sequential;
each is independently shippable.

**Phase A — Generalise the job model.** Introduce `Job` and `Task` as first-class,
with today's training run becoming `collab_lora_finetune` under the contract
above. Structurally large — it moves the round lifecycle, aggregation, and a
public endpoint behind the plugin boundary — but **numerically inert**: the
golden trace above is the entry criterion that proves the maths didn't move.
Scheduler v0: multiple jobs, FIFO within priority, machine labels matched against
job constraints, one task per machine, no preemption. Keep the pull-based claim.
No new user-facing features.

**Phase B — Accounts and identity.** `IdentityProvider` seam with a placeholder
provider (local user table + admin flag). Machine enrollment → durable
`machine_id` bound to a user. Per-machine keys. Contribution ledger:
reputation-only, append-only credit events in Weighted System Hours, versioned
formula, derived total. Anti-fraud v0: per-type `validate` is mandatory, a
sampled fraction of tasks run redundantly, reputation per `machine_id`.

**Phase C — Web UI.** Read-only first: fleet, jobs, rounds, your machines, your
contribution, leaderboard. Then job submission and management. Then account and
machine administration. Backend: read model + live-update channel.

**Phase D — Harden multi-tenancy.** Fair-share scheduling, quotas and budgets,
preemption, spot-check tasks with known answers, reputation-weighted aggregation.
Submitter allowlist administered from the web UI — an admin approves or denies a
submitter and the images pinned to them (Decision 9). Sandboxing at the
vetted-allowlist level: resource caps, default-deny egress with a per-job
allowlist, read-only mounts. gVisor / Kata deferred unless the allowlist opens
up.

**Phase E — A second job class for real.** Carry `batch_inference` end to end —
its own `validate`, no `reduce`, shard-level credit. Then one more genuinely
different type (dataset processing, or generic containerised batch) to confirm
the SDK isn't just "training with two spellings."

---

## Decisions taken

1. **M4b sequencing → interleave.** Phase A design starts now; M4b runs alongside
   it on rented hardware, against today's code, under the guardrail above.

2. **Points → reputation-only.** Append-only accrual log and a leaderboard. No
   debit path, no balance invariants. Kept clean enough to build spending on
   later without a migration.

3. **Job code → vetted allowlist.** A small set of manually-approved submitters.
   Threat model is a trusted author's mistake, not an adversary: resource caps,
   default-deny egress with a per-job allowlist, read-only mounts. Heavier
   isolation deferred.

4. **Parallel → across the fleet, one task per machine.** Concurrency is between
   hosts. The round model's dedicated-worker assumption and the calibrated
   throughput number stay intact.

5. **Identity → placeholder first.** A local user table with an admin flag,
   behind an `IdentityProvider` seam so real OAuth/OIDC drops in later.

6. **Crediting unit → Weighted System Hours.** Availability time × a per-machine
   weight for the whole provisioned system (GPU, CPU, RAM, bandwidth), not
   GPU-seconds. Coordinator-computed; job types contribute nothing to it.

7. **Redemption → none for now.** The leaderboard is the whole point. Revisit
   only alongside a spendable-points decision.

8. **Transport → pull only.** Workers poll. No inbound connectivity, no peer
   communication in this track; a Hivemind-style `SyncBackend` waits.

9. **Allowlist admin → web UI.** An admin approves or denies submitters, and the
   images pinned to them, from the UI. Needs the admin role from Decision 5.

10. **Scheduling → admin-ordered queue with capability backfill.** Not fixed
    tiers — an ordered list the admin reorders. Each free machine runs the first
    queued job whose constraints it satisfies, so a job deep in the queue still
    runs on hardware nothing ahead of it can use.

11. **Accrual basis → provisioned, not utilised.** A machine earns Weighted
    System Hours while enrolled, awake, and available — not only while leased —
    but *only while in good standing* (accepts offered leases, submissions pass
    `validate`). Idle-and-refusing accrues nothing.

12. **System weight → probe-derived.** Computed from the capability probe, not an
    admin table. Requires fixing the clamped `bench_score` seen in the 3060
    bring-up so the probe yields a real relative-performance number; the weight
    is a versioned function of that plus VRAM, CPU, RAM, and measured bandwidth.

13. **Priority → admin-decided, queue-shaped.** See Decision 10. Submitters do not
    set their own priority.

14. **Admin bootstrap → env var.** The first admin is named by a coordinator
    environment variable, matching the existing `COORDINATOR_HOST`-style config.

15. **Targeting → named nodes or spec predicates.** A job constraint is either an
    explicit list of `machine_id`s or a threshold predicate (`vram_gb >= 24`,
    `gpu_model in {...}`, OS, region). Implemented as an extension of the §6.8
    eligibility predicates, not a separate label system.

16. **Data plane → deferred.** Object store stays as-is through Phases A–E for
    open job *data*. Per-job encryption, egress accounting, and P2P transfer wait
    for §6.10 classification and the first large real dataset.

17. **Web UI → htmx.** Server-rendered templates + htmx over the existing FastAPI
    app; SSE for live updates. No JS build, one language. Reconsidered only if
    the UI outgrows dashboards and forms.

18. **Job payload → submitter-uploaded container.** An approved submitter uploads
    a Docker image to the coordinator, which stores and distributes it to
    workers. Job cancellation and allowlist revocation carry a `soft` (SIGTERM +
    grace) or `hard` (SIGKILL) flag set by whoever triggers them, delivered on
    the next heartbeat.

## Open questions

None of consequence. Residual detail, each settle-when-you-reach-it: the exact
job-spec fields a submitter fills alongside the image (extends the §8 task spec);
image size limits and retention; whether SSE needs a fallback poll.

---

## Build sequencing

The instinct is to fan a swarm of agents at this now. That fails, for four
concrete reasons, and the plan is built around clearing them.

### What blocks a parallel start

1. **The baseline run owns the machine.** `ganymede-baseline` is pinning the GPU
   at 100% for a few more hours and writing into this checkout. It is also the
   M4b reference. Nothing GPU-bound starts until it finishes.

2. **The golden trace doesn't exist yet.** The M4b guardrail makes a fixed-seed
   loss trace on today's code the *entry criterion* for Phase A. Capturing it
   needs the GPU — so Phase A is gated on the baseline finishing, plus ~30 min.

3. **Phase A is a refactor, and refactors don't fan out.** Moving the round
   lifecycle behind the plugin contract touches `app.py`, `rounds.py`,
   `closer.py`, `aggregate.py`, `db.py`, and a public endpoint. One agent does
   this serially or the merges are unmanageable. Every other workstream depends
   on the seam it creates.

4. **There is no component-level design.** This document records *decisions*, not
   *interfaces*. Before any fan-out there must be a frozen **data-model delta**
   (new and changed tables; `Job` / `Task` / the plugin contract as real Python
   signatures) and a frozen **API delta** (endpoint additions and changes, auth
   middleware, the heartbeat-carried soft/hard kill). Those two are the spine;
   every workstream hangs off both. They have to be one coherent hand — mine —
   not a committee of agents.

### The stages

**Stage 0 — Unblock (serial, now → baseline done).** The two spine docs are
written — `05-data-model.md` (schema delta, the `JobType` protocol, what Phase A
moves) and `06-api-delta.md` (endpoint delta, the three token kinds, the
heartbeat-carried kill). Still open in this stage: the baseline finishing, then
capturing and committing the golden trace, then cutting a `platform-expansion`
integration branch. Stage 1 can begin against the frozen spine as soon as those
two docs are reviewed; it does not wait on the GPU.

**Stage 1 — Component design (fan-out, ~6 agents, no GPU, no code).** One doc per
workstream, each importing the frozen spine and freezing its own seam: scheduler
(queue walk + eligibility-predicate extension + targeting); identity + enrollment
+ admin bootstrap; ledger + provisioned-hours accrual + good-standing gate;
job-type SDK + image upload/store/distribute + `batch_inference`; sandbox +
egress policy + soft/hard kill delivery; web UI (htmx page inventory + read model
+ SSE). I review all six for cross-consistency before Stage 3.

**Stage 2 — Phase A refactor (one agent, serial, gated on the golden trace).**
Round lifecycle behind the plugin contract. Numerically inert; golden trace
reproduced within tolerance. No other agent touches coordinator core until this
merges.

**Stage 3 — Implementation fan-out (many agents, after Stage 2 merges).** One git
worktree per workstream, each against the frozen spine and its Stage 1 design.
Integration order: identity + schema first (others need `machine_id` and users),
then ledger / scheduler / SDK in parallel, then web UI last (it reads
everything).

**Stage 4 — Integrate and regress.** Merge to `platform-expansion`; run the
standing M4b comparison; wire the UI to the real read model; curated merge onward.

### What can actually start today

Only Stage 0, and only the parts that are mine: the spine docs. The "lots of
agents" moments are Stage 1 and Stage 3, and both are gated on the spine being
frozen — which is gated on nothing but the writing. The GPU gate (golden trace →
Phase A) blocks only Stage 2. So the spine docs and the baseline run proceed in
parallel starting now; the first agent fan-out follows spine sign-off.

---

## Relationship to `03-roadmap.md`

This track subsumes Phase 2 items 1 (spot-check + reputation), 2 (concurrent
runs), and 9 (gVisor / Kata, TOPLOC, Byzantine-robust aggregation), and adds
accounts, the web UI, and the job-type contract on top. Items 3–8 of that list
(MLX, `IdleBackend`s, Postgres, Hivemind, MoE, `rl_rollout`) are unaffected and
stay where they are. A pointer to this document is already in that section; the
Phase 2 list stays authoritative until this draft firms up into scheduled work.
