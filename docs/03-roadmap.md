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

**3. The baseline.** One held-out eval set and a recorded single-node result per run.

### Exit criteria

- Trainer runs, loss descends, adapter round-trips through safetensors
- `ganymede-calibrate` produces a valid `calibration.json` for the first run config
- The run fits the target card at the chosen `base_precision`, with headroom
- A single-node `baseline.json` exists for that run

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

### Model: start much smaller than 8B

**Bring up on a ~1.5–4B dense Qwen in bf16, then scale to 8B once M4 passes.**

This is worth more than it sounds:

- **No `nf4` needed.** A 1.7B model in bf16 is ~3.4 GB of weights — it fits a 12 GB
  3060 and a 16 GB Mac comfortably. That removes the entire nf4/MPS exclusion problem
  (§6.8 Tier 3) during bring-up, so **every contributor is eligible for the bring-up
  run**, including the Macs.
- **You exercise the heterogeneity paths early**, which is where the interesting bugs
  are. On an 8B nf4 run the Macs sit idle and you'd find their bugs months later.
- **Rounds are minutes, not hours.** You can iterate the whole pipeline several times
  a day instead of once. Use ~10-minute rounds for bring-up rather than §3.3's 30–45.
- **Adapters are ~10–15 MB**, so the storage and bandwidth plumbing gets tested
  without waiting on transfers.

Then scale: same code, new run, `base_model` and `base_precision` changed in the run
config. That's the whole point of config-over-code, and scaling the model is a good
first proof that the property actually holds.

### Dataset: public and permissive for bring-up

You want something well-understood with a known-good outcome, so that a bad result
means your plumbing is wrong rather than your data. All of these are `open` class
(§6.10), which also keeps clearance out of the bring-up path.

| Dataset | Size | Notes |
|---|---|---|
| **Databricks Dolly 15k** | 15k | Human-written, tiny, CC-BY-SA. Best for the very first end-to-end loop — a full run takes minutes |
| **HuggingFace No Robots** | 10k | High-quality human-written SFT. Non-commercial licence — fine for bring-up, check before anything else |
| **UltraChat 200k** | 200k | The Zephyr SFT set. Big enough for meaningful bucket sharding and multi-round convergence |
| **Tulu 3 SFT mixture** | ~940k | Modern, strong, well-documented mixture. More than you need for bring-up |

**Verify the current licence on each dataset card before use** — these change, and
this table is a starting point rather than an authority.

**Decided:**

- **Bring-up (M0–M4): Qwen 1.7B (dense, bf16) + Dolly 15k.** Confirm the exact model
  ID at M0. No `nf4` anywhere in the bring-up path, so every contributor — Macs and
  3060s included — is eligible from the first run.
- **Buckets: 64** for Dolly (~230 samples each), per the sizing rule below.
- **Later:** scale to 7–8B and your own dataset, as a run-config change.

One caveat on Dolly worth knowing at M4: at 15k samples a round of any real size
covers a meaningful fraction of the set, so workers will repeat data across rounds
sooner than they would on a larger corpus. That's fine for proving the *machinery* —
it just means Dolly measures whether aggregation works, not how far the model can get.
If M4's convergence result looks suspiciously flat, dataset exhaustion is the first
thing to check, and swapping to UltraChat 200k is the test.

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
- CI that builds and pushes all three, pinning `torch-base` **by digest**

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
- **A second person installs it from `INSTALL.md` alone, with no help from you.** That
  test is the milestone; a working agent that only you can install isn't done
- `INSTALL.md` states a minimum free-disk figure, and cache eviction demonstrably
  holds the cache under its cap across two runs with different base models

---

## M4 — Multi-node convergence — *the milestone that proves the thesis*

**~3–4 days, plus wall-clock training time.**

Three or more real workers, sharded data, a real run to convergence, measured against
M0's baseline.

Exit criteria:
- Aggregated model **matches or beats** the M0 single-node baseline on the held-out set
- Wall-clock-to-target-loss is better than single-node — otherwise the entire system
  is an expensive way to train slower
- Both combine modes A/B'd (§5.2's research risk). If DiLoCo outer momentum doesn't
  beat plain weighted mean on LoRA adapters, **ship the plain mean** and record the
  negative result
- A worker killed mid-round costs one round and nothing else
- Per-round loss curve is visible and smooth; no divergence, no silent degradation
- Mixed-speed workers (at least two GPU classes, ideally the widest spread you have —
  a 3060 alongside an A100) both land near the round deadline, confirming §3.4's
  budgets rather than truncating the slower card every round
- No single worker exceeds the §3.4 dominance cap; removing the fastest worker
  mid-run degrades the round rather than wrecking it

Everything before this is plumbing. This is where you find out whether the idea works.
Budget for it to fail the first time and need a tuning pass — that is the normal
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
 │  training       ├──► M4 ──► M5
 └──► M1 ─────► M3 ┘
     coordinator
```

M1 depends on M0 only for the LoRA config shape (needed by the acceptance gates), so
the two tracks can overlap if there's a second person. M4 needs all four.

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
7. **`rl_rollout`** — needs its own spec review. Isaac Lab's image size and PyTorch
   pinning break the shared-layer story (Review Finding B), and shipping raw
   trajectories over WAN is a heavier data path than shipping weight deltas. Worth
   evaluating whether workers can send gradients or advantages instead of trajectories.
8. **gVisor/Kata, TOPLOC verification, Byzantine-robust aggregation** — when the pool
   includes people you don't know. §5.2's weighted mean is the drop-in point for
   trimmed-mean/median.

---

## Open questions — needed before M1

Blocking or near-blocking. Rough order of urgency.

1. **Base model — Qwen, latest generation. Two things to settle at M0.**

   **Pick the newest dense Qwen in the 7–8B range** and confirm the exact model ID on
   HuggingFace when you start M0 — the family moves quickly and my knowledge of what's
   current has a cutoff. Apache 2.0 across most of the family is why this is the
   cleanest choice for a project that ships adapters to many machines.

   **Dense, not MoE — this one matters.** Recent Qwen generations include MoE variants,
   and MoE interacts badly with DiLoCo-style averaging: which experts a worker trains
   depends on how *its* data shard routes, so different workers update different
   experts and averaging them is not the same operation it is for a dense model.
   Router state adds a second divergence path. Treat MoE + collaborative averaging as
   a research project, not a v1 default.

   **Check the chat template before generating any SFT data.** Recent Qwen models use a
   hybrid thinking/non-thinking template. Fine-tuning against the wrong template
   degrades behavior in ways that don't show up in training loss — you see it only in
   generation, which would silently invalidate M0's baseline. Verify it round-trips
   first. *Needed for M0.*

2. **The dataset(s).** What are they, how large, where do they live? Must shard into
   ~1000 stable buckets. Sensitivity is now handled per-run by §6.10's classification,
   so the remaining question is narrower: **which classification does the first run
   get, and who holds `internal` / `restricted` clearance?** *Needed for M0.*

3. ~~**Object store.**~~ **Resolved: self-hosted MinIO on the coordinator VM**, S3
   config in env vars so R2 is a later config swap (§6.6). Two follow-ons this raises:
   **where do off-box backups go?** (needs to be a different machine — a free
   object-storage tier is fine and is a painless way to start an R2 account early),
   and **how much volume to provision?** 200 GB is generous for 10 workers × 100
   rounds, but it depends on dataset size, which is question 2. *Needed for M1.*

4. **Coordinator hosting.** A small always-on VM, a domain, and TLS. Any of Hetzner /
   Fly / Vultr is fine; the requirements are modest and the choice isn't
   architectural. TLS is mandatory (§6.3). *Needed for M1.*

5. **Held-out eval set and target metric.** What number decides whether M4 passed?
   Define it during M0 while the baseline is being established, not after. *Needed
   for M0's exit criteria.*

6. **Contributor hardware inventory.** Concretely, a row per machine:

   | Field | Why it's needed |
   |---|---|
   | Label / owner | Who to ask when it misbehaves |
   | OS | Picks the packaging path — container, launchd, or Scheduled Task (§4.1) |
   | GPU / chip | With Apple Silicon, the chip variant matters (Air vs Max) |
   | VRAM or unified RAM | The main eligibility gate (§6.8 Tier 2) |
   | ~~Typical availability~~ | **Don't bother** — no reliable schedule exists. The coordinator measures participation instead (§3.2, §6.1) |
   | Upload bandwidth | An 85 MB adapter on a 1 Mbit/s uplink is 11 minutes of a 35-minute round — that machine needs longer rounds or a smaller model |
   | Who administers it | Decides `restricted`-run eligibility (§6.10) |

   Three eligible machines is the minimum for a meaningful M4 — and note "eligible"
   is per-run, so the count that matters is how many clear the bar for *that* run.
   Upload bandwidth is the field people forget and the one that quietly wrecks round
   pacing.

   **Availability is deliberately absent.** Machines come and go unscheduled, so the
   design treats fleet size as unknown at all times (§3.2) rather than planning around
   a roster. The coordinator measures who actually showed up; you don't predict it.
   *Needed for M3/M4.*

7. **Platform policy check** (deferrable to Phase 2, but cheap to do early): does
   running your own workload on a GPU listed for rent affect reliability or
   availability scoring on Vast/TensorDock? This determines whether the donated-host
   model works as designed (Review Finding M).

8. **Contributor agreement.** §6.10 gates `internal` and `restricted` runs on clearance,
   which implies contributors accept *something* before receiving non-public data. It
   needn't be elaborate, but it needs to exist before the first non-`open` run, and
   it's the natural place to also settle who owns the resulting adapters. Cheap now,
   awkward once data has already moved. *Needed before the first `internal` run, not
   for M0.*

Items 1, 2, and 5 gate M0, which gates everything. They're worth settling first.
