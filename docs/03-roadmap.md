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

**Decided: Qwen 1.7B, dense, bf16.** Confirm the exact model ID at M0 — the family
moves quickly. Scale to 7–8B after M4 passes, as a run-config change.

This is worth more than it sounds:

- **No `nf4` needed.** A 1.7B model in bf16 is ~3.4 GB of weights — it fits a 12 GB
  3060 and a 16 GB Mac comfortably. That removes the entire nf4/MPS exclusion problem
  (§6.8 Tier 3) during bring-up, so **every contributor is eligible for the bring-up
  run**, including the Macs.
- **You exercise the heterogeneity paths early**, which is where the interesting bugs
  are. On an 8B nf4 run the Macs sit idle and you'd find their bugs months later.
- **Rounds are minutes, not hours.** You can iterate the whole pipeline several times
  a day instead of once. Use ~10-minute rounds for bring-up rather than §3.4's 15–20.
- **Adapters are ~10–15 MB**, so the storage and bandwidth plumbing gets tested
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
afternoon.** At Qwen 1.7B almost anything with a GPU qualifies, and a few hours of
three small rentals costs about as much as lunch. That is a very cheap way to
de-risk the thesis before asking anyone to install anything.

Exit criteria:
- Aggregated model **matches or beats** the M0 single-node baseline on the held-out set
- Wall-clock-to-target-loss beats single-node — otherwise the system is an expensive
  way to train slower
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
7. **`rl_rollout`** — needs its own spec review. Isaac Lab's image size and PyTorch
   pinning break the shared-layer story (Review Finding B), and shipping raw
   trajectories over WAN is a heavier data path than shipping weight deltas. Worth
   evaluating whether workers can send gradients or advantages instead of trajectories.
8. **gVisor/Kata, TOPLOC verification, Byzantine-robust aggregation** — when the pool
   includes people you don't know. §5.2's weighted mean is the drop-in point for
   trimmed-mean/median.

---

## Open questions

Three remain. Two of the earlier ones closed themselves.

1. **The eval set and target metric.** What number decides M4b passed? Define it
   during M0 while the baseline is being established, not after — a baseline you can't
   compare against isn't one. *Needed for M0's exit criteria.*

2. **Your own dataset**, when bring-up is done: what it is, how large, where it lives,
   and which classification it warrants (§6.10). Dolly carries M0–M4; this is the
   first *real* run's input. *Not needed until after M4.*

3. **Contributor agreement**, before the first non-`open` run. §6.10 gates `internal`
   and `restricted` runs on clearance, which implies contributors accept something
   before receiving non-public data. It needn't be elaborate, but it's also the
   natural place to settle who owns the resulting adapters. Dormant while you're the
   only contributor. *Needed before the first non-`open` run.*

### Closed

- ~~**Base model**~~ → Qwen 1.7B dense bf16 for bring-up, 7–8B later.
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
