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

## M1 — Coordinator

**~5–6 days** (was 4–5; self-hosted storage adds MinIO standup, TLS, GC, and backup).

FastAPI + SQLite: schema (§6.1), API (§6.2), auth (§6.3), round state machine (§3.1),
lease manager, aggregator with acceptance gates (§5), and self-hosted MinIO with
presigned URL minting (§6.5).

Storage is self-hosted on the coordinator VM, on a **separate volume from the OS
disk**, with S3 config kept in environment variables so the later R2 swap is a config
change rather than a code change.

Deliverables:
- `ganymede/coordinator/` — app, models, rounds, leases, aggregate, auth
- `ganymede/coordinator/aggregate.py` with **both** combine modes (plain weighted mean
  and DiLoCo outer momentum), selectable per run — M4 needs to A/B them
- `ganymede/coordinator/store.py` — thin S3 wrapper, `boto3` with `endpoint_url` from
  env. Restricted to the portable API subset in §6.5
- `ganymede/coordinator/budget.py` — per-worker step budgets from observed and
  calibrated throughput (§3.4), plus cache-affinity run selection (§6.6)
- `deploy/` — MinIO service unit, reverse proxy + TLS config, separate volume mount
- `scripts/backup.py` — off-box SQLite dump + latest-adapter copy, on round close
- `scripts/gc.py` — drop worker submissions older than 3 closed rounds
- `scripts/newrun.py`, `scripts/issue-key.py` — admin CLI
- Test suite with a **fake worker**: concurrent claims, lease expiry, late submission,
  gate rejection, quorum-vs-deadline close, zero-submission reopen

Exit criteria:
- N fake workers concurrently claim, train (with stub adapters), and submit; rounds
  advance correctly
- No double-leasing under concurrent claim (the `BEGIN IMMEDIATE` path is explicitly
  tested)
- A NaN adapter, a wrong-shape adapter, and a pickle file are each rejected with the
  right reason
- Killing the coordinator mid-round loses nothing already submitted
- Two fake workers with 3× different throughput both finish a round near the deadline,
  and the aggregate weights them by actual steps (§3.4)
- **A presigned URL minted by the coordinator is fetchable by an external client**
  over the public hostname. This is the §6.5 signing footgun; catch it here, with one
  integration test, rather than in M2 against a real GPU
- Backup lands **off-box** and GC actually deletes. An untested backup script is
  indistinguishable from no backup script

Do this before any Docker work. Testing the state machine against fake workers is
vastly faster than testing it against real GPUs.

---

## M2 — Worker container

**~3–4 days.**

Image stack (§4.1), entrypoint loop (§4.2), SIGTERM/abandon (§4.4), isolation flags
(§4.5). The trainer is M0's, unchanged.

Deliverables:
- `docker/torch-base.Dockerfile`, `worker-core.Dockerfile`, `worker-llm.Dockerfile`
- `ganymede/worker/` — loop, client, GPU probe, signals
- CI that builds and pushes all three, pinning `torch-base` **by digest**

Exit criteria:
- One real GPU claims from a real coordinator, trains, submits, loops
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

**~2 days.**

`IdleBackend` protocol with the `local` implementation (§7.1), manifest reconciliation,
systemd timer + unit, the `/etc/ganymede/pause` kill switch.

Deliverables:
- `ganymede/host/` — agent, idle backends
- `packaging/ganymede-host.service` + `.timer`
- HF cache size cap with LRU eviction (§6.6) — many runs means many ~16 GB base
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
- Mixed-speed workers (at least two GPU classes) both land near the round deadline,
  confirming §3.4's budgets rather than truncating the slower card every round

Everything before this is plumbing. This is where you find out whether the idea works.
Budget for it to fail the first time and need a tuning pass — that is the normal
outcome, not a signal to abandon the approach.

---

## M5 — Operations

**~2 days.**

Per-round loss and per-contributor throughput surfaced on the read-only `/status`
page, coordinator alerting (round stalled, quorum missed repeatedly, gate rejection
rate spiking), a runbook, and verified restore-from-backup.

Exit criteria:
- You can answer "is it training, and who is contributing?" in one glance
- You get alerted about a stalled run without having to look
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
2. **`vast` / `tensordock` `IdleBackend`s** — needed when you move past own hardware.
   Small, given M3's interface. Verify the platform-policy question first (§7).
3. **Postgres migration** — when workers exceed a few hundred (§6.1).
4. **Hivemind `SyncBackend`** — when central bandwidth genuinely binds. At ~3.4 GB/hr
   for ten workers, that is a long way off. Do not do this early; it buys nothing at
   current scale and costs NAT traversal work.
5. **`rl_rollout`** — needs its own spec review. Isaac Lab's image size and PyTorch
   pinning break the shared-layer story (Review Finding B), and shipping raw
   trajectories over WAN is a heavier data path than shipping weight deltas. Worth
   evaluating whether workers can send gradients or advantages instead of trajectories.
6. **gVisor/Kata, TOPLOC verification, Byzantine-robust aggregation** — when the pool
   includes people you don't know. §5.2's weighted mean is the drop-in point for
   trimmed-mean/median.

---

## Open questions — needed before M1

Blocking or near-blocking. Rough order of urgency.

1. **Base model and license.** Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B? The license
   governs whether you can redistribute merged weights or adapters to contributors,
   which is an architectural constraint, not a legal footnote. *Needed for M0.*

2. **The dataset.** What is it, how large, where does it live? Two things matter
   beyond size: it must be shardable into ~1000 stable buckets, and **every worker
   sees it in plaintext**. If any of it is sensitive, that changes the trust model
   materially and pulls Phase 2 items forward. *Needed for M0.*

3. ~~**Object store.**~~ **Resolved: self-hosted MinIO on the coordinator VM**, S3
   config in env vars so R2 is a later config swap (§6.5). Two follow-ons this raises:
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

6. **Contributor count and hardware at M4.** Three is the minimum for a meaningful
   convergence test. Which specific cards? The slowest one sets round sizing (§3.3).
   *Needed for M3/M4.*

7. **Platform policy check** (deferrable to Phase 2, but cheap to do early): does
   running your own workload on a GPU listed for rent affect reliability or
   availability scoring on Vast/TensorDock? This determines whether the donated-host
   model works as designed (Review Finding M).

Items 1, 2, and 5 gate M0, which gates everything. They're worth settling first.
