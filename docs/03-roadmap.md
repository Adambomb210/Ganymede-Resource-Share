# Ganymede — Plan of Attack

Milestones for `llm_finetune` v1, per `02-architecture-v2.md`. This follows the
original spec's Appendix ordering, which was right — infra last, working training
first — with the milestones made concrete and given exit criteria.

Day estimates are focused-work days for one developer, and they assume the open
questions in §7 are answered. They are for sequencing, not for promising dates.

---

## M0 — Single-node training, no infrastructure

**~2–3 days. The most important milestone. Do not skip or compress it.**

One script, one GPU, no coordinator, no Docker, no network:

```
train_task.py --task task.json --base-adapter in.safetensors --out out.safetensors
```

This is exactly the §4.3 trainer, driven by a hand-written task spec file. It is the
same code path the worker will call — not a throwaway prototype.

Deliverables:
- `ganymede/train/lora.py` — the ~250-line trainer
- `ganymede/train/data.py` — bucket sharding, deterministic given `(seed, buckets)`
- A held-out eval set and a `baseline.json` recording single-node loss/metrics

Exit criteria:
- Trains, loss descends, adapter saves and reloads as safetensors
- Fits the target 24 GB card at the chosen `base_precision` with headroom
- **Measured throughput (steps/min, tokens/s) recorded** — this is what sets
  `local_steps_per_sync` for §3.3, and guessing it instead is how you end up with
  6-hour rounds nobody ever completes
- **A single-node baseline result exists.** Without it, M4 cannot tell whether
  distribution helped, hurt, or did nothing

Why this dominates everything: if 8B LoRA doesn't fit comfortably, or throughput makes
30-minute rounds impractical, that invalidates round sizing, bandwidth estimates, and
possibly the model choice. Find out with one script, not with a deployed swarm.

---

## M1 — Coordinator

**~4–5 days.**

FastAPI + SQLite: schema (§6.1), API (§6.2), auth (§6.3), round state machine (§3.1),
lease manager, aggregator with acceptance gates (§5), presigned URL minting, SQLite
backup-to-object-store on every round close.

Deliverables:
- `ganymede/coordinator/` — app, models, rounds, leases, aggregate, auth
- `ganymede/coordinator/aggregate.py` with **both** combine modes (plain weighted mean
  and DiLoCo outer momentum), selectable per run — M4 needs to A/B them
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
- `INSTALL.md` — the contributor-facing document. This is the actual product surface
  for everyone who isn't you; treat it as a deliverable, not an afterthought

Exit criteria:
- Machine idles → worker starts within one timer interval
- `touch /etc/ganymede/pause` → running container stops, no new ones start
- Manifest tag bump → agent pulls and restarts on the new tag with no manual step
- **A second person installs it from `INSTALL.md` alone, with no help from you.** That
  test is the milestone; a working agent that only you can install isn't done

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
- **Coordinator restore from backup has been performed at least once**, not merely
  configured. §6.4 names this as the system's single point of failure; an untested
  backup isn't one

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

3. **Object store.** Recommend **Cloudflare R2** — S3-compatible, and zero egress fees,
   which matters directly because every worker pulls a base adapter every round.
   Plain S3 works but you'll pay per-round egress that scales with contributor count.
   MinIO on the coordinator VM is viable for early testing and makes the VM a
   bandwidth bottleneck later. *Needed for M1.*

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
