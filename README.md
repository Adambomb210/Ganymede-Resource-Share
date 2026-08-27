# Ganymede Resource Share

Donated idle-GPU compute for collaborative model training.

Contributors run a worker on GPUs that would otherwise sit idle. A single small
always-on coordinator hands out work, combines the results, and keeps the durable
checkpoints. Workers are ephemeral by design — they claim work when the GPU is free
and disappear when it isn't.

**Status:** M1 in progress — the coordinator. The design docs below are current;
`ganymede/coordinator/` is the implementation.

## Documents

| Doc | What it is |
|---|---|
| [`docs/00-original-spec-v1.md`](docs/00-original-spec-v1.md) | The original draft spec, kept for reference |
| [`docs/01-spec-review.md`](docs/01-spec-review.md) | Review of v1 — findings, decisions, and alternatives considered |
| [`docs/02-architecture-v2.md`](docs/02-architecture-v2.md) | **Current architecture.** Supersedes v1 |
| [`docs/03-roadmap.md`](docs/03-roadmap.md) | Milestones, exit criteria, and open questions |
| [`INSTALL.md`](INSTALL.md) | **For contributors lending a machine.** Install, settings, and the off switch |

## Layout

| Path | What it is |
|---|---|
| `ganymede/coordinator/` | The coordinator: API, round state machine, aggregation, storage |
| `ganymede/trainer/` | The trainer a worker runs, plus the calibration and baseline harnesses |
| `ganymede/worker/` | The worker package a contributor installs: probe, client, loop |
| `ganymede/host/` | The host agent (§7): decides whether and what to run. Standard library only |
| `docker/` | The three-layer worker image stack (§4.1) |
| `configs/` | Run configs — one file feeds calibrate, baseline and `newrun` alike |
| `scripts/` | Admin CLI — create a run, issue a key, evaluate rounds, GC, back up and restore |
| `deploy/` | Container image, compose file, env template |
| `packaging/` | systemd, launchd and Task Scheduler units, and the three installers |
| `tests/` | Unit tests per module, plus a fake-worker integration suite |

### Coordinator modules

| Module | Responsibility |
|---|---|
| `config.py` | Settings from environment; the named tuning constants |
| `db.py` | Schema, WAL pragmas, and the `BEGIN IMMEDIATE` transaction helper |
| `auth.py` | Bearer keys, stored hashed |
| `budget.py` | Per-worker step budgets, bucket scaling, eligibility. Pure arithmetic |
| `aggregate.py` | Acceptance gates and the per-tensor outer step |
| `store.py` | S3-compatible object storage, restricted to the portable subset |
| `rounds.py` | Round lifecycle, leases, the work-based close rule |
| `closer.py` | Gating, aggregation, publishing the next round's base adapter |
| `invariants.py` | What "broken" means, checked over the database. Non-zero exit, so it works as a cron check |
| `eligibility.py` | Why a worker is never leased (M5). Verdicts recorded by the claim path, never re-derived |
| `app.py` | The HTTP surface |

### Trainer modules

| Module | Responsibility |
|---|---|
| `data.py` | Dataset resolution, bucketing, prompt format, loss masking. No model — pure enough to test in milliseconds |
| `model.py` | Base loading at the run's pinned precision, LoRA attach, adapter serialization |
| `train.py` | `run_task` and the one optimizer loop the baseline shares |
| `evaluate.py` | Held-out loss and the 20-prompt greedy smoke set |
| `calibrate.py` | `ganymede-calibrate` — fit probe, throughput, recommended `local_steps` |
| `baseline.py` | `ganymede-baseline` — multi-seed single-node reference for M4 |

### Worker modules

| Module | Responsibility |
|---|---|
| `probe.py` | The §6.9 self-test: allocation ceiling, precision support, bench score. Backends live in a registry, so AMD and Intel are one entry each |
| `client.py` | The §6.2 API over stdlib `urllib` — worker-core stays dependency-light |
| `control.py` | The §4.4 stop/pause sentinel files, with signals as an optimization |
| `loop.py` | The §4.2 entrypoint loop, and the `ganymede-worker` CLI |

### Host agent modules

Nothing here imports anything outside the standard library. The host agent runs
*outside* the container, on a machine where the only guaranteed thing is a
Python interpreter — so it installs by copying a directory, and the docker path
installs it with `--no-deps`.

| File | What it owns |
|---|---|
| `config.py` | Settings. A file first and environment second, because a timer's environment is whatever the init system decided |
| `idle.py` | §7.1's `IdleBackend`: pause sentinel, time window, GPU free, user idle across three platforms. Returns a reason, not just a bool |
| `cache.py` | §6.7's cache cap with LRU eviction, and the `ganymede-cache` CLI |
| `manifest.py` | §7 step 3 — which image this machine should run, including when two active runs disagree |
| `runtime.py` | Starting and stopping: Docker and native, behind one protocol (§4.1's delivery table) |
| `agent.py` | The tick, and the `ganymede-host` CLI a timer invokes |

```sh
ganymede-worker --probe-only    # what to ask a contributor for when they get no work
```

### Operating a live run

```sh
python3 -m ganymede.coordinator.invariants --db … --run-id …   # is anything wedged?
python3 -m scripts.evalround --run-id … --watch                # fill in held-out loss

ganymede-host --check                                          # what a contributor's machine resolved
ganymede-cache --cap-gb 40 --dry-run                           # what eviction would reclaim
```

Two of those answer the operator's standing questions:

```sh
ganymede-status --db …                    # is it training, and who is contributing?
ganymede-status --db … --alert            # silent when healthy, exit 1 when not -- for cron
ganymede-eligibility --db … --fleet       # why nobody is getting work
```

The first exits non-zero on a violation, so it belongs in cron. It reports the two
failures that are otherwise **silent** — a round wedged mid-close and a lease nobody
will reclaim — and deliberately says nothing about an idle fleet, which is normal
operation rather than an incident.

The second is separate from the coordinator on purpose: a forward pass over the
held-out split is minutes of CPU, and running it inside the close would hold one
arbitrary contributor's HTTP response open for all of it. It needs the trainer extra,
so it runs wherever that already lives.

The trainer's stack (`transformers`, `peft`, `datasets`) is an optional extra, not a
base dependency: the coordinator never loads a base model — `newrun.py` derives the
adapter manifest from `config.json` on a meta device — and the coordinator box is the
one machine in the system that should stay small.

```sh
pip install -e '.[trainer]'      # only on machines that train
```

## Development

```sh
pip install -e '.[dev]'
python3 -m pytest tests/ -q
```

The integration suite runs against an in-memory object store, so it needs no
containers. `tests/test_store.py` exercises the real S3 path against MinIO in
Docker and skips cleanly when Docker is unavailable.

The trainer suite is offline too: it trains a genuine 107k-parameter `Qwen3ForCausalLM`
with a real tokenizer, built in the fixture. Same architecture as the bring-up model,
so target modules, parameter naming and shapes are all the production ones — at a size
where a full run costs milliseconds. The claims that need a real model live in
`tests/test_trainer_cpu.py` behind `-m slow`.

`tests/test_worker_concurrency.py` is the M4a suite: three real worker *processes*
against a real coordinator and real MinIO, driving a multi-round run to completion,
plus a `SIGKILL` mid-round to check the blast radius. Also `-m slow`. Run it on its
own — the MinIO fixture uses a fixed container name and port, so a second pytest
process tears the container down underneath it.

## v1 scope

- **Job type:** `llm_finetune` only — LoRA on a dense Qwen base (bring up on `Qwen3-1.7B-Base`, scale to `Qwen3-8B-Base`)
- **Sync:** central aggregation (weighted mean + DiLoCo outer step), behind a
  `SyncBackend` seam so peer-to-peer can be swapped in later
- **Hosts:** own hardware first; donated and rented platform hosts behind the same
  `IdleBackend` interface
- **Storage:** self-hosted, S3-compatible (MinIO), with Cloudflare R2 or S3 reachable
  by a config change rather than a rewrite
- **Runs:** one active run at a time; concurrent runs deferred
- **Fleet:** heterogeneous by design. The worker is a pip-installable package that
  runs on almost any hardware; the container is one delivery path among several.
  Capabilities are probed, not enumerated, so unanticipated devices work without a
  coordinator change

- **Data:** sensitivity varies by run; a per-run classification gates who is eligible

**Platform compatibility and run eligibility are separate concerns.** A machine that
can't serve a particular run still joins the fleet and waits for one that suits it.
- **Trust:** trusted circle, with safetensors-only artifacts and sanity gates on every
  submission

`rl_rollout`, Hivemind, and the rest of the trust stack are Phase 2 — see
[`docs/03-roadmap.md`](docs/03-roadmap.md).

## License

MIT — see [LICENSE](LICENSE).
