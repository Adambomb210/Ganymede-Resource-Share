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

## Layout

| Path | What it is |
|---|---|
| `ganymede/coordinator/` | The coordinator: API, round state machine, aggregation, storage |
| `scripts/` | Admin CLI — create a run, issue a key, GC, backup |
| `deploy/` | Container image, compose file, env template |
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
| `app.py` | The HTTP surface |

## Development

```sh
pip install -e '.[dev]'
python3 -m pytest tests/ -q
```

The integration suite runs against an in-memory object store, so it needs no
containers. `tests/test_store.py` exercises the real S3 path against MinIO in
Docker and skips cleanly when Docker is unavailable.

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
