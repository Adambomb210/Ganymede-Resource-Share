# Ganymede Resource Share

Donated idle-GPU compute for collaborative model training.

Contributors run a worker on GPUs that would otherwise sit idle. A single small
always-on coordinator hands out work, combines the results, and keeps the durable
checkpoints. Workers are ephemeral by design — they claim work when the GPU is free
and disappear when it isn't.

**Status:** design phase. No implementation yet.

## Documents

| Doc | What it is |
|---|---|
| [`docs/00-original-spec-v1.md`](docs/00-original-spec-v1.md) | The original draft spec, kept for reference |
| [`docs/01-spec-review.md`](docs/01-spec-review.md) | Review of v1 — findings, decisions, and alternatives considered |
| [`docs/02-architecture-v2.md`](docs/02-architecture-v2.md) | **Current architecture.** Supersedes v1 |
| [`docs/03-roadmap.md`](docs/03-roadmap.md) | Milestones, exit criteria, and open questions |

## v1 scope

- **Job type:** `llm_finetune` only — LoRA on a dense 7–8B Qwen base
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
