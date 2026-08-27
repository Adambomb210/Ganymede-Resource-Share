# Ganymede — Sandbox, Image Pipeline & Kill Path

*Stage 1 component design (`04-platform-expansion.md`, Build sequencing). Imports the
frozen spine — `05-data-model.md` (the `images`, `jobs`, `submitters` tables; the
`tasks.status` / `jobs.status` enums) and `06-api-delta.md` (the `/v1/images/*`
endpoints, the heartbeat `cancel` field). Threat model is Decision 3: job code comes
from a handful of manually-approved submitters, so the adversary is **a trusted
author's honest mistake** — a baked-in credential, a runaway loop, an accidental fetch
to a private host — not someone attacking the runtime. That buys resource caps,
default-deny egress, and read-only mounts; it does not buy gVisor / Kata or the
assumption that job code is hostile (§5).*

---

## 1. Image pipeline: upload → finalize → scan → pull

### 1.1 Upload and finalize

`POST /v1/images/upload-url` (auth **submitter**) takes `{repo_tag, digest,
size_bytes}`, inserts an `images` row (`finalized_at NULL`, not worker-visible), and
returns `{image_id, url, digest_required}`. `url` is a presigned PUT against the
object store (`06`; big blobs never transit the API body), signed with a
content-length ceiling of `GANYMEDE_IMAGE_MAX_BYTES` (default 10 GiB).

**Digest.** The payload is a `docker save` archive. `digest` / `digest_required` is the
SHA-256 of that archive — the one value a worker can recompute from the bytes it pulls,
as opposed to the OCI manifest digest, which it cannot. The submitter's client computes
it and sends it in the `upload-url` body; the coordinator stores it in `images.digest`
and never re-hashes the body (it never sees it).

`POST /v1/images/{id}/finalize` sets `finalized_at` and `size_bytes`, sets
`scan_status='pending'` (`06`), and enqueues the scan. It rejects (`422`) if
`size_bytes` exceeds the cap or a `HEAD` on `object_ref` shows no object. Content
verification is deferred to the worker, not skipped — see §1.4 and §2.3.

### 1.2 `images` rows and how `jobs` reference them

An `images` row is immutable once finalized; a rebuild is a new row with a new digest.
`jobs.image_id → images(id)`; many jobs may pin one image. `NULL` only for first-party
built-ins (§4).

**Retention.** A GC cron (`list_prefix` + `delete`, the §6.6 pattern) keeps an image
while any non-terminal job references it, and for `image_keep_days` (default 30) after
the last referencing job reaches a terminal `jobs.status`. Un-finalized rows
(`finalized_at IS NULL`) are reaped after 24 h.

### 1.3 The scan (`pending → clean | flagged`)

Runs coordinator-side, out-of-band, after `/finalize`, as a queued unit of work inside
a throwaway confined container (it unpacks untrusted layers). Single pass, no network.
Writes `scan_status` and `scan_detail_json` (§ Spine deviations). At the honest-mistake
level it checks four things:

- **Manifest sanity.** Valid Docker / OCI schema; `linux/amd64`; layer count and
  uncompressed size within bounds (a decompression-bomb guard, not escape analysis).
- **Base-image provenance.** Walk the base layers' digests; the bottom of the stack
  must match a vetted set — the pinned `ganymede/torch-base` digests (§4.1), official
  CUDA runtime images, distroless. An unrecognised base → `flagged` for a human, not a
  permanent deny.
- **Obvious secrets.** Regex sweep of the image filesystem for the honest mistake:
  `id_rsa`, `.env`, `.git/`, AWS / GCP keys, `~/.docker/config.json`, bearer tokens.
- **Entrypoint sanity.** `ENTRYPOINT` / `CMD` present, non-empty, resolves to a real
  path; `USER` is non-root (the runtime forces `--user` regardless); no `--privileged`
  markers.

Explicitly **not** done: syscall tracing, behavioural malware analysis, supply-chain
attestation. Those answer the adversarial threat model and are deferred with
gVisor / Kata (§5).

### 1.4 A non-`clean` image cannot be scheduled

The claim-path queue walk (`06`, `_selectable_jobs`) skips any job whose `image_id`
resolves to `scan_status != 'clean'`:

- `pending` — the job sits in `queued`; `plan()` is not called, no tasks exist.
- `flagged` — blocked until an admin dispositions it (§ Spine deviations).

`POST /v1/jobs/{id}/enqueue` is still allowed against a `pending` image — the scan and
the queue advance in parallel — but no lease is ever issued against a non-`clean`
image, and the worker re-checks `digest` after pull (§2.3) as the last gate.

---

## 2. Runtime confinement on the worker

### 2.1 Who launches the job container

The worker owns the claim / heartbeat / submit loop (§4.2) and now also supervises the
job container. It does **not** get the host's Docker socket — that is root-equivalent
on the machine and defeats §4.6. The host agent starts the worker as it does today
(`runtime.run_argv` unchanged) plus one of:

- a **scoped runtime handle** — a socket proxy that permits only
  `create / start / stop / kill / inspect / rm` against a fixed flag template (§2.2–2.4),
  nothing else; or
- **rootless Podman** inside the worker container, for a host that will not expose
  even a proxied socket.

The worker launches the job image as a **sibling** container with those flags, stages
inputs into scratch, and signals the job directly on `cancel` (§3). The host agent's
existing one-shot tick (§7) is the backstop, not a second control path: on any tick it
reaps a job container whose supervising worker has exited or whose lease crumb (§3) is
stale, using the `status()` / `stop()` it already has. No new daemon, no new IPC.

First-party built-ins run as the worker process itself (§4) — none of §2 applies to
them.

### 2.2 Resource caps — deltas from the §4.6 baseline

§4.6 already gives `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--read-only`,
non-root `--user`, `--memory`, `--cpus`, `--pids-limit`, and Docker's default
seccomp / AppArmor. The job container inherits all of it. Added for submitter code:

| Resource | Mechanism | Default |
|---|---|---|
| Memory | `--memory` **and** `--memory-swap` equal (swap off) | host `memory` (§7), e.g. `16g`; OOM → task `failed` |
| GPU | the host's configured `--gpus` value stands (§7, default `all`); one task per machine (Decision 4) already makes the GPU exclusive. MIG / MPS partitioning out of scope (invariant 3) | — |
| Disk (scratch) | a quota'd mount, or `--storage-opt size=` where the driver allows, else a sized tmpfs | `job_scratch_gb`, default 50 |
| Wall time | `spec.max_runtime_sec`, enforced by the worker; `--stop-timeout` the hard backstop | ceiling `job_max_runtime_sec` |
| IPC / core | `--ipc=private`, `--ulimit core=0` | always |

### 2.3 Filesystem — scratch and nothing else

- `--read-only` rootfs. Writable: `/scratch` (the task scratch dir — quota'd, wiped on
  task exit), `/tmp` and `/run` as `--tmpfs`. Nothing else.
- **No host path is bind-mounted except the scratch dir.** In particular the worker's
  state dir is not visible to the job — it carries the kill channel, and §4.6's own
  reasoning is that a process which can write there can forge the switch.
- The §6.7 HF cache is not mounted into a submitter job (it ships its own deps). A job
  type that wants a shared model cache gets a **read-only** mount of a per-job-type
  prefix.
- I/O: the worker stages `input_ref` into `/scratch/in` via presigned GET before
  start; the job writes `/scratch/out`; the worker uploads from there via presigned
  PUT (`06` `upload-url` / `submit`). The job container does no object-store I/O
  itself.
- After pull, before `docker load`, the worker recomputes the archive SHA-256 and
  compares it to `image_digest` from the task payload (`06`). Mismatch → `abandon`,
  reason `image_digest_mismatch`, task re-queued. The loaded image is run by ID, not
  by the archive's embedded tag.

### 2.4 Egress — default-deny, per-job allowlist

- Default `--network none`. The common job (read `/scratch/in`, compute, write
  `/scratch/out`) needs no network — the worker does all transfer.
- A job that legitimately fetches declares `spec.egress_allow` (a list of hostnames).
  The worker attaches the job to an internal bridge whose only route is an HTTP
  `CONNECT` proxy it runs, permitting exactly those hosts on `:443` and
  denying-and-logging everything else. `HTTPS_PROXY` / `NO_PROXY` are injected into the
  job env. This is §4.6's "three destinations, all named, all HTTPS" generalised per
  job.
- The coordinator and object-store hosts are on the effective allowlist implicitly —
  they are the worker's path, not the job's. Nothing else is implicit.
- Denied connections are logged and surfaced on the job detail page. At the
  honest-mistake level the policy's job is to *tell the submitter* "your code tried to
  reach X and was blocked", not to fail silently.

---

## 3. The soft / hard kill (Decision 18)

**Normal path — worker healthy.**

1. `POST /v1/jobs/{id}/cancel {mode}` (owner or admin), or an admin cancel after
   allowlist revocation (`06`) → `jobs.status='cancelled'`, `jobs.cancel_mode=mode`.
2. The coordinator attaches `cancel: "soft" | "hard"` to the heartbeat response of
   every leased task of that job (`06`). This is the whole transport — pull-only
   (Decision 8).
3. On that heartbeat the worker acts on the job container:
   - **`soft`** → SIGTERM (`docker stop --time`). The job is expected to trap it,
     checkpoint the current unit, and exit within `cancel_grace_sec` (default 120, and
     never past the lease); a job that ignores it is SIGKILLed at the timeout. The
     checkpoint contract is the job-type SDK doc's. Then `POST /abandon`.
   - **`hard`** → SIGKILL now (`docker kill`), then `POST /abandon`.
4. The coordinator, seeing the lease dropped with a cancel outstanding, sets
   `tasks.status='cancelled'` (not `abandoned` / `expired`) — matches `05`.
5. A task with a cancel outstanding is **not re-dispatched** while it drains
   (`leased → cancelled` is terminal in `05`) — no second container on one unit.

Worst-case latency from cancel to the worker acting is one `heartbeat_interval_sec`
(`06`).

**Wedged worker — not heartbeating, not reaping the job container.**

The worker writes `lease.renewed_at` to its state dir on every heartbeat. On its next
tick the host agent treats a running job container whose `lease.renewed_at` is older
than `lease_seconds` as orphaned: `docker kill` the job container, `docker rm -f` the
worker, and let the following tick start a fresh one. Independently the coordinator
expires the lease at `lease_seconds` and the task goes `cancelled` (cancel
outstanding) or `expired`. In this path the soft / hard distinction collapses to hard
— a wedged worker gets no graceful drain — and the latency bound is one host-timer
interval (§7, default 900 s) rather than one heartbeat.

---

## 4. Interaction with the `required_image` path

- **First-party built-in job types** — `collab_lora_finetune` now, a first-party
  `batch_inference` next — carry `jobs.image_id IS NULL`. They keep the §4.1 / §7 path
  unchanged: the host reconciles the manifest's `required_image` tag, the worker runs
  in `ganymede/worker-*` (or a native install) under the §4.6 baseline. No
  `docker load`, no per-job pull, no egress bridge. This code is first-party and
  digest-pinned (§4.1); nothing in §1–§3 applies to it.
- **Submitter job types** always carry `image_id` and always take the uploaded-image +
  confinement path above. Never native. A host with no container runtime (macOS,
  native Windows / Linux — §4.1) fails the match with a new `worker_eligibility` reason
  `no_container_runtime`, recorded like any capability miss (`05`). That is the §4.6
  split made explicit: own hardware may run first-party work natively; submitter code
  only ever runs contained, on a host that opted into Docker.
- `_task_payload` already carries the distinction — `image_ref` / `image_digest` /
  `image_pull_url` are `null` for built-ins, set for submitter jobs (`06`).

---

## 5. Out of scope, and the upgrade path

- **gVisor / Kata** (`runsc`, `kata-runtime`) — deferred (Decision 3; `03` Phase 2
  item 9). Revisit when the submitter allowlist opens past authors the operator
  personally vouches for. Drop-in: a `--runtime` value in the sibling-container flags;
  the scan, egress, and kill machinery here is unchanged.
- **TEE / confidential compute** (SEV-SNP, TDX, H100 CC) — out. It defends job data
  against a hostile *host*, which is the data-plane threat model (Decision 16,
  deferred) and the inverse of this doc's. Revisit with §6.10 data classification and
  the first sensitive dataset.
- **Network-policy engines** (Cilium, Calico, Kubernetes NetworkPolicy, OPA) — out.
  There is no orchestrator: one job container per host, one `CONNECT` proxy. Revisit
  only if the fleet moves to multi-container pods.
- **Custom seccomp / AppArmor / SELinux authoring** — out at the honest-mistake level;
  Docker's default profiles plus `cap-drop=ALL` and `no-new-privileges` are the
  baseline (§4.6). Author a profile when the threat model flips to adversarial.
- **Image signing / SLSA provenance** (cosign) — not required; the trust anchor is the
  vetted submitter identity and the allowlist (Decisions 3, 9). A cheap add if the
  submitter pool grows.

---

## Spine deviations

No contradictions with `05` / `06`. The heartbeat `cancel` field,
images-through-the-object-store, presigned PUT / GET, and the four status enums are
used exactly as frozen. The following are additive:

- **Admin scan disposition.** Moving a `flagged` image to `clean` after review, or
  forcing a re-scan, has no endpoint in `06`. Proposed, additive, **auth: admin**,
  same conventions: `POST /v1/admin/images/{id}/scan`
  `{disposition: "clean" | "flagged" | "rescan", note}`. Nothing else in the frozen
  API moves.
- **`images` columns.** This doc adds `finalized_at TEXT`, `scanned_at TEXT`, and
  `scan_detail_json TEXT` — additive, within the allowance `05` grants a component doc
  over its own new tables. They land with the rest of `images` in migration 002–003.
- **`worker_eligibility` refusal reason** `no_container_runtime` — anticipated by `05`
  ("refusal reasons now include constraint misses"); named here so the scheduler doc
  and this one agree on the string.
