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

**That threat model is about the image's *code*, and it does not extend to the job
*spec* (review, 2026-09-21).** A spec names its inputs by object key —
`spec.shards[i].ref`, `spec.model_ref` — and `inputs_for` hands those keys straight to
`store.presign_get` to mint a URL the worker fetches. Nothing constrained them, so an
approved submitter could name **any** key and have the coordinator sign a read of it on
their behalf. The reachable targets are the coordinator's own reduce-internal state:
`runs/<id>/momentum.safetensors` (the DiLoCo outer momentum) and
`runs/<id>/rounds/<n>/base.safetensors`, neither of which any worker is handed through
the legitimate round protocol, and both of which may belong to a run whose
`data_classification` that submitter has no clearance for. The `run_id` needed to build
the key is not secret: `/ui/` renders every run's id to any authenticated contributor,
with none of the `clearance_and_terms_permit` filtering `/v1/manifest` applies to the
same data.

Closed by `store.is_reserved_key`: a spec may not address `runs/` or `images/`, checked
at submission (a 422, so the submitter is told) and again in the presign helpers as
defence in depth. Keys are normalised first — leading `/`, `./`, `//`, `..` segments,
backslashes and an `s3://bucket/` prefix all collapse — so the guard cannot be stepped
around by respelling. Case is deliberately *not* folded: S3 keys are case-sensitive, so
`Runs/x` is genuinely a different object and is not the coordinator's.

**Deliberately a deny-list, not an allow-list, and that is the part still open.** There
is no submitter namespace to permit: shards are placed in the bucket out of band and
this codebase has no convention for where they live. Giving submitter data its own
prefix and requiring specs to stay inside it is the real fix and is a design decision,
not a patch. **Also still open:** `readmodel.dashboard()` selects every `runs` row with
no clearance filter while `/v1/manifest` filters the same data — the two disagree about
what a given contributor may even learn a `run_id` for.

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
(`06`). The heartbeat thread acts on the container itself the instant a cancel
latches, rather than leaving it to the job body's next `should_stop()` poll — a hard
cancel that waits for the work to come round is not a hard cancel. It reaches the
container by name, derived from the task id, so it needs no handle on a container the
job type owns, and it is a harmless no-op if the cancel lands before the container
exists.

**Wedged worker — not heartbeating, not reaping the job container.**

The worker writes a **lease crumb** on every heartbeat — `task_id`, `renewed_at`, and
the job container's name — **one file per task**, at `leases/<task_id>.json` under the
job scratch root. Not the state dir, as an earlier draft of this section said: the
state dir is mounted read-only into the worker precisely so the worker cannot forge
the contributor's kill switch, so the scratch root is the one directory both the
worker and the host agent can see and the worker may write.

Per task rather than one file per host, because a host may hold several leases at
once (`14`). A shared file would have the second task's heartbeat overwrite the
first's record, leaving the reaper one task's liveness for two containers and no way
to tell which.

On its next tick the host agent weighs **every** crumb on its own. A running job
container whose crumb is older than `lease_seconds` is orphaned: `docker kill` then
`docker rm -f` **that job container** — not the worker, as an earlier draft said —
and clear that crumb. One sweep may reap more than one container. Independently the
coordinator expires each lease at `lease_seconds` and the task goes `cancelled`
(cancel outstanding) or `expired`. In this path the soft / hard distinction collapses to hard
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

---

## §1 status — the pipeline and the gate, built

`ganymede/coordinator/images.py`, the `/v1/images/*` endpoints, and
`ganymede-imagescan`. What landed:

| §1 | State |
|---|---|
| `upload-url` → presigned PUT, `images` row not worker-visible | **built.** The declared length is signed into the URL (asserted on the `X-Amz-SignedHeaders` of a real signature); whether a given store *enforces* a signed content-length is not something the suite shows, so the guard that actually holds is `finalize`'s `HEAD` — which is why an oversized or missing body leaves the row un-finalized rather than schedulable |
| `finalize` → `scan_status='pending'`, scan enqueued | **built.** The queue *is* the `images` table (`pending` + a `finalized_at`), not a second table that could disagree with it |
| The four checks — manifest sanity, base provenance, secrets, entrypoint | **built**, with the decompression-bomb guard enforced while streaming |
| A non-`clean` image cannot be scheduled (§1.4) | **built**, and asserted through the claim endpoint rather than the selector |
| Admin disposition / re-scan | **built** (`POST /v1/admin/images/{id}/scan`), audited, and recorded *beside* the scan's own findings rather than over them |
| Retention GC (§1.2) | **built** (migration 007). What is collected is the *archive*, not the row: a terminal job still records which image it ran, and that is worth more than the row it occupies. A collected image keeps its id, digest and verdict, loses `object_ref`, and gains `collected_at`; an upload that never finalized goes entirely, since nothing can reference one. Rides the scan sweep |

### Two deviations, both narrower than the design

**The scan runs in-process, not in a throwaway container.** §1.3 specifies a
confined container because the scan unpacks untrusted layers. What stands in for
it today is that *nothing is ever unpacked*: no archive path is ever joined to a
filesystem path, every member is read through a shared byte budget, and a
tripped budget is a `flagged` verdict rather than an exception. That makes the
bomb guard load-bearing rather than advisory, which is why it is enforced on the
**decompressed** side of the gzip stream — a compressed size is bounded by the
upload cap already and tells you nothing about what comes out. The container is
still the stronger boundary and still the target; it is a change of one call
site, and the checks do not move with it.

**The archive is read whole rather than streamed from the store.** `get_bytes`
today, which is fine at the sizes being scanned and wrong at the 10 GiB cap. The
scan itself takes a stream and never seeks backwards, so this is a `Store`
change when it matters, not a scan change.

### One thing worth knowing before §2

**With no `GANYMEDE_VETTED_BASE_DIFF_IDS` configured, every image is flagged.**
That is the fail-closed reading of §1.3 rather than an oversight: with no vetted
set there is no such thing as a recognised base, and the alternative — passing
everything until someone remembers to configure it — is the failure mode this
check exists to prevent. The operator either lists the vetted `diff_id`s or
dispositions by hand, and both are visible.

---

## §2 / §3 status — confinement and the kill path, built

`ganymede/worker/sandbox.py`, the cancel field on the heartbeat, and the host
agent's orphan reaper.

| §2 / §3 | State |
|---|---|
| Resource caps (§2.2) | **built.** `--memory` and `--memory-swap` equal, `--cpus`, `--pids-limit`, `--ipc=private`, `--ulimit core=0`, and the §4.6 baseline inherited whole |
| Filesystem (§2.3) | **built.** `--read-only`, one bind mount (the task's scratch), `/tmp` and `/run` as tmpfs, and *not* the worker's state dir |
| Digest verification after pull (§2.3) | **built**, and ordered before `docker load` — an archive that failed its check is never parsed by anything. The image runs by **id**, never by the tag the archive carries |
| Egress (§2.4) | **`--network none`, unconditionally.** The per-job allowlist and its CONNECT proxy are deferred — see below |
| Soft / hard cancel (§3) | **built.** The mode rides the heartbeat response; `soft` is `stop --time <grace>`, `hard` is `kill` |
| Wedged-worker backstop (§3) | **built.** The worker writes a lease crumb on every heartbeat and the host agent's tick reaps a job container whose crumb has gone stale |
| Storage quota (§2.2) | **built, off by default.** `--storage-opt size=` is a hard error on overlay2, the common driver, so opting in is a config flag |

### Three deviations, all recorded rather than hidden

**The socket proxy is not shipped.** §2.1 says the worker must not get the
host's Docker socket, and offers a scoped proxy or rootless Podman. What is
built is the *seam*: the worker invokes a runtime **binary** and never a socket
path, so `GANYMEDE_JOB_RUNTIME` plus a `DOCKER_HOST` pointing at a proxy — or at
a rootless Podman — is a deployment change, not a code change. Until an operator
does one of those, a worker configured with a plain `docker` against the host
socket **has more than §2.1 wants it to have**. That is the honest state of it.

**The egress proxy is deferred.** §2.4's default-deny is what shipped;
`spec.egress_allow`, the internal bridge and the CONNECT proxy are not. Nothing
in tree declares an allowlist — by §4 every first-party type carries
`image_id IS NULL` and takes none of this path — so the proxy would be built
against no consumer, and it is the piece most likely to churn when the first one
appears. Default-deny is a complete state by §2.4's own reasoning: the common
job reads `/scratch/in`, computes, writes `/scratch/out`, and the worker does
every transfer.

**Soft and hard collapse for first-party built-ins.** There is no job container
to signal, so both modes mean "stop the loop, abandon the shard" — which is what
the existing `control.should_stop()` path already did. The distinction is real
only for submitter code, which gets a SIGTERM and a grace period to checkpoint.
Worth stating because someone will otherwise expect `hard` to SIGKILL a training
worker mid-step.

### The consumer arrived: `contained_batch`

`ganymede/jobtypes/contained_batch/` (docs/10 §7) is the first caller
`JobContainer` has ever had. It stages `/scratch/in` from a presigned GET, pulls
and **hashes** the archive before `docker load`, starts the container detached,
supervises it, and reads `/scratch/out` back. The worker dispatches it as a
third body beside the two in-tree ones, and it reuses the existing submit path
unchanged.

Three things this doc had specified and nothing had exercised are now live:

- **The soft/hard distinction is real.** Below, this doc records that the two
  collapse for first-party built-ins because there is no container to signal.
  That deviation now has an exception: `soft` is `stop --time <grace>` and
  `hard` is `kill`, and the worker asks `cancelled()` before `should_drop()`
  precisely so a soft cancel is not silently promoted — which here would
  SIGKILL a job promised a grace period to checkpoint in.
- **§2.3's digest-before-load ordering**, on a real path rather than a unit
  test: an archive that fails its check is never parsed by anything, and the
  image is run by id — which is where the `docker load` output bug below
  surfaced, since getting an id out of a *tagged* archive turned out not to
  work at all.
- **§4's split, enforced rather than described.** See below.

**Now exercised against a real daemon.** `tests/test_contained_live.py` builds
an image, `docker save`s it, and runs it through the real path — pull, digest
check, `docker load`, container, bind mount, `/scratch` contract. It skips where
no daemon is reachable, since §4 refuses such a machine submitter jobs anyway.

§2.2–§2.4 are asserted **from inside the container**, which is the only place
they mean anything: the entrypoint probes its own environment and the test reads
`net=blocked`, `root=ro`, `uid=1000` back out of the output. Flipping
`--network none` to `bridge` makes that fail, so it is a real check.

**And it found a real bug immediately.** `docker load` prints
`Loaded image ID: sha256:...` only for an untagged archive; a tagged one prints
`Loaded image: name:tag`. `load_image` parsed only the first, so it raised on
every archive §1.1's upload path can produce — that path takes a `docker save`
payload and requires a `repo_tag`. No test with an injectable runner could see
it: a fake returns the string the parser expects, so a parser that is wrong
about the real tool's output looks correct forever. `load_image` now handles
both forms, resolving a tag through `inspect` immediately after the load that
set it — and still runs the **id**, so the property that a submitter's archive
must not choose which bytes execute is unchanged.

**A daemon that is merely stopped silently shrinks the suite.** Worth knowing
because it is invisible: `test_store.py`'s MinIO fixture and
`test_worker_concurrency.py`'s whole M4a fleet are gated on `docker info`
succeeding, so a dev box with Docker Desktop installed but not running reports
**20 skipped** and a green suite. Starting it takes the same tree to 896 passed,
0 skipped. Two consequences: read the skip count, not just the pass count; and
run only one pytest process at a time, because `test_store.py` starts *and
removes* a shared, named container (`ganymede-test-minio`) and a second run will
pull the store out from under the first. The symptom of getting that wrong is a
`WinError 10061` against `storage-test.local:9410` and a fleet that closes one
round instead of two — which is the storage-blip handler working, not failing.

What is still unproven is narrower: this ran on Docker Desktop's Linux engine on
Windows. `--storage-opt size=` is still off by default (§2.2), the socket proxy
of §2.1 is still not deployed, and no GPU was requested — the live test passes
`gpus=None`, because asking for one would make it fail on any box without the
container toolkit rather than testing anything about confinement.

**And §4 is now enforced in both directions, rather than described.** "Every
first-party type carries `image_id IS NULL`" was a statement about what gets
submitted: `POST /v1/jobs` accepts an `image_id` on any job type, and the claim
walk serves such a job to any worker reporting a container runtime. No in-tree
body reads `image_ref`, so one would have run that job to completion,
successfully, with exactly the confinement the image exists to provide absent
and nothing saying so.

`JobType.requires_image` is now the fact, and `can_honor` reads it three ways:

| Case | Refusal |
|---|---|
| an in-tree type handed an image | `contained_execution_unsupported` |
| a contained type with no image | `image_required` |
| a contained type on a machine reporting no runtime | `no_container_runtime` |

The third is belt-and-braces — the coordinator refuses it at claim off the
registered profile — but step 5 is where a *stale* profile surfaces, which is
what every other check there exists for. The middle one is the dangerous
direction: its failure without the check is not a crash but an empty output that
`validate` rejects on row count, spending an attempt and naming the wrong
thing.
