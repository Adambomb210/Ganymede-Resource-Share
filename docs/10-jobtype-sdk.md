# Ganymede — Job-Type SDK (component design, Stage 1)

*One Stage 1 workstream doc (`04-platform-expansion.md`, Build sequencing). It
imports the frozen spine — `05-data-model.md` (the `JobType` protocol, the tables)
and `06-api-delta.md` (the claim payload, `submit` running `validate()`) — and
freezes its own seam: the job-type package contract, registration/versioning, the
Phase A move of `collab_lora_finetune` behind that contract, and `batch_inference`
as the second type. Execution isolation for third-party code is the sandbox doc's;
the accrual formula behind `credit()` is the ledger doc's.*

---

## 1. Package layout and types

### The package

A job type is a package under `ganymede/jobtypes/<name>/`. The generic coordinator
imports the registry, never a type module directly.

```
ganymede/jobtypes/
  __init__.py                 # REGISTRY: dict[str, type[JobType]]; resolve()
  base.py                     # JobType Protocol; InputRefs / ReduceState / WorkUnits / Verdict; TaskSpec
  collab_lora_finetune/
    __init__.py               # class CollabLoraFinetune(JobType); name="collab_lora_finetune"; version=1
    plan.py                   # ex-rounds.py: open_round, should_close, reopen_empty_round, _pick_buckets, cadence
    claim.py                  # ex-rounds.py: the claim_task body — per-machine task sizing
    inputs.py                 # inputs_for
    validate.py               # ex-closer.gate_submission + aggregate.check_structural + expected_manifest
    aggregate.py              # ex-coordinator/aggregate.py, VERBATIM: combine / check_norms / dense_weights / …
    reduce.py                 # ex-closer._close_claimed_round body: the round close
  batch_inference/
    __init__.py               # class BatchInference(JobType); version=1
    plan.py  inputs.py  run.py  validate.py
```

### The protocol (reproduced from `05`, method set and signatures unchanged)

```python
class JobType(Protocol):
    name: str
    version: int

    def plan(self, job: JobRow, conn: Connection) -> list[TaskSpec]: ...
    def inputs_for(self, task: TaskRow, store: Store) -> InputRefs: ...
    def run(self, task: Task, inputs: InputRefs,
            on_step: StepCb, should_stop: StopCb) -> Result: ...
    def validate(self, task: TaskRow, result: Result,
                 conn: Connection, store: Store) -> Verdict: ...
    def reduce(self, job: JobRow, results: list[Result],
               conn: Connection, store: Store) -> ReduceState | None: ...
    def is_complete(self, job: JobRow, state: ReduceState | None) -> bool: ...
    def credit(self, task: TaskRow, result: Result) -> WorkUnits: ...
```

`plan` splits a job into `tasks` rows; `inputs_for` names what a worker fetches for
one task; `run` is the worker body (for `collab_lora_finetune` it *is* today's
`trainer.run_task`); `validate` is the §5.1 gate for that type; `reduce` is the
§5.2 combine, or `None` for embarrassingly parallel; `is_complete` ends the job;
`credit` is a trusted work-done figure feeding `credit_events.kind='work'` only —
normalised against provisioned Weighted System Hours, never banked raw (`04`,
`05`).

### New types (`base.py`), small by design

| Type | Shape | Notes |
| --- | --- | --- |
| `InputRefs` | `artifacts: dict[str, str]` (logical name → presigned GET), `params: dict[str, Any]` (small inline values) | what `run` pulls before it starts. `collab_lora_finetune`: `artifacts={"base_adapter": url}`, `params={"buckets", "num_buckets", "dataset_ref", "seed"}`. `batch_inference`: `artifacts={"model": url}`, `params={"shard_ref", "output_put_url", "decode", …}` — **no base adapter key**. |
| `ReduceState` | `epoch: int`, `result_ref: str \| None`, `metrics: dict[str, Any]`, frozen | the reduce-checkpoint record. `collab_lora_finetune`: `epoch`=round index, `result_ref`=combined adapter = next round's `base_adapter_ref`, `metrics`={divergence, contributors}. Persisted where the `rounds` row lives today. |
| `WorkUnits` | `unit: str` (`"tokens"` \| `"rows"` \| …), `count: int`, frozen | `credit()`'s return. Advisory, trusted. |
| `Verdict` | `accepted: bool`, `reason: str \| None` (stable slug, persisted), `detail: str \| None`, `compare_digest: str \| None` | `aggregate.GateResult` plus `compare_digest`, which a type attaches for the coordinator's redundant-execution comparison. The type never compares across an `attempt_group` itself. |

### Reused verbatim

- **`trainer.train.Task`** — `collab_lora_finetune`'s concrete `Task`: the parsed
  claim payload its `run` receives. `batch_inference` defines its own
  `batch_inference.run.InferTask`. `Task` in the signature is a nominal
  per-type name.
- **`trainer.train.TrainResult`** — `collab_lora_finetune`'s concrete `Result`.
  `batch_inference` defines `InferResult` (rows written, output ref, digest).
  `05`'s slashed "`Result`/`TrainResult`" already reads this way.
- **`rounds.TaskSpec`** — the coordinator→worker task descriptor. Widened, not
  redefined; see *Spine deviations*.

### The dispatcher (generic, stays in `coordinator/`)

- **plan** → for each returned `TaskSpec`, `INSERT` a `tasks` row (`job_id`,
  `input_ref_json`, `attempt_group`; `run_id`/`round_idx` `NULL` unless the type
  sets them at reduce).
- **claim** (`POST /v1/tasks/claim`, `06`) → walk queued jobs in `priority_rank`
  order, apply `constraints_json` then `budget.is_eligible`, take one unleased
  task for the first job this machine satisfies; call the type's claim seam (§3)
  for per-machine shaping; call `inputs_for`; assemble the payload.
- **submit** (`06`) → call `validate()`; record the `Verdict`; the
  redundant-group comparison is the coordinator's, not the type's.
- **close** → opportunistic, from the request path; call `reduce()`. A
  `ReduceState` back → call `is_complete(job, state)`. `None` back → the
  dispatcher's task-set rule decides completion (§5).
- **credit** → `credit()` result to `credit_events` `kind='work'`, normalised
  against provisioned hours.

---

## 2. Registration, versioning, discovery

`ganymede/jobtypes/__init__.py` holds `REGISTRY: dict[str, type[JobType]]`.
First-party types register on import. `collab_lora_finetune` **and**
`batch_inference` are both first-party, in-tree, `jobs.image_id` `NULL`, versioned
by the coordinator release. `resolve(job_type, version)` returns
`REGISTRY[job_type]()` after asserting `.version >= version`.

**Version binding.** At `POST /v1/jobs` the resolved pair is frozen into
`spec_json` as `spec_json.sdk = {"job_type": …, "version": …}` and never mutated.
`05` hands `spec_json`'s shape to the type and froze no `version` column on `jobs`;
none is added.

**Coordinator ↔ worker agreement.**

- *In-tree types:* a coordinator and its workers deploy from one release, so their
  `REGISTRY` matches by construction. A worker still checks
  `spec_json.sdk.version <= its REGISTRY version`; a higher value → abandon before
  download with refusal reason `job_type_version_unsupported`, recorded verbatim
  in `worker_eligibility` (now keyed by `job_id`, `05`), exactly as the
  `required_image` mismatch is handled today.
- *Image-backed types* (later — Phase E's third type): `spec_json.sdk.image_digest`
  pins `images.digest`; the claim payload carries `image_ref` / `image_digest` /
  `image_pull_url` (`06`); the worker verifies the digest after pull (`06`). The
  registry entry is created at `POST /v1/images/{id}/finalize` from the image's
  declared `job_type` / `version`. Coordinator-side hooks (`plan`, `validate`,
  `reduce`, `credit`) then run from that image inside the isolation the sandbox
  doc specifies — this doc freezes only the contract and the discovery rule.

Image upload/store/distribute is otherwise unchanged from `05`/`06`: presigned
PUT/GET against the object store, digest verified on pull, `scan_status`
transitions owned by the sandbox doc.

---

## 3. Phase A: the move, file by file

**Guiding rule: move, don't rewrite.** Every function below changes import path
only. Byte-identical bodies are the strongest guarantee of the numerical inertness
the M4b golden-trace gate (`04`) checks.

| Today | After Phase A |
| --- | --- |
| `rounds.py` — `expire_leases`, `abandon`, `record_submission`, `heartbeat` | **stay generic** (lease lifecycle); the 409 path gains the seam below |
| `rounds.py` — `open_round`, `current_round`, `round_progress`, `should_close`, `reopen_empty_round`, `_pick_buckets`, `task_seed`, `update_throughput` | → `collab_lora_finetune/plan.py` |
| `rounds.py` — `claim_task` body (`plan_budget` call, throughput floor, data-limited warning, bucket pick) | → `collab_lora_finetune/claim.py`. `budget.py` is **not** in `05`'s move table — `plan_budget` stays generic; the claim seam calls it. |
| `rounds.TaskSpec` | → `base.py`, widened |
| `closer.py` — `maybe_close`, the atomic `status='closing'` fence + reopen-on-exception, lease-expiry-on-close, run/job status advance, the dispatch to `reduce()` / `is_complete()` | **stay generic** (`coordinator/close.py`) |
| `closer.py` — `gate_submission`, `_record_verdict`, `expected_manifest`, `_MANIFEST_CACHE` | → `collab_lora_finetune/validate.py` |
| `closer.py` — `_close_claimed_round` body (gate 4, `combine`, momentum load/store, `adapter_divergence`, `base_adapter_key` write, next-round open, throughput fold-back) | → `collab_lora_finetune/reduce.py` |
| `coordinator/aggregate.py` — `combine`, `check_structural`, `check_norms`, `dense_weights`, `adapter_divergence`, `load_adapter`, `save_adapter`, `manifest_of`, `GateResult`, `REJECT_*` | → `collab_lora_finetune/aggregate.py`, **verbatim**. Already no DB/store deps. |
| `coordinator/eligibility.py` | stays generic; predicate set widened by the scheduler doc |
| `app.py` — `GET /v1/runs/{id}/rounds/current` | stays; delegates to `collab_lora_finetune` (reads its `rounds` row); 404 for non-training jobs |

### Two seams the move creates

**The claim seam.** Per-machine task sizing cannot come from the frozen
`plan(job, conn)` — it has no machine profile, and `local_steps` / bucket count
depend on the claiming machine's measured throughput (§3.5). One optional hook:

```
shape_claim(job, profile, settings, conn, now) -> TaskSpec | ClaimRefusal
```

`collab_lora_finetune` implements it as the moved `claim_task` body; a
`ClaimRefusal(reason)` for the throughput-floor / ineligible cases lets the
generic path record the refusal in `worker_eligibility` (recorded-not-recomputed,
`05`) exactly as `NotEligible` does today. `batch_inference` omits it — its `plan`
output is claimed as-is.

**The 409 seam.** `heartbeat` and `record_submission` raise `RoundClosed` (`06`'s
frozen 409) by reading `rounds.status` — state `05` just made type-private. After
the move the generic lease functions call one hook:

```
still_accepting(job, task, conn) -> None | RoundClosed
```

`collab_lora_finetune` checks its `rounds` row; `batch_inference` always returns
`None`. Without this, a worker that used to get 409 gets 200 — behaviourally
different while numerically inert.

### Inertness checklist — Phase A's entry criterion

The `04` fixture (a few hundred steps + one reduce over two synthetic submissions)
does **not** exercise the gates: `check_norms` early-returns at `n < 3`,
`dense_weights` skips the cap (`n >= 3` guard), momentum never carries (needs two
reduces), `combine`'s `w_sum == 0` guard never fires. Capture these golden
fixtures on `a1b4e36` first:

1. **Loss trace** — `train_loop`, fixed seed, ≥300 steps: the per-step loss list.
2. **Reduce, small cohort** — 2 synthetic submissions, one reduce: `combine`
   output adapter, `dense_weights` vector, `adapter_divergence`, and `should_close`
   decisions over an *injected* `now` timeline (it already takes `now` — the
   fixture supplies the timeline, not wall-clock).
3. **Reduce, gate-4 cohort** — ≥3 submissions including one norm outlier and one
   deliberately untouched tensor: `check_norms` verdicts, the post-cap
   `dense_weights` vector (convergence loop), `combine`'s zero-guard passthrough.
4. **Two consecutive reduces, `beta != 0`** — `outer_momentum_ref` bytes after
   reduce 1 and reduce 2.
5. **Structural gates** — malformed and valid adapters: `check_structural`
   `(accepted, reason slug, detail string)` for each.

Reproduction after the move:

- **Exact** (CPU fp32 math — bodies moved verbatim, reduction order preserved):
  `combine` (max abs elementwise diff = 0), `dense_weights`, `adapter_divergence`
  (≤ 1e-9), `check_norms` / `check_structural` verdict slugs and detail strings
  (persisted — must not drift), `base_adapter_key(run, round+1)` (result lands at
  the same key), `should_close` `(close?, reason)` per replayed timeline.
- **Tolerance band** (CUDA nondeterministic kernels): per-step loss rel err
  ≤ 1e-5, cumulative ≤ 1e-4. State the band in the golden file; a run outside it
  fails the criterion.
- **Structural:** moved functions show as pure relocation under rename detection —
  no body hunks. A body hunk needs sign-off and a re-run of the above.
- Existing coordinator test suite green with only import-path edits.
- **End-to-end:** replay a recorded round (base + 3 real submissions) through the
  old and new close path; `CloseResult` fields equal (`result_adapter_ref`
  digest, `accepted`, `rejected`, `total_steps`, `divergence`).

Fail any exact-equality item → Phase A stops until M4b closes and becomes the
reference instead (`04`, M4b guardrail).

---

## 4. `batch_inference` against the protocol

The proof the seam isn't welded to LoRA: no base adapter, no `reduce`, no round.

### `spec_json`

```json
{
  "sdk": {"job_type": "batch_inference", "version": 1},
  "model_ref": "<object-store key | hf://…>",
  "shards": [{"ref": "<key>", "rows": 4096}, "…"],
  "output_prefix": "<object-store prefix>",
  "prompt_template": "…{input}…",
  "decode": {"mode": "greedy", "max_new_tokens": 256},
  "output_schema": {"id": "str", "output": "str"},
  "redundancy": {"fraction": 0.05, "n": 2, "sample_rows": 8, "agree_on": "output"}
}
```

`validate()` runs this shape check on `POST /v1/jobs` (`06`). **The shard index —
refs and row counts — is submitter-declared**, because `plan(job, conn)` has no
`store` to enumerate the bucket, and `validate`'s row-count gate has no teeth
against a worker optimising for points (invariant 2) if the count comes from the
worker. A job that sets `redundancy` **must** use `decode.mode = "greedy"`: a
stochastic decode has no exact cross-task comparator.

### The seven methods

- **`plan(job, conn) → list[TaskSpec]`** — one `TaskSpec` per `spec.shards[i]`:
  `job_id` set, `input_ref` = shard ref, LoRA fields `None`, `run_id` /
  `round_idx` `None`. For a `fraction` of shards, emit `n` copies sharing an
  `attempt_group` (`05`). Called once, at enqueue. No rounds.
- **`inputs_for(task, store) → InputRefs`** — `artifacts = {"model": <presigned
  GET>}`, `params = {"shard_ref", "output_put_url": <presigned PUT>, "decode",
  "prompt_template", "output_schema"}`. No base adapter key — the seam.
- **`run(task, inputs, on_step, should_stop) → InferResult`** — load the model
  once; iterate shard rows in batches; write one output row per input row to a
  local file; upload to `output_put_url`. `on_step(rows_done, 0.0)` per batch
  drives the heartbeat. `should_stop`: soft → flush the current batch, upload,
  exit; hard → abort. `InferResult(rows, output_ref, digest = sha256 of the
  canonicalised `(id, output)` pairs, seconds)`.
- **`validate(task, result, conn, store) → Verdict`** — fetch `result.output_ref`;
  (a) row count `== spec.shards[i].rows`, else `Verdict(False,
  "row_count_mismatch")`; (b) every row satisfies `output_schema`, else
  `"schema_mismatch"`; (c) attach `compare_digest = result.digest`. The type does
  **not** compare across the `attempt_group` — the coordinator samples
  `redundancy.sample_rows` and requires agreement on `agree_on` across the `n`
  results (exact, since greedy); disagreement → re-dispatch and the offending
  machine to `standing='probation'` (`05`).
- **`reduce(job, results, conn, store) → None`** — always. Embarrassingly
  parallel: no combine, no artifact, no next round.
- **`is_complete(job, state) → bool`** — `state` is always `None`; **not
  consulted**, returns `True`. Completion is the dispatcher's (§5).
- **`credit(task, result) → WorkUnits`** — `WorkUnits("rows", result.rows)`.
  Trusted; `credit_events.kind='work'`; normalised against provisioned Weighted
  System Hours, never banked raw (`04`, `05`).

---

## 5. A round is a `reduce` checkpoint some types have

The generic layer knows **jobs**, **tasks**, and an optional per-job
**reduce-epoch** counter (`04`, invariant 4). Nothing else about rounds is
generic.

- **`collab_lora_finetune`:** reduce-epoch == round index. Each `reduce()` closes
  a round, publishes the combined adapter (= next round's `base_adapter_ref`),
  opens the next, until `is_complete(job, state)` (`round_idx >= target_rounds`).
  The `rounds` table — its `combine_mode` / `lr_outer` / `outer_beta` / momentum
  columns — and `closer`'s cadence are this type's private state (`05`).
- **`batch_inference`:** no `reduce`, reduce-epoch stays `NULL`, no round ever
  exists. Completion is "every `TaskSpec` from `plan` has an accepted verdict and
  every `attempt_group` has agreed," owned by the dispatcher because
  `is_complete` has neither `conn` nor a `ReduceState` to count from — which is
  exactly what `04`'s "`None` = embarrassingly parallel" buys: `plan` fixed a
  finite task set up front, so the generic layer can decide completion without
  type knowledge.

Endpoints:

- **`GET /v1/runs/{id}/rounds/current`** — unchanged (`06`), now documented as
  **`collab_lora_finetune`-specific**: it reads that type's `rounds` row via the
  type. A non-training job has no `runs` row → 404, indistinguishable from a
  missing run (`06`'s cross-tenant rule).
- Job-type-agnostic progress is `GET /v1/jobs/{id}` and the `jobs: [...]` block on
  `/status` (`06`). "Round" is a view a type may offer, not a platform concept.

---

## Spine deviations

1. **`TaskSpec` widened.** `05` lists it under "reused verbatim." This doc adds
   generic `job_id` / `input_ref` fields and defaults the LoRA-specific fields
   (`base_adapter_ref`, `lora_cfg`, `buckets`, `num_buckets`, `base_model`,
   `base_precision`, `dataset_ref`, `required_image`) to `None`. Additive, and it
   follows the shape `05` already froze on the `tasks` table — generic
   `input_ref_json` beside type-specific `buckets_json`, nullable `run_id` /
   `round_idx`. No wire or column change; every current call site passes the same
   fields.
2. **One added protocol member, optional.** `shape_claim` (per-machine claim-time
   sizing) with its sibling `still_accepting` (the 409 seam) is not in `05`'s
   seven-method table. It is optional — a type that omits it gets static `plan`
   output and no round-close 409. Rationale: it lets `claim_task`'s body and the
   `RoundClosed` path **move verbatim** rather than be rewritten, which is what
   the M4b inertness gate depends on. Widening `plan` / `heartbeat` signatures
   instead would be the larger change. Presented as one seam with two entry
   points, not two features.
3. **Dispatcher owns embarrassingly-parallel completion.** `05`'s `is_complete`
   takes no `conn`; a `reduce → None` type cannot answer completion itself. The
   rule lives in the generic close path. `is_complete` stays authoritative only
   for types that return a `ReduceState`.

Not deviations, recorded to save the check: `Task` / `Result` are the protocol's
nominal per-type names (`TrainResult` is `collab_lora_finetune`'s `Result`;
`batch_inference` defines `InferResult`) — `05`'s slashed "`Result`/`TrainResult`"
already says this. `spec_json.sdk` carrying `{job_type, version}` sits inside the
spec-shape latitude `05` grants the type; `jobs` gets no new column.

---

## Frozen here / open for later

**Frozen:** the package layout under `ganymede/jobtypes/`; the four new types
(`InputRefs`, `ReduceState`, `WorkUnits`, `Verdict`) and their fields; the in-tree
`REGISTRY` + `resolve()` + `spec_json.sdk` version binding and the
`job_type_version_unsupported` refusal; the Phase A move map, the two seams, and
the inertness checklist as Phase A's entry criterion; `batch_inference`'s
seven-method behaviour and its submitter-declared shard index with row counts.

**Open:** image-backed third-party type loading and its coordinator-side sandbox
(sandbox doc); the redundant-execution comparator's exact sampling and the
probation thresholds (anti-fraud, Phase D); `InputRefs.params` exact keys per
future type; `batch_inference` decode options beyond greedy.
