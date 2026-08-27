# Ganymede — Data Model Delta (spine doc 1 of 2)

*One half of the frozen spine the platform-expansion fan-out builds against
(`04-platform-expansion.md`, Build sequencing). The other half is
`06-api-delta.md`. Nothing below is optional-to-agree: a component design doc may
add columns and indexes inside its own new tables, but table names, key columns,
foreign keys, status enums, and the `JobType` protocol signature are fixed here.*

*Stage 1 status: the six component docs (`07`–`12`) are written. Their spine
extensions are folded in below — see [Stage 1 reconciliation](#stage-1-reconciliation--changes-folded-in-after-the-component-docs).*

Delta against the schema in `ganymede/coordinator/db.py` as of `a1b4e36`.

---

## Prerequisite: a real migration mechanism

`db.py` today is **additive-only on purpose** — `_ADDED_COLUMNS` adds nullable
columns and nothing else, and the docstring says a rename, drop, or type change
"need a real migration with a version number and a plan … pretending otherwise
here would mean writing a migration framework before there is a second thing to
migrate."

The second thing has arrived. This delta needs, at minimum:

- `tasks.run_id` / `tasks.round_idx` going from required to nullable (non-training
  jobs have neither).
- `workers.id` changing meaning — from a value derived from
  `(contributor, device fingerprint)` in `app.py:register` to an
  enrollment-minted identifier that survives a hardware change (invariant 1).

So **Stage 0 also delivers a minimal migration runner**: a `schema_version`
table, numbered migration scripts, forward-only, each wrapped in the existing
`immediate()` transaction helper. Not a framework — a list of SQL blocks with a
version cursor. The identity workstream owns migration 004 (the `workers.id`
rework) and its backfill plan; everything else here is additive and lands in
002–003.

---

## Naming

The code says `contributors` and `workers`; the platform language is **users** and
**machines**. Renaming tables touches every module and buys nothing this quarter,
so: **table names stay, domain terms change in new code.** New tables and columns
use `user_id` / `machine_id`; they reference `contributors(id)` / `workers(id)`.
A later migration can rename once the churn is worth it.

---

## Existing tables — changes

### `contributors`  (the user record)

| Column | Change | Notes |
| --- | --- | --- |
| `auth_provider` | **add** `TEXT NOT NULL DEFAULT 'local'` | `local` = the placeholder provider (Decision 5). `github`, `oidc` later, same table. |
| `auth_subject` | **add** `TEXT` | provider's stable user id; `NULL` for `local` rows keyed by `name`. Unique with `auth_provider`. |
| `is_admin` | **add** `INTEGER NOT NULL DEFAULT 0` | the admin role (Decisions 9, 13). First admin set by env var at bootstrap (Decision 14), not through the API. |
| `email` | **add** `TEXT` | optional; the placeholder login may use it. |
| `clearance` | unchanged | §6.10 machinery stays as-is. |

Contributor keys (`key_hash`) remain — they are now **operator / admin / CLI**
credentials. Machines stop using them (see `machine_keys`).

### `workers`  (the machine record)

| Column | Change | Notes |
| --- | --- | --- |
| `id` | **semantics change** (migration 004) | enrollment-minted, stable across GPU swap / OS reinstall / re-enroll. Not derived from hardware. |
| `display_name` | **add** `TEXT` | what the owner called it at enrollment. |
| `enrolled_at` | **add** `TEXT` | distinct from `first_seen` (first poll). |
| `hardware_fingerprint_json` | **add** `TEXT` | GPU / CPU / board. **Advisory only** — drives "this looks like a different computer" fraud review, never identity, never credit (invariant 1). |
| `standing` | **add** `TEXT NOT NULL DEFAULT 'good'` | `good` \| `probation` \| `revoked`. Gates provisioned accrual (Decision 11) and how much unverified work the machine may hold (invariant 2). |
| `reputation` | **add** `REAL NOT NULL DEFAULT 0.25` | the `[0,1]` trust scalar, earned slowly and lost fast (invariant 2). Owned by the identity subsystem; its inputs and its `standing` transitions are the ledger doc's (`09`). Added in Stage 1 reconciliation — `09` needs it, `08` owns the `workers` record. |
| `last_available_at` | **add** `TEXT` | last poll that counted toward provisioned hours. |
| `compute_profile_json` | unchanged | still the probe result (§6.9). |

### `runs`  →  a `collab_lora_finetune` job's private state

| Column | Change | Notes |
| --- | --- | --- |
| `job_id` | **add** `TEXT REFERENCES jobs(id)` | every `runs` row gets a parent `jobs` row (migration 003 backfills). |

Everything else in `runs`, and the whole of `rounds`, is now **private to the
`collab_lora_finetune` job type**. No schema change to `rounds`. `combine_mode`,
`lr_outer`, `outer_beta`, DiLoCo momentum — all stay, all type-specific.

### `tasks`  (generalised unit of work)

| Column | Change | Notes |
| --- | --- | --- |
| `job_id` | **add** `TEXT REFERENCES jobs(id)` | required after migration 003. |
| `run_id`, `round_idx` | **now nullable** | set only for `collab_lora_finetune`. |
| `buckets_json` | keep; **add** `input_ref_json TEXT` | generic input handle (a shard ref, a dataset slice). `buckets_json` stays for the training type. |
| `attempt_group` | **add** `TEXT` | tasks sharing this are the same logical unit dispatched to N machines for redundant execution (invariant 2). `NULL` = singleton. |
| `status` | **enum extended** | `leased` \| `submitted` \| `abandoned` \| `expired` \| **`cancelled`** (soft/hard kill, Decision 18). |

### `worker_eligibility`  (the diagnostic, §M5)

| Column | Change | Notes |
| --- | --- | --- |
| `run_id` | **generalise to `job_id`** | still one row per `(machine, job)`, still upserted per poll, still recorded-not-recomputed. Refusal reasons now include constraint misses (Decision 15). |

### `throughput`

Unchanged for training. Not the basis for credit — see `machine_weight`.

### `audit`

Unchanged. Keeps accumulating; it is the raw material for reputation scoring.

---

## New tables

### `schema_version`
`version INTEGER PRIMARY KEY, applied_at TEXT`. One row. The migration cursor.

### `enrollments`
Pending machine-enrollment tokens.
`id`, `user_id → contributors(id)`, `token_hash` (sha256, like `auth.hash_key`),
`display_name`, `created_at`, `consumed_at`, `machine_id → workers(id)` (set on
claim). Token is shown once, at issue.

### `machine_keys`
Per-machine bearer credential (replaces the hand-issued contributor key for
workers).
`machine_id → workers(id)`, `key_hash UNIQUE`, `enabled INTEGER DEFAULT 1`,
`created_at`. Revocation is `enabled = 0`, matching `auth.py`.

### `jobs`
The generic parent. A `collab_lora_finetune` job has a child `runs` row; a
`batch_inference` job has none.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | TEXT PK | |
| `owner_id` | TEXT → contributors(id) | |
| `job_type` | TEXT NOT NULL | `collab_lora_finetune` \| `batch_inference` \| … registered types only. |
| `spec_json` | TEXT NOT NULL | the job type validates its own shape; extends the §8 task spec. |
| `image_id` | TEXT → images(id) | the payload container (Decision 18). `NULL` only for first-party built-ins. |
| `status` | TEXT NOT NULL | `draft` \| `queued` \| `running` \| `paused` \| `done` \| `failed` \| `cancelled`. |
| `priority_rank` | INTEGER NOT NULL | position in the admin-ordered queue (Decision 10). Lower = sooner. Admin-set only (Decision 13). |
| `constraints_json` | TEXT NOT NULL DEFAULT '{}' | Decision 15 — either `{"machine_ids": [...]}` or a predicate `{"vram_gb": {">=": 24}, "gpu_model": {"in": [...]}, "os": ..., "region": ...}`. |
| `cancel_mode` | TEXT | `soft` \| `hard`, set when moved to `cancelled` (Decision 18). |
| `created_at` | TEXT | |

### `submitters`
The vetted allowlist (Decisions 3, 9).
`user_id → contributors(id) PK`, `status` (`pending` \| `approved` \| `denied` \|
`revoked`), `decided_by → contributors(id)`, `decided_at`, `note`. Only
`approved` users may `POST /v1/images` and `POST /v1/jobs`.

### `images`
Uploaded payload containers.
`id`, `submitter_id → contributors(id)`, `digest` (SHA-256 of the `docker save`
archive — the value a worker can recompute from the bytes it pulls, not the OCI
manifest digest), `size_bytes`, `object_ref` (object-store key — images go to
storage, never through the API body), `uploaded_at`, `scan_status` (`pending` \|
`clean` \| `flagged`). Jobs reference `images.id`; the digest is what a worker
verifies after pull.

Stage 1 reconciliation added, from `11`: `finalized_at TEXT` (worker-visible only
once set), `scanned_at TEXT`, `scan_detail_json TEXT` (the scan's findings, for
the admin disposition surface). An `images` row is immutable once finalized; a
rebuild is a new row with a new digest.

### `credit_events`
Append-only. The ledger (Decisions 6, 7, 11). Never updated, never deleted.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER PK AUTOINCREMENT | |
| `machine_id` | TEXT → workers(id) | |
| `user_id` | TEXT → contributors(id) | denormalised owner at time of accrual. |
| `kind` | TEXT | `provisioned` (primary) \| `work` (secondary signal, from `credit()`). |
| `weighted_hours` | REAL | the credited amount. `0.0` on every `kind = 'work'` row. |
| `raw_seconds` | INTEGER | availability seconds in the period. **On a `kind = 'work'` row this carries the `credit()` `WorkUnits` scalar instead** (Stage 1 reconciliation, from `09`) — the row's `job_type`, recoverable via `tasks`/`jobs`, fixes the unit. `05` froze these columns, so `work` rows overload `raw_seconds` rather than add a `work_units` column. Safe because every banked total filters `kind = 'provisioned'`. |
| `system_weight` | REAL | the multiplier applied (from `machine_weight`). `0.0` on `work` rows. |
| `formula_version` | INTEGER | so a re-weighting is auditable, not retroactive. |
| `period_start`, `period_end` | TEXT | the accrual window. Equal, and set to `created_at`, on `work` rows. |
| `created_at` | TEXT | |

Running total = `SUM(weighted_hours) WHERE kind = 'provisioned'` per machine or
user. No balance column, no debit row — reputation-only.

### `availability_ticks`
The integral input for provisioned accrual. Appended on every poll / heartbeat.
`machine_id`, `at`, `leased INTEGER`, `in_good_standing INTEGER`. The accrual
engine sums good-standing time between runs and writes one `credit_events` row
per window, then these rows are GC-eligible (like `audit`, keep N days).

### `machine_weight`
The probe-derived per-machine multiplier (Decision 12).
`machine_id → workers(id) PK`, `weight REAL`, `components_json` (the GPU /
VRAM / CPU / RAM / bandwidth terms), `formula_version INTEGER`, `computed_at`.
**Depends on fixing the clamped `bench_score`** seen in the 3060 bring-up — until
that yields a real relative-performance number, the weight function can't be
trusted, and this table's `formula_version` is how a recompute rolls forward.
`09` fixes `formula_version = 0` as an explicitly interim GPU-class lookup and
sets the exit criteria (a de-saturated bench + measured monotonicity across ≥ 2
real GPU classes) before any `formula_version ≥ 1` reads `bench_score`.

### `sessions`
Web-UI session tokens (Stage 1 reconciliation — `05` delegated the session
*mechanism* to `08`, and the table is the mechanism). Created in migration 004.
`token_hash TEXT PRIMARY KEY` (sha256, like `auth.hash_key`), `user_id → contributors(id)`,
`created_at TEXT`, `expires_at TEXT NOT NULL`, `last_used_at TEXT`. Absolute
expiry, no sliding renewal in the placeholder; expired rows GC'd on the `audit` /
`availability_ticks` cron.

---

## The `JobType` protocol

The seam every job type implements. Lives in `ganymede/jobtypes/`. Method → where
it runs → what it replaces (`04-platform-expansion.md`, "The job-type contract").

```python
class JobType(Protocol):
    name: str
    version: int

    # coordinator: split a job into dispatchable tasks
    def plan(self, job: JobRow, conn: Connection) -> list[TaskSpec]: ...

    # coordinator: what the worker must fetch for this task
    def inputs_for(self, task: TaskRow, store: Store) -> InputRefs: ...

    # worker: execute. This IS today's trainer.run_task for collab_lora_finetune.
    def run(self, task: Task, inputs: InputRefs,
            on_step: StepCb, should_stop: StopCb) -> Result: ...

    # coordinator: acceptance. Per-type §5.1 gates.
    def validate(self, task: TaskRow, result: Result,
                 conn: Connection, store: Store) -> Verdict: ...

    # coordinator: optional reduce step. None => embarrassingly parallel.
    def reduce(self, job: JobRow, results: list[Result],
               conn: Connection, store: Store) -> ReduceState | None: ...

    # coordinator: is the job finished?
    def is_complete(self, job: JobRow, state: ReduceState | None) -> bool: ...

    # coordinator, TRUSTED: work-done units for the secondary 'work' signal only.
    # NEVER the provisioned accrual, NEVER banked directly (04, credit() section).
    def credit(self, task: TaskRow, result: Result) -> WorkUnits: ...

    # --- optional claim seam (Stage 1 reconciliation, from 10) -----------------
    # plan() has no machine profile; per-machine task sizing (local_steps, bucket
    # count from measured throughput) and the round-closed 409 both need one.
    # A type that omits these gets static plan() output and never returns 409.
    # They exist so claim_task's body and the RoundClosed path MOVE VERBATIM in
    # Phase A rather than being rewritten — which is what the inertness gate needs.
    def shape_claim(self, job, profile, settings, conn, now) -> TaskSpec | ClaimRefusal: ...
    def still_accepting(self, job, task, conn) -> None | RoundClosed: ...
```

`Task`, `Result`/`TrainResult` already exist in `ganymede/trainer/train.py` and
are reused as each type's concrete `run` payload / return (nominal per-type names
— `batch_inference` defines `InferTask` / `InferResult`). **`TaskSpec` is
widened, not reused verbatim** (Stage 1 reconciliation, from `10`): it gains the
generic `job_id` / `input_ref` fields and defaults the LoRA-specific fields to
`None`, following the shape already frozen on the `tasks` table. `InputRefs`,
`ReduceState`, `WorkUnits`, `Verdict`, `ClaimRefusal` are new and small; their
field lists are `10`'s.

### What Phase A physically moves

`10` has the function-level map; the shape:

| Today | After Phase A |
| --- | --- |
| `coordinator/rounds.py` — lease lifecycle (`expire_leases`, `abandon`, `record_submission`, `heartbeat`) | **stays generic**; the 409 path calls `still_accepting` |
| `coordinator/rounds.py` — round lifecycle, `claim_task` body, `TaskSpec` | → `jobtypes/collab_lora_finetune/` (`plan.py`, `claim.py`); `TaskSpec` → `jobtypes/base.py`, widened |
| `coordinator/closer.py` — `maybe_close` / the close fence / status advance / dispatch to `reduce()` | → **`coordinator/close.py`**, generic. (`07` calls this remnant `advance_job`; the module is `close.py`, the per-poll entry point is `advance_job`.) |
| `coordinator/closer.py` — `gate_submission`, `expected_manifest` | → `jobtypes/collab_lora_finetune/validate.py` (the type's `validate`) |
| `coordinator/aggregate.py` (`combine`, `check_structural`, `check_norms`, `adapter_divergence`, …) | → `jobtypes/collab_lora_finetune/aggregate.py`, **verbatim** (no DB/store deps) |
| `coordinator/eligibility.py` | stays generic; predicate set widened (Decision 15) |
| `app.py` `/runs/{id}/rounds/current` | stays, delegates to the type; 404 for non-training jobs |

Phase A is **numerically inert** — the golden trace (`04`, M4b guardrail) is the
entry criterion. `10`'s inertness checklist enumerates the golden fixtures to
capture on `a1b4e36` first and the exact-equality vs tolerance-band checks after
the move.

---

## State machines

- **`jobs.status`**: `draft → queued → running → (done | failed | cancelled)`;
  `running ⇄ paused`. `cancelled` carries `cancel_mode`.
- **`tasks.status`**: `leased → (submitted | abandoned | expired | cancelled)`.
  `cancelled` is the pull-delivered kill; `expired` is lease timeout (unchanged).
- **`workers.standing`**: `good ⇄ probation → revoked`. `probation` on failed
  spot-checks / validation; `revoked` is terminal for accrual, set by admin or by
  the fraud rules. Only `good` and `probation`-with-limits accrue provisioned
  hours; `revoked` accrues nothing.
- **`submitters.status`**: `pending → (approved | denied)`; `approved → revoked`.

---

## Frozen vs. open for component docs

**Frozen here:** every table name and every column above; the four status enums;
the `JobType` protocol method set and signatures (including the optional claim
seam added in reconciliation); `credit()` being coordinator-side and non-banking;
the identity-is-not-hardware rule.

**Specified in the Stage 1 component docs:** indexes; the accrual formula, the
`machine_weight` weight function, and the reputation inputs / `standing`
transitions (`09`); the constraint-predicate grammar (`07`); `InputRefs` /
`ReduceState` / `Verdict` shapes and the Phase A function map (`10`); the image
scan pipeline and `scan_status` transitions (`11`); enrollment token format /
TTL, the session mechanism, and migration 004's backfill (`08`); the `/ui/*`
page set and SSE schema (`12`).

---

## Stage 1 reconciliation — changes folded in after the component docs

The six Stage 1 docs (`07`–`12`) were written against this spine; where they had
to extend it, the extension is now here rather than only in the component doc:

1. **`workers.reputation REAL DEFAULT 0.25`** — `09` needs the trust scalar; `08`
   owns the `workers` record. Added above.
2. **`sessions` table** — `05` delegated the session mechanism to `08`; the table
   is that mechanism. Added above, created in migration 004.
3. **`images` gains `finalized_at` / `scanned_at` / `scan_detail_json`** — `11`'s
   scan pipeline needs them. Additive, within a component doc's latitude over its
   own new table; recorded here so the API doc and SDK doc agree.
4. **`JobType` optional claim seam** (`shape_claim` + `still_accepting`) — `10`
   needs it so `claim_task` and the `RoundClosed` path move verbatim in Phase A.
   A type may omit both. Added to the protocol above.
5. **`TaskSpec` widened, not verbatim** — generic `job_id` / `input_ref` fields,
   LoRA fields default `None`. Follows the `tasks`-table shape already frozen.
6. **`credit_events.raw_seconds` overload on `kind = 'work'` rows** — carries the
   `WorkUnits` scalar; no new column, since `05` froze `credit_events`. Every
   banked query filters `kind = 'provisioned'`.
7. **The generic close remnant is `coordinator/close.py`**, per-poll entry point
   `advance_job` — reconciling `07`'s and `10`'s names for the same code.
