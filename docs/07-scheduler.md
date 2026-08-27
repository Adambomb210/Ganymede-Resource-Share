# Ganymede — Placement Scheduler (component design)

*Stage 1 component doc. Imports the frozen spine (`05-data-model.md`,
`06-api-delta.md`), freezes its own seam: the claim-path queue walk, the
constraint predicate grammar, refusals into `worker_eligibility`. Owns Decisions
4, 10, 13, 15 of `04-platform-expansion.md`. Design only.*

The scheduler is not a matcher on a timer. It is the claim endpoint, walking an
admin-ordered queue once per poll and leasing the first job the calling machine
can run. There is no placement state; the queue order is the only policy input,
and the admin owns it (Decision 13).

---

## 1. The claim path becomes a queue walk

`POST /v1/tasks/claim` — auth **machine** (`06`). Identity from the machine key;
`ClaimRequest.worker_id` accepted and ignored. `ClaimRequest.run_id` is **kept**
(`06` compatibility): a v1 worker pinning a `run_id` still works — the walk maps
it to its parent via `runs.job_id` (`05`) and pins that job. `ClaimRequest.job_id`
is the new pin.

### `_selectable_runs` → `_selectable_jobs`

`_selectable_jobs(conn, pinned_job_id, cached) -> list[JobRow]`:

- Rows: `jobs WHERE status IN ('queued','running')`. **Both** — a
  `collab_lora_finetune` job flips to `running` on its first lease but fans out
  tasks every round after; walking only `queued` hands it one task ever.
  `draft` / `paused` / terminal states are skipped. (`06` says "queued jobs in
  `priority_rank` order"; this is that phrase read precisely.)
- Order: `priority_rank ASC` primary (lower = sooner, `05`), then the
  cache-affinity tiebreak **within a rank** (`06`), then `created_at ASC`.
- Affinity now has two axes, both advisory, neither crossing a rank boundary:
  base-model-on-disk (today's behaviour) and image-digest-already-pulled
  (Decision 18 images are 100s of MB to GBs — a warm image is worth a warm base
  model). The affinity function can be refined without touching the walk.
- `pinned_job_id` short-circuits to that one row if selectable, else 204.

### The walk loop in `app.py`

Replaces `for run_id in candidates`. Per iteration, in queue order:

1. **`advance_job(conn, store, job, settings)`** — the generic remnant of
   `closer.maybe_close`. For `collab_lora_finetune` it runs the round
   close/reopen decision as today (kept on the poll: a round can become
   closeable through time alone, and the submit path would never notice). For an
   embarrassingly-parallel type it is an `is_complete` check that may flip the
   job to `done`.
2. **Constraint gate (Decision 15, §2)** —
   `check_constraints(job.constraints_json, machine_id, profile) -> (ok, reason)`.
   Pure: reads only `(machine_id, profile)`, no DB, no write lock — which is why
   it sits in the `app.py` walk, not inside `claim_task`. `not ok` → append
   `Verdict(job_id, REFUSED, reason)` and **`continue`** — never `break`.
3. **`claim_task` / the type's plan+lease path** — existing generic gates run
   here unchanged and in their current order: clearance (§6.10),
   `budget.is_eligible` (§6.8), image tag, throughput floor. `NotEligible` →
   `Verdict(REFUSED, why)`, `continue`. A `TaskSpec` → `Verdict(LEASED)`, flip
   `queued → running` if needed, record, return payload. `None` (eligible,
   nothing to hand out) → `Verdict(IDLE)`, `continue`.
4. After the loop: `eligibility.record(conn, machine_id, verdicts)`, then
   204 + `Retry-After`.

### Capability backfill falls out of not breaking

The loop `continue`s past every refusal, so a machine no high-rank job accepts
still reaches the first lower-rank job it fits (Decision 10). No early `break`,
no "stop after N refusals" — the backfill *is* the absence of that cutoff. A job
deep in the queue running on hardware nothing ahead of it can use is intended,
not a fairness leak.

### `BEGIN IMMEDIATE` safety is unchanged

Each `claim_task` still opens its own `immediate()` transaction and takes the
write lock at statement one (`db.py`); two machines racing one job serialise
there, the loser falls through to `IDLE` or the next job. Walking more jobs is
just more short independent transactions per poll — the same shape as today's
per-run loop.

**`eligibility.record` stays outside every `immediate()` block.** It calls
`conn.commit()` unconditionally and swallows `sqlite3.Error`; inside a
transaction it would silently commit the caller's half-done work. Called once
after the walk (and once on the early lease-return path, after that transaction
commits). Freeze this ordering — a later tidy-up of the loop is what breaks it.

### One lease per machine (Decision 4) — explicit

**Invariant: a machine holds at most one `leased` task across all jobs.**

- Cheap pre-check in `app.py` before the walk:
  `SELECT id, job_id FROM tasks WHERE worker_id=? AND status='leased' LIMIT 1`.
  A hit skips the walk and re-serves that task.
- Authoritative re-check as the first statement inside the claim transaction —
  under the write lock, so two concurrent polls from one machine cannot both
  lease. This **replaces** `claim_task`'s round-scoped held-lease check
  (`… AND run_id=? AND round_idx=?`) with a global one.
- **Re-serving is per-type.** The `tasks` row alone cannot rebuild the payload
  (`_task_payload` needs `base_adapter_ref` off the round). The resume path
  dispatches to the owning job type — `inputs_for(task)` plus a **fresh**
  presign, never a replay of the expired URLs. A worker retrying after a blip
  resumes; it never forks a second task.
- Consequence (invariant 3): the scheduler never puts a second task on a busy
  host, so `calibration.json`'s throughput number stays meaningful. MPS/MIG
  partitioning stays out of scope.
- `rounds.expire_leases` unchanged — a crashed holder's task returns to the pool.

---

## 2. The constraint predicate grammar (Decision 15)

`jobs.constraints_json` — `TEXT NOT NULL DEFAULT '{}'` (`05`). Set at
`POST /v1/jobs`, editable by owner/admin; the job type validates its shape at
write time. Two mutually exclusive forms.

### Form A — pin

```json
{"machine_ids": ["a1b2…", "c3d4…"]}
```

`machine_ids` is the only key (any sibling key → submit-time 422). The calling
`machine_id` must be in the list; empty list matches nothing. A pin does **not**
waive §6.8/§6.10 — a pinned machine that cannot fit the model still refuses at
the capability gate. "Only these machines", not "regardless of fit".

### Form B — predicate

```json
{"vram_gb": {">=": 24}, "gpu_model": {"in": ["NVIDIA A100", "NVIDIA H100"]},
 "os": "linux", "region": "eu-west"}
```

`field → condition`, all fields **AND**; an absent field is no constraint on
that axis (the `is_eligible` convention). A condition is an object of
`{op: operand}` pairs (all AND — `{">=": 24, "<": 80}` is a range) or a **bare
scalar meaning `==`** — not an invention, it is `05`'s frozen example (`"os": …`,
`"region": …` appear there bare) read literally.

**Operator set** — small, total, typed. Unknown operator → submit-time 422.

| op | operand | holds when |
| --- | --- | --- |
| `>=` `>` `<=` `<` | number | field coerces to a number and orders that way |
| `==` `!=` | string / number | equal / not equal |
| `in` / `not_in` | array | field value is / is not a member |
| `contains` | scalar | field is a list and the operand appears in it (mirrors the `supports` check in `is_eligible`) |

No OR, no nesting, no regex. Disjoint hardware classes use `in`, or are two jobs
— Decision 13 keeps the grammar flat.

### Field resolver — constraint field → `compute_profile_json`

Evaluated against a flattened view of `workers.compute_profile_json` (the
`ComputeProfile` of `06`) plus two enrollment-derived fields. Frozen here — an
implementer cannot guess it.

| field | source | notes |
| --- | --- | --- |
| `vram_gb` | `probe.alloc_max_mb` else `vram_mb`, ÷ 1024 | prefers the probed ceiling over the spec-sheet claim, as `is_eligible`'s `min_vram_mb` does (`budget.py`) |
| `vram_mb` | same, no division | raw form |
| `gpu_model` | `compute_profile.device_name` | |
| `backend` | `compute_profile.backend` | `cuda` \| `mps` \| `cpu` |
| `compute_capability` | `compute_profile.compute_capability` | |
| `supports` | `compute_profile.supports` (list) | use with `contains` |
| `bench_score` | `probe.bench_score` | Decision 12: clamped in the 3060 bring-up — unreliable until fixed |
| `driver` / `torch_ver` | `compute_profile.*` | |
| `os` | probe profile (`hardware_fingerprint_json`, `05`) | identity/sandbox docs own its capture |
| `region` | machine record, **declared at enrollment** | self-reported, advisory. Never a compliance control — data residency is `data_classification` + `clearance` (§6.10) |

**Unknown field name → submit-time 422**, deliberately unlike `is_eligible`
(which ignores unknown `requires` keys). A job naming an unevaluable field would
silently never place — the "told eligible, never gets work" failure
`eligibility.py` exists to prevent.

**Missing value on a known field → fail closed** for *every* operator, `!=` and
`not_in` included. Same rationale as `is_eligible` / `clearance_permits`: `os !=
windows` must not place on a machine whose OS was never probed.

### Composition with `budget.is_eligible` — sequential, not merged

`06` froze the order: constraints **first**, then `is_eligible`. Two gates:

1. `check_constraints` — the job's **targeting** (where it may run).
2. `budget.is_eligible(profile, requires_json)` — the run's **capability floor**
   (§6.8: precision, VRAM, capabilities). Unchanged.
3. then clearance, image tag, throughput floor — unchanged, after.

Not merged: `requires_json` is per-*run*, config-derived, about feasibility;
`constraints_json` is per-*job*, admin-editable, about policy, and stays `'{}'`
for most jobs (one dict check). Merging would weld the two grammars together
forever and lose the ability to tell a contributor *which* thing excluded them.
Overlap on an axis (both can name VRAM) is intended — `requires` is the floor,
`constraints` an admin narrowing on top ("the big job runs only on the A100s
even though a 4090 fits").

---

## 3. Generic vs job-type-owned; refusals into `worker_eligibility`

`eligibility.py` stays **entirely generic and recorder-only** — recorded, not
recomputed. It never decides; it writes what the walk decided, on the real poll,
verbatim. Changes:

- `worker_eligibility.run_id` → `job_id` (`05`, migrations 002–003). PK stays one
  row per `(machine, job)`. `Verdict.run_id` → `Verdict.job_id`; `explain()`,
  `for_contributor()`, `fleet_summary()` SQL updated. `LEASED` / `IDLE` /
  `REFUSED` unchanged — `REFUSED` now also covers constraint misses. `_shape()`
  grouping still works (see reasons below).
- **`explain()` filters to non-terminal jobs.** Rows pile up for `done` /
  `cancelled` jobs; unfiltered, a contributor's diagnostic grows without bound
  and mixes in jobs that finished last month. Terminal-job rows become
  GC-eligible, like `audit`.
- **Record only changed verdicts.** `eligibility.py`'s docstring argues "one
  upsert per (worker, run) per poll … twenty rows for a ten-machine two-run
  fleet". Keyed by `job_id` over a real queue, a lease returns early so the
  verdict list is *how deep the machine walked* — but an idle machine 204s and
  walks the whole queue. Skip the upsert when `outcome` and `reason` match the
  stored row; a uniformly-idle fleet against a 200-job queue is then a handful of
  writes per interval once steady.

**A job type owns** its `validate()` (acceptance gates, was §5.1 — writes
`submissions.reject_reason`, a different table and lifecycle) and its `plan()`
(whether a task exists → `IDLE` vs `LEASED`). It never writes an eligibility
verdict and never raises its own "not eligible": eligibility is capability +
clearance + targeting + throughput floor, all generic. A machine-property gate a
type wants is expressed as `constraints_json`, not code.

**New refusal reason strings** — produced by `check_constraints` on the poll,
wrapped `Verdict(job_id, REFUSED, reason)` by the walk, written verbatim:

```
machine not in pin list (<n> pinned)
<field> <value> fails >= <operand>          # and >, <=, <
<field> <value> fails == <operand>          # and !=
<field> <value> not in job allow-list       # in
<field> <value> in job deny-list            # not_in
<field> missing from probe profile          # fail-closed on absent value
```

Ordering-op reasons echo the number (neutralised by `_shape()`'s digit strip, so
`vram_gb 12 fails >= 24` and `vram_gb 8 fails >= 24` group as one problem in
`fleet_summary`). `in` / `not_in` use the stable phrasing above, not the array,
so they group too. Existing reasons carry over unchanged (`clearance … <
classification …`, `missing capability: …`, `backend … not in …`, `vram_mb N <
N`, `run requires image …`, `below throughput floor for this run: …`).

---

## 4. Starvation and fairness (Decision 13)

Strict `priority_rank` order is the whole policy — no aging, no weighting, no
per-submitter quota in this track. Starvation is bounded structurally by the
backfill in §1: a low-rank job runs whenever a machine no higher-rank job
accepts is free. A job starves only if *every* free machine is continuously
absorbed by higher-rank jobs it also fits — the admin's call. `GET
/v1/admin/queue` reports leased-task counts; a job stuck at zero is the signal to
`POST /v1/admin/queue/reorder`.

**Where weighted fair-share slots in later:** the sort key of `_selectable_jobs`
is the single seam. Today `(priority_rank, affinity_miss, created_at)`; a future
fair-share replaces the primary term with a dynamic priority — `priority_rank`
adjusted by the submitter's recent share of leased-task-seconds, from `tasks` or
a share-accounting table. The walk is untouched: head-first, backfilling, one
task per machine. Preemption is separate and larger (it aims the Decision 18
heartbeat `cancel` at a task, not a job) — Phase D. `reopen_empty_round` and the
stall detector are `collab_lora_finetune`'s, below the scheduler, unchanged.
This paragraph is the whole note — named, not designed.

---

## 5. Priority is admin-write-only

`jobs.priority_rank` is set **only** by `POST /v1/admin/queue/reorder`
(`{job_id, before | after | rank}`, auth **admin**, `06`).

- `POST /v1/jobs` (submitter) creates `status='draft'`, accepts no
  `priority_rank` — a value in the body is ignored (additive-safe, not a 422).
- `POST /v1/jobs/{id}/enqueue` moves `draft → queued` and defaults
  `priority_rank` to the tail (`MAX(priority_rank)+1`). Only then is the job
  visible to `_selectable_jobs`. The admin then adjusts with `reorder`.
- No priority hint, no "urgent" flag — Decision 13 is explicit that submitters do
  not set their own priority. A future fair-share does not change this: the admin
  ordering stays the base, fair-share is an adjustment the admin turns on.
- Ranks are integers, gaps allowed, no uniqueness constraint (ties break on the
  secondary sort). Sparse spacing (10, 20, 30) lets `before` / `after` insert
  without renumbering; worst case is an `O(queue length)` rewrite in one
  `immediate()` transaction.

---

## Frozen here

- The walk: `_selectable_jobs` over `status ∈ {queued, running}` in
  `priority_rank` order, affinity tiebreak within a rank, no early `break`,
  capability backfill, one lease per machine enforced inside the claim
  transaction, `eligibility.record` outside it.
- `run_id` pin retained and mapped to its parent job; `job_id` the new pin.
- The constraint grammar: pin vs predicate, the operator set, bare-scalar-`==`,
  all-AND, no OR/nesting, fail-closed on missing values, unknown field/operator =
  submit-time 422.
- The field resolver map; `region` advisory and non-compliance.
- Gate order: constraints → `is_eligible` → clearance → image → throughput floor.
- The new refusal reason strings; `worker_eligibility` generic and recorder-only,
  keyed by `job_id`, record-only-changed, `explain()` filtered to non-terminal
  jobs.

## Open / other docs

- Probe capture of `os` and any further predicate-nameable profile fields —
  identity + sandbox docs, against `05`'s `hardware_fingerprint_json`.
- The fair-share formula and its share-accounting table — Phase D.
- Preemption and the task-scoped `cancel` — Phase D.
- `GET /v1/admin/queue` exact response body — web-UI doc.
- How an embarrassingly-parallel type's `plan` distinguishes `IDLE` from `done` —
  job-type SDK doc.
