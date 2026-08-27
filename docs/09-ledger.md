# Ganymede — Contribution Ledger & Provisioned Accrual (component design)

*Stage 1 component design (`04-platform-expansion.md`, Build sequencing). Imports
the frozen spine: `05-data-model.md` (`credit_events`, `availability_ticks`,
`machine_weight`, `workers.standing`) and `06-api-delta.md` (`/v1/me`,
`/v1/leaderboard`). Owns Decisions 2, 6, 7, 11, 12; sits against invariant 2
("acceptance gates flip from sanity checks to fraud checks"). Design only.*

Freezes: the accrual window and its good-standing gate; the Weighted System Hours
formula v1; the `machine_weight` function including its interim form and
roll-forward rule; the reputation score and its `workers.standing` transitions;
the fields of `/v1/me` and `/v1/leaderboard`. Adds no table and no column beyond
the spine.

---

## 1. The provisioned-accrual engine (Decision 11)

Primary reputation accrues from *provisioned* time — enrolled, awake, available —
not time under lease and not anything job code reports. The engine is a periodic
sweep run from cron, like `invariants.py` and `status.py --alert` (§6.4): no
daemon, no work on the claim path.

### 1.1 The signal — `availability_ticks`

One row appended on **every poll and every heartbeat** (`05`): `machine_id`,
`at`, `leased`, `in_good_standing`. This is the fleet-wide, job-independent
version of the per-`(machine, job)` row `worker_eligibility` already writes each
poll, and reuses its rationale — a poll is the only event every worker generates
whether or not it gets work (`scripts/status.py:awake_workers`).

`leased` is informational: it drives the leased-vs-idle split in `/v1/me` and
never changes the credited amount — Decision 11 counts idle-available time.
`in_good_standing` is the gate (§1.3).

### 1.2 The integration window

- **3600 s, aligned to wall-clock hours.** One `credit_events` row per
  `(machine_id, window)`, `kind = 'provisioned'`, idempotent on
  `(machine_id, kind, period_start)`.
- **Per-tick cap: `AWAKE_WINDOW_SEC` (900 s)** — reused from `scripts/status.py`,
  not a new constant. A tick at `t_i` contributes `min(t_{i+1} − t_i, 900)` s to
  its window, and only if its own `in_good_standing = 1`. 900 s is already the
  fleet's judged "still awake since the last poll", so a machine that polls,
  sleeps six hours, polls again banks 900 s, not six hours. A pair straddling an
  hour boundary splits at the boundary.
- **`SETTLE_DELAY_SEC` (1800 s):** a window is written only once its `period_end`
  is >1800 s in the past, so straggler heartbeats have landed.

### 1.3 What counts as "available" — the good-standing gate

`in_good_standing = 1` iff **all** of:

1. `workers.standing != 'revoked'`.
2. Since the machine's previous tick it has **not** abandoned a claimed task,
   gone no-show on one (claimed, no heartbeat, lease expired), or had a
   submission rejected by `validate()` (`tasks.status`, `worker_eligibility`,
   `audit`). In the pull model (Decision 8) this is Decision 11's "accepts the
   leases it is offered" — claim work then drop or fail it and you are refusing
   work.
3. Outstanding unverified accepted tasks are within the ceiling for its standing
   (§5.4).

A capability `REFUSED` in `worker_eligibility` (predicate miss, Decision 15) is
neutral — it does not clear the gate. The machine cannot help the queue; it is
not farming.

**Probation.** Probation ticks keep `in_good_standing = 1` (condition 1 bars only
`revoked`), but at settle the engine scales the whole window by
`PROBATION_FACTOR` (0.5) and enforces `PROBATION_MONTHLY_CAP_HOURS` — `05`'s
"`probation`-with-limits accrue". The factor is read from `workers.standing` **at
settle time**; a machine back to `good` before settle gets the full window.
Boundary error is at most one window and accepted.

### 1.4 Settle and GC

Once a window is settled for a machine, its ticks with `at < period_end` are
GC-eligible; a sweep drops ticks older than `TICK_RETENTION_DAYS` (3), matching
`audit` / `worker_eligibility`. `credit_events` is never GC'd (`05`: "never
updated, never deleted").

---

## 2. Weighted System Hours — formula v1 (Decision 6)

Coordinator-computed; job types contribute nothing.

```
weighted_hours = (raw_seconds / 3600.0) * system_weight
```

per `(machine_id, window)`, where `raw_seconds` is the good-standing seconds from
§1.2 (INTEGER ≤ 3600, times `PROBATION_FACTOR` if §1.3 applies), and
`system_weight` / `formula_version` are read from `machine_weight` at settle and
**copied into the row** (`05` columns). The row also carries `raw_seconds`,
`period_start`, `period_end`, `created_at`, and `user_id` denormalised from
`workers.contributor_id`.

Running total = `SUM(weighted_hours) WHERE kind = 'provisioned'` — derived at read
time, never a stored balance (§7).

---

## 3. The `machine_weight` weight function (Decision 12)

`machine_weight` is a **current-value** table (`weight`, `components_json`,
`formula_version`, `computed_at`): one row per machine, recomputed on enrollment
and on re-probe. Not append-only — history lives in the `credit_events` rows it
stamped.

### 3.1 The probe must yield a real number first

`ganymede/worker/probe.py:bench_score` times 8 forward+backward passes over a
fixed block (`BATCH=4, SEQ=128, DIM=256, HEADS=4`). That shape is small enough
that a 3060 finished in ~16 ms (`8 / 0.016 ≈ 500`) and any faster card measures
the same Python-loop-plus-launch floor. The number **saturates** — non-monotonic
across GPU classes, unusable as a weight input. There is no literal clamp; the
ceiling is the benchmark's own overhead.

**Exit criteria before any `formula_version ≥ 1` reads `bench_score`:**

1. A new `BENCH_VERSION` (the score is version-scoped by design).
2. A shape where the **fastest card in the fleet** still spends the bulk of each
   iteration in compute (target ≥ 50 ms/iter).
3. **Demonstrated monotonic separation across ≥ 2 real GPU classes** (3060 vs
   3090, or 3090 vs 4090) on hardware in hand — measured and recorded like the
   golden trace (`04`, M4b guardrail).

Until all three hold, `formula_version = 0` (§3.2).

### 3.2 Interim — `formula_version = 0`, a coarse GPU-class lookup

**Explicitly interim.** It lets the ledger accrue from Phase B without waiting on
the probe reshape, and deliberately under-provisions unknown hardware. `weight`
is a static lookup on fields `compute_profile_json` already carries (`backend`,
`device_name`, `compute_capability`, `vram_mb`):

| Class | base |
| --- | --- |
| `cpu` backend / no GPU | 0.10 |
| GPU, `vram_mb < 8000` or `compute_capability < 7.0` | 0.40 |
| GPU, 8–12 GB (3060 class) — **the 1.0 anchor** | 1.00 |
| GPU, 16–24 GB (3090 / 4080 class) | 2.00 |
| GPU, ≥ 24 GB (4090 / A100 / H100 class) | 3.50 |
| GPU present, unclassified | 0.75 |

then `weight = base × clamp(vram_mb / 12000, 0.5, 1.5)`. `components_json` records
the matched class and terms. There is **no admin-override column**: an admin
correction is a recompute of the machine's `machine_weight` row (`weight` and
`computed_at` change, `formula_version` stays 0), effective forward under §3.4.

### 3.3 `formula_version ≥ 1` — probe-derived, target shape

```
weight = bench_rel^0.60 * (vram_gb/12)^0.20 * (bw_mbps/1000)^0.10
       * (ram_gb/32)^0.05 * cpu_rel^0.05
```

clamped to `[0.05, 8.0]`, with `bench_rel = bench_score / BENCH_REF[bench_version]`
against the fleet's 3060-class anchor and `cpu_rel` a fixed CPU micro-bench
normalised the same way. Exponents are `formula_version`-scoped and tunable; a
coefficient change **is** a new version. Requires probe work not owned here:
reshaped GPU bench, a CPU micro-bench, RAM total (reachable via
`_system_memory_mb`), a sustained host↔coordinator bandwidth sample — new
`compute_profile_json` keys, no coordinator schema change (§6.9).

### 3.4 `formula_version` roll-forward — never retroactive

A re-weighting is a **new `formula_version` applied to windows settled after the
recompute**. Past `credit_events` rows keep the `system_weight` /
`formula_version` they were stamped with — never recomputed, never rewritten
(`05`: append-only). `SUM(weighted_hours)` legitimately spans versions and is a
true history, not a re-price. A gross historical error is fixed by an admin
compensating row — `kind = 'provisioned'`, `raw_seconds = 0`,
`system_weight = 0`, `period_start = period_end = now`, `weighted_hours = ±X` —
with the reason in `audit` (`event = 'ledger_adjustment'`). Never an edit.

---

## 4. The secondary `work` signal

`JobType.credit(task, result) → WorkUnits` (`05`; coordinator-side, trusted).
Returns work done: tokens trained, rows inferred.

### 4.1 Recorded, never banked

On each accepted submission, one `credit_events` row:

- `kind = 'work'`
- `raw_seconds` = the integer `WorkUnits` scalar (the row's `job_type`,
  recoverable via `tasks`/`jobs`, fixes the unit)
- `weighted_hours = 0.0` — an unfiltered `SUM` stays inert
- `system_weight = 0.0`; `formula_version` = the work-normalisation version
- `period_start = period_end = created_at` = submission time

**Invariant, enforced in review:** every balance, total, and leaderboard query
filters `kind = 'provisioned'`.

Why not banked: `credit()` is third-party job code (Decision 3); banking it lets
a job type mint reputation. It stays a signal the coordinator observes and
normalises against provisioned hours (`04`, `credit()` section).

### 4.2 What it feeds

- **Reputation corroboration (§5).** The coordinator holds an expected
  work-per-`system_weight`-hour envelope. A machine whose `work` rows sit far
  outside it — far above (inflated `credit()`) or far below (claiming work,
  producing little) — is flagged into the reputation inputs as an anomaly, not a
  weighted term.
- **Optional output leaderboard.** A cosmetic `SUM(raw_seconds) WHERE
  kind = 'work'` by machine and `job_type` unit, separate from Weighted System
  Hours and never mixed in.

---

## 5. Reputation score per machine

One scalar per `machine_id` in `[0, 1]`, **earned slowly, lost fast** (invariant
2). Stored in the machine record (identity subsystem's); this doc fixes its
inputs and what it drives. It is a cached rollup, recomputed on a schedule from
`audit` and recorded outcomes.

### 5.1 Inputs

1. **`audit` rejections** — `validate()` rejection rate over a trailing window
   (the raw material `audit` has gathered since M1 — §5.1, §6.3).
2. **Redundant-execution disagreement** — tasks sharing `tasks.attempt_group`
   (`05`) sent to N machines; the coordinator owns the comparison (`04`). A
   minority machine takes a hard hit; no one in the group is credited `work`
   until it resolves.
3. **Spot-check outcomes** — known-answer tasks issued indistinguishably from
   real work (Phase D, invariant 2). A wrong answer is the largest single
   penalty.

### 5.2 Dynamics

Enrollment starts at `REP_ENROLL = 0.25` — a new machine is low-trust. Each clean
accepted submission and passed spot-check raises it a small increment (asymptotic
to 1.0). A spot-check failure or minority disagreement drops it sharply (≈ ×0.5
plus a floor subtraction); a `validate()` rejection drops it modestly.

### 5.3 `workers.standing` transitions

`good ⇄ probation → revoked` (`05`).

| Transition | Trigger |
| --- | --- |
| `good → probation` | score `< REP_GOOD` (0.60), **or** one spot-check failure, **or** one minority redundancy disagreement |
| `probation → good` | score `≥ REP_GOOD` **and** a clean `PROBATION_RECOVERY_DAYS` (7) window with ≥ 1 passed spot-check and no rejection |
| `probation → revoked` | score `< REP_REVOKE` (0.15), **or** a second spot-check failure while on probation |
| `→ revoked` (any) | admin action or the fraud rules |
| `revoked` | terminal for accrual; reinstatement is admin-only, out of scope |

`revoked` accrues nothing (§1.3.1); `probation` accrues at `PROBATION_FACTOR`
(§1.3).

### 5.4 Unverified-work ceiling — the tie to §1

"Unverified" = accepted by `validate()`, not yet corroborated by redundancy or a
spot-check.

| Standing | Unverified ceiling | Redundancy sampling |
| --- | --- | --- |
| `good` | `K_GOOD` (48) | `F_GOOD` (5–10 %) |
| `probation` | `K_PROBATION` (3) | 100 % |
| `revoked` | 0 | — |

**Named rule:** when outstanding unverified tasks exceed the ceiling, the next
tick is written `in_good_standing = 0` (§1.3.3), so provisioned accrual
**pauses** until verification catches up. A machine cannot bank Weighted System
Hours faster than its output can be checked. This is the join between the
reputation subsystem and the accrual engine.

---

## 6. `/v1/me` and `/v1/leaderboard`

Both **auth: user** (`06`); additive, response shapes new.

### 6.1 `GET /v1/me`

```
user            : { id, name, auth_provider, is_admin,
                    submitter_status }        # approved|pending|denied|revoked|null
totals          : { weighted_hours,           # SUM WHERE kind='provisioned', all caller machines
                    machines }                # count, excluding retired
machines[]      : { machine_id, display_name,
                    standing,                 # good|probation|revoked
                    reputation,               # 0..1
                    enrolled_at, last_available_at,
                    system_weight, formula_version,
                    weighted_hours_total,     # SUM WHERE kind='provisioned' AND machine_id=…
                    accrued_current_window,   # advisory: good-standing s so far this hour × weight / 3600
                    leased_now, in_good_standing_now,
                    unverified_tasks, unverified_ceiling }
recent_events[] : { id, machine_id, kind,     # provisioned|work
                    weighted_hours, raw_seconds,
                    system_weight, formula_version,
                    period_start, period_end, created_at }   # last 50, newest first
```

Retired machines (`06`, `/v1/machines/{id}/retire`) still appear with frozen
totals; `last_available_at` stops advancing.

### 6.2 `GET /v1/leaderboard`

```
generated_at
formula_version_current                       # machine_weight.formula_version in force now
by_machine[] : { rank, machine_id, display_name, user_id, user_name,
                 weighted_hours,              # SUM WHERE kind='provisioned' GROUP BY machine_id
                 system_weight, standing }
by_user[]    : { rank, user_id, user_name,
                 weighted_hours,              # SUM over the user's machines, same filter
                 machines }
by_work[]    : { rank, machine_id, display_name, job_type, work_units }  # SUM(raw_seconds) WHERE kind='work'; optional
```

- The ranking sum spans `formula_version`s by construction (§3.4) and is never
  re-priced.
- Visible to any authenticated user — Decision 7's "the leaderboard is the whole
  point", and the one place `06`'s 404-not-403 cross-tenant rule is deliberately
  relaxed. Exposed fields are `display_name` / `user_name` and the sums only — no
  email, no profile, no machine internals.
- A machine or user may opt out; then ranked as `"anonymous"` with totals still
  counted (opt-out storage is the identity doc's).
- `limit` / `offset`, default `limit = 50`, ordered `weighted_hours DESC`.

---

## 7. Reputation-only (Decisions 2, 7)

- **No `balance` column, no debit row, no redemption endpoint.** `credit_events`
  is append-only; the total is `SUM(weighted_hours)` at read time. No expiry. The
  only signed row is the §3.4 admin compensating entry, itself a *credit*, not a
  spend.
- **The path to spendable points is preserved, not designed.** Every field a
  points economy needs is already on the row — amount, `system_weight`,
  `formula_version`, `machine_id`, `user_id`, period bounds. A future spend
  system is a **separate** append-only table referencing `credit_events`, with
  balance = `SUM(credits) − SUM(spends)`. Nothing here has to be unwound. Out of
  scope; revisited only with a redemption decision.

---

## Constants

| Name | Value | Note |
| --- | --- | --- |
| `ACCRUAL_WINDOW_SEC` | 3600 | hour-aligned |
| per-tick credit cap | 900 | **reused** `AWAKE_WINDOW_SEC`, `scripts/status.py` |
| `SETTLE_DELAY_SEC` | 1800 | 2× the per-tick cap |
| `TICK_RETENTION_DAYS` | 3 | matches `audit` / `worker_eligibility` GC |
| `PROBATION_FACTOR` | 0.5 | `05` "probation-with-limits" |
| `PROBATION_MONTHLY_CAP_HOURS` | tunable | hard ceiling while on probation |
| `REP_ENROLL` / `REP_GOOD` / `REP_REVOKE` | 0.25 / 0.60 / 0.15 | §5 |
| `PROBATION_RECOVERY_DAYS` | 7 | §5.3 |
| `K_GOOD` / `K_PROBATION` | 48 / 3 | §5.4 |
| `F_GOOD` | 0.05–0.10 | redundancy sampling, `good` |
| `BENCH_REF[bench_version]` | measured | 3060-class anchor, set when §3.1 clears |

Values are tunable from observed data; changing the weight **function** is a
`formula_version` bump (§3.4).

---

## Spine deviations

**None.** No table, column, key, enum, or endpoint departs from `05` / `06`.

One noted semantic choice, no schema change: a `kind = 'work'` row carries the
`credit()` `WorkUnits` scalar in `raw_seconds` and sets `weighted_hours = 0.0`.
This overloads `raw_seconds` ("availability seconds") for `work` rows only;
`provisioned` rows are unaffected. A dedicated `work_units` column was rejected
because `05` freezes `credit_events`'s columns. The `kind = 'provisioned'` filter
on every banked query (§4.1) makes the overload safe.

---

## Frozen vs. open

**Frozen here:** the 3600 s hour-aligned accrual window and 900 s per-tick cap;
the three-condition good-standing gate and `probation` handling; the
`weighted_hours = (raw_seconds / 3600) × system_weight` formula and its
non-retroactive `formula_version` roll-forward; `formula_version = 0` as an
explicitly interim GPU-class lookup gated on a de-saturated `bench_score` with a
measured-monotonicity exit criterion; the `work`-signal recording rule and its
exclusion from every banked total; the reputation inputs, `standing` transition
table, and the unverified-work ceiling that pauses accrual; the field lists of
`/v1/me` and `/v1/leaderboard`.

**Open (other docs):** the reputation score's exact curve and storage (identity);
spot-check generation and indistinguishability (sandbox / SDK); `attempt_group`
dispatch and comparison (scheduler); the reshaped GPU bench, CPU micro-bench, and
bandwidth probe (worker probe); leaderboard opt-out storage (identity); the SSE
`credit_events` delta shape (web-UI).
