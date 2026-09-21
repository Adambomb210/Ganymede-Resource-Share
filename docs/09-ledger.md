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
   penalty. Designed in **`13-fairness.md`** §5: the known answer is an
   already-accepted shard re-issued to a different machine, which is how
   "indistinguishable" is met — it is real work.

### 5.2 Dynamics

Enrollment starts at `REP_ENROLL = 0.25` — a new machine is low-trust. The score
is a **pure function of the trailing 30-day window**: `ledger.reputation_for`
takes the five counts that window yields (accepted, passed probes, rejections,
failed probes, minority disagreements) and returns a score. Nothing carries over
between sweeps.

Each clean accepted submission and passed spot-check closes a fixed *share of the
remaining distance to 1.0* — `REP_CLEAN_STEP`, and twice that for a corroborated
pass — so the score is asymptotic to 1.0 in the **volume of clean work in the
window**, and reaches `REP_GOOD` at 32 accepted submissions. A spot-check failure
or minority disagreement drops it sharply (≈ ×0.5 plus a floor subtraction); a
`validate()` rejection drops it modestly. Convictions apply worst-first, so a
machine with several lands where the worst one puts it rather than where the
order of evaluation left it.

> **Why the word "asymptotic" changed meaning here (review, 2026-09-21).** This
> section and §5's "a cached rollup, recomputed on a schedule from recorded
> outcomes" used to be in tension, and the implementation fell into the gap.
> "Recomputed from recorded outcomes" describes a *pure function of history*;
> "each submission raises it a small increment, asymptotic to 1.0" describes an
> *accumulator*. `recompute_reputation` did both at once: it started from the
> **stored** score and applied **trailing-window totals** — not deltas; there is
> no watermark column — so every sweep re-convicted a machine for the same
> historical events. The per-rejection recurrence `s' = (s + inc)*0.5 −
> REP_REJECT_FLOOR` has fixed point `inc − 0.20`, and with spot checks off (the
> default) `inc ≤ 0.16` sat below it, so any machine with one rejection in its
> window decayed to 0.0 and was `revoked` — terminal, admin-only to reverse —
> in about two sweeps, at a recommended cadence of once a minute. Standing was a
> function of cron cadence rather than of conduct.
>
> Resolved in favour of the pure function. "Asymptotic to 1.0" survives, but it
> is now asymptotic *in the volume of clean work inside the window* rather than
> in the number of times the sweep has run, which is the reading that makes both
> sentences true at once. The `min(acc, 8)` cap is gone with the thing it existed
> to bound. Pinned by `test_the_reputation_sweep_is_idempotent` and
> `test_repetition_alone_never_moves_standing`.

### 5.3 `workers.standing` transitions

`good ⇄ probation → revoked` (`05`).

Standing is a **classification of the trailing window**, not a walk through
states — the same property §5.2 demanded of the score, for the same reason.
Every row below is a question about the window alone, so running the sweep twice
over an unchanged history lands in the same place. The one exception is
`probation → good`, which asks what the machine has done *since*, and is
idempotent because its answer only changes when the machine does something.

| Transition | Trigger |
| --- | --- |
| `good → probation` | score `< REP_GOOD` (0.60), **or** one spot-check failure, **or** one minority redundancy disagreement |
| `probation → good` | score `≥ REP_GOOD` **and** a clean `PROBATION_RECOVERY_DAYS` (7) window with ≥ 1 passed spot-check and no rejection |
| `→ revoked` | **two** spot-check failures in the trailing window |
| `→ revoked` (any) | admin action or the fraud rules |
| `revoked` | terminal for accrual; reinstatement is admin-only, out of scope |

Three things about that table are deliberate and were not obvious (review,
2026-09-21):

**Revocation is conduct-only; `score < REP_REVOKE` is gone as a trigger.** It
could not survive the move to a window score. A low score there is dominated by
low *volume*, not by bad conduct: `REP_ENROLL` (0.25) sits barely above
`REP_REVOKE` (0.15), so a contributor whose very first submission failed
validation scored 0.025 and was permanently banned — on the most sympathetic
case in the system. Against an accumulator that only sank that far through
repeated misconduct the threshold meant something; against a 30-day window it
means "new and unlucky", and a trigger that cannot tell those apart must not be
terminal. A low score still suppresses accrual through the §6 reputation
weighting — just never terminally. `REP_REVOKE` remains defined as the published
floor of the probation band.

**"A second failure" is counted in the window, not against probation entry.**
There is no `standing_changed_at` column, and the stateful substitute — "on
probation, and a failure in the last 7 days" — re-read the *same* failure on
every sweep, which is how revocation-by-cron happened. The cost of the window
form is real and accepted: two failures 30 days apart now revoke even if the
machine recovered to `good` in between. Adding a column to recover that
precision is a migration, and is not worth it to preserve a rule that was
firing on its own trigger.

**Minority disagreements never revoke, at any count** — only spot-check failures
do. A minority is merely outvoted; a failed probe was caught against an answer
already known to be right. That ordering is asserted by
`test_a_failed_probe_costs_more_than_a_minority_disagreement`, and the old table
blurred it by letting a low enough score revoke on any path.

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
