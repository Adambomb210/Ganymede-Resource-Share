# 13 — Fair-share, quotas, preemption, and anti-fraud v1

The component doc for the rest of Phase D. `04-platform-expansion.md` lists seven
things under "Harden multi-tenancy"; `11-sandbox.md` took one of them
(sandboxing) and Phase C took another (the submitter allowlist administered from
the web UI — `12` C2 ships approve / deny / revoke as buttons on
`/ui/submitters`). This doc owns the five left:

1. Fair-share scheduling
2. Quotas and budgets
3. Preemption
4. Spot-check tasks with known answers
5. Reputation-weighted aggregation

Three other docs deferred their formula to exactly here, and this is the doc they
were deferring to:

- `07-scheduler.md` §4 — "The fair-share formula and its share-accounting table —
  Phase D", and "Preemption and the task-scoped `cancel` — Phase D".
- `09-ledger.md` §5.1 — reputation input 3, "spot-check outcomes … known-answer
  tasks issued indistinguishably from real work (Phase D, invariant 2)".
- `10-jobtype-sdk.md` "Open" — "the redundant-execution comparator's exact
  sampling and the probation thresholds (anti-fraud, Phase D)".

---

## 0. The shape of the whole thing

**Everything here is off by default.** Every one of the five is gated on a
setting whose default reproduces today's behaviour exactly, and in three of the
five that inertness is a test, not a claim. This is not timidity: this is a
fleet with one contributor and no contention, so there is nothing to tune any of
these against yet. Shipping them on would be shipping untuned policy, and untuned
scheduling policy is worse than none — it moves work for reasons nobody can
explain.

What "finished" means for Phase D, then, is that the *mechanism* exists, is
tested, and is one environment variable away; not that the fleet is running
under it.

The one place that split gets uncomfortable is automatic preemption, and §4.6
says so plainly rather than burying it.

| Slice | Setting | Default | Inert when off? |
| --- | --- | --- | --- |
| Fair-share | `GANYMEDE_FAIRSHARE_SPREAD` | `0.0` | Provably — the sort key gains `+ 0.0` |
| Quotas | per-submitter row | no row | Structurally — no row, no cap |
| Preemption (manual) | — | always on | It is an explicit admin action |
| Preemption (automatic) | `GANYMEDE_AUTOPREEMPT` | `false` | The sweep step returns early |
| Spot-checks | `GANYMEDE_SPOTCHECK_RATE` | `0.0` | The dice never come up |
| Reputation weighting | `GANYMEDE_REPUTATION_WEIGHTED_AGG` | `false` | Provably — and under uniform reputation, inert even when *on* |

---

## 1. Share accounting

The substrate for §2 and §3 both. One question: **how much of the fleet has each
submitter had lately?**

### 1.1 The unit is the lease, not the work

A submitter's consumption is measured in **leased task-seconds**: the wall-clock
time their jobs held a machine, whether or not the machine produced anything.

The alternative — charging for accepted work — is more flattering and less
correct. A machine leased for fifteen minutes is fifteen minutes nobody else
could have that machine. If a job leases a card and the task is abandoned, the
fleet still lost the time, and a submitter whose jobs abandon constantly would
otherwise be charged nothing for a great deal of damage. Charge the reservation.

**"A machine" is stale as of `14`, and the gap is real.** A lease is now a set of
*devices*, not a machine. For `collab_lora_finetune` nothing changes: `14` §7 gives a
multi-GPU box one lease per card, so four cards are four task rows and accrue four
times — already correctly weighted, and `14` §8.1 now caps that type at one device per
task anyway. But a **static** type (`batch_inference`, `contained_batch`) with
`gpu_count > 1` allocates several devices to a *single* task row, and `_task_seconds`
charges per task: that lease's seconds are counted once, identically to a one-card lease
of the same duration, while holding N times the hardware. A submitter running wide
static jobs therefore pays 1x the fairness cost for Nx the consumption and is
under-demoted by `effective_rank`.

Left unfixed deliberately, and recorded here rather than in a commit message. The fix is
to multiply by `tasks.gpu_count`, which changes what the accrued number *means* — and
this section's whole point is that such a change is a `formula_version` bump with a
migration, not a silent redefinition. It is also currently inert: nothing in the fleet
submits a wide static job yet. Fold it into `formula_version = 1` alongside
`machine_weight`, where the unit is being restated anyway.

`formula_version = 0` is unweighted seconds. A second of a 4090 and a second of a
3060 count the same, which is wrong, and `machine_weight` (`09` §3.2) is sitting
right there to fix it — but weighting the share by machine class makes the
fairness formula depend on the *weight* formula's version, and `machine_weight`
is itself explicitly interim at `formula_version = 0`. Two interim formulas
multiplied together produce a number nobody can reason about. Unweighted first;
`formula_version = 1` is the weighted one, and the column exists so the change is
a migration rather than a silent redefinition.

### 1.2 `tasks.leased_at`

Not derivable today. `tasks.created_at` is the lease time for
`collab_lora_finetune` (its `claim.py` inserts the row already `leased`), but a
static type's tasks are planned at `enqueue` and leased minutes or days later, so
`created_at` is the wrong number by an unbounded margin.

Migration 008 adds `tasks.leased_at TEXT`, written at every transition to
`leased` — both the static reserve and the collab insert — and backfilled to
`created_at` for existing rows. The backfill is exact for every
`collab_lora_finetune` row that exists and approximate for nothing, because no
static-type task has ever been leased on any live database. Recorded because a
future reader will otherwise assume it is a guess.

### 1.3 Seconds per task

```
end     = COALESCE(submission.received_at, lease_expires_at)
seconds = max(0, min(now, end) - leased_at)
```

An in-flight lease is charged for what it has held *so far*, which means a
submitter's share rises while their work runs rather than in a step when it
finishes. An abandoned lease is charged in full to `lease_expires_at`, per §1.1.
A submission that landed early ends the charge at the submission — the machine
was free from that moment.

### 1.4 Decay, not a window

```
share_seconds(owner) = Σ  seconds(task) × 0.5 ^ (age_hours(task) / SHARE_HALF_LIFE_HOURS)
```

`SHARE_HALF_LIFE_HOURS = 24`.

A trailing fixed window is the obvious alternative and it has a bad failure mode:
when a large job falls off the back edge of the window, the queue reorders itself
and nobody did anything. An operator watching the queue move sees an event with
no cause. Exponential decay is continuous — the ordering drifts, it never jumps.

Only tasks with `leased_at` within `SHARE_LOOKBACK_DAYS = 7` are summed. That is
a computational bound, not a policy one: at a 24-hour half-life a seven-day-old
second contributes 2⁻⁷ ≈ 0.8% of itself, and the truncation error is far below
the noise in the thing being measured.

### 1.5 A rollup, decayed forward on read

```sql
CREATE TABLE share_accounting (
    owner_id        TEXT PRIMARY KEY REFERENCES contributors(id),
    decayed_seconds REAL NOT NULL,
    formula_version INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
)
```

Recomputed on the existing sweep, alongside the ledger and the image scan — same
cadence, same cron entry, one more call. `_selectable_jobs` runs inside the hot
claim path under `BEGIN IMMEDIATE`; it reads one row per owner and does no
aggregation.

**`decayed_seconds` is as-of `updated_at`, and the reader ages it forward:**

```
current(owner) = decayed_seconds × 0.5 ^ ((now - updated_at) / SHARE_HALF_LIFE_HOURS)
```

This is the property that makes a cached rollup safe here. A sweep that has not
run for an hour does not hand the scheduler an hour-stale number — it hands it a
correctly-aged one, missing only the leases *taken during* that hour. The rollup
degrades toward "everyone's share decays uniformly", which is the neutral answer,
rather than toward "whoever was ahead an hour ago is still ahead". A sweep that
has stopped entirely fades to zero and fair-share turns itself off, which is the
right direction for a mechanism to fail in.

---

## 2. Fair-share scheduling

### 2.1 It replaces the primary sort term, and it is never a filter

`07` §4 named the seam precisely: the sort key of `_selectable_jobs`, today
`(priority_rank, affinity_miss, created_at)`, and "the walk is untouched:
head-first, backfilling, one task per machine."

```python
effective_rank = priority_rank + spread * share_fraction(owner)
key = (effective_rank, affinity_miss, created_at)
```

**A sort term, not a filter.** The image gate immediately above it in the same
function *is* a filter, and deliberately — an unscannable image must never be
merely unlikely to be picked. Fair-share is the exact opposite case and the
inverse rule binds: a submitter who has used a lot of the fleet must be **late**,
never **unschedulable**. Filtering them would break the capability backfill —
their job would stop being offered to a machine no other job fits, and the fleet
would idle to punish them. Demotion is the whole mechanism.

### 2.2 `share_fraction` and the spread

`share_fraction(owner) ∈ [0, 1]` is that owner's share of the fleet's total
current decayed seconds. Zero recent usage → 0.0. Sole recent user → 1.0. No
recent usage by anyone → 0.0 for everyone.

`spread` is `GANYMEDE_FAIRSHARE_SPREAD`, default **`0.0`**, and it is the whole
control. It says, in units of `priority_rank`, **how much of the ordering the
admin is delegating to the formula.** Ranks are sparse by convention (10, 20,
30 — `07` §5), so:

- `spread = 0` — off. `+ 0.0 * anything` is `+ 0.0`; the key is today's key with
  an int promoted to a float, and floats compare identically to the ints they
  came from at these magnitudes. Inertness here is arithmetic, and there is a
  test that asserts the *order* is byte-identical with the setting off.
- `spread = 10` — a submitter monopolising the fleet slips at most one sparse
  rank slot. The recommended first value.
- `spread = 100` — fair-share dominates and `priority_rank` is a tiebreak.

`07` §5 says "the admin ordering stays the base, fair-share is an adjustment the
admin turns on." A dial with a documented unit keeps that literally true instead
of merely asserting it.

### 2.3 Fair-share only ever demotes

`share_fraction ≥ 0`, so `effective_rank ≥ priority_rank` always. Fair-share can
make a job run *later* than the admin ordered it. It can never make one run
*sooner*.

This asymmetry is deliberate and worth the paragraph. `07` §5 is emphatic that
submitters do not set their own priority; a formula that could promote a job
above its admin rank would be a back door into exactly that — starve yourself
deliberately, get promoted. There is no such move here. The admin's rank is a
ceiling on how well any job can do.

### 2.4 What this does not do

No aging. A job at rank 90 does not creep toward rank 10 because it has waited.
`07` §4's structural argument still carries the anti-starvation load: the backfill
means a low-rank job runs whenever a machine no higher-rank job accepts is free,
and a job that never runs is one every free machine is continuously absorbed away
from — the admin's call, visible as a zero in `GET /v1/admin/queue`.

Aging and fair-share solve different problems (waiting vs. hogging) and stacking
both on one sort term makes the ordering unexplainable. One at a time.

---

## 3. Quotas and budgets

Two different things, and conflating them is the usual mistake:

- **Quota** — a cap on what you may hold *at once*. Self-clearing. "No more than
  four machines."
- **Budget** — a cap on what you may consume *in total over a period*. Does not
  self-clear. "No more than 200 machine-hours this month."

### 3.1 The table

```sql
CREATE TABLE submitter_quotas (
    user_id              TEXT PRIMARY KEY REFERENCES contributors(id),
    max_concurrent_tasks INTEGER,   -- NULL = uncapped
    monthly_task_hours   REAL,      -- NULL = uncapped
    note                 TEXT,
    updated_by           TEXT REFERENCES contributors(id),
    updated_at           TEXT NOT NULL
)
```

**No row means no cap**, and no row is the default for everybody, including
every submitter that already exists. The feature is therefore inert on arrival
without needing a flag: an admin turns it on for one submitter by creating one
row. `NULL` in a column is that one dimension uncapped, so a concurrency cap
without a budget is a row with one column filled.

Admin-write-only, by the same argument `07` §5 makes about `priority_rank`:
`POST /v1/admin/submitters/{user_id}/quota {max_concurrent_tasks,
monthly_task_hours, note}`, and it renders on `/ui/submitters` beside the
allowlist status, which is the page an admin is already on when they think about
a submitter.

### 3.2 Enforced in the walk, with a recorded reason — not as a filter

Both caps are checked inside the claim walk, as a `continue` with a recorded
`eligibility.Verdict`, using the two new refusal reasons:

```
over_concurrency_quota
over_monthly_budget
```

A `continue` in the walk is every bit as absolute as a filter — the job is not
offered, full stop — and it is the shape the backfill is built around. What it
buys over a filter is the *explanation*: the refusal lands in
`worker_eligibility` and surfaces through `explain()` and the admin queue view,
so a submitter whose jobs have stopped moving is told "you are at your cap"
rather than watching a queued job sit there silently. A quota nobody can see
hitting is a support ticket.

The counts are computed once per claim and memoised by `owner_id` for the walk —
a queue of thirty jobs from three submitters is three queries, not thirty.

### 3.3 What each cap counts

- **Concurrency**: `COUNT(*) FROM tasks WHERE status = 'leased'` joined to jobs
  by owner. Live, exact, cheap.
- **Budget**: the *undecayed* sum of §1.3 seconds since the start of the current
  UTC calendar month. Deliberately not the decayed share — a budget is an
  accounting quantity and it has to match what a human gets when they add the
  month up by hand. Decay belongs to fairness, which is about *recency*; a budget
  is about a *total*.

The month boundary is UTC and hard. No proration, no rollover, no partial first
month. Every one of those is a billing feature, and `04` Decision 2 keeps this
system firmly out of billing — this is a brake, not an invoice.

### 3.4 Admission control at enqueue

A job whose owner is already over their monthly budget is refused at `POST
/v1/jobs/{id}/enqueue` with a `409` naming the budget, rather than being allowed
to queue and then never lease. The concurrency quota is *not* checked there — it
is transient by nature, and refusing an enqueue because four tasks happen to be
running right now would be nonsense.

---

## 4. Preemption

`07` §4: "it aims the Decision 18 heartbeat `cancel` at a task, not a job."

### 4.1 The transport does not change

This is the point. Decision 8 is that a cancel rides the heartbeat response and
nothing else; `11` §3 built that for the job-scoped case. A preemption is the
*same* `{"cancel": "soft"|"hard"}` field on the *same* heartbeat response,
handled by the *same* worker latch. No new endpoint, no new field, no push.

What changes is only who can be the *cause* of that field appearing.

### 4.2 The marker: `tasks.preempt_mode`

Migration 008 adds `tasks.preempt_mode TEXT` (`soft` | `hard`, `NULL` = not
preempted).

`rounds.cancel_outstanding(conn, task_id)` widens from job-keyed to
**task-first**:

```
the task's own preempt_mode, if set
else the job-level cancel (jobs.status = 'cancelled' → jobs.cancel_mode or 'soft')
else None
```

Task first, because a preemption targets a task on a job that is still perfectly
alive — `queued` or `running` — so the existing `WHERE j.status = 'cancelled'`
would find nothing. The job-level clause stays as the fallback and is unchanged.

### 4.3 A preempted task lands on `preempted`, and that is a third status

`rounds.abandon()` currently chooses between `cancelled` and `abandoned` off
`cancel_outstanding`. It gains a third answer, and both existing answers are
wrong for this case for different reasons:

- **`abandoned` is wrong** because `ledger._infraction_since` counts `abandoned`
  and `expired` against the machine's standing. A machine that was told to stop
  by the scheduler did nothing wrong. This is the identical argument that split
  `cancelled` out in `11` §3, and it lands the identical way.
- **`cancelled` is wrong** because it is *terminal*. A cancelled task's work is
  not wanted. A preempted task's work is wanted very much — it was interrupted,
  not abandoned as an idea — and it has to go back in the pool.

So: `preempted`, which is non-terminal and re-servable. `expire_leases` gets the
same three-way split for the case where a preempted worker simply stops
heartbeating instead of acknowledging.

### 4.4 A preemption does not burn an attempt

`_reserve_static_task` bounds re-serving with `attempts < close.MAX_TASK_ATTEMPTS`
(5) and increments on every reserve. Left alone, five preemptions would fail a
shard that nothing was ever wrong with — the scheduler would have quietly
consumed a budget meant for *the shard's* failures, not for its own decisions.

The reserve therefore becomes:

```sql
attempts = attempts + CASE WHEN status = 'preempted' THEN 0 ELSE 1 END
```

and `'preempted'` joins `('planned', 'expired', 'abandoned')` in the re-serve
`WHERE`.

Every preemption writes an `audit` row (`task_preempted`, with the task, the
worker, the mode, and the cause), so a shard being pushed around repeatedly is
visible even though it burns no attempts. A hard cap on preemptions per task is
**not** in this version: the bound that matters is on the *policy* (§4.6, one per
sweep), and a cap here would silently convert "this shard keeps getting
preempted" into "this shard failed", which is a worse thing to discover.

### 4.5 Manual preemption

`POST /v1/admin/tasks/{task_id}/preempt {mode}` — admin only, `mode` is `soft` or
`hard`, exactly the `11` §3 grammar. Sets `preempt_mode`, and the next heartbeat
carries it. This is the whole mechanism, and it is always available.

### 4.6 Automatic preemption is built, and it is off

Ships behind `GANYMEDE_AUTOPREEMPT`, default **false**. The mechanism above is
complete without it; this is a policy on top, and it is the one place in this doc
where "off by default" is covering for something real rather than just being
careful. Say it plainly: **this policy has never run against a fleet with actual
contention, because no such fleet exists yet.** The numbers below are reasoned,
not measured. Turning it on before there is contention to watch would be
turning on an untested scheduler.

The policy, run once per sweep, at most **one preemption per sweep**:

1. Find a **starved** job: `queued` or `running`, zero leased tasks, and waiting
   longer than `AUTOPREEMPT_STARVE_MIN` (30 minutes).
2. Find leases held by jobs whose `effective_rank` is at least
   `AUTOPREEMPT_RANK_MARGIN` (10 — one sparse slot) *worse* than the starved
   job's.
3. Keep only those held by a machine that could plausibly take the starved job:
   one with **no current `refused` verdict** for it in `worker_eligibility`.
4. Preempt the **longest-running** of those, `soft`.

Step 3 is the part worth pointing at. It needs "would this machine accept that
job?", which is the constraint gate — and re-evaluating the gate here would mean
reaching for a probe profile the sweep does not have. But the claim path has
already answered that question for every (machine, job) pair it has walked, and
written the answer down. `worker_eligibility` is a recorder (`07` "Frozen here")
and this is the first thing to read it as an input. Absence of a refusal is a
weaker signal than a fresh evaluation — a machine that has never polled while
this job was queued has no verdict either way — but it fails in the safe
direction: no candidate, no preemption.

One per sweep, `soft`, and only against a job at least a full rank slot better
off: three separate brakes, because the failure mode of an over-eager preemption
policy is a fleet that spends its time stopping and restarting work.

---

## 5. Spot-check tasks

`09` §5.1 input 3: known-answer tasks, "issued indistinguishably from real work",
where a wrong answer is the largest single penalty.

### 5.1 A spot-check is a re-issue of an already-accepted shard

The design that suggests itself first — synthesize a fake task with a canned
expected output — fails the requirement in the sentence that states it. A
synthetic task is distinguishable: it is not on any real job, or it is on one and
pollutes the aggregation, and either way a machine looking for the tell will find
one.

So instead: **a spot-check is a real, already-accepted shard of a real job,
re-issued to a different machine.** Nothing about it can be detected, because
there is nothing to detect — it is genuine work, on a genuine job, with a genuine
payload. The known answer is the answer a trusted machine already gave.

This makes spot-checks a near-relative of the `attempt_group` redundancy that
already exists, and the difference is exactly the interesting one. Redundancy
issues N copies concurrently and compares them against each other: a
disagreement is *ambiguous* — a minority is suspected, not convicted, and `09`
§5.1 rates it a hard hit rather than the maximum. A spot-check compares one new
answer against one **already-accepted** answer. That asymmetry is why `09` §5.1
rates it the largest single penalty: there is no question about which side is
wrong.

### 5.2 Only deterministic types

"Known answer" is meaningless for `collab_lora_finetune`. Training is stochastic;
two honest machines produce different adapters, which is the entire reason `05`
has a norm gate and a divergence metric instead of an equality check.

A type opts in by being deterministic. Today that is `batch_inference`, which
decodes greedily (`10`). A type that is not deterministic is never spot-checked,
and this is a property of the type, not a configuration — the coordinator looks
for `shape_claim` (a dynamic, per-machine-sized type is not a candidate) and for
the type's own comparator.

**Amendment (Phase E): the opt-in is explicit, `JobType.spot_checkable`,
default off.** As first built there was no gate: probes are issued from the
static-task reserve path and `batch_inference` was the only type on it, which
made *static* an accurate proxy for *deterministic* by accident.
`contained_batch` (`10` §7) is static too and runs a submitter's image, so the
proxy stopped holding — and `judge` reaches for `batch_inference`'s comparator
regardless of the job's own type, so an honest machine on a job that samples,
threads or stamps a timestamp would have been convicted by it, with `09` §5.1's
largest single penalty. Gating issuance rather than making `judge` polymorphic
is the smaller change, and it makes that hardcoded comparator correct *by
construction*: nothing but `batch_inference` is ever judged.

One tension this section did not anticipate, recorded rather than resolved: for
a *generic* containerised type, determinism is a property of the submitter's
image and therefore per-**job**, not per-type. A per-job opt-in is deferred;
`contained_batch` says no at the type level, full stop, which keeps this
section's rule intact and costs nothing yet.

### 5.3 Issue and judge

```sql
CREATE TABLE spot_check_issues (
    task_id        TEXT PRIMARY KEY REFERENCES tasks(id),
    source_task_id TEXT NOT NULL REFERENCES tasks(id),
    issued_at      TEXT NOT NULL,
    outcome        TEXT,          -- NULL until judged | passed | failed
    decided_at     TEXT
)
```

**Issue.** On the static claim path, with probability
`GANYMEDE_SPOTCHECK_RATE` (default `0.0`), the coordinator inserts a fresh
`tasks` row duplicating the `input_ref_json` of an accepted task on that job
**done by a different machine**, leases it to the claimant, and records the
`spot_check_issues` row. Different machine is load-bearing: a machine checked
against itself agrees with itself.

**Judge.** On submit, both artifacts are parsed and compared with the type's own
comparator — for `batch_inference`, `validate.sample_agreement`, the exact
function the `attempt_group` path uses, with the job's own `agree_on` and
`sample_rows` from its spec. One comparator, one definition of "the same
answer"; a spot-check that used a stricter test than redundancy would convict
machines redundancy would acquit.

### 5.4 A probe is excluded from job completion

`_advance_parallel_job` requires every task on the job to be accepted. A failed
spot-check would otherwise wedge the job forever — the probe never passes, so the
job is never done. And a *passed* one would join an `attempt_group` it was never
part of.

Probe tasks are therefore excluded from both, by one clause:

```sql
... FROM tasks WHERE job_id = ? AND id NOT IN (SELECT task_id FROM spot_check_issues)
```

The probe's work is still real and still credited to the machine that did it. It
just does not participate in deciding whether the job is finished — the shard it
duplicates already did that.

### 5.5 Into the reputation score

`ledger.recompute_reputation`'s docstring already reserves the slot: "spot-checks
and redundant-execution disagreement are Phase D and enter here as the extra
penalty terms the moment they exist." Both arrive now.

| Input | Effect on the score |
| --- | --- |
| Passed spot-check | `+0.04`, asymptotic to 1.0 — twice a clean submission, because it is corroborated |
| Failed spot-check | `×0.5 − 0.20` — the largest single penalty (`09` §5.1) |
| Minority in an `attempt_group` disagreement | `×0.5 − 0.15` — a hard hit, below a spot-check because a minority is suspected, not convicted |

And the standing transitions `09` §5.3 already froze, which until now had no
trigger that could fire them:

- `good → probation` on **one** spot-check failure, regardless of score.
- `probation → revoked` on a **second** spot-check failure while on probation.
- `probation → good` additionally requires **≥ 1 passed spot-check** in the clean
  `PROBATION_RECOVERY_DAYS` window — which is why recovery cannot be waited out
  in silence: a machine that stops working stops being able to earn its way back.

The minority side of a redundancy disagreement is identified from the
`attempt_group_disagreement` audit event, which `close.py` already writes with
each member's `worker_id` captured *before* re-dispatch nulls it — recorded at
the time as "a Phase-D scorer can still find the offending machine". This is that
scorer.

---

## 6. Reputation-weighted aggregation

The only numerically live slice in this doc. Everything else moves scheduling
and bookkeeping; this one moves the loss.

### 6.1 The change

`aggregate.dense_weights(steps, keys, cap)` gains an optional
`reputation: list[float] | None`. When present:

```
raw_i = steps_i × reputation_i        (instead of steps_i)
```

and then the identical share, dominance-cap and renormalise loop runs on top,
untouched. The cap still means what it meant — "no worker carries more than
`cap` × the median contribution" — it is just measuring a contribution that now
accounts for how much the coordinator trusts the machine that made it.

`reduce.py` passes the submitting machines' `workers.reputation` when
`GANYMEDE_REPUTATION_WEIGHTED_AGG` is set, and `None` otherwise. Default off.

### 6.2 Why it is off, and the two tests that matter

`03`'s golden trace is an entry criterion for exactly this kind of change: the
platform work is supposed to be numerically inert, and this is the one piece that
is not. Two tests, and they check different things:

1. **Uniform reputation is exactly inert, even with the feature on.** Every
   machine enrolls at `REP_ENROLL = 0.25`; a cohort that has all enrolled and
   none diverged has a constant reputation vector, and a constant factor
   cancels in the normalisation. `weights(steps, rep=[c]*n) == weights(steps)`
   to floating-point equality. This is the test that says the *formula* is a
   reweighting and not a rescaling.
2. **The default path is untouched with the feature off** — `reduce` passes
   `reputation=None`, and `None` is asserted identical to not passing the
   argument at all.

   The obvious third test is the golden trace, and it cannot be written: `04`
   ("the golden trace doesn't exist yet") records that it has never been
   captured, because capturing it needs the M4b hardware. So the standing
   entry criterion for turning this flag on is **capture the trace first, then
   turn it on and re-run it** — not "the tests pass". Two unit tests about a
   weight vector are not evidence about a loss curve, and this section would be
   overclaiming if it implied otherwise.

The interesting case — mixed reputations across a real cohort — is not something
a unit test can validate. It changes the answer, on purpose, and whether it
changes it for the better is an empirical question about a fleet that has run
long enough for reputations to diverge. That is an M4b-and-after question, and
the flag is how it gets asked.

### 6.3 It does not touch credit

Reputation weighting changes the *merge*, not the *ledger*. `09` §1.3 already
scales accrual by standing (`PROBATION_FACTOR`), and that stays the only place
reputation touches what a contributor earns. Weighting the merge by reputation
*and* the accrual by standing off the same signal would compound one number into
two penalties for one fault.

---

---

## Status — built

All five slices are in, behind their defaults. The code:

| Slice | Where |
| --- | --- |
| §1 share accounting | `coordinator/fairness.py`, migration 008, the ledger sweep |
| §2 fair-share | `fairness.effective_rank` in `app._selectable_jobs`'s sort key |
| §3 quotas | `fairness.quota_refusal` in the claim walk; `budget_exhausted` at enqueue |
| §4 preemption | `tasks.preempt_mode`, `rounds.cancel_outstanding`, `POST /v1/admin/tasks/{id}/preempt`, `fairness.autopreempt` |
| §5 spot-checks | `coordinator/spotcheck.py`, issued in `_claim_static_task`, judged on submit |
| §5.5 reputation inputs | `ledger.recompute_reputation` — all three of `09` §5.1's inputs now live |
| §6 weighted merge | `aggregate.dense_weights(..., reputation=)`, threaded through `reduce_close` |

### Five deviations from the design above

Recorded here rather than only in commit messages, because a deviation nobody
can find is a lie in the doc.

1. **A redundancy disagreement penalises the whole group, not the minority.**
   §5.5's table says "minority", and the code cannot identify one. The
   comparator is `sample_agreement`, which answers *did they agree* and not
   *who was right*; `10` leaves minority identification open. Guessing would
   apply a hard penalty to whichever machine happened to be listed first in an
   audit event. Penalising every member is the conservative reading and it
   matches what redundancy already does — nobody in a disagreeing group is
   credited until it resolves. The moment a comparator can name a minority, this
   should narrow to it.

2. **Automatic preemption's constants are reasoned, not measured.** §4.6 already
   says this; repeating it here because it is the one place "off by default" is
   covering for something real rather than being careful. `AUTOPREEMPT_STARVE_MIN`
   and `AUTOPREEMPT_RANK_MARGIN` want a fleet with contention.

3. **No in-tree job runs a spot-check end to end.** §5 needs a deterministic
   type, which means `batch_inference`, which has no live jobs yet — its first
   real run is Phase E. The mechanism is unit-tested against a fixed rng and a
   seeded store; it has never been exercised by a worker.

4. **A voided probe is retired on the sweep, not at the moment its task dies.**
   §5.3 did not say when. `spotcheck.void_stale` runs before
   `evaluate_reputation` on the same sweep, so a probe whose machine vanished is
   never counted as anything — but between the task expiring and the next sweep
   the row sits at `outcome IS NULL`. Nothing reads it in that state; the
   reputation query filters `passed` / `failed`.

5. **`reduce_close` gained a defaulted keyword.** `10` §3 freezes the reduce
   signature; `rep_weighted=False` is additive, so a type that never heard of §6
   keeps its existing call. Called out because "frozen" and "we added a
   parameter" want to be seen together.

### What is off, and what turning it on costs

- `GANYMEDE_FAIRSHARE_SPREAD=10` — one sparse rank slot of authority. Needs the
  ledger sweep running, or every share reads zero and the queue does not move.
- A `submitter_quotas` row — per submitter, no flag, immediate.
- `GANYMEDE_AUTOPREEMPT=1` — see deviation 2.
- `GANYMEDE_SPOTCHECK_RATE=0.05` — needs a deterministic type with accepted work
  on the same job; below that it silently issues nothing, which is correct and
  looks identical to being off.
- `GANYMEDE_REPUTATION_WEIGHTED_AGG=1` — moves the loss. Not before a fleet whose
  reputations have diverged, and not before the golden trace exists to re-run
  (`04`: it has not been captured yet). This is the one flag with a hardware
  prerequisite rather than just a fleet-size one.

---

## Frozen here

- Share is **leased task-seconds**, exponentially decayed at a 24-hour half life,
  charged for the reservation and not the work, `formula_version = 0` unweighted.
- The rollup is decayed forward **on read** from `updated_at`, so a stale sweep
  fades fairness out rather than freezing it.
- Fair-share is `priority_rank + spread × share_fraction`, a **sort term**, and it
  **only ever demotes**. `spread` defaults to `0.0`.
- Quotas are a **row per submitter**, absent by default, admin-write-only,
  enforced as a walk refusal with a recorded reason — never as a silent filter.
- Budget is an undecayed UTC-calendar-month sum. No proration, no rollover.
- Preemption rides the **existing** heartbeat `cancel` field. Task marker first,
  job cancel as fallback.
- `preempted` is a third task status: not an infraction, not terminal,
  re-servable, and it **burns no attempt**.
- A spot-check is a **re-issue of an accepted shard to a different machine**,
  judged by the **type's own comparator**, and excluded from job completion and
  from `attempt_group` comparison.
- Spot-check failure is the largest single reputation penalty and forces
  `probation` on the first occurrence.
- Reputation weighting multiplies `steps` before the existing cap-and-normalise
  loop, and is exactly inert under a uniform reputation vector.

## Open / other docs

- **`formula_version = 1` for share** — machine-weighted seconds. Gated on
  `machine_weight` leaving its own interim `formula_version = 0` (`09` §3.3).
- **Aging** — a second fairness axis (waiting, not hogging). Deliberately not
  stacked on the same sort term (§2.4).
- **The automatic-preemption constants** — `AUTOPREEMPT_STARVE_MIN`,
  `AUTOPREEMPT_RANK_MARGIN`, and the one-per-sweep rate are reasoned, not
  measured (§4.6). They want a fleet with contention.
- **Spot-checks for a stochastic type** — needs a distributional test rather than
  a comparator, and no such thing is designed. `collab_lora_finetune` is covered
  by the norm gate and the divergence metric instead.
- **A hard per-shard failure path** (dead-letter, job-fail) — still open, still
  named in `close.py`. Preemption deliberately does not consume its budget (§4.4)
  but does not supply it either.
- **Whether reputation weighting helps** — an empirical question for a fleet with
  diverged reputations (§6.2).
