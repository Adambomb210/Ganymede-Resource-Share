# Ganymede — Multi-GPU Hosts

*Devices, not machines, are the unit of allocation. This document replaces
Decision 4 (`07` §1) and amends `04`'s "one task per machine" scoping.*

A machine with four cards currently registers as one card and runs one task. This
document makes a multi-GPU host a first-class thing **fleet-wide** rather than a
special case for one server: a worker registers a *device inventory*, a lease holds a
*set* of devices, and a host may hold several leases at once so long as their device
sets do not overlap.

A single-GPU contributor becomes a host with one device. Nothing about their
experience changes, and that is the test this design has to pass — the point is not to
support one donated machine, it is that the donated machine is a node like any other.

---

## 1. What Decision 4 was protecting, and what replaces it

`07` §1 froze: **a machine holds at most one `leased` task across all jobs.** It was
not arbitrary. It bought three things:

1. **A meaningful throughput number.** `plan_budget` sizes `local_steps` from
   `calibration.json`'s steps/min, measured on an idle card. `04` §3 states the
   consequence plainly: *"under contention that number is fiction."*
2. **A simple exclusion rule.** One `SELECT ... WHERE worker_id=? AND status='leased'`
   is the whole admission check.
3. **No partitioning machinery.** MPS/MIG stayed out of scope because a lease *was*
   the whole machine.

**The replacement: a device holds at most one task.**

This keeps (1) intact, which is the one that actually mattered. A task holding a whole
exclusive device runs at the rate that device was calibrated at, whether or not a
sibling card next to it is busy. Contention was never about the *machine* — it was
about the *card*. Decision 4 conflated the two because, until now, they were the same
thing.

(2) becomes a ledger rather than a predicate; §3 and §4 are that ledger, and the
uniqueness property is enforced by the database rather than by application logic. (3)
is unchanged: whole devices only, no partitioning — see §8.2.

**Invariant, frozen:** *every allocated device is held by exactly one non-terminal
task.* Everything in §4 exists to make that true under concurrent claims.

---

## 2. The device inventory

A worker reports `devices: [{index, name, vram_mb, compute_capability, supports,
alloc_max_mb, bench_score}]` alongside its existing flat `ComputeProfile`.

**The flat fields stay, populated from device 0.** They feed the `uuid5` fingerprint
that derives `worker_id` (`app.register`), and changing them would re-register every
machine in the fleet as a new one, orphaning its reputation, its enrollment and its
accrual history. They also keep every `constraints_json` predicate written before this
document meaning what it meant.

**Enumeration is per-backend and belongs on the `Backend` table** (`worker/probe.py`),
which already carries a per-backend `describe`:

| backend | count | in-process pin | container pin |
|---|---|---|---|
| `cuda` | `torch.cuda.device_count()` | `CUDA_VISIBLE_DEVICES` | `--gpus device=N` |
| `rocm` | same `torch.cuda` API | `HIP_VISIBLE_DEVICES` + `CUDA_VISIBLE_DEVICES` | `--device=/dev/kfd --device=/dev/dri/renderD{128+N} --group-add video` |
| `xpu` | `torch.xpu.device_count()` | `ZE_AFFINITY_MASK` | `--device=/dev/dri/renderD{128+N}` |
| `mps` | always 1 | n/a | none — no container path on Apple (`11` §4) |
| `cpu` | 1, operator-raisable | n/a | `--cpuset-cpus` |

MPS is correct at one device by construction: unified memory, no index, and its OOM is
a SIGKILL rather than a catchable allocation failure.

**CPU slots default to 1** even though a CPU box could run several tasks
(`GANYMEDE_CPU_SLOTS` raises it). `_describe_cpu` reports `vram_mb` as total system RAM,
so raising it is an operator's deliberate act rather than something the probe guesses at.

**A pooled backend's per-device figures are a share, not a copy.** CPU slots and Apple's
unified memory draw on one pool, flagged `shares_memory_pool` on the `Backend` table, and
their `vram_mb` / `alloc_max_mb` are divided by the device count. `constraints.total_vram_gb`
*sums* the per-device figures, so four slots each reporting the whole machine's RAM would
advertise four times the memory that exists, and a submitter predicate like
`total_vram_gb >= 100` would match a 32 GB box. Dividing is also the honest per-slot
budget: a slot can only use its share without starving its siblings. Discrete-memory
backends (CUDA, ROCm, XPU) are never divided — each card's VRAM is genuinely its own.

The **flat** `vram_mb` is deliberately *not* divided. It describes the machine, and it is
a `uuid5` fingerprint input: dividing it would re-register every existing CPU worker as a
new machine. The single-slot default therefore leaves device 0 exactly equal to the flat
field, which is the backward-compatibility property the flat-field freeze rests on.

**Probe devices sequentially.** `allocation_ceiling_mb` doubles until OOM; four
concurrent probes each measure a fraction of the real ceiling, and that number is
preferred over `vram_mb` by both `budget.is_eligible` and `constraints._vram_mb`. A
concurrent probe therefore *understates* capacity and silently makes the machine
ineligible for work it could do.

**A backend with no known pinning form refuses to launch** rather than falling back to
"all devices". Handing a container every card on the box while the coordinator believes
it holds one is exactly the silent double-booking this ledger exists to prevent, and
fail-closed is the convention everywhere else in the claim path.

**That refusal is for named devices this host cannot honour — not for a coordinator
that named none.** The two look alike and want opposite answers. `devices.allocate`
raises on a non-positive count, so a coordinator that allocates always names at least
one device; an empty list therefore means the coordinator predates this document, and
*that* coordinator is still enforcing one lease per machine — which makes `--gpus all`
both safe and exactly what the host used to get. Emitting nothing instead would run a
submitter's GPU job on CPU: it "works", far slower, and nobody finds out. So on a
discrete backend (`cuda`, `rocm`, `xpu`) an empty device list falls back to every
device with a logged warning, while `cpu` and `mps` correctly get nothing — there is no
accelerator to hand over in the first place. A *new* coordinator reaching that branch is
a bug on its side, and `invariants.py`'s `lease_without_device` catches it there rather
than this branch papering over it.

**`GANYMEDE_JOB_GPUS` now defaults to unset, not `"all"`.** The old default was only
safe under Decision 4's one-task-per-machine invariant, which this document replaces. An
operator who leaves it unset gets the per-lease pin; setting it explicitly still wins
outright, with a warning when it overrides a real allocation. Step 7 applies this
fail-closed rule to the in-process pin too (`worker/loop.py`'s `_pin_env`), not only the
container one this paragraph was written about — the reasoning is about the backend
having no known pin at all, not about which launch mechanism is asking.

**Deviations recorded during step 7 (the worker-side supervisor), not assumed by
anything above:**

- **A child is spawned per *lease*, never a long-lived child per device slot.**
  `devices.allocate` hands out "the lowest count indices from the free set" — there is
  no way for a worker to ask for a specific index on a later claim — and a process's
  `CUDA_VISIBLE_DEVICES` (or the ROCm/XPU equivalent) cannot be changed once that
  process's CUDA-family context has initialized against one visible set. A persistent
  per-slot child could therefore be handed a *different* physical device on its second
  lease and have no way to follow it. Spawning fresh every lease sidesteps that
  entirely: each child reads its own lease's `devices` field and pins exactly that,
  every time. The cost is that the trainer's process-lifetime model cache
  (`trainer/modelcache.py`) no longer amortizes a base-model load across several leases
  on a multi-device host the way it does on a single-device one — a known, accepted
  regression on that path only, not fixed by this step.
- **A single-device worker (`Worker._slot_count() <= 1`) never spawns a child process at
  all.** `Worker.run` dispatches to `_run_single`, `run`'s pre-step-7 body verbatim, so
  the fleet's overwhelming majority — one donated card, the common case §1 opens on —
  keeps the model-cache reuse above and every other single-device behavior byte-for-byte
  unchanged. One direct consequence worth stating plainly: an unhandled exception in a
  single-device worker's training loop still exits that worker process, exactly as it
  does today. The "one lease's crash does not take the box down" property this step adds
  is therefore a property of a worker reporting more than one device, not a fleet-wide
  invariant — see the step's own report for the reasoning.
- **A CPU host with `GANYMEDE_CPU_SLOTS` raised above 1 has no in-process pin token at
  all** (the table above already says "n/a" on that row) — this step makes the
  consequence concrete rather than leaving it implicit: such a host's children run
  unpinned, contending for the same cores rather than isolated onto their own. Not
  addressed here; a future step could shell out to an OS-level affinity mechanism
  (`taskset`/`sched_setaffinity`, Windows `SetProcessAffinityMask`), but nothing in the
  claim protocol or `probe.py`'s `_cpu_devices` currently reports which cores a "slot"
  should get, so there is nothing yet to pin *to*.

**Deviations recorded during step 8 (backend-aware container device pinning,
`worker/sandbox.py`'s `device_argv` and the `contained_batch` launch path), not
assumed by anything above:**

- **The `cpu` row's container pin is `--cpuset-cpus` in the table above, and step 8
  does not implement it.** The reason is the same one step 7 already recorded for the
  in-process pin, two paragraphs up: `probe._cpu_devices` hands out `range(slots)` as
  ordinal labels, not physical core ids, and nothing in the claim protocol or the probe
  reports which core a "slot" should get. Fabricating `--cpuset-cpus` from the index
  would pin from a mapping that does not exist — two sibling containers could land on
  the same literal core while each believed itself isolated, which is worse than the
  accepted gap it would replace. `device_argv("cpu", ...)` emits no flags, matching
  `_pin_env`'s treatment of the same row. **The table's own value for this row should
  be read as aspirational, not implemented, until something reports a real core
  mapping to pin to.**
- **`mps`'s container pin is a hard refusal**, not the "none" the table's word choice
  undersells the consequence of. `device_argv("mps", indices)` returns `None` — the
  same fail-closed signal an unrecognised backend gets — whenever `indices` is
  non-empty, which for `mps` (always exactly one device) is effectively always. This is
  deliberately *not* the same answer `_pin_env` gives the in-process pin for `mps`
  (`{}`, a no-op): a process with no `CUDA_VISIBLE_DEVICES`-equivalent set still sees
  the one GPU there is (torch's MPS backend has no visible-device concept to restrict),
  but a *container* on Docker Desktop for Mac cannot reach that GPU through any flag at
  all — there is no Metal passthrough into the Linux VM the container runs in. Silently
  starting the container anyway would mean a submitter's job runs on CPU while
  everyone, including the coordinator's own image-eligibility check, believes it got a
  GPU. Refusing is the honest answer; it does mean a host that reports `mps` cannot run
  `contained_batch` at all today, which §4's `no_container_runtime` gate does not catch
  because Docker Desktop itself may be genuinely present and working on such a host.
- **`--gpus device=N,M,...` needs literal embedded double quotes, not shell quoting.**
  Docker's own `--gpus` value parser is a CSV reader over the `count=` / `capabilities=`
  / `driver=` / `device=` grammar, so an unquoted multi-index value
  (`--gpus device=1,2,3`) is ambiguous with that grammar and silently drops everything
  after the first comma rather than erroring. The fix — confirmed against Docker's own
  documented example at `docs.docker.com/engine/containers/gpu/`
  (`--gpus '"device=0,2"'`, where the outer quotes are the shell's and the inner ones
  are literal characters in the flag's value) — is emitted as the argv element
  `'"device=1,2,3"'` here, i.e. the string itself carries the double-quote characters,
  since `subprocess.run` bypasses the shell entirely and there is nothing else that
  would add them.
- **The `renderD{128+N}` formula (`rocm` and `xpu` rows) assumes device index `N` is
  also DRM render-node order**, i.e. that node 128 is whatever `torch` calls device 0,
  node 129 is device 1, and so on. On a box with integrated graphics alongside a
  discrete card, the integrated GPU commonly claims the lowest render node, which would
  make index 0 pin the *wrong* device. Nothing in `probe.py` or this table correlates a
  torch device index with a DRM render node number to confirm or refute this, and
  neither `rocm` nor `xpu` hardware was available to test against. Implemented exactly
  as the table specifies; flagged here as unverified rather than silently trusted.
- **`--group-add video` (the `rocm` row) may be the wrong group on a modern kernel.**
  Recent distros commonly gate `/dev/dri/renderD*` behind a `render` group rather than
  `video`; with `--user 1000:1000` forced (§4.6), the wrong group would leave the node
  present but unreadable by the job, a failure that looks like a driver problem rather
  than a permissions one. Implemented as the table specifies, for the same
  cannot-verify-without-hardware reason as the render-node formula above.
- **`GANYMEDE_JOB_GPUS`'s default changed from `"all"` to unset-means-defer.** Before
  this step, `SandboxConfig.gpus` defaulted to `"all"`, which was safe only because
  Decision 4's one-task-per-machine invariant meant the one task on a box already owned
  every card on it. That invariant is exactly what this document replaces, so an
  unconditional `"all"` default now would hand a container every card on a multi-lease
  host, including ones a sibling lease holds — the double-booking §1 and §4 exist to
  prevent, reintroduced at the container launch after being closed everywhere else. An
  operator who leaves `GANYMEDE_JOB_GPUS` unset now gets the lease's own `devices`,
  pinned through `device_argv`, instead. An operator who sets it explicitly still gets
  it verbatim — `run_argv` logs a warning when that override and a non-empty lease
  allocation are both present, since trusting an explicit operator setting over the
  ledger is deliberate but should not be silent.

**A device index is worker-local, and both pins compose through the worker's own
visibility restriction (review-added).** `probe.run_probe` enumerates
`range(torch.cuda.device_count())` — whatever *this worker process* can see. So on a box
where the worker was launched with `CUDA_VISIBLE_DEVICES=4,5,6,7` (an operator lending
Ganymede half an 8-card machine), the inventory reports four devices as indices **0-3**,
and those are the numbers the coordinator allocates and returns in a lease.

Neither pin inherits that restriction. `CUDA_VISIBLE_DEVICES` does not nest — a process
that sets it has the value read against the box's *full physical* device list — and the
NVIDIA container runtime addresses cards by absolute physical index or UUID, ignoring the
worker's own setting entirely. Written raw, a lease holding local index `2` would send
both the in-process child and the container to physical card **2**, a card deliberately
withheld from Ganymede, rather than card 6. Nothing errors; the lease simply runs on
hardware it does not hold, contending with whatever else is there.

So the ambient list, where one is set, is the translation table: local index `i` means
its `i`-th entry. `sandbox.compose_visible` is that translation, shared by
`loop._pin_env` and `device_argv` so the two cannot drift, and positional rather than
numeric so a UUID-valued list composes unchanged (`--gpus device=` accepts a UUID
wherever it accepts an index). A local index with no entry in the ambient list is a
**refusal**, the same answer both pins already give an unpinnable backend — guessing is
how a lease lands on a card it does not hold. With nothing set, every local index is
already physical and both pins emit exactly their previous values, byte for byte, which
is every ordinary deployment.

The ROCm and XPU container branches are **not** composed this way, and that is a named
gap: they need a DRM render-node number, an ambient list may hold UUIDs no `renderD`
formula can consume, and the `128 + N` formula is itself still unverified on real
hardware (above). Compounding two unverified mappings would raise the risk rather than
lower it.

---

## 3. Schema (migration 009)

```sql
CREATE TABLE worker_devices (
    worker_id          TEXT    NOT NULL REFERENCES workers(id),
    device_index       INTEGER NOT NULL,
    device_name        TEXT    NOT NULL,
    vram_mb            INTEGER NOT NULL,
    compute_capability TEXT,
    supports_json      TEXT    NOT NULL DEFAULT '[]',
    alloc_max_mb       INTEGER,
    bench_score        REAL,
    retired_at         TEXT,
    PRIMARY KEY (worker_id, device_index)
);

CREATE TABLE task_devices (
    id             INTEGER PRIMARY KEY,   -- surrogate; see below
    task_id        TEXT    NOT NULL REFERENCES tasks(id),
    worker_id      TEXT    NOT NULL REFERENCES workers(id),
    device_index   INTEGER NOT NULL,
    allocated_at   TEXT    NOT NULL,
    released_at    TEXT,
    release_reason TEXT
);
CREATE UNIQUE INDEX idx_task_devices_busy
    ON task_devices(worker_id, device_index) WHERE released_at IS NULL;
CREATE INDEX idx_task_devices_history ON task_devices(worker_id, allocated_at);
-- `release` looks a task's live rows up by task id on every terminal path;
-- neither index above serves that. Partial on the same predicate as
-- idx_task_devices_busy, so it stays the size of the held set rather than of
-- the append-only history behind it (§4, §8.3).
CREATE INDEX idx_task_devices_live
    ON task_devices(task_id) WHERE released_at IS NULL;

CREATE TABLE device_reservations (
    worker_id    TEXT    NOT NULL REFERENCES workers(id),
    device_index INTEGER NOT NULL,
    job_id       TEXT    NOT NULL REFERENCES jobs(id),
    reserved_at  TEXT    NOT NULL,
    expires_at   TEXT    NOT NULL,
    PRIMARY KEY (worker_id, device_index)
);

ALTER TABLE jobs  ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE tasks ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE submitter_quotas ADD COLUMN max_concurrent_gpus INTEGER;
```

**`task_devices` is append-only.** A release stamps `released_at`; rows are never
deleted. The partial unique index is what makes that work: a released row stops
occupying the card while staying on the record. One table carries both the invariant
and the history.

**The key is a surrogate rowid, not `(task_id, device_index)`.** Task ids are recycled:
`_claim_static_task` re-leases the *same* `tasks` row after an expiry, an abandon or a
preemption rather than minting a new id. A task that lands on the same device twice
therefore produces two legitimate rows for one (task, device) pair — one released, one
live — and a composite key over those columns would reject the second allocation.
History has no natural key here. The *invariant* lives entirely in the partial unique
index, which is the only uniqueness this table should enforce.

That history is not decoration. It is the per-device utilisation record `09`'s
provisioned-hours accounting needs once a contributor can provision four cards instead
of one, and it is the only way to answer "why is card 2 dark" without reading logs —
the row that never got its `released_at` names the task that failed to release.

**`max_concurrent_gpus`** is what actually expresses "three cards for project B" once a
job can be wider than one card. `max_concurrent_tasks` counts tasks, and with
multi-GPU jobs those stop being the same number.

**Two data steps inside the migration**, both required for correctness rather than
tidiness: synthesize one `worker_devices` row per existing worker from its flat
profile, and backfill `task_devices` for every currently-`leased` task at
`device_index = 0`, taking `allocated_at` from the existing `tasks.leased_at` so the
history starts honest rather than stamped with the migration's own clock. A migration
that leaves live leases unaccounted lets the very next claim double-book.

---

## 4. The allocation ledger

`coordinator/devices.py`, sitting beside the claim walk the way `constraints.py` and
`fairness.py` do.

- `inventory(conn, worker_id)` — live devices, `retired_at IS NULL`.
- `free_devices(conn, worker_id, job_id, now)` — inventory, minus *unreleased*
  `task_devices`, minus **unexpired** `device_reservations` held by **other** jobs.
  It checks `expires_at` itself rather than trusting the sweep to have run: since
  `expire_reservations` only fires from the claim poll, an unswept reservation left by
  a dead job would otherwise block a device that is genuinely free — the mirror image
  of double-booking, and the same dark-card bug.
- `allocate(conn, worker_id, task_id, count, free)` — inserts inside the caller's
  existing write transaction; `None` if the unique index rejects (a lost race).
- `release(conn, task_id, reason)` — stamps `released_at` / `release_reason` where
  `released_at IS NULL`. Idempotent, never destructive.
- `device_history(conn, worker_id, since)` — the operator and ledger read path.
- `reserve(...)` / `expire_reservations(conn)` — §6.

**Every query against `task_devices` in the allocation path carries
`released_at IS NULL`.** Forgetting it on one of them is the single most likely way to
dark-card a machine, which is why it lives behind `free_devices` rather than being
written out at each call site. The same predicate carries the two partial indexes —
`idx_task_devices_busy` for the invariant, `idx_task_devices_live` for `release`'s
lookup by task id — so both stay the size of the held set rather than of the history.

**`allocate` raises on `count <= 0`** rather than returning `None`: a non-positive
device count is a caller bug, not a lost race, and the two must not be answered the
same way. The claim walk must therefore validate `gpu_count` **before** reaching the
allocator — an exception raised inside the walk is a `break`, and §5.2 requires every
refusal there to be a `continue`.

**`expire_reservations` is called from the claim path**, beside the existing
`rounds.expire_leases`. Deliberately *not* from `scripts/ledger.py`: that script is the
only caller of `recompute_shares` and `autopreempt`, it has no console entry point in
`pyproject.toml` and no timer in `packaging/`, and consequently neither of those has
ever run in a deployment. A sweep that depends on an installer nobody installed is how
a reservation becomes a permanently-dark card. The claim poll is already the
sweep-on-poll seam, and a stale reservation only matters when someone is claiming.

---

## 5. The claim path

1. **The held-lease pre-check becomes a multi-lease reconcile.** `ClaimRequest` gains
   `active_task_ids`; the coordinator re-serves any leased task for this worker *not*
   in that list — which is the crash-recovery case the current re-serve exists for —
   and otherwise falls through to the walk instead of returning. An absent field means
   an empty list, which reproduces today's behaviour for a v1 worker exactly.
2. **A capacity gate**, immediately after the constraint gate, following its discipline
   exactly: a recorded `REFUSED` verdict and a `continue`, **never a `break`**. The walk
   reaching a lower-rank job this host can still fit *is* the capability backfill
   (`07` §1, Decision 10). Refusal reason: `insufficient_free_devices`.
3. **Allocation happens inside the transaction that creates the lease**, not around it.
   Both claim paths already open `immediate(conn)`; the allocation goes inside that same
   block. The unique index means two concurrent claims from one worker cannot
   double-book even if the free-set computation raced.
4. `_task_payload` gains `devices: [int]`.
5. **Release on every terminal path**, each with its own `release_reason`. There are
   **six**, not the five an earlier draft of this section listed: submit, abandon, the
   three `expire_leases` outcomes (expired, cancelled, preempted), and — the one that
   is easy to miss — `reduce.py`'s round close, which expires any straggler still
   `leased` when a round closes early on step count. That last one is the *ordinary*
   case for any multi-contributor round, not an edge case, and `expire_leases` never
   reaches it: the straggler's lease is nowhere near its TTL, because the round ended
   early rather than late. Its reason is `round_closed`, distinct from `expired`, so
   the ledger can tell "reclaimed because the work finished" from "reclaimed because
   nobody heartbeated".
6. **Reconcile on register — and on enrollment.** A worker that comes back reporting
   fewer devices has its allocations for the missing ones released with reason
   `reconciled`. Note there are **two** paths that create a `workers` row:
   `POST /v1/workers/register` and `POST /v1/machines/claim-enrollment`. Both must
   reconcile. A machine that enrolls without one would hold zero `worker_devices`
   rows, hence zero free devices, hence a refusal on every job forever — it enrolls
   successfully and then silently never works.
7. **`free_devices` is a required argument with no default** on every claim entry
   point. A caller that omits it is a bug at that call site, not a machine with no
   cards, and the two must not look alike: defaulting to an empty list turns a
   forgotten argument into a permanent silent refusal, where the worker gets 204
   forever and nothing anywhere errors. The resume paths pass `[]` deliberately — a
   resume hands back a lease whose devices are already allocated, and an empty free set
   is also the right safety answer if such a call ever fell through to the minting
   path.

**Two known gaps, both live:**

- **`spotcheck.maybe_issue` does not allocate.** It mints and leases a probe task
  outside the allocating claim paths, so the probe's device reads as free while the
  probe runs and the next claim can double-book it. Harmless only because
  `spotcheck_rate` defaults to `0.0`. **This is a blocker on turning spot-checks on**,
  not a cosmetic gap.
- The `invariants.py` check formerly named `_one_lease_per_worker_per_round` has been
  retired: under §7 a multi-GPU host holding two leases in one round is correct.
  What replaces it is `_every_lease_holds_a_device`, which hunts the state the unique
  index cannot see — a task leased without an allocation, whose card reads as free
  while it is in use. That check is what makes the spot-check gap above fail loudly
  rather than quietly.

**Constraints** gain `gpu_count` and `total_vram_gb` resolvers. Both read from the
**profile**, never from `worker_devices`: `check_constraints` is pure by contract —
reads only `(machine_id, profile)`, no DB, no write lock — and §2 puts `devices` on the
profile precisely so that stays true. The existing fail-closed-on-missing rule applies
for every operator, `!=` and `not_in` included; a pre-009 worker synthesizes one device,
so `gpu_count` resolves to 1 rather than to nothing.

---

## 6. Reservation with backfill

A 4-card job on a busy 4-card box can starve forever: single-card jobs keep taking each
device as it frees. Head-of-queue blocking would fix it and is rejected — it idles cards
and contradicts the capability-backfill principle the whole scheduler is built on.

Instead: a blocked wide job accumulates a reservation on devices as they free, with a
TTL so a dead job cannot hold cards. A smaller job may still take a reserved device if
its `tasks.max_runtime_sec` fits before `expires_at` — that column already exists, so
this needs no new estimate plumbing.

**Who may reserve has to be specified exactly, or four wide jobs deadlock the box
holding one card each.** The walk has no notion of "head" — its list is
`_selectable_jobs` sorted by `effective_rank`. So:

> **Only the first job in walk order refused for `insufficient_free_devices` may
> reserve, at most one job may hold reservations on a given worker at a time, and a
> job may only reserve on a worker whose own live inventory could ever satisfy it.**

The `device_reservations` primary key gives per-device uniqueness but not that second
property; `devices.reserve` enforces it under its own write lock.

**Which half actually prevents the deadlock**, since an earlier draft of this section
got the emphasis wrong: it is the *second* rule, enforced in the ledger. `reserve`
refuses outright when another job already holds a reservation on that worker, so four
wide jobs cannot end up holding one card each no matter what the walk does. The
walk-order rule is a cost optimization on top of that — it stops every lower-rank wide
job on a busy poll from opening a write transaction only to be refused.

**Only a job with `gpu_count > 1` may reserve.** A single-card job refused here has
zero free devices, and the very next device anything releases satisfies it immediately
through ordinary backfill — it never needs to survive across polls. Reserving on its
behalf would hold a card no wider job needs, for no reason. Reservation is for the case
ordinary backfill structurally cannot serve: a job needing several devices at once,
where single-card jobs taking each one as it frees *is* the starvation.

**Only a job this worker could ever satisfy may reserve — `gpu_count <= len(inventory)`
for *this* worker, not the fleet's widest.** Added by review after the rest of this
section shipped; the original two conditions were written against the deadlock case
("four wide jobs holding one card each") and never contemplated a job that cannot fit
the worker at all. `create_job` only checks a new job against `max_inventory_width`, the
fleet-wide maximum, so a `gpu_count = 4` job is admitted while any 4-card host exists —
and then meets every narrower host for the rest of its life, refused
`insufficient_free_devices` there structurally, forever. Since reservation admits one
holder per worker and one reserver per poll, that job would take the 3-card box's
reservation slot on *every* poll, and the `gpu_count = 3` job behind it in walk order —
which fits exactly — would never accumulate anything. That is this section's own
starvation, pointed at the wrong job. Comparing against the worker's own live inventory
is the whole fix: a job that cannot run here never holds here.

**What gets reserved is the plain free set, never the backfill-widened one.** A device
this job can see only because some other short job might finish in time is not a device
it may claim as reserved for itself.

**A reservation ends when the job claims.** The accumulation it existed for is over the
moment the job fits, and what it won is now `task_devices` rows that exclude everyone
on their own. Left behind, the rows would keep excluding other jobs for the rest of the
TTL *after* the winning task released its cards, while the reserver — which sees through
its own reservation — went unaffected. A wide job needing another task's worth of
devices re-accumulates from scratch.

**TTL: `GANYMEDE_DEVICE_RESERVATION_TTL_SEC`, default 300s.** A blocked job re-reserves
on every poll, so the TTL only has to survive one poll gap rather than the whole
accumulation window — and staying well under `lease_duration_sec` (900s) means a dead
reserver dark-cards a device for minutes, not for most of a lease.

### 6.1 Re-registration, and what a vanished card must not keep holding

`devices.reconcile_inventory` runs inside `register`'s existing write transaction and
brings `worker_devices` in line with what the worker just reported. A device index the
worker reported before and does not report now — a box that reboots with a dead card,
or comes back with one card lent to something else — is **retired**, and everything
still standing against that index is dropped with it:

1. `worker_devices.retired_at` is stamped, so `inventory` and therefore `free_devices`
   stop offering it;
2. any live `task_devices` row on it is released with reason `reconciled`, so the card
   is not permanently allocated to a task that will never submit or expire against it;
3. **any `device_reservations` row on it is deleted.**

(3) is not tidiness. `reserve`'s holder check is *worker-wide* — "at most one job may
hold reservations on a given worker at a time" — and it purges only rows past
`expires_at`. An unexpired reservation orphaned on a retired index therefore keeps its
job reading as the holder of the entire worker, so every *other* job is refused a
reservation there, for up to the reservation TTL, on the strength of a claim over a card
that no longer exists. `has_other_reservations` answers `True` for that window too, so
the walk also pays for a backfill peek that cannot find a target. It self-heals at the
TTL rather than dark-carding permanently, which is exactly what made it easy to miss.

---

## 7. Collaborative training on a multi-GPU host

A multi-GPU box takes as many `collab_lora_finetune` leases as it has free devices, and
its submissions carry full weight. **The dominance cap is unchanged.**

`aggregate.dense_weights` receives one entry per submission and clamps each at
`cap * median`. Four leases produce four *normal-sized* entries — each budgeted from
per-card throughput, each on a whole exclusive device — so nothing trips, and nothing
needs to. Three reasons to leave it alone:

- Four leases are four **independent** local runs over four different bucket
  assignments (`_pick_buckets`, least-trained-first). That is four independent local
  optima, which is exactly the diversity the outer average is built on — not one node
  dominating, but four contributions that share a power supply.
- Weighting by steps is the correct thing. The box did 4x the steps on 4x the data.
  Clamping it makes a donated machine deliver a quarter of what it is.
- Grouping the cap by `contributor_id` would be poor Sybil defense anyway: identity is
  a placeholder local user table (`08`, Decision 5), so an attacker registers four
  accounts while an honest donor eats the tax. That is the worst trade available.

`distinct_contributors` already keys on the contributor, so a four-lease box correctly
reports a cohort of one — the diversity signal was never at risk.

**What replaces a cap change is visibility.** The genuine risk is cohort *composition*:
a round where one box is four of five members is mostly one machine's view of the data.
Record `max_contributor_share` on the round beside `distinct_contributors`, surface it,
and alert on it. Two levers ship **inert** so the decision can be revisited with
evidence rather than re-litigated from first principles:

- `dominance_group_by = submission | contributor`, default `submission`.
- `max_leases_per_worker` per job type, default **unlimited**.

---

## 8. Deferred, with the reasoning recorded

### 8.1 In-process multi-device training

One 4-card `collab_lora_finetune` task. Deferred by decision, not by oversight.

**And enforced, as of review, rather than merely unbuilt.** `CollabLoraFinetune` declares
`max_gpu_count = 1` and `create_job` refuses a wider job with a 422 — a third `gpu_count`
check beside the two in §9, read as an optional class attribute exactly the way
`requires_image` already is, so a type that says nothing is bounded only by the fleet.
Without it the deferral was silent: a `gpu_count = 4` collab job passed submission
(the fleet *is* that wide), `claim_task` allocated four cards to one task, and
`trainer.model.pick_device` then trained on one of them — three cards allocated, idle,
and correctly reported busy by the ledger for the whole lease. Refusing at submission
rather than at claim follows the same reasoning as the fleet-width check: the submitter
gets a reason now instead of a row that polls forever. §7's several one-card leases
remain the supported way to fill a wide box with this job type.

Scope it to **model parallelism**, not data parallelism: for LoRA the memory cost is the
*frozen base*, since the adapter is tiny and `ADAPTER_DTYPE` is fp32. DDP gives 4x speed
but still needs the base to fit on one card — the exact constraint this would exist to
break.

**Set the expectation plainly: `device_map="auto"` is naive pipeline parallelism, one
card active at a time.** A 4-card task is *slower per step* than a 1-card task on a model
that fits. The win is fit, not speed.

The seam is already the right shape — `load_base` passes `device_map={"": device.type}`
on the quantized path today. What ripples outward, and why this is not small:

- **Throughput keys must carry the device count.** `calibration.json`'s `throughput` and
  `fits` maps key on `device_name` alone — `ganymede/device.py` calls it "a join key in
  four places" — and a 4-card model-parallel rate is a different number for the same card
  name. A key mismatch has no symptom beyond budgets that quietly never improve.
- **Eligibility becomes aggregate.** `budget.is_eligible`'s `min_vram_mb` compares
  against one device; `runs.requires_json` would need `min_total_vram_mb`. Without it a
  1-card host claims a 4-card run and OOMs on load.
- **Adapter key names are a silent-failure surface.** `trainer/model.py` requires
  byte-identical key names and acceptance gate 2 checks every submission against them. A
  sharded load is exactly the sort of change that perturbs parameter naming, and the
  failure mode is every submission on every round rejected.

**And a consequence worth stating before anyone builds it:** a run whose base needs four
cards can only be claimed by four-card hosts. In a fleet of single-GPU contributors that
is a cohort of one, `distinct_contributors = 1` every round, and the round machinery
becomes ceremony around what is really single-node training. That is an argument for
giving big-model training its own job type rather than borrowing the collaborative one.

**Trigger condition:** the single-card nf4 fit ladder. If nf4 on one card already covers
the models that matter, four independent leases beat one 4-card task on every axis and
this never needs building. Measure before committing.

### 8.2 Fractional / shared devices

Whole devices only. `modelcache.DEFAULT_CAPACITY = 1` is per *process*, so two training
tasks on one 8 GB card means two resident base copies — an OOM, not a config. Plausible
only for small inference, and `04` §3's contended-throughput argument applies in full the
moment two tasks share a card.

The schema is shaped so this is an increment rather than a rewrite: the unit would become
*VRAM on a device* rather than the device, with whole-device as the default reservation.

### 8.3 `task_devices` retention

The table grows one row per (task, device). Fold a prune into the existing `11` §1.2
retention work rather than building a second mechanism.

---

## 9. Step 9: submission surface, operator visibility, deploy

Everything through §8 makes a multi-GPU host schedulable. Nothing before this step lets
a submitter *ask* for more than one card, lets an operator *cap* a submitter's cards, or
lets anyone *see* which card holds which task — the feature worked and was invisible.
This section is that surface, plus the deviations found wiring it.

**`gpu_count` on `POST /v1/jobs`.** `JobCreateRequest.gpu_count` defaults to `1`, so a
job submitted with no opinion is byte-for-byte today's job. `create_job` — not the
endpoint — validates it, because `create_job` is the one function both `POST /v1/jobs`
and the web UI's submission form call (its own docstring says so); a check placed in the
endpoint alone would leave the form unguarded. Two checks: `gpu_count < 1` is a caller
bug regardless of what hardware exists, checked unconditionally. `gpu_count` wider than
the fleet's widest live inventory (`devices.max_inventory_width`) is checked **only when
that width is nonzero** — a fresh coordinator, or one whose one donated box has not yet
registered, reports a widest inventory of zero, and an unconditional check would 422
every job, including the default `gpu_count = 1`, on day one. This is exactly the
"CAREFUL" §5 already flags for the capacity gate, applied one step earlier, at
submission rather than at claim.

**`max_concurrent_gpus` on `POST /v1/admin/submitters/{id}/quota`**, enforced in
`fairness.quota_refusal` beside `max_concurrent_tasks`, same shape: a `continue` with a
recorded verdict in the claim walk, `None` on no row (uncapped, the default for every
submitter that already exists). Counted from `task_devices` — live, unreleased rows
joined out to the owner through `tasks`/`jobs` — rather than from
`SUM(tasks.gpu_count) WHERE status = 'leased'`, because the two sources can already
disagree: `spotcheck.maybe_issue` (§4, §5) mints a leased probe task without allocating
through the ledger, so a `tasks`-only count could overstate what an owner holds by
however many probes are outstanding. `task_devices` is the one source that cannot
disagree with `devices.allocate` itself, since the partial unique index is what
`allocate` enforces the invariant against.

**Known limitation, named rather than fixed here:** the check is a coarse pre-check —
"is this owner already at or over the cap" — not "would granting this job's `gpu_count`
push them over it." That distinction did not exist before this step: every prior claim
could only ever add one task (one device) at a time, so "at the cap" and "about to
exceed the cap" were the same question. A wide job's single claim can now allocate
several devices in one transaction, so a submitter sitting one device under
`max_concurrent_gpus` can still be granted a job that lands several over it in that one
step — bounded by how many devices any one machine can ever offer one task (never
unbounded), but real. Tightening this to
`gpus_used(...) + job["gpu_count"] > cap` is possible but was not done in this step: the
claim walk currently memoizes `quota_refusal` once per owner per poll (`quota_seen`),
which is correct only because the answer does not depend on which job is being
evaluated; making it depend on `job["gpu_count"]` means jobs from the same owner with
different widths can get different verdicts in the same poll, which needs the walk's
memoization to change shape, not just this function's body. Left for the operator to
size caps with this in mind until it is.

**Per-device operator visibility.** `GET /v1/fleet` gains a `devices` array per worker —
`{index, name, vram_mb, task_id}`, `task_id` null for a free card — built from one query
across `task_devices` (`released_at IS NULL`) rather than one query per worker, joined in
Python against `devices.inventory`. `GET /v1/admin/queue` gains `gpu_count` (already a
column on `jobs`, never previously surfaced) and `leased_devices`, a per-job count of
live `task_devices` rows, following the same correlated-subquery shape the existing
`leased_tasks` count already uses. `scripts/status.py` gains `fleet_devices()` and a
`devices:` section in both the plain-text render and `--json`, grouped by worker with a
busy/free count per card — the CLI-side mirror of `/v1/fleet`'s new field, for the
operator who works from a terminal rather than the read-only `/ui/` pages. Neither `/ui/`
page was touched: both read through `webui.py`'s own `readmodel` queries, not these two
endpoints, so adding fields here changes nothing about what the browser dashboard shows
or how it is built.

**`_gpu_busy` becomes per-device (`host/idle.py`).** Before this step, one non-Ganymede
compute process anywhere on the box read as "the machine is busy" — correct when a lease
was the whole machine (`07` §1, the decision §1 of this document replaces), wrong once a
lease is a device: on a real 4-card donated box, a contributor's own game on card 2 would
silently park cards 0, 1 and 3 idle forever, because the worker never gets the chance to
start and report an inventory the coordinator could route around the busy one. The
rewrite asks `nvidia-smi` twice — `--query-gpu=index,uuid` for the index/uuid map,
`--query-compute-apps=gpu_uuid,pid,process_name` for who is running where, joined on
uuid — because compute-apps rows carry no physical index of their own. `_gpu_busy`'s
aggregate answer changes meaning on a multi-GPU box: "busy" now means *every* enumerated
card carries a foreign process, not merely one of several. On a single-GPU host the two
readings are identical (one busy card is the only card), so the fleet's overwhelming
majority — one donated card, this document's own opening line — sees no behavioural
change at all.

**A compute process is one nvidia-smi attributes memory to (review-added, and it fixed
a live bug on Windows).** `--query-compute-apps` now asks for `used_memory` as well, and
a row carrying no figure — `[N/A]`, `[Insufficient Permissions]` — is not counted. The
spec's rule has always been about CUDA processes specifically (`01`: *"No non-Ganymede
CUDA process holds the GPU"*), and this is what makes the implementation match it.

On Linux nothing changes: the driver reports real per-process memory, so every genuine
CUDA client keeps its row. On **Windows/WDDM it is the difference between a check that
works and one that can never pass.** Measured on a real RTX 3060 / Windows 11 box:
`--query-compute-apps` returned **forty** rows — `explorer.exe`, the shell, a browser,
Discord, Slack, Steam — because WDDM enumerates every process holding a graphics
context, not just compute clients; and it attributed `[N/A]` memory to *all* of them,
including a genuine `torch` CUDA process started on the same box for the comparison. So
nothing in that output distinguishes the desktop compositor from a training run.

Because `require_gpu_free` defaults to `True`, the old rule meant `_gpu_check` returned
`IdleReport(idle=False, reason="gpu in use: [Insufficient Permissions]")` on any Windows
desktop — permanently. The host agent never started the worker, on the platform most
likely to have an idle gaming GPU to donate, and the reason string named a process
nvidia-smi could not even read. Every existing test passed throughout, because all of
them fed the parser hand-written Linux-shaped CSV; `tests_host/test_host_idle.py` now
carries a fixture captured verbatim from the real machine.

The residual risk is stated plainly: on Windows this gate can no longer prove the card
busy at all, so a contributor's own headless CUDA job is not detected by it. They are
still protected by the pause sentinel, the active window, and `_user_idle_check` — and
anyone actively at the machine is caught by the last of those. That is the trade this
module's own rule already prescribes: *"the check can only ever prove the GPU busy,
never prove it free"*, and treating "I can't check" as "busy" *"would quietly exclude
every one of those from ever contributing, which is exactly backwards."*

**This is a start/stop gate, not a routing decision, and that gap is deliberate rather
than closed here.** `_gpu_busy` answers "should the worker container run at all," never
"which card should it use" — a card it finds busy is not communicated to the worker's
own probe once the worker starts. `worker/probe.py` enumerates every device its backend
reports unconditionally; nothing tells it to omit an index `nvidia-smi` found occupied by
someone else. So the coordinator can still see that card in the worker's inventory and
allocate it to a job, which would then contend with whatever the contributor was already
running on it. Closing this needs a channel from the host agent's tick (which already
runs `nvidia-smi` once) into the worker process it starts — an exclusion list the probe
subtracts from its own enumeration — and nothing in `probe.py`, `agent.py`'s
`worker_env()`, or the claim protocol carries one today. Recorded here rather than built,
matching this document's own convention (§2's CPU-affinity gap, §4's spot-check gap): a
known, named limitation is safer than an unverified fix bolted on under time pressure.

**Tested without an NVIDIA GPU** the same way `test_host_idle.py` already tested the
single-call version: every `nvidia-smi` invocation is a monkeypatched
`subprocess.run` returning canned CSV, dispatched by which flag the argv carries
(`_fake_nvidia_smi` in `tests_host/test_host_idle.py`) rather than by call order, so the
tests are indifferent to which of the two queries runs first.
