"""The per-device allocation ledger (docs/14). A device, not a machine, is the
unit of allocation from here on -- §1 replaces Decision 4 (`07` §1: "a machine
holds at most one leased task") with the tighter invariant this module exists
to make true under concurrent claims:

    **Every allocated device is held by exactly one non-terminal task.**

Sits beside the claim walk the way ``constraints.py`` and ``fairness.py`` do:
``constraints.py`` answers "may this job run here", ``fairness.py`` answers
"whose turn is it", and this module answers "which cards on this box are
actually free right now, and who has which one". Nothing here decides
*whether* to allocate -- that is the claim path (docs/14 §5, a later step) --
only the bookkeeping once the walk has decided.

**Scope note (updated for step 5):** ``app.py``'s claim walk, ``register`` and
every terminal task path (submit, abandon, ``expire_leases``) call
``allocate`` / ``release`` / ``free_devices`` / ``held_devices`` from inside
their existing write transactions, and ``register`` calls
``reconcile_inventory`` to keep ``worker_devices`` in step with what a worker
reports. ``reserve`` / ``expire_reservations`` (§6, reservation-with-backfill)
are wired too, as of this step: the claim walk sweeps expired reservations
beside ``rounds.expire_leases``, reserves on behalf of the first wide job it
refuses for ``insufficient_free_devices`` each poll, and asks
``free_devices`` for a backfill-relaxed set when a smaller job's own refusal
might be rescuable. The walk-order policy (which refused job, if any, gets to
call ``reserve``) and the backfill peek both live in ``app.py`` -- this
module answers "which devices, right now" and "who may hold what," never
"whose turn is it," the same division §4 already draws for allocation.

``task_devices`` is append-only (docs/14 §3): a release stamps ``released_at``
rather than deleting the row, and the partial unique index
``idx_task_devices_busy`` -- unique on ``(worker_id, device_index) WHERE
released_at IS NULL`` -- is what makes a released row stop occupying its slot
while staying on the record. **Every query against ``task_devices`` in the
allocation path carries ``released_at IS NULL``.** Forgetting it on one call
site is the single most likely way to dark-card a machine (docs/14 §4), which
is why ``free_devices`` exists as a function here rather than as inline SQL
wherever a free set is needed.

Task ids are recycled: ``_claim_static_task`` (``app.py``) re-leases the
*same* ``tasks`` row after an expiry, an abandon or a preemption rather than
minting a new id, so one ``(task_id, device_index)`` pair can legitimately
appear more than once in the history -- one released row, one live. That is
why ``task_devices`` keys on a surrogate rowid rather than a composite key,
and why ``release`` below stamps by ``released_at IS NULL``, never by
matching the pair.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from ganymede.coordinator.db import immediate
from ganymede.coordinator.rounds import _iso, utcnow


class AllocationRaced(Exception):
    """Raised by a caller, inside its own ``immediate()`` block, when
    ``allocate`` returns ``None`` (docs/14 §4's "an ordinary, expected
    outcome, not a failure") and that caller has *other* writes earlier in
    the same transaction -- a task row already flipped to ``leased``, bucket
    counters already bumped -- that a plain early ``return`` would leave
    committed. Raising instead of returning is what makes ``immediate()``
    roll the whole block back (``db.immediate`` re-raises through a
    ``ROLLBACK``), so a lost race never leaves a task ``leased`` with no
    device behind it -- a phantom lease nothing would ever release (docs/14
    §5.3).

    ``allocate`` itself never raises this -- it already answers a lost race
    with a plain ``None``, which is correct for a caller with nothing else to
    undo. This exists only for the claim-path callers (docs/14 §5.3:
    ``collab_lora_finetune.claim.claim_task``, ``app._claim_static_task``)
    that mint a task row in the same transaction as the allocation and must
    undo *that* too.
    """


# --------------------------------------------------------------------------
# Inventory and the free set (docs/14 §4)
# --------------------------------------------------------------------------


def inventory(conn: sqlite3.Connection, worker_id: str) -> list[sqlite3.Row]:
    """This worker's live devices -- ``retired_at IS NULL`` -- in index order.

    Nothing sets ``retired_at`` yet (migration 009's own comment says so); the
    filter is here from day one anyway, because a worker that never retires a
    device should not be the only reason this predicate gets written correctly
    later, under time pressure, at the one call site that matters.
    """
    return conn.execute(
        """SELECT * FROM worker_devices
            WHERE worker_id = ? AND retired_at IS NULL
            ORDER BY device_index""",
        (worker_id,),
    ).fetchall()


def max_inventory_width(conn: sqlite3.Connection) -> int:
    """The widest live device inventory any worker in the fleet reports right
    now, or ``0`` if nobody has ever reconciled one.

    The one fact ``app.create_job`` needs to reject a job wider than any host
    could ever run, as a 422 at submission rather than a row that queues
    forever and never moves (docs/14 §5's own "CAREFUL"). ``0`` is not "the
    fleet's widest machine has zero devices" -- ``reconcile_inventory`` always
    writes at least one row for a real worker (§2's "an empty list means the
    coordinator predates this document") -- it is "no worker has ever
    registered," which is the ordinary state of a fresh coordinator and must
    not reject every job on day one. The caller is the one that has to read
    ``0`` that way; this function only reports the fact.
    """
    row = conn.execute(
        """SELECT MAX(n) AS n FROM (
             SELECT COUNT(*) AS n FROM worker_devices
              WHERE retired_at IS NULL GROUP BY worker_id)"""
    ).fetchone()
    return int(row["n"] or 0)


def free_devices(conn: sqlite3.Connection, worker_id: str, job_id: str,
                 now: datetime | None = None,
                 max_runtime_sec: int | None = None) -> list[int]:
    """This worker's live device indices, minus what is unavailable to
    ``job_id`` right now (docs/14 §4): unreleased ``task_devices`` rows (any
    job -- a busy card is busy regardless of who is asking), and
    ``device_reservations`` held by **another** job. A job sees through its
    own reservation -- reserving a card is how a blocked wide job eventually
    gets it, and a reservation that made its own holder unable to see the
    device would defeat the point (docs/14 §6).

    ``now`` gates the reservation subtraction on ``expires_at`` so a lapsed
    reservation nobody has swept yet does not make this job wait behind a
    ghost -- deliberately not trusting ``expire_reservations`` to have run
    first, the same way ``share_fractions`` ages its rollup forward rather
    than trusting the sweep. docs/14 §4 specifies exactly this ("minus
    **unexpired** ``device_reservations`` held by **other** jobs ... it checks
    ``expires_at`` itself rather than trusting the sweep to have run"); it was
    written up as a deviation while the doc still said otherwise, and the doc
    has since been amended, so this is the documented behaviour, not a
    departure from it.

    ``max_runtime_sec`` is backfill (docs/14 §6), and ``None`` -- the
    default -- is the plain rule above with no exception: every call site
    that does not ask for backfill gets exactly today's behaviour, byte for
    byte. Given a value, a device reserved by another job is *not* excluded
    when this job's own deadline -- ``now + max_runtime_sec`` -- falls
    *before* that reservation's ``expires_at``: this job's task will have
    finished and released the device again, on its own, strictly before the
    point the reserving job's own already-granted TTL window says it might
    come to collect it, so the reserving job never waits past what it was
    already promised. A reservation whose ``expires_at`` is at or before this
    job's deadline stays excluded -- this job would still be running (or
    finishing exactly) when the reserving job's window says it may need the
    card back, which is the starvation §6 exists to prevent, so fail-closed
    here means "still excluded," not "still free." The caller (the claim
    walk) is responsible for never passing a job's own guess forward as fact
    once a task is actually selected -- see ``_claim_static_task``'s re-check
    under its own write lock.

    Returns a sorted ``list[int]``, not rows -- this is what feeds ``allocate``
    directly and what ``_task_payload`` will eventually serialise as
    ``devices: [int]`` (docs/14 §5.4).
    """
    now = now or utcnow()
    now_iso = _iso(now)
    live = {row["device_index"] for row in inventory(conn, worker_id)}
    busy = {
        row["device_index"] for row in conn.execute(
            """SELECT device_index FROM task_devices
                WHERE worker_id = ? AND released_at IS NULL""",
            (worker_id,),
        ).fetchall()
    }
    reserved_rows = conn.execute(
        """SELECT device_index, expires_at FROM device_reservations
            WHERE worker_id = ? AND job_id != ? AND expires_at > ?""",
        (worker_id, job_id, now_iso),
    ).fetchall()
    if max_runtime_sec is None:
        reserved_by_others = {row["device_index"] for row in reserved_rows}
    else:
        deadline_iso = _iso(now + timedelta(seconds=max_runtime_sec))
        reserved_by_others = {
            row["device_index"] for row in reserved_rows
            if row["expires_at"] <= deadline_iso
        }
    return sorted(live - busy - reserved_by_others)


def has_other_reservations(conn: sqlite3.Connection, worker_id: str, job_id: str,
                           now: datetime | None = None) -> bool:
    """True if some job other than ``job_id`` holds an unexpired reservation
    on this worker right now.

    The claim walk's cheap pre-check before it bothers peeking at a backfill
    candidate (docs/14 §6): with nothing reserved, backfill has no target and
    the peek buys nothing. This is what keeps a fleet where every job is
    ``gpu_count == 1`` -- where ``device_reservations`` never gains a row in
    the first place, since §6's reservation gate only ever fires for a wide
    job -- from paying for a feature it structurally cannot use: the walk
    calls this, gets ``False`` every time, and never issues the extra queries
    a peek would cost. Not folded into ``free_devices`` itself, which must
    answer "what is free" regardless; this only answers "is it worth asking
    a harder question than that."
    """
    now = now or utcnow()
    row = conn.execute(
        """SELECT 1 FROM device_reservations
            WHERE worker_id = ? AND job_id != ? AND expires_at > ? LIMIT 1""",
        (worker_id, job_id, _iso(now)),
    ).fetchone()
    return row is not None


# --------------------------------------------------------------------------
# Allocation and release (docs/14 §4-§5)
# --------------------------------------------------------------------------


def allocate(conn: sqlite3.Connection, worker_id: str, task_id: str,
            count: int, free: list[int]) -> list[int] | None:
    """Take ``count`` devices from ``free`` for ``task_id``. Returns the
    allocated indices, or ``None`` if it cannot.

    **Runs inside the caller's existing write transaction.** Both claim paths
    already open ``immediate(conn)`` (docs/14 §5.3); this function issues no
    ``BEGIN`` and no ``COMMIT`` of its own, so it composes with that block
    instead of nesting a transaction inside it -- SQLite has no nested
    transactions, and ``immediate()`` is the only lock primitive this codebase
    uses.

    Takes the *lowest* ``count`` indices from ``free``, deterministically.
    Under a raced free-set computation this maximises the chance two
    claimants on the same worker overlap -- which is correct, not wasteful:
    the partial unique index is the arbiter of who actually wins a card, and
    ``None`` is an ordinary, expected outcome for the loser (docs/14 §4),
    not a failure to avoid.

    ``sqlite3.IntegrityError`` from ``idx_task_devices_busy`` means a
    concurrent claim on this worker won a device in ``free`` first. Because
    this function may need several inserts for ``count > 1`` and a
    mid-batch failure must not leave the earlier inserts of *this same
    call* live -- that would hand the task a partial, undersized allocation
    that nothing will ever release on its own -- the rows already inserted
    by this call are deleted by ``id`` before returning ``None``. This is a
    plain ``DELETE ... WHERE id IN (...)`` rather than a ``SAVEPOINT``:
    nothing else in the codebase uses savepoints, ``immediate()`` is the one
    transaction primitive in use, and undoing this function's own inserts by
    id needs nothing more.
    """
    if count <= 0:
        raise ValueError(f"allocate: count must be positive, got {count}")
    if len(free) < count:
        return None

    indices = sorted(free)[:count]
    now = _iso(utcnow())
    inserted_ids: list[int] = []
    for idx in indices:
        try:
            cur = conn.execute(
                """INSERT INTO task_devices
                     (task_id, worker_id, device_index, allocated_at)
                   VALUES (?, ?, ?, ?)""",
                (task_id, worker_id, idx, now),
            )
        except sqlite3.IntegrityError:
            if inserted_ids:
                placeholders = ",".join("?" * len(inserted_ids))
                conn.execute(
                    f"DELETE FROM task_devices WHERE id IN ({placeholders})",
                    inserted_ids,
                )
            return None
        inserted_ids.append(cur.lastrowid)
    return indices


def release(conn: sqlite3.Connection, task_id: str, reason: str) -> int:
    """Stamp ``released_at`` / ``release_reason`` on ``task_id``'s live
    ``task_devices`` rows. Idempotent -- a second call finds nothing with
    ``released_at IS NULL`` and stamps nothing -- and never destructive: the
    row stays, which is the append-only history docs/14 §3 wants.

    Filters by ``released_at IS NULL`` rather than by ``device_index`` or by
    matching some prior allocation, because task ids are recycled (module
    docstring): a task that has been leased, expired and re-leased onto the
    same or a different card owns several ``task_devices`` rows, and only the
    live one may be touched. Stamping by task id alone, without that filter,
    would re-stamp an already-released row from a previous lease of this same
    task and corrupt its history.

    No ``immediate()`` of its own, by design (unlike ``allocate``, this one
    self-contains a single read-free ``UPDATE``, so there is nothing to
    protect between a check and a write). ``isolation_level=None``
    (``db.connect``) means a bare statement commits itself when called
    standalone, and composes cleanly when a caller -- a terminal path that
    already holds ``immediate(conn)`` for the status transition it is making
    at the same time -- calls it from inside that block. Wiring those callers
    is docs/14 §5.5, a later step; this function only has to work either way,
    which a lock of its own would not let it do.

    Returns the number of rows stamped, so a caller (or a test) can tell a
    real release from a no-op.
    """
    return conn.execute(
        """UPDATE task_devices SET released_at = ?, release_reason = ?
            WHERE task_id = ? AND released_at IS NULL""",
        (_iso(utcnow()), reason, task_id),
    ).rowcount


def held_devices(conn: sqlite3.Connection, task_id: str) -> list[int]:
    """The live device indices ``task_id`` currently holds -- what
    ``_task_payload`` serialises as ``devices: [int]`` (docs/14 §5.4).

    Filters by ``released_at IS NULL`` for the same reason ``release`` does
    (module docstring): task ids are recycled, so a task id can own more than
    one ``task_devices`` row over its lifetime, and only the live one is
    "what this task holds *right now*". Backed by ``idx_task_devices_live``,
    the same partial index ``release`` uses to find the row it stamps.

    Empty for a task that holds none right now -- a legitimate answer, not a
    sentinel for "not found": a spot-check probe (``spotcheck.maybe_issue``)
    does not allocate through this module yet (a known gap; see the step 4
    report), so its task legitimately has no live row here.
    """
    return [
        row["device_index"] for row in conn.execute(
            """SELECT device_index FROM task_devices
                WHERE task_id = ? AND released_at IS NULL
                ORDER BY device_index""",
            (task_id,),
        ).fetchall()
    ]


def reconcile_inventory(conn: sqlite3.Connection, worker_id: str, profile: dict,
                        now: datetime | None = None) -> None:
    """Bring ``worker_devices`` in line with what a ``register`` call reports
    (docs/14 §5.6). Runs inside the caller's existing write transaction --
    ``app.register`` already opens ``immediate(conn)`` for the ``workers``
    upsert, and this is more bookkeeping under the same lock, not a second
    one.

    ``profile["devices"]`` is the per-device report docs/14 §2 puts on the
    profile (worker-side reporting is a later step -- nothing sends this
    yet). Its absence means "one device, synthesized from the flat profile
    fields", exactly what migration 009's own backfill did once, at migration
    time, for every worker that already existed. Repeating that synthesis
    here, on *every* register rather than only once, is what keeps a worker
    that registers for the first time *after* migration 009 ran from ending
    up with zero ``worker_devices`` rows -- and therefore zero free devices,
    forever, since ``free_devices`` starts from ``inventory()``, and
    ``inventory()`` cannot report a device nobody ever wrote a row for.

    A device index this worker reported before but does not report now had
    its ``worker_devices`` row retired, and any ``task_devices`` allocation
    still open on it is released with reason ``reconciled``: a box that
    reboots with a dead card must not leave that card permanently allocated
    to a task that will never submit or expire cleanly against it (docs/14
    §5.6 -- "a box that reboots with a dead card").
    """
    now = now or utcnow()
    now_iso = _iso(now)
    reported = profile.get("devices")
    if not isinstance(reported, list) or not reported:
        # Pre-009 shape (§2's "flat fields stay"): one synthesized device,
        # the same fallback migration 009's own backfill uses and for the
        # same reason -- ``compute_profile_json`` carries no richer a
        # fingerprint than this until a worker actually reports ``devices``.
        probe = profile.get("probe")
        probe = probe if isinstance(probe, dict) else {}
        supports = profile.get("supports")
        supports = supports if isinstance(supports, list) else []
        device_name = profile.get("device_name")
        device_name = (device_name if isinstance(device_name, str) and device_name
                       else "unknown")
        try:
            vram_mb = int(profile.get("vram_mb") or 0)
        except (TypeError, ValueError):
            vram_mb = 0
        reported = [{
            "index": 0, "name": device_name, "vram_mb": vram_mb,
            "compute_capability": profile.get("compute_capability"),
            "supports": supports, "alloc_max_mb": probe.get("alloc_max_mb"),
            "bench_score": probe.get("bench_score"),
        }]

    by_index: dict[int, dict] = {}
    for d in reported:
        if not isinstance(d, dict):
            continue
        try:
            idx = int(d.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = d

    # Read before the upsert below -- this is the "before" set the vanished
    # computation needs, and it must come from the same live-device
    # definition ``free_devices`` uses (``retired_at IS NULL``), not from a
    # raw row count, or a previously-retired index that this call revives
    # would be wrongly counted as newly vanished.
    live_before = {row["device_index"] for row in inventory(conn, worker_id)}

    for idx, d in by_index.items():
        supports = d.get("supports")
        supports = supports if isinstance(supports, list) else []
        conn.execute(
            """INSERT OR REPLACE INTO worker_devices
                 (worker_id, device_index, device_name, vram_mb,
                  compute_capability, supports_json, alloc_max_mb, bench_score,
                  retired_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (worker_id, idx, d.get("name") or "unknown",
             int(d.get("vram_mb") or 0) if d.get("vram_mb") is not None else 0,
             d.get("compute_capability"), json.dumps(supports),
             d.get("alloc_max_mb"), d.get("bench_score")),
        )

    for idx in live_before - set(by_index):
        conn.execute(
            "UPDATE worker_devices SET retired_at = ? "
            "WHERE worker_id = ? AND device_index = ?",
            (now_iso, worker_id, idx),
        )
        conn.execute(
            """UPDATE task_devices SET released_at = ?, release_reason = 'reconciled'
                WHERE worker_id = ? AND device_index = ? AND released_at IS NULL""",
            (now_iso, worker_id, idx),
        )
        # And drop any reservation standing on the card that just vanished.
        # Not merely tidiness: ``reserve``'s holder check is *worker-wide*
        # ("at most one job may hold reservations on a given worker",
        # docs/14 §6), and it purges only rows past ``expires_at``. An
        # unexpired reservation left behind on a retired index therefore
        # keeps its job reading as the holder of this whole worker, so every
        # *other* job is refused a reservation here -- on the strength of a
        # claim over a card that no longer exists -- until the TTL lapses.
        # ``has_other_reservations`` answers True for that window too, so the
        # walk also pays for a backfill peek that can never find a target.
        # Same reasoning as the ``task_devices`` release above: a box that
        # reboots with a dead card must not keep holding anything against it.
        conn.execute(
            "DELETE FROM device_reservations WHERE worker_id = ? AND device_index = ?",
            (worker_id, idx),
        )


# --------------------------------------------------------------------------
# The ledger read path (docs/14 §4)
# --------------------------------------------------------------------------


def device_history(conn: sqlite3.Connection, worker_id: str,
                   since: datetime) -> list[sqlite3.Row]:
    """This worker's ``task_devices`` rows -- released and live -- allocated
    at or after ``since``, oldest first. The operator / ledger read path: "why
    is card 2 dark" is answered by the row whose ``released_at`` never got
    set (docs/14 §3).

    Deliberately not filtered by ``released_at`` -- unlike every other query
    in this module -- because this is the one path where the released rows
    *are* the point, not noise to exclude. Ordered by ``(allocated_at, id)``,
    which ``idx_task_devices_history`` serves directly; ``id`` is the
    tiebreak for two rows landing in the same second, so the order is total
    and deterministic rather than left to SQLite's unspecified tie behaviour.
    """
    return conn.execute(
        """SELECT * FROM task_devices
            WHERE worker_id = ? AND allocated_at >= ?
            ORDER BY allocated_at, id""",
        (worker_id, _iso(since)),
    ).fetchall()


# --------------------------------------------------------------------------
# Reservation with backfill (docs/14 §6)
# --------------------------------------------------------------------------


def release_reservations(conn: sqlite3.Connection, worker_id: str,
                         job_id: str) -> int:
    """Drop this job's reservations on this worker. Returns rows removed.

    Called the moment the reserving job successfully claims (docs/14 §6). A
    reservation exists to let a job that cannot yet fit *accumulate* devices
    across polls; the accumulation is over the instant it fits, and what it
    won is now recorded as ``task_devices`` rows, which exclude every other
    job on their own.

    Leaving the rows behind would give the reserver an asymmetric hold §6
    never intended: it sees through its own reservation and is unaffected,
    while every other job stays excluded for the remainder of the TTL *after*
    the winning task has already released its cards. A wide job that needs
    another task's worth of devices re-accumulates from scratch, which is the
    fair outcome -- by then the rest of the queue has had its turn at the
    cards it was holding.

    Opens its own ``immediate()``, matching ``reserve``.
    """
    with immediate(conn):
        return conn.execute(
            "DELETE FROM device_reservations WHERE worker_id = ? AND job_id = ?",
            (worker_id, job_id),
        ).rowcount


def reserve(conn: sqlite3.Connection, worker_id: str, job_id: str,
           indices: list[int], ttl: float) -> bool:
    """Reserve ``indices`` on ``worker_id`` for ``job_id`` until ``ttl``
    seconds from now. ``True`` if granted, ``False`` if refused because
    another job already holds a reservation on this worker.

    **Two uniqueness properties, docs/14 §6.** Per-device is the
    ``device_reservations`` primary key -- ``(worker_id, device_index)`` --
    and SQLite enforces that one for free. "At most one job may hold
    reservations on a given worker at a time" is *not* a key constraint
    (nothing in the schema names the pair ``(worker_id, job_id)``), so this
    function enforces it explicitly: **inside one ``immediate(conn)`` block**,
    read the distinct ``job_id``s already holding a live reservation on this
    worker, and refuse if that set is non-empty and is not just ``{job_id}``.
    The write lock is what makes that check-then-insert atomic against a
    second ``reserve`` call racing in from another job -- without it, two
    calls could both pass the holder check before either has inserted, and
    both would win. This is the concrete mechanism docs/14 §6 leaves
    unspecified ("it is enforced separately") -- an explicit application-level
    check under the existing write lock, not a schema constraint, because the
    schema (migration 009) is already shipped and its own comment says this
    half of the invariant is deferred to "a later step's application logic",
    which is this one.

    A schema-level alternative -- e.g. a second unique index on
    ``(worker_id)`` alone -- was not used: SQLite has no partial-unique-by-
    distinct-value construct that would let *one* job hold several rows
    (one per device) on a worker while still being the only job that may.
    A trigger could enforce it, but this codebase's convention (``db.py``,
    every write path in ``fairness.py``) is application-level checks under
    ``immediate()``, not triggers, and this module should not be the first
    file to break that pattern.

    **Expired rows on this worker are purged first, in the same transaction,
    before the holder check.** Without this, a dead job's own un-swept,
    already-expired reservation would (a) still count as a "holder" and
    wrongly refuse every other job, and (b) collide with this insert via the
    primary key even for the *same* job trying to re-reserve after its own
    TTL lapsed. This function does not depend on ``expire_reservations``
    having run -- docs/14 §4 already declines to run that sweep from anywhere
    but the claim poll, so a function that assumed it had would be wrong on
    the very first call of a session.

    The same job re-reserving, or widening its reservation to more devices,
    is allowed: ``INSERT OR REPLACE`` on rows already owned by ``job_id``
    simply refreshes ``reserved_at`` / ``expires_at`` rather than colliding
    with the primary key.

    Does **not** check that ``indices`` are currently free (via
    ``free_devices``) -- that is the caller's contract per docs/14 §6 ("a
    blocked wide job accumulates a reservation on devices *as they free*"),
    and duplicating that check here would give this function an opinion
    about scheduling policy it should not have. Likewise, *which* job is
    entitled to call ``reserve`` at all -- docs/14 §6's "only the first job
    in walk order refused for ``insufficient_free_devices``" -- is a walk-
    order policy that lives in ``app.py``'s claim walk (docs/14 §5), not in
    the ledger: this function only enforces that whichever job the walk
    decides to call it for is the *only* one that ends up holding.
    """
    now = utcnow()
    now_iso = _iso(now)
    expires_iso = _iso(now + timedelta(seconds=ttl))
    with immediate(conn):
        conn.execute(
            "DELETE FROM device_reservations WHERE worker_id = ? AND expires_at <= ?",
            (worker_id, now_iso),
        )
        holders = {
            row["job_id"] for row in conn.execute(
                "SELECT DISTINCT job_id FROM device_reservations WHERE worker_id = ?",
                (worker_id,),
            ).fetchall()
        }
        if holders and holders != {job_id}:
            return False
        for idx in indices:
            conn.execute(
                """INSERT OR REPLACE INTO device_reservations
                     (worker_id, device_index, job_id, reserved_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (worker_id, idx, job_id, now_iso, expires_iso),
            )
    return True


def expire_reservations(conn: sqlite3.Connection,
                        now: datetime | None = None) -> int:
    """Delete every ``device_reservations`` row past its ``expires_at``.
    Returns the number of rows removed.

    docs/14 §4: called from the claim path (``app.claim``), beside
    ``rounds.expire_leases``, and deliberately *not* from ``scripts/ledger.py``
    -- that script has no installed timer in any deployment (docs/14 §4's own
    reasoning), and a sweep that depends on an installer nobody runs is how a
    reservation becomes a permanently-dark card. Safe to call standalone too
    (its own ``immediate()``, not nested) and repeatedly (deleting rows that
    are already gone is a no-op, not an error) -- both a test and the claim
    path lean on that.
    """
    now = now or utcnow()
    with immediate(conn):
        return conn.execute(
            "DELETE FROM device_reservations WHERE expires_at <= ?",
            (_iso(now),),
        ).rowcount
