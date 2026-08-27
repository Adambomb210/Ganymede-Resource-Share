# Ganymede — Identity, Enrollment & Admin (component design)

*One of the Stage 1 component docs (`04-platform-expansion.md`, Build sequencing).
Imports the frozen spine — `04` Decisions 5, 9, 14; `05` (`contributors` /
`workers` deltas, `enrollments`, `machine_keys`, `schema_version`, migration
004); `06` ("Auth: three token kinds", "Machine enrollment"). Freezes its own
seam: the `IdentityProvider` interface, token formats and TTLs, the session
mechanism, migration 004's backfill, the bootstrap-admin upsert. Anything that
departs from `05` / `06` is in [Spine deviations](#spine-deviations) and nowhere
else.*

Invariant 1 (`04`) — inventory is derived, not maintained; it dies with the
account — stops holding for machines the moment a machine carries an accrued
total. This doc is where the maintained record starts: an enrollment-minted
`machine_id`, bound to a user, stable across the hardware changing underneath it.

---

## The `IdentityProvider` seam

`06` picks a placeholder provider (Decision 5) so accounts ship in Phase B with
no OAuth dependency. The seam is one interface; the placeholder and a later
GitHub / OIDC provider are implementations of it. The coordinator owns the
`contributors` row and the session — the provider only turns a submitted
credential into a verified identity.

```python
# ganymede/coordinator/identity.py

@dataclass(frozen=True)
class ProviderIdentity:
    auth_provider: str        # 'local' | 'github' | 'oidc'
    auth_subject: str | None  # provider's stable id; None for 'local' (keyed by name)
    name: str
    email: str | None

class IdentityProvider(Protocol):
    name: str

    # Redirect providers return an authorize URL + opaque state; 'local' returns None.
    def begin(self, redirect_uri: str) -> AuthChallenge | None: ...

    # Verify a credential or raise AuthError.
    #   'local'  -> {"username", "secret"}  checked against contributors
    #   'github' -> {"code", "state"}       code-for-token exchange + GET /user
    def verify(self, credential: dict, conn: Connection) -> ProviderIdentity: ...
```

`POST /v1/auth/session` (`06`, user-issuing):

1. `provider = PROVIDERS[settings.auth_provider]` — env `GANYMEDE_AUTH_PROVIDER`,
   default `local`.
2. `ident = provider.verify(body, conn)`.
3. Resolve the row:
   `SELECT id, enabled, is_admin FROM contributors
    WHERE auth_provider = ? AND (auth_subject = ? OR (? IS NULL AND name = ?))`.
   External providers **JIT-provision** on first login (`INSERT`, `is_admin = 0`);
   `local` never provisions here — local rows come from the bootstrap var or an
   admin.
4. Mint a session (below); `Set-Cookie` + JSON body.

### The `local` placeholder

`verify({"username", "secret"})`:

- `row = SELECT id, key_hash, enabled FROM contributors
   WHERE auth_provider = 'local' AND (name = :username OR email = :username)`
- `hmac.compare_digest(row["key_hash"], auth.hash_key(secret))`, `row["enabled"]`,
  else `AuthError`.

The **secret is the contributor's issued key** — the same 32-byte token whose
`hash_key` already sits in `contributors.key_hash` (`auth.py`). No password
column, no KDF: `hash_key`'s own docstring is the argument — these are random
tokens, not passwords, so there is no dictionary to stretch. The placeholder is a
TTL'd, cookie-deliverable wrapper around a credential the coordinator already
verifies. When a real provider lands, the oddity leaves with it.

### GitHub / OIDC drop in later, unchanged behind it

Untouched: the endpoint, `sessions`, `authenticate()`, every `require_*`
dependency, the `contributors` schema. Added by that workstream: an
`IdentityProvider` class, `GANYMEDE_AUTH_PROVIDER=github`, provider env
(`GANYMEDE_OIDC_ISSUER`, client id / secret). `begin()` gets a real body (the
authorize redirect); `verify()` exchanges the code and reads `auth_subject` from
`sub` / the GitHub user id. `contributors.(auth_provider, auth_subject)` is
already the composite key for it (`05`).

### Session token

| Property | Value |
| --- | --- |
| Format | `secrets.token_urlsafe(32)`, no prefix |
| At rest | `sessions.token_hash = auth.hash_key(token)` — a DB dump is not a set of live sessions |
| Delivery | `Set-Cookie: ganymede_session=…; HttpOnly; Secure; SameSite=Lax; Path=/` **and** `{session_token, expires_at}` in the body — cookie for `/ui/*`, body for non-browser callers |
| TTL | `GANYMEDE_SESSION_TTL_SEC`, default 43200 (12 h). Absolute expiry; no sliding renewal in the placeholder |
| Revoke | `DELETE /v1/auth/session` (this caller) deletes the row; expiry is checked lazily in `authenticate()` |

`sessions` (this doc's addition, created by migration 004):
`token_hash TEXT PRIMARY KEY`, `user_id TEXT NOT NULL REFERENCES contributors(id)`,
`created_at TEXT`, `expires_at TEXT NOT NULL`, `last_used_at TEXT`. GC drops
expired rows on the `audit` / `availability_ticks` cron.

`/ui/*` mutations: `SameSite=Lax` blocks cross-site form posts; state-changing
`/ui/*` requests must additionally carry `X-Ganymede-UI: 1` (htmx sets it
globally), which a cross-origin form cannot forge. No CSRF token store.

---

## Machine enrollment

`app.py:register` mints `worker_id = uuid5(contributor, device fingerprint)`.
That derivation is deleted (invariant 1). Identity is minted once, at enrollment,
and never recomputed.

Flow: `enroll` (user auth) → one-time `enroll_token` → operator pastes it into
the host config → `claim-enrollment` (token only) → `{machine_id, machine_key}`
written to the host config; the worker uses the machine key for all `/v1/tasks/*`
thereafter.

### `POST /v1/machines/enroll` — auth: user

- `enroll_token = "gme_" + secrets.token_urlsafe(32)`.
- `INSERT INTO enrollments (id, user_id, token_hash, display_name, created_at,
  consumed_at, machine_id) VALUES (uuid4, :user, hash_key(enroll_token),
  :display_name, now, NULL, NULL)`.
- Response `{enroll_token, enroll_id, expires_at}` — `enroll_token` shown once.
- **TTL** `GANYMEDE_ENROLL_TTL_SEC`, default 3600 (1 h — it goes into a config
  within the minute). Expiry is `created_at + TTL`, evaluated at claim; no
  `expires_at` column, because `05` freezes `enrollments`.

### `POST /v1/machines/claim-enrollment` — auth: none (token in body)

All inside `immediate(conn)`:

1. `row = SELECT * FROM enrollments WHERE token_hash = hash_key(enroll_token)`;
   `hmac.compare_digest` guard, as in `auth.authenticate`.
2. Reject → **404** if `row is None`, `consumed_at IS NOT NULL`, or
   `now > created_at + TTL`. One body for all three (see
   [404-not-403](#404-not-403-across-machines-jobs-and-enrollments)).
3. `machine_id = uuid4().hex` — minted, not a function of `compute_profile` or
   anything else.
4. `INSERT INTO workers (id, contributor_id, compute_profile_json, display_name,
   enrolled_at, first_seen, last_seen, hardware_fingerprint_json, standing)
   VALUES (:machine_id, :user, :profile, :display_name, now, now, now,
   :fingerprint, 'good')`.
5. `machine_key = "gmk_" + secrets.token_urlsafe(32)`;
   `INSERT INTO machine_keys (machine_id, key_hash, enabled, created_at)
   VALUES (:machine_id, hash_key(machine_key), 1, now)`.
6. `UPDATE enrollments SET consumed_at = now, machine_id = :machine_id
   WHERE id = :id AND consumed_at IS NULL` — rowcount 0 lost the race →
   `ROLLBACK`, 404. This is the one-time gate.
7. Response `{machine_id, machine_key}` — `machine_key` shown once. Hook:
   enqueue a `machine_weight` recompute (Decision 12), owned by the ledger doc.

`hardware_fingerprint_json` is coordinator-derived from the submitted
`compute_profile`: `{gpu_model, gpu_uuid?, cpu_model, cpu_count, total_ram_mb,
board_serial?}`, whatever the probe saw. **Advisory only** (invariant 1):
recomputed on every `/v1/workers/register` re-probe; a material divergence writes
`audit(event='fingerprint_drift')` for fraud review and does nothing else. It
never changes `machine_id`, never gates accrual, never reaches `credit()`. A
legitimate GPU upgrade changes it — that is the expected case, not the alarm.

Retry hazard: the response carries `machine_key` once. A host that loses it
retries and gets 404 (token consumed); the operator re-enrolls. The host writes
`machine_key` to disk before ACKing to shrink the window. Fine at v0 fleet size.

### `POST /v1/workers/register` — auth: machine (kept, deprecated, `06`)

Looks the worker up by its machine key, updates `compute_profile_json`,
re-derives `hardware_fingerprint_json`. Mints nothing. The `uuid5` block in
`app.py:register` is removed by this workstream.

---

## `machine_keys` — issuance, rotation, revocation

| Event | Mechanism |
| --- | --- |
| Issue | One row at `claim-enrollment`. `key_hash UNIQUE`, `enabled = 1`. |
| Rotate | `POST /v1/machines/{id}/rotate-key` (auth: user, owner) — `immediate`: insert a new enabled row, `UPDATE machine_keys SET enabled = 0 WHERE machine_id = :id AND key_hash != :new`. Immediate swap; the host is reconfigured with the returned key. |
| Revoke | `enabled = 0`, never `DELETE` — matches `auth.py` ("revocation is `enabled = 0` … so the audit trail survives"). Triggered by `retire`, `rotate-key`, or an admin `/v1/admin/*` action. |
| Retire | `POST /v1/machines/{id}/retire` (`06`) — `standing = 'revoked'`, every key `enabled = 0`. `credit_events` rows stay (append-only). |

Standing and key-enabled are **separate axes**. A `standing = 'revoked'` machine
keeps a working key on purpose — it must still authenticate a heartbeat to
receive `cancel: hard` (`06`) and to be told it is getting no work. Revoking
*standing* does not disable *keys*; only `retire`, `rotate-key`, or an explicit
admin key action does.

### `authenticate()` gains the machine path

```python
Principal = Contributor | Machine

@dataclass(frozen=True)
class Machine:
    id: str
    owner_id: str    # contributors.id
    standing: str     # good | probation | revoked

def authenticate(conn, header, *, cookie=None) -> Principal:
    token = parse_bearer(header) or cookie
    if token is None:
        raise AuthError("missing credential")
    digest = hash_key(token)

    # 1. machine key — highest QPS (every heartbeat, every worker)
    row = conn.execute(
        """SELECT w.id, w.contributor_id, w.standing, k.enabled
             FROM machine_keys k JOIN workers w ON w.id = k.machine_id
            WHERE k.key_hash = ?""", (digest,)).fetchone()
    if row is not None:
        if not row["enabled"]:
            raise AuthError("machine key revoked")
        return Machine(row["id"], row["contributor_id"], row["standing"])

    # 2. contributor key — CLI, operators, admins
    row = conn.execute(
        "SELECT id, name, clearance, is_admin, enabled, key_hash "
        "FROM contributors WHERE key_hash = ?", (digest,)).fetchone()
    if row is not None:
        if not hmac.compare_digest(row["key_hash"], digest):
            raise AuthError("unknown key")
        if not row["enabled"]:
            raise AuthError("key revoked")
        return Contributor(row["id"], row["name"], row["clearance"], bool(row["is_admin"]))

    # 3. session token — the web UI
    row = conn.execute(
        """SELECT c.id, c.name, c.clearance, c.is_admin, c.enabled, s.expires_at
             FROM sessions s JOIN contributors c ON c.id = s.user_id
            WHERE s.token_hash = ?""", (digest,)).fetchone()
    if row is not None and row["enabled"] and now_iso() < row["expires_at"]:
        return Contributor(row["id"], row["name"], row["clearance"], bool(row["is_admin"]))

    raise AuthError("unknown credential")
```

Three sha256 lookups worst case, all on `UNIQUE` indexes. `Contributor` gains
`is_admin: bool` (`05` column). A token is exactly one kind — the `gmk_` / `gme_`
prefixes and table separation keep the spaces disjoint; the fall-through order is
a micro-optimisation for the machine-key hot path, not a correctness property.

FastAPI dependencies map the `06` auth classes:

| `06` class | Dependency | Resolves / rejects |
| --- | --- | --- |
| machine | `require_machine` | `Machine`; a `Contributor` on `/v1/tasks/*` → 404 (existence not confirmed) |
| user | `require_user` (today's `require_contributor`) | any `Contributor` — key or session |
| submitter | `require_submitter` | `Contributor` with `submitters.status = 'approved'`, else 404 |
| admin | `require_admin` | `Contributor` with `is_admin`; otherwise **404 on the whole `/v1/admin/*` tree** |

---

## Migration 004 — `workers.id` becomes enrollment-minted

The hard one (`05`, "Prerequisite"). Forward-only, wrapped in `immediate()`.
Table creation for `machine_keys` / `enrollments` rides in the additive 002–003;
**004 is the `id` semantics change, its backfill, and `sessions`.**

**Before:** `workers.id = uuid5(NAMESPACE_OID, json([contributor_id, device_name,
backend, vram_mb]))` (`app.py:register`). A GPU swap changes `device_name` /
`vram_mb` → a new id → the accrued total is orphaned. Re-enrollment has nowhere
to land.

**After:** `workers.id = uuid4().hex`, minted at `claim-enrollment`, stored,
never a function of anything mutable. Stable across GPU swap, OS reinstall, a
month offline, and re-enroll — a re-enroll of the *same physical box* is a new
row and a new id by design; the owner retires the stale one, and the fingerprint
match flags it for the owner to reconcile rather than auto-merging.

### Backfill

Existing ids are already unique primary keys — derived, not *wrong*. 004 keeps
them and rewrites no FK (`tasks.worker_id`, `credit_events.machine_id`, …
untouched). For each existing `workers` row:

1. `enrolled_at = first_seen`; `standing = 'good'`;
   `display_name = COALESCE(image_tag, 'machine-' || substr(id, 1, 8))`;
   `hardware_fingerprint_json` derived from `compute_profile_json`.
2. Synthesize a consumed `enrollments` row:
   `token_hash = 'migrated:' || id` (a `:` makes it un-matchable — every real
   `hash_key` output is 64 hex chars), `user_id = contributor_id`,
   `created_at = consumed_at = first_seen`, `machine_id = id`. Every machine now
   has a uniform enrollment record.
3. **No machine key is minted** — there is no channel to deliver one.
   Transitional auth rule instead (see [Spine deviations](#spine-deviations)): a
   pre-004 `workers` row with zero `machine_keys` authenticates `/v1/tasks/*`
   with its owner's **contributor** key, logged `audit(event='legacy_worker_auth')`.
   A later migration removes the rule once every live worker has re-enrolled.

The fleet that exists when 004 runs is the author's own two or three machines
(`03-roadmap.md`). The grace path is politeness; the operator re-runs `enroll`
per box and is done in a minute. `scripts/reenroll-fleet.py` prints one
`enroll_token` per orphaned worker for that.

`sessions` is created here (columns above). It is not in `05`'s table list —
`05` delegated the session mechanism to this doc, and the table is the mechanism.

---

## `GANYMEDE_BOOTSTRAP_ADMIN` — first admin at coordinator start

Decision 14: the first admin is named by env, not created through the API.
`06`'s admin surface has no `is_admin` writer, so **the env var is the only way
`is_admin` is first set.**

- `Settings.from_env()` reads `GANYMEDE_BOOTSTRAP_ADMIN` (optional; unset →
  skip). Value: comma-separated `name` or `name:secret`.
- `bootstrap()` calls `ensure_bootstrap_admin(conn, spec)` after `init_schema` +
  migrations, inside `immediate`:

```python
for entry in spec.split(","):
    name, _, secret = entry.strip().partition(":")
    row = conn.execute(
        "SELECT id FROM contributors WHERE auth_provider = 'local' AND name = ?",
        (name,)).fetchone()
    if row is None:
        key = secret or generate_key()
        conn.execute(
            "INSERT INTO contributors "
            "(id, name, key_hash, enabled, clearance, is_admin, auth_provider, created_at) "
            "VALUES (?, ?, ?, 1, 'restricted', 1, 'local', ?)",
            (uuid4().hex, name, hash_key(key), now))
        if not secret:
            print(f"[bootstrap] minted admin key for {name!r}: {key}")  # once, cf. scripts/issue-key.py
    else:
        conn.execute("UPDATE contributors SET is_admin = 1 WHERE id = ?", (row["id"],))
        if secret:
            conn.execute("UPDATE contributors SET key_hash = ? WHERE id = ?",
                         (hash_key(secret), row["id"]))
```

Idempotent — safe on every boot. Promotes an existing local row or creates one;
`clearance = 'restricted'` so an admin sees every run (`budget.clearance_permits`).
**Never demotes**: removing the var does not strip `is_admin`. Removing admin
from someone is a manual `UPDATE` until a `/v1/admin/users` surface exists.

### Admin gating

`require_admin` reads `Contributor.is_admin`. `/v1/admin/*` returns **404** for an
authenticated non-admin — the admin API is not confirmed to exist — and 401 for
the unauthenticated. Session-auth'd admins (`/ui/queue`, `/ui/submitters`,
Decision 9) go through the identical check.

---

## 404-not-403 across machines, jobs, and enrollments

`06` carries the rule forward from `app.py` (`_worker_for_task`,
`worker_eligibility`): a resource the caller may not see is **indistinguishable
from one that does not exist** — same status, same body, no timing tell (every
lookup is a hash- or PK-indexed point query). Extended:

| Surface | Non-owner, non-admin result |
| --- | --- |
| `GET /v1/machines/{id}`, `POST …/retire`, `…/rotate-key` | 404 `unknown machine` (`workers.contributor_id != principal`) |
| `GET /v1/jobs/{id}`, `POST …/cancel`, `…/enqueue` | 404 `unknown job` (`jobs.owner_id != principal`; admin sees all, `06`) |
| `/v1/tasks/{id}/*` | 404 unless the caller is the `Machine` holding the task (`_worker_for_task`, re-keyed from `contributor_id` to `machine_id`) |
| `/v1/admin/*` | 404 for any non-admin principal |
| `POST /v1/machines/claim-enrollment` | 404 for unknown / consumed / expired token — never distinguished, so a probe cannot learn a token was ever valid |

A `Machine` principal acting on another machine's endpoint, or on any job, gets
the same 404 — a leaked machine key is scoped to its own task lifecycle and
nothing else.

---

## Spine deviations

Everything not listed here conforms to `05` / `06`.

1. **`DELETE /v1/auth/session`** — additive method on a spine-named path (logout /
   session revoke), auth: user. `06` names only `POST`.
2. **`POST /v1/machines/{id}/rotate-key`** — additive path, auth: user (owner).
   `retire` was the spine's only key-kill and is too coarse for a leaked key with
   a live machine still behind it.
3. **`workers.standing = 'revoked'` on owner `retire`** — `05` sets `revoked`
   "by admin or by the fraud rules"; this doc also sets it on owner-initiated
   retire. No schema change; `audit.event` separates `owner_retire` from
   `fraud_revoke`.
4. **Transitional `/v1/tasks/*` auth** — a pre-004 `workers` row with no
   `machine_keys` resolves a `Machine` from the owner's contributor key
   (`audit event='legacy_worker_auth'`). Removed by a later migration once the
   fleet has re-enrolled; the spine's `auth: machine` end state is unchanged.
5. **`sessions` table** — not in `05`'s list; `05` delegated the session
   mechanism here. Columns are fixed above for the web-UI and ledger docs.

---

## Frozen for downstream

- **Token shapes**: session `token_urlsafe(32)` (cookie + bearer, no prefix);
  `enroll_token` = `gme_` + `token_urlsafe(32)`; `machine_key` = `gmk_` +
  `token_urlsafe(32)`. All sha256 at rest via `auth.hash_key`, all shown once.
- **Env / TTL**: `GANYMEDE_AUTH_PROVIDER=local`, `GANYMEDE_SESSION_TTL_SEC=43200`,
  `GANYMEDE_ENROLL_TTL_SEC=3600`, `GANYMEDE_BOOTSTRAP_ADMIN` (unset).
- **`IdentityProvider`** Protocol + `ProviderIdentity` shape.
- **`authenticate()`** returns `Contributor | Machine`; `Machine(id, owner_id,
  standing)`; `Contributor` gains `is_admin: bool`.
- **Auth-class → dependency** table.
- **`sessions`** columns; **migration 004** step list and backfill.
- **404 extension** table.

**Open, deferred to the doc named:** real OAuth / OIDC provider env and the
`begin()` body (identity, when built); per-key / per-session rate limits (§6.3 —
still required, numbers TBD); the `machine_weight` recompute trigger (ledger);
an admin / user CRUD surface (later); session sliding-renewal (deferred —
absolute TTL for now).
