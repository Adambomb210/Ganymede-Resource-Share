# Ganymede — Web UI (component design)

*Stage 1 component doc. Imports the frozen spine — `05-data-model.md` (schema),
`06-api-delta.md` (endpoints) — and freezes its own seam: the `/ui/*` page set,
the read model that feeds those pages plus `/v1/me` and `/v1/leaderboard`, and
the `/v1/events` SSE schema. Design only, per `04-platform-expansion.md` Decision
17 and Phase C. No code.*

Delta against `ganymede/coordinator/app.py` and `scripts/status.py` as of
`a1b4e36`.

---

## Stack

- **Server-rendered Jinja2 + htmx**, served from the existing FastAPI app under
  `/ui/*` (`06`, htmx pages). Separate namespace from `/v1/*` JSON; same
  process, same `create_app`, same connection-per-request (`get_conn`).
- **No build toolchain.** `htmx.min.js` and the htmx **SSE extension** are
  vendored into the repo and served from `/ui/static/`. No CDN, no npm, no
  bundler, no transpile step. One language on the server (Python), zero on the
  client.
- **CSP is `script-src 'self'`.** No inline `<script>`, no `eval`. `hx-*`
  attributes are CSP-safe and are the whole client. `hx-vals` with the `js:`
  prefix is banned (eval-shaped); values come from the server-rendered DOM.
- **Explicitly not React / not an SPA.** No client router, no virtual DOM, no
  client state store. Reconsidered only if the UI outgrows dashboards and forms
  (Decision 17).
- **Auth is the session cookie** from the placeholder provider (`06`,
  `POST /v1/auth/session`). `/ui/*` reuses it; it does not accept bearer keys.
- **No WebSocket.** Live updates are SSE only (`/v1/events`), matching the
  pull-only transport ethos (Decision 8).

### Reconciliation with §6.2

`02-architecture-v2.md` §6.2 lists `/status` and `/metrics` as "read-only HTML".
That parenthetical is **superseded by `06`**, not deviated from here: `/status`
stays JSON and gains `jobs`; `/metrics` is still not built; the HTML operator
view now lives at `/ui/*`. `06` is the newer spine doc and already overrode it.

---

## Page inventory

Served under `/ui/`. Role column: **user** = any authenticated contributor,
**submitter** = `submitters.status = approved`, **admin** = `is_admin = 1`. A
page a role may not see returns `303 → /ui/` (admin-only) or `404` (another
user's resource — `06` cross-tenant rule), never a naked `403`.

| Path | Reads | Role | POSTs |
| --- | --- | --- | --- |
| `/ui/login` | nothing | none | `POST /v1/auth/session` → set cookie, `303 → /ui/` |
| `/ui/` (dashboard) | `/status` (runs + jobs), `eligibility.fleet_summary`, `awake_workers`, `stalls`, `invariants.check`, queued-job count, leased-task count | user | none |
| `/ui/jobs` | `jobs` where `owner_id = caller` (admin: all); status, `job_type`, `priority_rank`, leased-task count | user | none |
| `/ui/jobs/{id}` | `jobs` row, `tasks` for the job (status, holder machine, `attempt_group`); **`collab_lora_finetune` only:** `rounds` (idx, status, `distinct_contributors`, `eval_loss`, `adapter_divergence`, age) + `invariants.coverage` | user (owner) / admin; else `404` | `POST /v1/jobs/{id}/enqueue` (owner submitter), `POST /v1/jobs/{id}/cancel {mode}` (owner submitter / admin) |
| `/ui/queue` | `GET /v1/admin/queue` — jobs in `priority_rank` order + leased-task counts | admin | `POST /v1/admin/queue/reorder {job_id, before\|after\|rank}` |
| `/ui/submitters` | `submitters` ⋈ `contributors` (status, `decided_by`, `decided_at`, `note`); `images` pinned per submitter; each submitter's running jobs | admin | `POST /v1/admin/submitters/{user_id} {status, note}`; `POST /v1/admin/jobs/{id}/cancel {mode}` |
| `/ui/machines` | `GET /v1/me` — caller's machines, `standing`, `weighted_hours_total`, `system_weight`, `last_available_at`, recent `credit_events` | user (own only) | `POST /v1/machines/enroll {display_name}`; `POST /v1/machines/{id}/retire` |
| `/ui/leaderboard` | `GET /v1/leaderboard` — rank, display name, weighted-hours total | user | none |

### Page notes

- **Dashboard "healthy" is not re-derived.** `AWAKE_WINDOW_SEC = 900`,
  `STALL_GRACE_MULT`, `COHORT_FLOOR`, `awake_workers()`, `stalls()`,
  `invariants.check()` stay owned by `scripts/status.py` / `invariants.py`; the
  dashboard imports and renders them. Forking "stalled" into a template is the
  failure mode `03` M5 warns against. The admin view adds the per-refusal-reason
  breakdown (`fleet_summary`) and stall detail.
- **`/ui/jobs/{id}` per-round loss/divergence is type-specific**, like
  `/v1/runs/{id}/rounds/current` in `06`. Post-Phase-A the round lifecycle is
  behind the plugin boundary (`05`); the generic job page renders tasks and job
  status and delegates the round table to the type. Types without `reduce` show
  tasks only.
- **`/ui/queue` reorder is move-up / move-down buttons**, not drag — htmx cannot
  drag and Decision 17 forbids a build step. Each button is one
  `POST /v1/admin/queue/reorder {job_id, before|after: <neighbour_id>}`, exactly
  the body `06` froze. A vendored `Sortable.js` emitting the same body is an
  optional C2 enhancement; buttons ship first.
- **`/ui/machines` shows `enroll_token` exactly once**, inline in the `POST
  /v1/machines/enroll` response fragment. No `GET` returns it, a refresh loses
  it, it never enters a URL or redirect (`05`, `enrollments`).
- **`/ui/submitters` revoke does not kill jobs** (`06`): a revoked submitter
  cannot enqueue more; running jobs are cancelled explicitly with a `mode`, a
  separate per-job action on the page.

---

## Read model

The pages and `/v1/me` / `/v1/leaderboard` read through a **query layer**: a
module of named read-only `SELECT`s (optionally registered as SQLite `VIEW`s —
which are query macros, not stored state). It holds nothing.

**Off the hot claim path.** The claim path takes `BEGIN IMMEDIATE` via
`immediate()`. The read model:

- runs only read-only `SELECT`s, on the per-request connection from `get_conn`,
  never inside `immediate()`, so it never acquires the write lock and adds zero
  contention on `BEGIN IMMEDIATE`;
- relies on WAL (`db.connect` pragmas) — readers and the writer do not block
  each other;
- never writes. Anything materialized — `credit_events`, `machine_weight`,
  `availability_ticks` rollups — is written by the **existing accrual engine**
  on its own schedule (`05`), never by a page render or an SSE handler;
- holds no long-lived read transaction: a stream or a slow page must not pin a
  WAL snapshot and stall checkpointing.

Pagination is keyset (`created_at, id` for `/ui/jobs`; `weighted_hours, id` for
the leaderboard), not `OFFSET`.

### `/v1/me` and `/v1/leaderboard` — field ownership

`06` assigns these to "ledger + web-UI docs". Split:

**Stage 1 reconciliation:** `06` split ownership — **`09` (ledger) owns the field
lists** of `/v1/me` and `/v1/leaderboard`; its shape is the fuller one (`user`,
`totals`, `machines[]` with reputation / current-window / unverified counts,
`recent_events[]`; leaderboard `by_machine[]` / `by_user[]` / optional
`by_work[]`). This doc renders that shape and owns the **envelope**: keyset
pagination (`weighted_hours, id`, not `OFFSET`), the `?scope=machines|users`
switch, `next_cursor`. The minimal shape sketched in earlier drafts of this doc
is superseded by `09`'s.

**The leaderboard is not fleet enumeration.** Rank, display name, weighted hours.
No `machine_id`s, no hardware profiles, no standing of machines you do not own —
the same boundary `_worker_for_task`'s 404 rule enforces. `scope=machines` lists
machine display names owned by the viewer plus opaque rank rows for the rest.

---

## SSE — `GET /v1/events`

Auth class **user** (`06`). `EventSource` cannot set an `Authorization` header,
so the stream authenticates by the session cookie — which `/ui/*` already
carries. `async def` handler; it holds **no** DB connection for the life of the
stream (see read model). It awaits an `asyncio.Queue` fed by the in-process hub
and writes SSE frames.

### Envelope schema

Events carry a **tiny envelope**, never rendered HTML — per-subscriber rendering
inside the hub does not scale.

```
id:    <monotonic int, process-wide>
event: <type>
data:  { "type": <type>, "id": <same int>, <one entity id field> }
```

| `type` | Emitted when | `data` id field | Audience | UI reaction |
| --- | --- | --- | --- | --- |
| `job.status` | `jobs.status` changes | `job_id` | owner + admin | `hx-get` job row (`/ui/jobs`) / job header (`/ui/jobs/{id}`) |
| `round.close` | a `collab_lora_finetune` round closes | `job_id` | owner + admin | `hx-get` the rounds fragment on `/ui/jobs/{id}` |
| `fleet.delta` | a machine's presence/standing rollup changes the dashboard counts | *(none)* | all users | `hx-get` the dashboard fleet-health fragment |
| `standing.change` | `workers.standing` changes | `machine_id` | machine owner + admin | `hx-get` the `/ui/machines` row |
| `queue.change` | `priority_rank` reorder, or a job enters/leaves the queue | `job_id` | admin only | `hx-get` `/ui/queue` |
| `submitter.change` | `submitters.status` changes | `user_id` | admin + that submitter (own row) | `hx-get` `/ui/submitters` / a status badge |

### htmx swap

Trivial payloads (a badge count) may use `sse-swap`. Everything real uses
`hx-trigger="sse:<type>"` + `hx-get` against the page's own fragment endpoint —
the swap re-reads current truth through the read model. Fragments are
idempotent: applying the same swap twice is harmless.

### Emit and fan-out

Endpoints that mutate state call `events.publish(envelope)` **after** their
`immediate()` transaction commits — a synchronous non-blocking `put` onto each
subscriber's queue. The hub holds, per subscriber, `(user_id, is_admin,
owned_machine_ids)`. Authorization is applied **at emit time, per subscriber**:
the `06` 404-not-403 rule extends to the stream — a non-admin never receives
`queue.change`, and nothing signals that the event exists.

### Reconnect and backfill

- The hub keeps a **bounded in-process ring buffer** of recent envelopes (last
  N, a few minutes). No new table — `05` is untouched.
- On reconnect the htmx SSE extension sends `Last-Event-ID`. If that id is still
  in the ring, the hub replays the envelopes after it.
- If the id is older than the ring tail, **or** the coordinator restarted (ids
  reset, buffer empty), the hub emits one `sync` event carrying no data. Every
  live fragment on the page has `hx-trigger="sse:sync"` + `hx-get`, so the page
  pulls current state for each fragment. An idempotent fragment refetch **is**
  complete recovery; the ring is an optimization to avoid a full refetch on
  every transient blip, not a correctness requirement.
- **Single-process assumption.** The ring and the monotonic id counter are
  per-process. Multi-worker `uvicorn` breaks both. The current deployment is a
  single process (§6.5); scaling out later means Redis pub/sub or a durable
  `events` table (a `05` migration at that point), and is out of scope here.
- SSE fallback poll (if SSE proves unreliable behind a proxy): a
  `settle-when-you-reach-it` item per `04` Open questions — a meta-refresh or an
  htmx `every Ns` trigger on the same fragment endpoints, no schema impact.

---

## Auth, sessions, CSRF

- **`/ui/*` unauthenticated → `303 → /ui/login`.** `/v1/*` unauthenticated stays
  `401` JSON. `require_contributor` raises the 401; the `/ui` routes catch the
  auth failure and redirect instead. `06` froze auth *classes*, not the HTML
  failure mode — that is this doc's.
- **CSRF is new surface.** Bearer tokens were immune; a session cookie plus htmx
  `POST`s is not. **Resolved in Stage 1 reconciliation (`06`): `08`'s scheme
  stands** — `SameSite=Lax` cookie **plus** a static `X-Ganymede-UI: 1` request
  header that htmx sets globally (`htmx.config.headers`) and a cross-origin form
  cannot forge. The coordinator rejects a cookie-authed mutating request that
  lacks it. No per-session CSRF token, no token store. (This doc's earlier
  per-session-token proposal was dropped.)
- `/ui/*` `GET`s are read-only; all mutation is an explicit `POST` to a `/v1/*`
  endpoint. SSE is display-only — it never mutates.

---

## Progressive rollout — matches Phase C

**C1 — read-only operator view.** `/ui/`, `/ui/jobs`, `/ui/jobs/{id}`,
`/ui/leaderboard`, `/ui/machines` (standing / hours / credit history, view
only), `/ui/login`. `/v1/events`, `/v1/me`, `/v1/leaderboard` live. No `POST`
except login. This is the `scripts/status.py` operator view in a browser, plus
the ledger read surface.

**C2 — admin surface.** `/ui/queue` reorder, `/ui/submitters`
approve/deny/revoke, job `enqueue` / `cancel`. Needs the admin role (Decision 5)
and the submitter allowlist (`05`). Job submission (image upload, `POST
/v1/jobs`) is a submitter-facing form landing here too.

**C3 — account and machine management.** `/ui/machines` enrollment (token shown
once) and `retire`; account / session management. Completes `04` Phase C's
"then account and machine administration".

Each step is independently shippable and adds only additive routes.

---

## Not built — deliberately

- **No `/metrics` Prometheus endpoint.** Still out of scope (§6.2, `06`). The
  dashboard is the operator view; a Prometheus/Grafana integration is a separate
  thing nobody has asked for.
- **No contributor-facing product beyond `/ui/leaderboard`.** No public
  profiles, no per-contributor stat pages, no social features, no embeddable
  widgets. `03` M5 scope: "the operator view, not a contributor-facing product."
- **No public / unauthenticated pages** beyond the existing `/status` JSON and
  `/healthz`. `/ui/login` is the only unauthenticated addition and it renders no
  data. No public dashboard, no unauthenticated leaderboard, no marketing
  surface.
- **No SPA, no client router, no JS build, no WebSocket.** SSE only; every
  mutation is a server round-trip.
- **No server-push mutation.** The stream cannot change state; it only tells the
  browser to re-read.

---

## Spine deviations

**None.** The ring-buffer transport avoids a new `events` table, so `05` is
untouched. `/ui/*`, `/v1/events`, `/v1/me`, `/v1/leaderboard`, and `/status`
staying JSON are all named in `06`. The `§6.2` "read-only HTML" note is
superseded by `06` (reconciliation above), not deviated from here.

### Cross-doc dependencies

- **Session cookie + CSRF token** — identity doc (`06`, placeholder provider).
  This doc assumes `SameSite` cookies and a per-session CSRF token surfaced to
  templates.
- **Enrollment token format / TTL** — identity doc (`05`, `enrollments`).
- **Accrual formula, `system_weight` function, good-standing gate** — ledger
  doc. This doc freezes only the `/v1/me` and `/v1/leaderboard` response shapes.
- **Constraint-predicate grammar** rendered on `/ui/jobs/{id}` and `/ui/queue` —
  scheduler doc.
- **Round signals behind the plugin boundary** — Phase A (`05`). The generic
  dashboard needs a job-level health signal that survives `stalls()` becoming
  partly type-private; per-round loss/divergence stays a `collab_lora_finetune`
  fragment.

---

## Frozen vs. open

**Frozen here:** the `/ui/*` page set and each page's role, reads, and `POST`
targets; the `/ui/*` `303-to-login` failure mode (vs. `/v1/*` `401`); the
`/v1/events` envelope schema (type set, monotonic `id`, one entity-id field, no
HTML on the wire) and per-subscriber emit-time authorization; ring-buffer
transport with `Last-Event-ID` replay and `sync`-event + idempotent-refetch
backfill; the `/v1/me` and `/v1/leaderboard` response shapes; no JS build,
vendored htmx + SSE extension, `script-src 'self'`.

**Yours to specify later:** Jinja layout and fragment decomposition; CSS;
keyset-cursor encoding; whether `Sortable.js` progressive enhancement ships in
C2 or never; dashboard widget selection and any thresholds beyond the
`status.py` constants; the SSE fallback-poll interval.
