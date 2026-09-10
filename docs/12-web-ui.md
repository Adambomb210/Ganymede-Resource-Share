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

## Status — built, and driven in a real browser

All three rollout steps are in: `ganymede/coordinator/webui.py`, the templates
under `coordinator/templates/`, vendored `htmx.min.js` / `sse.js` / `json-enc.js`,
and the SSE hub on `/v1/events`.

| Step | State |
|---|---|
| C1 — read-only operator view | **built.** All eight pages, `303`-to-login for anonymous, `404` (not `403`) cross-tenant, CSP `default-src 'self'` with no `unsafe-inline` |
| C2 — admin surface | **built** except job submission, below. Queue reorder, submitter approve/deny/revoke, job enqueue/cancel |
| C3 — account and machine management | **built.** Enroll (token inline, exactly once, never on a `GET`) and retire |
| `/v1/events` SSE | **built.** Ring buffer, `Last-Event-ID` replay, `sync` on connect, per-subscriber emit-time authorization |

Mutations post **directly to the frozen `/v1` endpoints** from htmx rather than
through `/ui/*` handlers, with `X-Ganymede-UI: 1` as the CSRF marker — so there
is one implementation of every action and the UI cannot drift from the API.
`POST /ui/login` and `/ui/logout` are the only `/ui` writers.

### The job submission form, and the one `/ui` POST that is not login

C2's "job submission ... landing here too" is now built, minimally:
`frags/new_job.html` on `/ui/jobs`, shown only to an approved submitter.
Job type (from `REGISTRY`), an optional image picked from the caller's own
finalized images, and the spec as JSON text. It creates a **draft**; enqueueing
stays the separate action it already was.

**It posts to `/ui/jobs/new`, not to `/v1/jobs`, and that is forced.** htmx's
`json-enc` encodes a form as a *flat* object of strings; `POST /v1/jobs` takes a
nested `spec` object. With `script-src 'self'` and no build step there is no way
for a browser form to produce that body — which is the likeliest reason this
page was specified and never built. So the spec arrives as text, is parsed
server-side, and goes on to `app.create_job`, which is the same function
`POST /v1/jobs` calls. **Nothing about which jobs are allowed lives in the UI.**

The second reason is the error path: a 422 from `/v1` swapped into the page by
htmx puts a raw JSON error blob in front of a person, and a malformed spec is
the *expected* outcome of typing JSON into a textarea, not an exceptional one.
The type's own `validate_spec` message already names the field, so it is shown
verbatim beside the input with the submitter's text preserved.

This is a **deviation from "each page's POST targets"** in Frozen vs. open,
recorded rather than quiet: one `/ui` writer beyond login, justified by an
encoding mismatch the frozen API cannot express through a CSP-compliant form.

**Still API-only: image upload.** The picker lists images the submitter already
uploaded; it does not upload one. A browser upload means PUTting to a presigned
URL, which needs JS beyond htmx — a real decision, not an oversight, and the
natural place for whatever larger submission system replaces this form.

### Four forms had the wrong encoding, and why nothing caught it

Found by starting the server and driving it over HTTP — not by reading it, and
not by the suite.

htmx posts `application/x-www-form-urlencoded` unless `json-enc` is on the
element or an ancestor. Every mutating `/v1` endpoint declares a Pydantic body,
so FastAPI answers **422 before the handler runs**. Machine enrollment and both
job-cancel buttons had no encoder: in a browser, enrolling a machine and
cancelling a job both failed. `queue/reorder` and `submitters/{id}` had it and
worked, and the `enqueue` button correctly omits it because that endpoint takes
no body — so the author had reasoned about the encoder and simply missed the two
forms that send fields.

Every test around them passed. `test_enroll_token_shown_once_via_html_fragment`
posts `json={"display_name": "box"}` through `TestClient` — it supplies exactly
what the browser would have had to produce. The test asserted the *endpoint*;
nobody asserted the *form*. That is the same shape as `10` §4's `FakeWorker` and
`11`'s fake container runner: a stand-in cannot disagree with the code under
test about what the real client sends.

`test_every_form_posts_the_encoding_its_endpoint_accepts` now checks the
templates against the app's own routes, **both ways**: a route whose
`body_field` is a JSON `Body` must have the encoder, and one whose `body_field`
is `Form` must not. Both directions are 422s in a browser and silent in a
`TestClient` suite. It earned the second direction immediately — the submission
form takes `Form` fields, and the first version of this test flagged it.

It reads `route.body_field` rather than the handler's signature deliberately:
`app.py` uses `from __future__ import annotations`, so every annotation is a
string and a naive `issubclass` check matches nothing and passes by being inert.
The test asserts its own detection is non-empty in both directions for exactly
that reason.

### The browser pass — six defects, none of them visible to the suite

`tests/test_webui_browser.py` drives a real Chromium against a real coordinator
process. It exists because everything else here runs through `TestClient`, which
never executes a line of the vendored htmx: the suite and the handler agree
about the request by construction, because the suite *is* the request. Six
things were wrong. Five were invisible in every existing test, and the sixth
could not have been found by reading either side alone.

**The suite is three mechanisms, not eight pages.** `json-enc` + the CSRF header
(enrollment), plain `Form` encoding (the submission form), and
SSE → `hx-trigger` driven by a *second* client. Every other page is one of those
three at a different URL, and a browser suite slow enough to skip is worse than
none. It asserts the **request the browser makes** — intercepted, its
`content-type` read — rather than htmx's internals, because introspecting
`htmx.config` would just be a different stand-in.

The load-bearing check is not a test: the page fixture fails any test whose
console logged an error or whose page threw. CSP blocking a script does not fail
an assertion — it prints a violation and leaves a page that renders correctly
and does nothing.

**1. Every page 500ed intermittently, and it was not a UI bug.** `get_conn` is a
*sync generator* dependency, so FastAPI runs its three phases through
`run_in_threadpool` separately: `__enter__`, the endpoint, `__exit__`. anyio
hands each whichever worker thread is idle, and sqlite3's own guard then raises
`ProgrammingError` on the first query or on `close()`. A browser opens several
connections at once for a page and its assets, which is what spreads the phases
across threads; `TestClient` drives everything through one portal thread and
cannot reproduce it. This affected `/v1` exactly as much as `/ui` — the whole
coordinator, under any real concurrency. `db.connect` grew `same_thread=False`
for this one caller. The phases are strictly sequential, so a connection is
still only ever used by one thread at a time; what is disabled is a check that
cannot tell "sequentially, on three threads" from "concurrently".

**2. Five live fragments deleted the element that listened for updates.** A
`hx-swap="outerHTML"` element whose fragment route returns only the *inner*
content replaces itself with content carrying no `id` and no `hx-trigger`. Since
`sse:sync` is emitted per subscriber on connect, that happened on page load: the
jobs list, machines list, queue, submitters table and rounds table each went
live for exactly one event and then silently stopped. Three fragments —
`fleet`, `job_header`, `job_row` — had it right, and the difference is where the
wrapper lives. The wrapper now lives in the fragment everywhere, which is the
only version that survives its own swap.

`test_every_sse_driven_fragment_replaces_itself_with_a_listener` fetches each
one and asserts the response carries the id and the trigger back. The defect is
in what the *next* event does, so neither response is wrong on its own — which
is why nothing saw it.

**3. `frags/rounds.html` had never been rendered, and could not be.** It was
missing an `{% endfor %}`. Jinja reports that when it *parses* the file, and
`{% include %}` parses lazily, so the page's `{% if rounds %}` guard meant a job
with no rounds never touched it — and no test ever built a job with a round. The
first live run would have 500ed the one page you watch while a run is live.

The guard **moved inside the wrapper** rather than going away: the section has
to exist from page load or the `sse:round.close` refetch has nothing to target
when the first round closes, but a `batch_inference` job must not sprout an
empty Rounds heading either (the read model above: types without `reduce` show
tasks only). Both are asserted now, and
`test_a_job_with_rounds_renders_its_rounds_table` is the first test in the suite
that makes the fragment render at all.

A seventh, found while reading rather than running, and folded in here because
it is the same neighbourhood: `hub.publish` returned the `Envelope` instead of
`env.id` on the one path where no subscriber is authorized, against its own
`-> int`. No caller reads the value, which is why it sat there.

**4. Enrollment negotiated on a header htmx does not send.** The endpoint keyed
the HTML fragment on `Accept: text/html`; htmx sets no `Accept` at all, so the
XHR default `*/*` won and the browser got JSON — which `hx-swap="outerHTML"`
then swapped in, replacing the `#enroll` section with the one-time token as raw
text. This survived the *previous* round of fixes: adding `json-enc` corrected
the request, and nobody ever exercised the response. It keys on `HX-Request`
now, which htmx always sends, with `Accept` still honoured for callers that do
ask. `test_enroll_token_shown_once_via_html_fragment` passes either way, because
it hands the endpoint `Accept: text/html` by hand.

**5. htmx does not swap a 4xx, so the submission form's error path did
nothing.** Typing a malformed spec and pressing the button produced no visible
change whatsoever — the response was logged as a "Response Status Error Code"
and discarded. That path is the entire reason `/ui/jobs/new` exists rather than
posting to `/v1/jobs`, and it had never run. `static/ui.js` opts **422 from a
`/ui/` path** back into swapping, and nothing else: a 422 from `/v1` is a
FastAPI validation blob, and putting that in front of a person is what this
endpoint was built to avoid. Vendored rather than inline, because `script-src
'self'` allows a served file and nothing else.

**6. CSP blocked htmx's own indicator stylesheet on every page load.** htmx
injects a `<style>` element for `.htmx-indicator` at startup; `style-src 'self'`
refuses it. Harmless today — no template uses an indicator — but it was a
console violation on every navigation, which is precisely the noise that hides
the next real one. `base.html` sets `includeIndicatorStyles: false` through the
`htmx-config` meta tag and the three rules moved into `style.css` verbatim.

**What the pass confirmed working**, which is worth recording because it was all
unverified: the three vendored scripts load and run under `script-src 'self'`;
`json-enc` produces `application/json` and `hx-headers` carries
`X-Ganymede-UI: 1`; the submission form correctly sends
`application/x-www-form-urlencoded` to its `Form` endpoint; the EventSource
connects under `connect-src 'self'`; `sse:` trigger names match the hub's
`event:` names, dots included; and a change made by an entirely separate client
reaches an open page and updates it without a reload.

**Cost.** Six seconds, six tests. Playwright is a separate `browser` extra
rather than part of `dev`, because a ~150 MB Chromium download does not belong
in the path of someone running the unit suite; without it the module skips with
a reason that says the browser pass is not running, rather than a bare "no
playwright".

**Still not exercised:** Firefox and Safari (Chromium only), any viewport but
the default, and the `Last-Event-ID` replay path — the tests reconnect nothing.
Image upload remains API-only.

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
