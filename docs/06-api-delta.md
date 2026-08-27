# Ganymede — API Delta (spine doc 2 of 2)

*The other half of the frozen spine (`04-platform-expansion.md`, Build
sequencing; `05-data-model.md` is doc 1). A component design doc may specify
request/response bodies beyond what is named here and add fields, but **paths,
methods, the auth class of each endpoint, the `/v1` prefix discipline, and the
heartbeat `cancel` field are fixed.***

*Stage 1 status: the six component docs (`07`–`12`) are written. Endpoints they
added and the one conflict they surfaced (CSRF) are folded in — see
[Stage 1 reconciliation](#stage-1-reconciliation--endpoints-added-after-the-component-docs).*

Delta against `ganymede/coordinator/app.py` as of `a1b4e36`.

---

## Conventions carried forward

Unchanged and non-negotiable:

- **`/v1` prefix**, additive only. No endpoint below changes an existing
  response shape; every new field is optional or on a new path.
- **Bearer auth, TLS mandatory** (`require_contributor` in `app.py`;
  `x-forwarded-proto` for the proxy case).
- **404, not 403, on cross-tenant** — a resource another user owns must be
  indistinguishable from one that does not exist (`_worker_for_task`,
  `worker_eligibility`).
- **Big blobs never transit the API body.** Weights use presigned URLs today;
  **container images use the same pattern** (presigned PUT/GET against the object
  store).
- **`204 + Retry-After`** stays the "nothing for you right now" answer on claim.
- `/metrics` is still not built and still out of scope.

---

## Auth: three token kinds

`authenticate()` in `auth.py` gains two paths beside the existing contributor
key:

| Token | Backs | Used by | Resolves to |
| --- | --- | --- | --- |
| contributor key (`contributors.key_hash`) | existing | CLI, operators, **admins** (`is_admin`) | `Contributor` |
| machine key (`machine_keys.key_hash`) | new (`05`) | the worker process | `Machine` |
| session cookie / token | new | the web UI, via the placeholder provider | `Contributor` |

Endpoint auth classes below: **machine**, **user** (any authenticated
contributor), **submitter** (`submitters.status = approved`), **admin**
(`contributors.is_admin = 1`).

### Placeholder identity provider (Decision 5)

- `POST /v1/auth/session` — body `{username, secret}` (local table; `secret` is
  the contributor's existing issued key, hashed to `contributors.key_hash`), sets
  a session token — `Set-Cookie` (HttpOnly, Secure, SameSite=Lax) **and** in the
  body for non-browser callers. Real OAuth/OIDC later swaps the body and the
  verification, not the endpoint. **user**-issuing.
- `DELETE /v1/auth/session` — logout / revoke the calling session. **user**.
  (Stage 1 reconciliation, from `08`.)
- First admin is **not** created through the API — it is named by
  `GANYMEDE_BOOTSTRAP_ADMIN` at coordinator start (Decision 14), which upserts
  one `contributors` row with `is_admin = 1`.

**CSRF** (Stage 1 reconciliation — `08` and `12` had picked different schemes;
`08`'s stands): bearer callers are immune; session-cookie callers are guarded by
`SameSite=Lax` **plus** a static `X-Ganymede-UI: 1` request header that htmx sets
globally and a cross-origin form cannot forge. State-changing `/ui/*` and
cookie-authed `/v1/*` requests must carry it. No per-session CSRF token, no token
store.

---

## Machine enrollment (replaces fingerprint registration)

`app.py:register` derives `worker_id` from `(contributor, device fingerprint)`.
That becomes:

| Method | Path | Auth | Body → Response |
| --- | --- | --- | --- |
| POST | `/v1/machines/enroll` | user | `{display_name}` → `{enroll_token}` (shown once) |
| POST | `/v1/machines/claim-enrollment` | — (token in body) | `{enroll_token, compute_profile}` → `{machine_id, machine_key}` |
| POST | `/v1/workers/register` | machine | **kept, deprecated** — re-probe path; updates `compute_profile_json` only, never mints identity |
| GET | `/v1/machines` | user | caller's machines + `standing` + accrued hours |
| POST | `/v1/machines/{id}/retire` | user | owner removes a machine; accrual stops, ledger rows stay |
| POST | `/v1/machines/{id}/rotate-key` | user (owner) | issue a new machine key, disable the old — for a leaked key with a live machine still behind it (Stage 1 reconciliation, from `08`) |

`claim-enrollment` writes the `workers` row (`id` = minted, not derived),
`machine_keys` row, and consumes the `enrollments` row (`05`).

---

## Claim path — now a queue walk across all jobs

`POST /v1/tasks/claim` — **auth: machine** (identity from the machine key; the
`worker_id` body field becomes optional / ignored).

Behaviour change:

- Iterates **queued jobs in `priority_rank` order** (Decision 10), not just
  `status='active'` runs. `_selectable_runs` generalises to `_selectable_jobs`,
  keeping the cache-affinity tiebreak within a rank.
- For each job, applies `constraints_json` (Decision 15) **before** the existing
  `budget.is_eligible` predicate check:
  - `{"machine_ids": [...]}` → this machine must be in the list;
  - predicate → `vram_gb`, `gpu_model`, `os`, `region`, … compared against the
    probe profile. New refusal reasons, recorded verbatim in `worker_eligibility`
    (now keyed by `job_id`).
- **Capability backfill**: a machine no high-rank job accepts still gets the
  first lower-rank job whose constraints it meets. This falls out of walking the
  ordered list and not stopping at the first refusal.
- One task at a time per machine (Decision 4) — unchanged, now explicit.

Response payload (`_task_payload`) gains, alongside today's fields:

| Field | Meaning |
| --- | --- |
| `job_id`, `job_type` | which type's worker code runs |
| `image_ref`, `image_digest`, `image_pull_url` | presigned GET for the payload container (Decision 18); `null` for first-party built-ins |
| `input_ref` | generic input handle; `buckets` still present for `collab_lora_finetune` |

`base_adapter_url`, `seed`, `local_steps`, `lora_cfg`, `hyperparams` etc. remain
exactly as they are for the training type.

---

## Task lifecycle — generalised, one new field

`/v1/tasks/{id}/heartbeat`, `/upload-url`, `/submit`, `/abandon` keep their
paths, methods, auth (**machine**), and response shapes. Two changes:

1. **Heartbeat carries the kill (Decision 18).** Response, today
   `{lease_expires_at}`, may now also include:
   ```json
   { "lease_expires_at": "...", "cancel": "soft" | "hard" }
   ```
   `soft` → the worker finishes or checkpoints the current unit, then exits.
   `hard` → the worker aborts now. This is the whole cancellation transport —
   pull-only (Decision 8), so worst-case latency is one `heartbeat_interval_sec`.
   The existing `409 RoundClosed` / `410 LeaseLost` are unchanged.

2. **`submit` runs the job type's `validate()`**, not `closer.gate_submission`
   directly. Response keys (`accepted`, `reject_reason`, `next_action`,
   `round_closed`) are unchanged; `round_closed` is meaningful only for types
   with a `reduce` step.

---

## Job submission & management (new)

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| POST | `/v1/images/upload-url` | submitter | → `{image_id, url, digest_required}`; presigned PUT to object store |
| POST | `/v1/images/{id}/finalize` | submitter | worker-visible only after this; sets `scan_status='pending'` |
| GET | `/v1/images` | submitter | caller's uploads |
| POST | `/v1/jobs` | submitter | `{job_type, spec, image_id, constraints}` → `{job_id, status:'draft'}`; job type validates `spec` |
| POST | `/v1/jobs/{id}/enqueue` | submitter | `draft → queued`; admin sets `priority_rank` |
| GET | `/v1/jobs` / `/v1/jobs/{id}` | user | own jobs; admin sees all |
| POST | `/v1/jobs/{id}/cancel` | submitter (own) / admin | `{mode: "soft" \| "hard"}` → sets `jobs.cancel_mode`, propagates to leased tasks via next heartbeat |

---

## Admin surface (auth: admin)

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/v1/admin/queue` | jobs in `priority_rank` order, with leased-task counts |
| POST | `/v1/admin/queue/reorder` | `{job_id, before \| after \| rank}` — the only way `priority_rank` is set (Decision 13) |
| POST | `/v1/admin/submitters/{user_id}` | `{status: "approved" \| "denied" \| "revoked", note}` (Decisions 3, 9) |
| POST | `/v1/admin/jobs/{id}/cancel` | `{mode}` — same as owner cancel, any job |
| POST | `/v1/admin/images/{id}/scan` | `{disposition: "clean" \| "flagged" \| "rescan", note}` — disposition a `flagged` image or force a re-scan (Stage 1 reconciliation, from `11`) |
| GET | `/v1/admin/fleet` | existing `fleet_summary` refusal grouping + standings |

Revoking a submitter does **not** auto-kill their running jobs — the admin
cancels those explicitly with a chosen `mode` (Decision 18). A revoked submitter
simply cannot enqueue more.

---

## Observability / web-UI read model

| Method | Path | Auth | Change |
| --- | --- | --- | --- |
| GET | `/status` | none | **kept JSON.** Add `jobs: [{id, job_type, status, priority_rank}]` beside `runs`. |
| GET | `/v1/fleet` | user | add per-machine `standing`, `weighted_hours_total`, `system_weight` |
| GET | `/v1/me` | user | caller's machines, standings, accrued Weighted System Hours, recent `credit_events`. **Field list owned by `09`** (the richer shape); `12` renders it. |
| GET | `/v1/leaderboard` | user | machines / users by `SUM(weighted_hours) WHERE kind='provisioned'`; the point of the whole ledger (Decision 7). `?scope=machines\|users`, keyset-paginated. Fields owned by `09`. The one place the cross-tenant 404 rule is relaxed — rank + display name + hours only, no machine internals. |
| GET | `/v1/runs/{id}/rounds/current` | user | **unchanged**; documented as `collab_lora_finetune`-specific |
| GET | `/v1/events` | user | **new** — SSE stream, session-cookie auth (`EventSource` can't set headers). Envelope schema (tiny, no HTML on the wire; `job.status` / `round.close` / `fleet.delta` / `standing.change` / `queue.change` / `submitter.change`) and ring-buffer reconnect are `12`'s. Per-subscriber emit-time authorization — a non-admin never receives `queue.change`. |

### htmx pages (Decision 17)

Server-rendered, served under `/ui/*`, separate namespace from `/v1/*` JSON,
same FastAPI app, same session auth. Inventory (read-only first, then the rest):

`/ui/` dashboard · `/ui/jobs` + `/ui/jobs/{id}` · `/ui/queue` (admin reorder) ·
`/ui/submitters` (admin) · `/ui/machines` (my machines + enrollment) ·
`/ui/leaderboard`. No JS build; htmx + SSE extension only.

---

## Versioning / compatibility

Everything above is additive. `register`, `/runs/{id}/rounds/current`, the
task-lifecycle response shapes, and `/status`'s existing keys are preserved so a
v1 worker and the existing `ganymede-status` CLI keep working through the
transition. `API_VERSION` stays `v1`.

---

## Frozen vs. open for component docs

**Frozen:** every path, method, and auth class above; the three token kinds; the
heartbeat `cancel: soft|hard` field and its pull-delivery semantics; images going
through the object store, not the body; `priority_rank` being admin-write-only;
the `SameSite` + `X-Ganymede-UI` CSRF scheme.

**Specified in the Stage 1 component docs:** request/response bodies beyond the
named fields; the SSE envelope schema and reconnect (`12`); the enrollment token
format/TTL, session mechanism, and `authenticate()` return type (`08`); the
constraint predicate grammar (`07`); image size caps, `scan_status` transitions,
and pull verification (`11`); `/v1/me` and `/v1/leaderboard` field lists (`09`
owns the fields, `12` the envelope).

---

## Stage 1 reconciliation — endpoints added after the component docs

1. **`DELETE /v1/auth/session`** (`08`) — session logout. **user**.
2. **`POST /v1/machines/{id}/rotate-key`** (`08`) — leaked-key recovery without
   retiring the machine. **user (owner)**.
3. **`POST /v1/admin/images/{id}/scan`** (`11`) — admin disposition of a
   `flagged` image, or forced re-scan. **admin**.
4. **CSRF scheme fixed** — `SameSite=Lax` + a static `X-Ganymede-UI: 1` header
   (`08`), not a per-session token (`12`'s alternative, dropped).
5. **`/v1/me` / `/v1/leaderboard` ownership split** — `09` owns the field lists
   (its shape is the richer one), `12` owns the envelope and rendering.
6. `/status` and `02`'s §6.2 "read-only HTML" note: `06` already superseded it —
   `/status` stays JSON + gains `jobs`; the HTML operator view is `/ui/*`.
