"""The web UI with JavaScript actually executing (docs/12 "Still not verified").

Everything else in `test_webui.py` drives the app through `TestClient`, which
never runs a line of the vendored htmx. That suite therefore cannot see the one
class of bug this UI has already shipped three times: a form whose *encoding* is
wrong. `TestClient` supplies the body the endpoint wants, so it agrees with the
handler by construction -- a stand-in cannot disagree with the code under test
about what the real client sends (docs/10 §4, docs/11, docs/12).

So this suite asserts the **request the browser actually makes**, not htmx's
internals. Reading `htmx.config` or checking that `window.htmx` exists would
just be a different stand-in; intercepting the outgoing request and reading its
`content-type` is the thing no unit test can fake.

Three mechanisms cover every distinct way this UI talks to the server, and the
rest of the pages are these same three at different URLs:

  * `json-enc` + the CSRF header  -- machine enrollment (broken once)
  * plain form encoding           -- the submission form (the inverse rule)
  * SSE -> `hx-trigger="sse:..."` -- a change made by *another* client

The fourth check is not a test: `browser_page` fails any test that logged a
console error or an uncaught exception. A CSP that blocks the vendored scripts
does not fail an assertion -- it prints a violation to the console and leaves a
page that looks fine and does nothing. Collecting that in the fixture rather
than per-test is deliberate; a check you have to remember to write is the blind
spot this suite exists to close.

Marked ``slow`` and ``browser``: it starts uvicorn and a real Chromium.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.browser]

REPO_ROOT = Path(__file__).resolve().parents[1]

# A loud skip on purpose. A bare "no playwright" reason is how a suite quietly
# shrinks -- this session already found 20 tests skipping behind a stopped
# Docker daemon while the run still printed green (docs/11).
sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="playwright is not installed: the browser pass is NOT running. "
           "Install it with `pip install -e .[browser]` and "
           "`python -m playwright install chromium`.",
)

ADMIN_SECRET = "rootpw"


# ---------------------------------------------------------------------------
# The server under the browser
# ---------------------------------------------------------------------------


class _NoStore:
    """A store that fails loudly if anything reaches for it.

    ``bootstrap`` calls ``store.ensure_bucket()``, so the real factory cannot
    start without a live MinIO -- and none of the pages this suite drives touch
    object storage. Rather than assume that, this raises: if a ``/ui`` route
    ever does presign something, the test says so instead of passing against a
    fake that quietly answered.
    """

    def __getattr__(self, name: str):
        raise AssertionError(
            f"the browser fixture's store was asked for {name!r} -- a /ui route "
            "now touches object storage and this fixture needs a real one"
        )


def browser_app():
    """uvicorn factory for this suite (``--factory``).

    A near-copy of ``app.bootstrap`` minus ``ensure_bucket``. It lives here, and
    is named in the ``uvicorn`` argv below, so the subprocess imports this
    module rather than a scratch file that could drift from it.
    """
    from ganymede.coordinator.app import create_app
    from ganymede.coordinator.config import Settings
    from ganymede.coordinator.db import connect, init_schema

    settings = Settings.from_env()
    conn = connect(settings.db_path)
    init_schema(conn)
    conn.close()
    return create_app(settings, _NoStore())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def _seed(db_path: str) -> str:
    """One admin who is also an approved submitter, and two jobs."""
    from ganymede.coordinator.auth import hash_key
    from ganymede.coordinator.db import connect, init_schema
    from ganymede.coordinator.rounds import _iso, utcnow

    conn = connect(db_path)
    init_schema(conn)
    now = _iso(utcnow())
    uid = uuid.uuid4().hex
    conn.execute(
        """INSERT INTO contributors
             (id, name, key_hash, enabled, clearance, is_admin, auth_provider, created_at)
           VALUES (?, 'root', ?, 1, 'open', 1, 'local', ?)""",
        (uid, hash_key(ADMIN_SECRET), now),
    )
    conn.execute(
        "INSERT INTO submitters (user_id, status, decided_at) VALUES (?, 'approved', ?)",
        (uid, now),
    )
    # A real spec, not `{}`: enqueue re-reads it to plan tasks, and a job the
    # API could never have created would 500 there for a reason that has
    # nothing to do with what this suite is testing.
    spec = json.dumps({
        "model_ref": "hf://test-model",
        "shards": [{"ref": "s0", "rows": 4}],
        "output_prefix": "out/seeded",
        "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
    })
    for jid, status in (("job1", "draft"), ("job2", "queued")):
        conn.execute(
            """INSERT INTO jobs (id, owner_id, job_type, spec_json, image_id, status,
                        priority_rank, constraints_json, cancel_mode, created_at)
               VALUES (?, ?, 'batch_inference', ?, NULL, ?, 1, '{}', NULL, ?)""",
            (jid, uid, spec, status, now),
        )
    conn.commit()
    conn.close()
    return uid


@pytest.fixture(scope="module")
def live_ui(tmp_path_factory):
    """A real coordinator process with a seeded database."""
    tmp_path = tmp_path_factory.mktemp("browser")
    port = _free_port()
    db_path = str(tmp_path / "ui.db")
    owner = _seed(db_path)

    env = {
        **os.environ,
        "GANYMEDE_DB": db_path,
        # Never dialled: _NoStore raises before boto3 would. Settings.from_env
        # requires them to be set, so they are set.
        "STORAGE_HOST": "http://127.0.0.1:1",
        "S3_BUCKET": "unused",
        "S3_ACCESS_KEY": "unused",
        "S3_SECRET_KEY": "unused",
        "COORDINATOR_HOST": f"http://127.0.0.1:{port}",
        "GANYMEDE_REQUIRE_TLS": "0",
        "PYTHONPATH": str(REPO_ROOT),
    }

    log = tmp_path / "coordinator.log"
    handle = log.open("w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "tests.test_webui_browser:browser_app",
         "--factory", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT), env=env,
        stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for(f"{url}/healthz"):
            server.terminate()
            server.wait(timeout=5)
            handle.close()
            pytest.fail(f"coordinator did not start: {log.read_text()[-2000:]}")
        yield {"url": url, "db": db_path, "owner": owner, "log": log}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        handle.close()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - the reason must survive
            pytest.fail(
                "chromium is not installed: the browser pass is NOT running. "
                f"Run `python -m playwright install chromium`. ({exc})"
            )
        try:
            yield b
        finally:
            b.close()


@pytest.fixture
def browser_page(browser, live_ui):
    """A page whose console is an assertion.

    Every test gets a fresh context (so no session leaks between them) and fails
    if the page logged an error or threw. CSP blocking a vendored script is a
    console message, not an exception: without this the page would render, do
    nothing, and every DOM assertion below would time out with a message about
    a selector rather than about the policy that broke it.
    """
    context = browser.new_context(base_url=live_ui["url"])
    page = context.new_page()
    problems: list[str] = []
    # A test that expects a non-2xx answer declares it here. Chromium logs every
    # failed resource load as a console error and htmx logs its own "Response
    # Status Error Code", so a deliberate 422 would otherwise fail its own test.
    # Empty by default and named per test, so the strictness stays: nothing is
    # filtered by pattern, only by a status the test says it is asking for.
    page.tolerated_statuses = set()

    def on_console(msg):
        if msg.type != "error":
            return
        # The one blanket exemption, and it is not a judgement call: no template
        # links a favicon, so Chromium requests /favicon.ico on every navigation
        # and logs the 404.
        if "favicon.ico" in msg.text:
            return
        # Substring, not a parse: a message carrying those three digits for some
        # other reason would also be swallowed. Acceptable while one test uses
        # one status, and the narrower alternative is parsing Chromium's and
        # htmx's two different phrasings of the same fact.
        if any(str(code) in msg.text for code in page.tolerated_statuses):
            return
        problems.append(f"console.{msg.type}: {msg.text}")

    page.on("console", on_console)
    page.on("pageerror", lambda exc: problems.append(f"pageerror: {exc}"))
    try:
        yield page
    finally:
        context.close()
    assert problems == [], "the browser reported:\n  " + "\n  ".join(problems)


def _login(page, secret: str = ADMIN_SECRET) -> None:
    page.goto("/ui/login")
    page.fill("input[name=username]", "root")
    page.fill("input[name=secret]", secret)
    page.click("button[type=submit]")
    page.wait_for_url("**/ui/")


def _api(live_ui, method: str, path: str, body: dict | None = None) -> int:
    """A second client, outside the browser, authenticated by bearer key.

    Bearer callers are exempt from the ``X-Ganymede-UI`` CSRF header
    (``app._principal``), which is what makes this usable as "somebody else
    changed something" for the SSE test.
    """
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{live_ui['url']}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {ADMIN_SECRET}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


# ---------------------------------------------------------------------------
# The pages load, and the scripts CSP allows actually ran
# ---------------------------------------------------------------------------


def test_every_page_renders_with_htmx_running_and_a_clean_console(browser_page):
    """The load-bearing check, and the cheapest one.

    `script-src 'self'` is the policy the vendored scripts were chosen for
    (docs/12 "Stack"), but no browser had ever enforced it against them. If any
    of the three were blocked, this fails twice: `window.htmx` is undefined, and
    the fixture reports the violation Chromium logged.
    """
    page = browser_page
    _login(page)
    for path in ("/ui/", "/ui/jobs", "/ui/machines", "/ui/leaderboard",
                 "/ui/queue", "/ui/submitters", "/ui/jobs/job1"):
        page.goto(path)
        assert page.evaluate("typeof window.htmx") == "object", path


def test_the_sse_stream_connects_under_connect_src_self(browser_page, live_ui):
    """`hx-ext="sse"` on <body> opens an EventSource against `/v1/events`.

    Two ways this fails silently: the extension never registers (htmx ignores an
    unknown `hx-ext` and carries on), or `connect-src 'self'` refuses the
    stream. Both leave a page that renders correctly and never updates.
    """
    page = browser_page
    with page.expect_event("request", lambda r: r.url.endswith("/v1/events")) as info:
        _login(page)
    assert info.value.url.endswith("/v1/events")


# ---------------------------------------------------------------------------
# json-enc + the CSRF header -- the encoding that shipped broken
# ---------------------------------------------------------------------------


def test_enrolling_a_machine_sends_json_and_swaps_in_the_token(browser_page):
    """The exact form that was broken: `hx-post` to a `/v1` endpoint with a
    Pydantic body, which answers 422 to anything form-encoded.

    Asserting the request rather than the outcome is the point -- a 422 would
    also leave the DOM unchanged, so a DOM-only assertion could not tell "the
    encoder is missing" from "the swap target is wrong".
    """
    page = browser_page
    _login(page)
    page.goto("/ui/machines")

    with page.expect_request("**/v1/machines/enroll") as info:
        page.fill("input[name=display_name]", "gpu-box-1")
        page.click("#enroll button[type=submit]")
    request = info.value

    assert request.headers.get("content-type", "").startswith("application/json"), \
        "json-enc did not encode the body"
    assert request.headers.get("x-ganymede-ui") == "1", "the CSRF header is missing"
    assert json.loads(request.post_data)["display_name"] == "gpu-box-1"

    # hx-swap="outerHTML" against hx-target="#enroll": the same id comes back
    # carrying the one-time token.
    page.wait_for_selector("#enroll .token")
    assert "Machine enrolled" in page.inner_text("#enroll h2")
    assert page.inner_text("#enroll .token").strip() != ""


# ---------------------------------------------------------------------------
# Plain form encoding -- the inverse rule
# ---------------------------------------------------------------------------


def test_the_submission_form_posts_form_fields_and_swaps_the_card(browser_page):
    """`/ui/jobs/new` takes `Form()` fields, so this form must NOT carry
    `json-enc` -- the mirror image of the rule above, and a 422 in a browser
    either way round."""
    page = browser_page
    _login(page)
    page.goto("/ui/jobs")

    spec = {
        "model_ref": "hf://test-model",
        "shards": [{"ref": "s0", "rows": 4}],
        "output_prefix": "out/browser",
        "prompt_template": "{input}",
        "decode": {"mode": "greedy", "max_new_tokens": 4},
        "output_schema": {"id": "str", "output": "str"},
    }
    with page.expect_request("**/ui/jobs/new") as info:
        page.select_option("#new-job select[name=job_type]", "batch_inference")
        page.fill("#new-job textarea[name=spec]", json.dumps(spec))
        page.click("#new-job button[type=submit]")

    ctype = info.value.headers.get("content-type", "")
    assert ctype.startswith("application/x-www-form-urlencoded"), \
        f"the form sent {ctype!r} to an endpoint that takes Form() fields"

    page.wait_for_selector("#new-job .ok")
    assert "Created" in page.inner_text("#new-job .ok")


def test_a_malformed_spec_comes_back_in_the_card_with_the_typing_intact(browser_page):
    """The expected outcome of typing JSON into a textarea, and the reason this
    endpoint is a `/ui` POST at all: htmx swaps the response into the page, so a
    raw `/v1` error blob would land in front of a person."""
    page = browser_page
    # The endpoint answers 422 on purpose (test_webui.py asserts it), and htmx
    # swaps a /ui 422 only because static/ui.js opts it back in -- without that
    # script this click does nothing at all, which is what shipped.
    page.tolerated_statuses.add(422)
    _login(page)
    page.goto("/ui/jobs")

    page.fill("#new-job textarea[name=spec]", "{not json")
    page.click("#new-job button[type=submit]")

    page.wait_for_selector("#new-job .error")
    assert "not valid JSON" in page.inner_text("#new-job .error")
    # Re-rendered, not cleared: the swap replaced the card and the form still
    # holds what was typed.
    assert "{not json" in page.input_value("#new-job textarea[name=spec]")


# ---------------------------------------------------------------------------
# SSE -> hx-trigger, driven from outside the browser
# ---------------------------------------------------------------------------


def test_a_change_by_another_client_updates_the_open_page(browser_page, live_ui):
    """The highest-value test here, and the only one that proves the whole SSE
    path at once.

    The browser sits on `/ui/jobs`, whose list carries
    `hx-trigger="sse:job.status"` + `hx-get="/ui/frag/jobs-table"`. A *separate*
    client then enqueues a job over HTTP, which publishes `job.status`
    (`app.py`). For the row to change, four things must all be right: the
    EventSource connected, the hub's `event:` name matches the `sse:` trigger
    name in the template, the extension routes it to the element, and the swap
    target resolves. Any one of them wrong is a page that silently goes stale.

    The emit has to come over HTTP to the *same* process: `events.py` keeps the
    hub in module state, so calling `publish` in the test process would reach a
    hub nobody is subscribed to.
    """
    page = browser_page
    _login(page)
    # Wait for the stream to be live before changing anything, and prove it the
    # way the page itself does: `sse:sync` is emitted per subscriber on connect,
    # and every live fragment answers it with an idempotent refetch. That
    # response landing means there is a subscriber to publish to. Without this
    # the enqueue below races the EventSource and the test is a coin flip.
    with page.expect_response("**/ui/frag/jobs-table"):
        page.goto("/ui/jobs")
    row = page.locator("#job-row-job1")
    sync_api.expect(row).to_contain_text("draft")

    assert _api(live_ui, "POST", "/v1/jobs/job1/enqueue") == 200

    # This row, not the table: the tests share one database, so another test's
    # created draft is legitimately on the page and "no draft anywhere" would
    # assert the wrong thing.
    #
    # No reload, no click: the only thing that can change this text is the
    # stream. Playwright polls the expectation, so a stale page fails on time.
    sync_api.expect(row).to_contain_text("queued", timeout=15_000)
