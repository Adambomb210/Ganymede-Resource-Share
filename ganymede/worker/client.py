"""The coordinator client (docs/02-architecture-v2.md 6.2), over the standard library.

`urllib` rather than `requests` or `httpx`, and that is a deliberate cost. The
worker-core layer is ~30 MB on top of an ~8 GB torch base (4.1), and its whole
value is that it can be rebuilt and shipped often without disturbing the pinned
torch underneath. Every dependency added here is a dependency in that layer, on
every contributor's machine, on three platforms, forever -- and this client makes
seven request shapes. Verified in M1: an external client speaking only `urllib`
runs a full round through the presigned URLs.

Retries and what is not retried
-------------------------------
Network failures are retried with exponential backoff plus jitter. HTTP status
codes mostly are not, and the exceptions are the interesting part:

- **409 and 410 are not errors.** They are the round-closed and lease-lost
  signals, and the loop acts on them (drop work, re-claim). Retrying either
  would mean a worker arguing with a coordinator that has already moved on.
- **401 and 403 are not retried.** A bad key does not become a good one, and a
  worker hammering an auth endpoint is exactly what a rate limiter is for.
- **5xx is retried**, because a coordinator restart (6.4) is expected to be
  survivable without losing the fleet.
"""

from __future__ import annotations

import json
import random
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

USER_AGENT = "ganymede-worker"

DEFAULT_TIMEOUT_SEC = 60
# Artifact transfers get their own, much longer timeout: a ~25 MB adapter over a
# volunteer's domestic uplink is a different animal from a JSON round trip, and
# one timeout covering both would either abort real uploads or let a wedged API
# call hang the loop.
DEFAULT_TRANSFER_TIMEOUT_SEC = 900

DEFAULT_MAX_RETRIES = 4
RETRY_BASE_SEC = 2.0
RETRY_MAX_SEC = 60.0

# Status codes that mean "try again shortly", as opposed to "this will never
# work" or "stop and do something else".
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class CoordinatorError(RuntimeError):
    """An HTTP response the client will not retry and cannot interpret."""

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"{status} from {url}: {body[:400]}")
        self.status = status
        self.body = body
        self.url = url


class RoundClosed(RuntimeError):
    """409: the round closed under us. Drop the work and re-claim (3.2)."""


class LeaseLost(RuntimeError):
    """410: the lease expired and the shard was re-issued. Drop the work."""


class GateRejected(RuntimeError):
    """422: the submission failed an acceptance gate (5.1)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Response:
    status: int
    body: dict[str, Any]
    headers: dict[str, str]


class CoordinatorClient:
    """One worker's conversation with one coordinator.

    Stateless beyond the base URL and key: every method is a request. The loop
    holds whatever state there is, which keeps this file testable against a stub
    server and keeps retry logic out of the loop.
    """

    def __init__(
        self,
        base_url: str,
        key: str,
        *,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        transfer_timeout: int = DEFAULT_TRANSFER_TIMEOUT_SEC,
        max_retries: int = DEFAULT_MAX_RETRIES,
        verify_tls: bool = True,
        opener: Any | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.transfer_timeout = transfer_timeout
        self.max_retries = max_retries
        # The TLS context has to be baked into the opener's HTTPSHandler:
        # `OpenerDirector.open` takes no `context` argument -- that belongs to
        # the module-level `urlopen`, and passing it here is a TypeError on
        # every single request.
        #
        # Disabling verification is an explicit escape hatch rather than a
        # global override, because it should be visible in one place; a
        # self-signed certificate on a home coordinator is a real
        # early-deployment case (6.5), and the alternative people reach for is
        # setting it process-wide, which then also silently covers HuggingFace
        # and the object store.
        if opener is not None:
            self._opener = opener
        elif verify_tls:
            self._opener = urllib.request.build_opener()
        else:
            self._opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
            )

    # ---------------- plumbing ----------------

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), RETRY_MAX_SEC))
                return
            except ValueError:
                pass  # HTTP-date form; fall through to backoff
        delay = min(RETRY_BASE_SEC * (2**attempt), RETRY_MAX_SEC)
        # Jitter, because a coordinator restart wakes every worker in the fleet
        # at once and a fixed backoff would have them all return in lockstep --
        # turning one outage into a thundering herd on every subsequent retry.
        time.sleep(delay * (0.5 + random.random() * 0.5))

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        expect: tuple[int, ...] = (200,),
    ) -> Response:
        url = f"{self.base_url}{path}"
        payload = json.dumps(body).encode() if body is not None else None

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=payload, method=method)
            request.add_header("Authorization", f"Bearer {self.key}")
            request.add_header("User-Agent", USER_AGENT)
            if payload is not None:
                request.add_header("Content-Type", "application/json")

            try:
                with self._opener.open(request, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return Response(
                        status=resp.status,
                        body=json.loads(raw) if raw else {},
                        headers={k.lower(): v for k, v in resp.headers.items()},
                    )
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode(errors="replace")
                if exc.code in expect:
                    return Response(
                        status=exc.code,
                        body=json.loads(raw) if raw.strip() else {},
                        headers={k.lower(): v for k, v in exc.headers.items()},
                    )
                self._raise_for_status(exc.code, raw, url)
                if exc.code in RETRY_STATUSES and attempt < self.max_retries:
                    last_error = exc
                    self._sleep(attempt, exc.headers.get("Retry-After"))
                    continue
                raise CoordinatorError(exc.code, raw, url) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt)

        raise CoordinatorError(0, f"network failure after {self.max_retries} retries: "
                                  f"{last_error}", url)

    @staticmethod
    def _raise_for_status(status: int, body: str, url: str) -> None:
        """Turn the statuses that *mean* something into their own exceptions.

        409 and 410 are protocol, not failure: the coordinator is telling the
        worker its work no longer counts. Conflating them with 500s would make
        the loop retry a round that has already closed.
        """
        if status == 409:
            raise RoundClosed(body)
        if status == 410:
            raise LeaseLost(body)
        if status == 422:
            raise GateRejected(body)

    # ---------------- endpoints ----------------

    def manifest(self) -> dict[str, Any]:
        return self.request("GET", "/v1/manifest").body

    def register(self, compute_profile: dict[str, Any], image_tag: str | None = None) -> dict[str, Any]:
        return self.request(
            "POST", "/v1/workers/register",
            {"compute_profile": compute_profile, "image_tag": image_tag},
        ).body

    def claim(
        self,
        worker_id: str,
        *,
        capabilities: dict[str, Any] | None = None,
        cached_base_models: list[str] | None = None,
        run_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Ask for work.

        Returns ``(task, retry_after_sec)``. A ``None`` task with a retry delay is
        the **204** case and is entirely normal (6.2 L4): there is no eligible
        work right now, which on an unscheduled volunteer fleet is the common
        state rather than an error.
        """
        body: dict[str, Any] = {"worker_id": worker_id}
        if capabilities is not None:
            body["capabilities"] = capabilities
        if cached_base_models:
            body["cached_base_models"] = cached_base_models
        if run_id:
            body["run_id"] = run_id

        resp = self.request("POST", "/v1/tasks/claim", body, expect=(200, 204))
        if resp.status == 204:
            return None, _retry_after(resp.headers)
        return resp.body, 0

    def heartbeat(self, task_id: str, steps_completed: int,
                  loss_ewma: float | None = None) -> dict[str, Any]:
        return self.request(
            "POST", f"/v1/tasks/{task_id}/heartbeat",
            {"steps_completed": steps_completed, "loss_ewma": loss_ewma},
        ).body

    def upload_url(self, task_id: str) -> dict[str, Any]:
        return self.request("POST", f"/v1/tasks/{task_id}/upload-url", {}).body

    def submit(self, task_id: str, artifact_key: str, steps_completed: int,
               tokens_seen: int = 0, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request(
            "POST", f"/v1/tasks/{task_id}/submit",
            {"artifact_key": artifact_key, "steps_completed": steps_completed,
             "tokens_seen": tokens_seen, "metrics": metrics or {}},
        ).body

    def abandon(self, task_id: str) -> dict[str, Any]:
        return self.request("POST", f"/v1/tasks/{task_id}/abandon", {}).body

    # ---------------- object storage ----------------
    #
    # Presigned URLs are pre-authenticated, so these deliberately carry no
    # Authorization header: sending the coordinator bearer token to the object
    # store would leak it to a second service for no benefit. They also do not
    # go through `request`, since the payloads are megabytes of adapter rather
    # than JSON and the retry semantics differ.

    def download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, method="GET")
                request.add_header("User-Agent", USER_AGENT)
                with self._opener.open(request, timeout=self.transfer_timeout) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt)
        raise CoordinatorError(0, f"download failed after {self.max_retries} retries: "
                                  f"{last_error}", url)

    def upload(self, url: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """PUT to a presigned URL.

        Not retried on an HTTP error, only on a network one. A presigned PUT that
        comes back 403 has almost always expired or been signed against a
        different hostname (6.6's footgun), and retrying an expired signature
        just spends the round's remaining time proving it is still expired.
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, data=data, method="PUT")
                request.add_header("Content-Type", content_type)
                request.add_header("User-Agent", USER_AGENT)
                with self._opener.open(request, timeout=self.transfer_timeout):
                    return
            except urllib.error.HTTPError as exc:
                raise CoordinatorError(exc.code, exc.read().decode(errors="replace"), url) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._sleep(attempt)
        raise CoordinatorError(0, f"upload failed after {self.max_retries} retries: "
                                  f"{last_error}", url)


def _retry_after(headers: dict[str, str]) -> int:
    raw = headers.get("retry-after")
    if not raw:
        return 0
    try:
        return max(0, int(float(raw)))
    except ValueError:
        return 0
