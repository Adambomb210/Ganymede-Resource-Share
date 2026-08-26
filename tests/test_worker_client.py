"""The coordinator client: what it retries, and what it deliberately does not.

The retry policy is the whole substance of this module, and most of it is about
*not* retrying. 409 and 410 are protocol rather than failure -- the coordinator
telling a worker its work no longer counts -- and a client that treated them as
transient would have workers arguing with a round that has already closed.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from ganymede.worker import client as client_mod
from ganymede.worker.client import (
    CoordinatorClient,
    CoordinatorError,
    GateRejected,
    LeaseLost,
    RoundClosed,
)


class FakeResponse(io.BytesIO):
    def __init__(self, status: int, body: dict | None = None, headers: dict | None = None):
        super().__init__(json.dumps(body).encode() if body is not None else b"")
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeOpener:
    """Replays a scripted sequence of responses and records what was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        result = self.responses.pop(0) if self.responses else FakeResponse(200, {})
        if isinstance(result, Exception):
            raise result
        return result


def _http_error(code: int, body: str = "", headers: dict | None = None):
    return urllib.error.HTTPError(
        "http://c/x", code, "err", headers or {}, io.BytesIO(body.encode())
    )


@pytest.fixture
def no_sleep(monkeypatch):
    """Retries are real; waiting for them is not worth six seconds a test."""
    monkeypatch.setattr(client_mod.time, "sleep", lambda _: None)


def make_client(*responses, **kwargs) -> tuple[CoordinatorClient, FakeOpener]:
    opener = FakeOpener(*responses)
    return CoordinatorClient("http://c", "key", opener=opener, **kwargs), opener


# --------------------------------------------------------------------------
# Auth and shape
# --------------------------------------------------------------------------


def test_every_request_carries_the_bearer_token():
    client, opener = make_client(FakeResponse(200, {"ok": True}))
    client.request("GET", "/v1/manifest")
    assert opener.requests[0].get_header("Authorization") == "Bearer key"


def test_presigned_transfers_deliberately_omit_the_token():
    """The presigned URL is already authenticated.

    Sending the coordinator bearer token to the object store would hand a second
    service a credential it has no use for, on every round, from every worker.
    """
    client, opener = make_client(FakeResponse(200), FakeResponse(200))
    client.download("http://storage/obj?X-Amz-Signature=abc")
    client.upload("http://storage/obj?X-Amz-Signature=abc", b"bytes")

    for request in opener.requests:
        assert request.get_header("Authorization") is None


# --------------------------------------------------------------------------
# The statuses that mean something
# --------------------------------------------------------------------------


def test_409_is_round_closed_not_a_failure(no_sleep):
    client, opener = make_client(_http_error(409, "round closed"))
    with pytest.raises(RoundClosed):
        client.heartbeat("t1", 5)
    assert len(opener.requests) == 1  # not retried


def test_410_is_lease_lost(no_sleep):
    client, _ = make_client(_http_error(410, "lease expired"))
    with pytest.raises(LeaseLost):
        client.heartbeat("t1", 5)


def test_422_is_a_gate_rejection_carrying_its_reason(no_sleep):
    client, _ = make_client(_http_error(422, "norm_outlier"))
    with pytest.raises(GateRejected) as caught:
        client.submit("t1", "key", 10)
    assert "norm_outlier" in caught.value.reason


def test_401_is_not_retried(no_sleep):
    """A bad key does not become a good one, and a worker hammering an auth
    endpoint is exactly what a rate limiter exists to stop."""
    client, opener = make_client(*[_http_error(401, "unknown key")] * 5)
    with pytest.raises(CoordinatorError) as caught:
        client.manifest()
    assert caught.value.status == 401
    assert len(opener.requests) == 1


def test_503_is_retried_then_succeeds(no_sleep):
    """A coordinator restart (6.4) must be survivable without losing the fleet."""
    client, opener = make_client(
        _http_error(503), _http_error(503), FakeResponse(200, {"worker_id": "w1"})
    )
    assert client.register({"backend": "cpu"})["worker_id"] == "w1"
    assert len(opener.requests) == 3


def test_a_network_failure_is_retried_then_reported(no_sleep):
    client, opener = make_client(*[urllib.error.URLError("connection refused")] * 9,
                                 max_retries=3)
    with pytest.raises(CoordinatorError, match="network failure"):
        client.manifest()
    assert len(opener.requests) == 4  # the first try plus three retries


# --------------------------------------------------------------------------
# Claim: 204 is the normal case
# --------------------------------------------------------------------------


def test_204_means_no_work_not_an_error():
    """On an unscheduled volunteer fleet this is the common state (3.2)."""
    client, _ = make_client(_http_error(204, "", {"Retry-After": "45"}))
    task, retry_after = client.claim("w1")
    assert task is None
    assert retry_after == 45


def test_204_without_a_retry_after_leaves_the_delay_to_the_caller():
    client, _ = make_client(_http_error(204, "", {}))
    task, retry_after = client.claim("w1")
    assert task is None and retry_after == 0


def test_an_unparseable_retry_after_does_not_crash_the_claim():
    """It may legitimately be an HTTP-date rather than a number."""
    client, _ = make_client(_http_error(204, "", {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}))
    assert client.claim("w1") == (None, 0)


def test_200_returns_the_task_spec():
    client, opener = make_client(FakeResponse(200, {"task_id": "t1", "buckets": [1, 2]}))
    task, retry_after = client.claim("w1", capabilities={"backend": "cuda"},
                                     cached_base_models=["Qwen/Qwen3-1.7B-Base"])
    assert task["task_id"] == "t1" and retry_after == 0

    body = json.loads(opener.requests[0].data)
    assert body["worker_id"] == "w1"
    assert body["capabilities"]["backend"] == "cuda"
    # 6.2: the coordinator prefers a run whose base model this worker already
    # holds -- the difference between starting in seconds and after a 16 GB pull.
    assert body["cached_base_models"] == ["Qwen/Qwen3-1.7B-Base"]


# --------------------------------------------------------------------------
# Transfers
# --------------------------------------------------------------------------


def test_upload_puts_the_bytes():
    client, opener = make_client(FakeResponse(200))
    client.upload("http://storage/obj", b"adapter-bytes")
    request = opener.requests[0]
    assert request.method == "PUT"
    assert request.data == b"adapter-bytes"


def test_an_expired_presigned_url_fails_immediately(no_sleep):
    """A 403 on a presigned PUT has almost always expired or been signed against
    a different hostname (6.6's footgun). Retrying just spends the round's
    remaining time proving the signature is still expired."""
    client, opener = make_client(*[_http_error(403, "SignatureDoesNotMatch")] * 5)
    with pytest.raises(CoordinatorError) as caught:
        client.upload("http://storage/obj", b"x")
    assert caught.value.status == 403
    assert len(opener.requests) == 1


def test_a_flaky_download_is_retried(no_sleep):
    client, opener = make_client(
        urllib.error.URLError("reset"), FakeResponse(200, {"a": 1})
    )
    assert client.download("http://storage/obj")
    assert len(opener.requests) == 2


def test_transfers_get_their_own_longer_timeout():
    """A ~25 MB adapter over a domestic uplink is not a JSON round trip, and one
    timeout covering both would either abort real uploads or let a wedged API
    call hang the loop."""
    client, _ = make_client()
    assert client.transfer_timeout > client.timeout


def test_disabling_tls_verification_builds_its_own_opener():
    """`OpenerDirector.open` takes no `context` argument -- that belongs to the
    module-level `urlopen` -- so the context has to be baked into the handler."""
    insecure = CoordinatorClient("https://c", "k", verify_tls=False)
    secure = CoordinatorClient("https://c", "k")
    assert insecure._opener is not secure._opener
