"""A reusable fake worker client for the coordinator's integration suite.

This wraps the raw HTTP calls a real ``ganymede-worker`` would make (register,
claim, heartbeat, submit, abandon) so scenario tests read as "worker does X"
rather than as a wall of ``client.post(...)`` calls. It talks to the app
through the same ``TestClient`` + ``FakeStore`` fixtures every other test
uses -- there is no separate transport here, just a thinner call surface.

The one liberty taken with the real worker protocol: a real worker PUTs its
artifact bytes through the presigned URL the coordinator hands back from
``upload-url``. ``FakeStore`` doesn't serve HTTP, so ``submit()`` fetches the
derived key from that endpoint (exercising the real endpoint) and then writes
the bytes straight into the store, which is exactly what conftest's
docstring says the presigned PUT stands in for in this suite.
"""

from __future__ import annotations

import io
from typing import Any

import torch
from safetensors.torch import save as _st_save

from ganymede.jobtypes.collab_lora_finetune.aggregate import save_adapter

# --------------------------------------------------------------------------
# Adapter construction
# --------------------------------------------------------------------------
#
# Deliberately duplicated from conftest.make_adapter rather than imported: this
# module is meant to be usable standalone (it is its own file with its own
# ownership boundary), and importing conftest would tie its behaviour to
# pytest's collection machinery for no benefit.


def _base_adapter(scale: float = 1.0, seed: int | None = None,
                   dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """The same miniature two-layer adapter shape conftest.make_adapter uses."""
    g = torch.Generator().manual_seed(seed if seed is not None else 0)
    return {
        f"layers.{i}.{proj}.lora_{ab}.weight": (
            torch.randn(4, 8, generator=g) * scale
        ).to(dtype)
        for i in range(2)
        for proj in ("q_proj", "v_proj")
        for ab in ("A", "B")
    }


_CORRUPT_KINDS = (
    "nan", "inf", "wrong_shape", "missing_key", "extra_key",
    "wrong_dtype", "pickle", "huge_norm",
)


def _corrupt_bytes(adapter: dict[str, torch.Tensor], kind: str) -> bytes:
    """Build a deliberately bad artifact, one of ``_CORRUPT_KINDS``."""
    if kind not in _CORRUPT_KINDS:
        raise ValueError(f"unknown corrupt kind: {kind!r} (want one of {_CORRUPT_KINDS})")

    if kind == "pickle":
        # torch.save's pickle format, NOT safetensors -- gate 1 must reject
        # this without ever unpickling it (aggregate.load_adapter's whole
        # reason for existing). See test_aggregate.py's equivalent case.
        buf = io.BytesIO()
        torch.save(adapter, buf)
        return buf.getvalue()

    adapter = {k: v.clone() for k, v in adapter.items()}
    first_key = sorted(adapter)[0]

    if kind == "nan":
        adapter[first_key][0, 0] = float("nan")
    elif kind == "inf":
        adapter[first_key][0, 0] = float("inf")
    elif kind == "wrong_shape":
        adapter[first_key] = torch.zeros(adapter[first_key].shape[0] + 1,
                                         adapter[first_key].shape[1])
    elif kind == "missing_key":
        del adapter[first_key]
    elif kind == "extra_key":
        adapter["bogus.extra.lora_A.weight"] = torch.zeros(4, 8)
    elif kind == "wrong_dtype":
        adapter[first_key] = adapter[first_key].to(torch.float16)
    elif kind == "huge_norm":
        # ~50x every other worker's scale -- a blown-up local update, the
        # shape the cohort norm gate (5.1 gate 4) exists to catch.
        adapter = {k: v * 50.0 for k, v in adapter.items()}

    return _st_save(adapter)


class APIResult(dict):
    """A JSON response body that also carries its HTTP status code.

    Lets a test read the payload like a normal dict (``r["reject_reason"]``)
    while still asserting on the transport-level outcome (``r.status_code``),
    which is exactly what the acceptance-gate and lease-failure tests need --
    a 409/410/422 response still has a body worth reading (FastAPI's
    ``{"detail": ...}``), and forcing every caller to unpack a tuple would be
    noise for the common (200-with-a-payload) case.
    """

    status_code: int


def _wrap(resp) -> APIResult:
    try:
        body = resp.json() if resp.content else {}
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {"value": body}
    result = APIResult(body)
    result.status_code = resp.status_code
    return result


class FakeWorker:
    """One simulated worker: a client, a store, a contributor key, a profile.

    Holds at most one claimed task at a time, matching the real protocol (one
    lease per worker -- rounds.claim_task's "resume, don't fork" rule).
    """

    def __init__(self, client, store, key: str, *, device: str = "RTX 3060",
                 backend: str = "cuda", vram_mb: int = 12288,
                 supports: list[str] | None = None, probe: dict | None = None) -> None:
        self.client = client
        self.store = store
        self.key = key
        self.device = device
        self.backend = backend
        self.vram_mb = vram_mb
        self.supports = supports if supports is not None else ["bf16", "fp16", "nf4"]
        self.probe = probe if probe is not None else {
            "alloc_max_mb": vram_mb - 1000, "bench_score": 40.0,
        }
        self.headers = {"Authorization": f"Bearer {key}"}
        self.worker_id: str | None = None
        self.task: dict | None = None  # last claimed task payload
        self.last_response: Any = None  # raw httpx.Response from the last call

    # -- profile / registration -------------------------------------------

    def _profile(self) -> dict:
        return {
            "backend": self.backend, "device_name": self.device, "vram_mb": self.vram_mb,
            "supports": self.supports, "probe": self.probe,
        }

    def register(self) -> str:
        resp = self.client.post(
            "/v1/workers/register", headers=self.headers,
            json={"compute_profile": self._profile()},
        )
        self.last_response = resp
        assert resp.status_code == 200, resp.text
        self.worker_id = resp.json()["worker_id"]
        return self.worker_id

    # -- lifecycle ----------------------------------------------------------

    def claim(self, run_id: str | None = None) -> dict | None:
        """POST /tasks/claim. Returns the task payload, or None on 204.

        ``self.last_response`` always carries the raw response afterwards, so
        a test that needs the 204's ``Retry-After`` header (or any other
        transport detail) can still get at it even though the convenience
        return value collapses 204 to ``None``.
        """
        if self.worker_id is None:
            self.register()
        body: dict[str, Any] = {"worker_id": self.worker_id}
        if run_id is not None:
            body["run_id"] = run_id
        resp = self.client.post("/v1/tasks/claim", headers=self.headers, json=body)
        self.last_response = resp
        if resp.status_code == 204:
            self.task = None
            return None
        assert resp.status_code == 200, resp.text
        self.task = resp.json()
        return self.task

    def heartbeat(self, steps: int) -> APIResult:
        assert self.task is not None, "heartbeat() with no claimed task"
        resp = self.client.post(
            f"/v1/tasks/{self.task['task_id']}/heartbeat",
            headers=self.headers, json={"steps_completed": steps},
        )
        self.last_response = resp
        return _wrap(resp)

    def submit(self, steps: int, *, adapter: dict[str, torch.Tensor] | None = None,
               scale: float = 0.02, seed: int | None = None,
               steps_per_min: float = 30.0, corrupt: str | None = None,
               tokens_seen: int = 0, artifact_key: str | None = None) -> APIResult:
        """Upload an artifact (well-formed unless ``corrupt`` is set) and submit.

        ``artifact_key`` overrides the derived key the coordinator hands back
        from ``upload-url`` -- the escape hatch a well-behaved worker never
        uses, but exactly what the "can't point a submission at someone
        else's object" gate (5.1 gate: derived key only) test needs to try.
        """
        assert self.task is not None, "submit() with no claimed task"
        task_id = self.task["task_id"]

        up = self.client.post(f"/v1/tasks/{task_id}/upload-url", headers=self.headers)
        self.last_response = up
        assert up.status_code == 200, up.text
        derived_key = up.json()["key"]
        key = artifact_key if artifact_key is not None else derived_key

        if adapter is not None:
            raw = save_adapter(adapter)
        elif corrupt is not None:
            raw = _corrupt_bytes(_base_adapter(scale=scale, seed=seed), corrupt)
        else:
            raw = save_adapter(_base_adapter(scale=scale, seed=seed))

        # The worker writes straight into the store -- see module docstring:
        # this is what the presigned PUT stands in for in this suite.
        self.store.put_bytes(key, raw)

        resp = self.client.post(
            f"/v1/tasks/{task_id}/submit", headers=self.headers,
            json={
                "artifact_key": key, "steps_completed": steps, "tokens_seen": tokens_seen,
                "metrics": {"steps_per_min": steps_per_min, "gpu_model": self.device},
            },
        )
        self.last_response = resp
        result = _wrap(resp)
        if resp.status_code == 200:
            self.task = None
        return result

    def abandon(self) -> APIResult:
        assert self.task is not None, "abandon() with no claimed task"
        resp = self.client.post(f"/v1/tasks/{self.task['task_id']}/abandon", headers=self.headers)
        self.last_response = resp
        result = _wrap(resp)
        if resp.status_code == 200:
            self.task = None
        return result

    def run_task(self, steps: int | None = None) -> dict | None:
        """claim + heartbeat + submit, the common case. None if claim got 204."""
        task = self.claim()
        if task is None:
            return None
        n = steps if steps is not None else task["local_steps"]
        self.heartbeat(n)
        return self.submit(n)
