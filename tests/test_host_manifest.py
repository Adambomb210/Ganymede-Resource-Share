"""Manifest reconciliation (docs/02-architecture-v2.md 7 step 3).

No network anywhere: `fetch` takes an opener, so the HTTP boundary is a fake
and every test here runs in microseconds. What is actually worth testing is the
*decision* -- what happens when two active runs want different images -- and
that has no HTTP in it at all.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from ganymede.host import manifest as manifest_mod
from ganymede.host.config import HostConfig


def _config(**kw) -> HostConfig:
    base = {"coordinator_url": "https://coordinator.example", "key": "k"}
    base.update(kw)
    return HostConfig(**base)


def _manifest(*runs: dict) -> manifest_mod.Manifest:
    return manifest_mod.parse({"api_version": "v1", "runs": list(runs)})


def _run(run_id: str, image: str | None = None, base_model: str = "Qwen/Qwen3-1.7B") -> dict:
    return {
        "run_id": run_id,
        "base_model": base_model,
        "base_precision": "bf16",
        "required_image": image,
        "current_round": 1,
        "target_rounds": 10,
    }


# --------------------------------------------------------------------------
# resolve: the interesting half
# --------------------------------------------------------------------------


def test_a_single_required_image_is_the_answer():
    decision = manifest_mod.resolve(_manifest(_run("a", "ganymede/worker-llm:v3")), _config())
    assert decision.image == "ganymede/worker-llm:v3"
    assert decision.constrained


def test_disagreeing_runs_are_settled_by_majority_not_by_row_order():
    """Two runs on v3, one on v4. The fleet is doing v3 work; one new run does
    not get to turn every contributor over."""
    decision = manifest_mod.resolve(
        _manifest(
            _run("a", "ganymede/worker-llm:v3"),
            _run("b", "ganymede/worker-llm:v3"),
            _run("c", "ganymede/worker-llm:v4"),
        ),
        _config(),
    )
    assert decision.image == "ganymede/worker-llm:v3"
    assert "v4" in decision.reason  # the losing side is reported, not hidden


def test_majority_is_unaffected_by_the_order_rows_arrive_in():
    runs = [_run("a", "img:v3"), _run("b", "img:v4"), _run("c", "img:v3")]
    forwards = manifest_mod.resolve(_manifest(*runs), _config()).image
    backwards = manifest_mod.resolve(_manifest(*reversed(runs)), _config()).image
    assert forwards == backwards == "img:v3"


def test_a_tie_moves_the_fleet_forward_rather_than_oscillating():
    """One run each. Whichever way the coordinator orders the rows, every host
    has to reach the same answer or the fleet thrashes -- and the answer has to
    be the newer tag, or the last step of a migration never completes."""
    decision = manifest_mod.resolve(
        _manifest(_run("a", "ganymede/worker-llm:v3"), _run("b", "ganymede/worker-llm:v4")),
        _config(),
    )
    assert decision.image == "ganymede/worker-llm:v4"


def test_a_local_pin_outranks_the_fleet():
    decision = manifest_mod.resolve(
        _manifest(_run("a", "ganymede/worker-llm:v3")), _config(image_tag="ganymede/worker-llm:v1")
    )
    assert decision.image == "ganymede/worker-llm:v1"
    assert "pinned" in decision.reason


def test_a_bare_pinned_tag_is_qualified_with_the_configured_repo():
    decision = manifest_mod.resolve(_manifest(), _config(image_tag="v7"))
    assert decision.image == "ganymede/worker-llm:v7"


def test_no_active_runs_is_unconstrained_not_an_error():
    decision = manifest_mod.resolve(_manifest(), _config())
    assert decision.image is None
    assert decision.constrained is False
    assert "no active runs" in decision.reason


def test_runs_that_name_no_image_constrain_nothing():
    decision = manifest_mod.resolve(_manifest(_run("a", None), _run("b", None)), _config())
    assert decision.image is None
    assert decision.constrained is False


def test_one_run_naming_an_image_decides_for_runs_that_do_not():
    decision = manifest_mod.resolve(_manifest(_run("a", None), _run("b", "img:v2")), _config())
    assert decision.image == "img:v2"


def test_differs_from_is_false_when_either_side_is_unknown():
    """A decision with no image, or a runtime with no image, is not evidence of
    a mismatch -- and acting on it would restart a healthy worker."""
    decision = manifest_mod.ImageDecision("img:v2", "x")
    assert decision.differs_from("img:v1")
    assert not decision.differs_from("img:v2")
    assert not decision.differs_from(None)
    assert not manifest_mod.ImageDecision(None, "x").differs_from("img:v1")


def test_base_models_names_what_the_cache_must_protect():
    m = _manifest(_run("a", base_model="Qwen/Qwen3-1.7B"), _run("b", base_model="meta/Llama"))
    assert m.base_models == frozenset({"Qwen/Qwen3-1.7B", "meta/Llama"})


# --------------------------------------------------------------------------
# parse: tolerance, because a host agent is installed once and updated rarely
# --------------------------------------------------------------------------


def test_unknown_fields_are_ignored_so_a_stale_agent_keeps_working():
    m = manifest_mod.parse(
        {"api_version": "v1", "runs": [{**_run("a", "img:v1"), "something_new": 42}],
         "a_field_from_the_future": True}
    )
    assert m.runs[0].run_id == "a"


def test_a_missing_runs_key_is_an_empty_manifest():
    assert manifest_mod.parse({"api_version": "v1"}).runs == []


def test_a_runs_value_that_is_not_a_list_is_an_error():
    with pytest.raises(manifest_mod.ManifestError):
        manifest_mod.parse({"runs": "nope"})


# --------------------------------------------------------------------------
# fetch: the HTTP boundary
# --------------------------------------------------------------------------


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_fetch_sends_the_bearer_key_and_parses_the_body():
    seen = {}

    def opener(request, timeout=None, context=None):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return _Resp(json.dumps({"api_version": "v1", "runs": [_run("a", "img:v1")]}).encode())

    m = manifest_mod.fetch(_config(), opener=opener)
    assert seen["url"] == "https://coordinator.example/v1/manifest"
    assert seen["auth"] == "Bearer k"
    assert m.runs[0].required_image == "img:v1"


def test_an_unreachable_coordinator_names_the_url_it_tried():
    def opener(request, timeout=None, context=None):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(manifest_mod.ManifestError) as exc:
        manifest_mod.fetch(_config(), opener=opener)
    assert "coordinator.example" in str(exc.value)


def test_an_http_error_carries_the_status():
    def opener(request, timeout=None, context=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    with pytest.raises(manifest_mod.ManifestError) as exc:
        manifest_mod.fetch(_config(), opener=opener)
    assert "401" in str(exc.value)


def test_a_body_that_is_not_json_is_a_manifest_error_not_a_traceback():
    def opener(request, timeout=None, context=None):
        return _Resp(b"<html>proxy error</html>")

    with pytest.raises(manifest_mod.ManifestError):
        manifest_mod.fetch(_config(), opener=opener)


def test_a_trailing_slash_on_the_coordinator_url_does_not_double_up():
    seen = {}

    def opener(request, timeout=None, context=None):
        seen["url"] = request.full_url
        return _Resp(b'{"runs": []}')

    manifest_mod.fetch(_config(coordinator_url="https://c.example/"), opener=opener)
    assert seen["url"] == "https://c.example/v1/manifest"
