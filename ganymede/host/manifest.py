"""Manifest reconciliation (docs/02-architecture-v2.md 7, step 3).

```
3. GET /v1/manifest; if required image tag != local, pull it
```

Why this lives on the host rather than in the worker
----------------------------------------------------
4.2 step 3 has no version check, and the note there says that is deliberate:
"see Finding E and 7". This module is what that sentence defers to. A worker
that discovered its own image was wrong could only abandon and exit -- it is
already inside the image in question, so it cannot fix anything. The host agent
is outside, holds the Docker socket, and can pull. So the *decision* about what
image the fleet should be running belongs here, before a container starts, and
4.2 step 5's in-container check is the backstop for a race rather than the
primary path.

Deciding, not acting
--------------------
Nothing here pulls, starts or stops anything. `resolve` returns an
`ImageDecision` and the agent acts on it. The split exists because the
interesting half is the decision -- what happens when two active runs want
different images, what a pin means, what "nothing to reconcile" is as distinct
from "reconciliation failed" -- and a function that both decides and acts can
only be tested with a Docker daemon.

Standard library only, like the rest of ``ganymede/host``.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from ganymede.host.config import HostConfig

DEFAULT_TIMEOUT_SEC = 30


class ManifestError(RuntimeError):
    """The manifest could not be fetched or made sense of.

    Carries the URL because the single most common cause is a coordinator URL
    with a typo or a missing scheme, and an error that does not say which
    address it tried is one the contributor cannot act on.
    """

    def __init__(self, url: str, detail: str):
        super().__init__(f"{url}: {detail}")
        self.url = url
        self.detail = detail


@dataclass(frozen=True)
class Run:
    run_id: str
    base_model: str
    base_precision: str
    required_image: str | None
    current_round: int
    target_rounds: int


@dataclass(frozen=True)
class Manifest:
    api_version: str
    runs: list[Run]
    heartbeat_interval_sec: int | None = None

    @property
    def base_models(self) -> frozenset[str]:
        """What the cache should be protecting from eviction (6.7).

        A model an active run needs is a model this machine is about to be
        asked for. Evicting it to make room means re-downloading ~16 GB before
        any work happens, which is potentially longer than the round.
        """
        return frozenset(r.base_model for r in self.runs if r.base_model)


@dataclass(frozen=True)
class ImageDecision:
    """What to run, and whether that differs from what is running now."""

    image: str | None
    reason: str
    # False when there is simply nothing to reconcile against -- no active runs,
    # or no run naming an image. Distinct from an exception, which means the
    # question could not be asked at all. The agent treats the two differently:
    # nothing to reconcile is a normal quiet tick, a failure is not.
    constrained: bool = True

    def differs_from(self, running: str | None) -> bool:
        if self.image is None or running is None:
            return False
        return self.image != running


def fetch(
    config: HostConfig,
    *,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    opener: Callable[..., Any] | None = None,
) -> Manifest:
    """``GET /v1/manifest`` with the contributor's bearer key.

    ``urllib`` rather than the worker's own ``CoordinatorClient``: that client
    lives in the worker package, which on the container path is not installed
    on the host at all. The host agent's whole delivery story is "copy the
    package and run it", and one HTTP GET is not worth breaking that for.
    """
    url = f"{config.coordinator_url.rstrip('/')}/v1/manifest"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {config.key}", "Accept": "application/json"},
        method="GET",
    )

    context: ssl.SSLContext | None = None
    if not config.verify_tls:
        # Only ever reachable from an explicit `verify_tls: false`, which exists
        # for a self-signed coordinator on a home network during bring-up. The
        # bearer key is in this request, so this is genuinely unsafe over any
        # network the contributor does not own -- 6.3 is explicit that auth is
        # bearer-over-TLS and this switch is the documented exception, not a
        # convenience.
        context = ssl._create_unverified_context()

    try:
        opened = (opener or urllib.request.urlopen)(request, timeout=timeout, context=context)
    except TypeError:
        # A test double, or an opener that does not take a context. Retrying
        # without it keeps this function usable from a fake without forcing
        # every fake to mirror urlopen's exact signature.
        opened = (opener or urllib.request.urlopen)(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = _read_body(exc)
        raise ManifestError(url, f"HTTP {exc.code}: {body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise ManifestError(url, f"unreachable: {exc.reason}") from exc
    except OSError as exc:
        raise ManifestError(url, str(exc)) from exc

    with opened as response:
        raw = _read_body(response)

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ManifestError(url, f"response was not JSON: {raw[:200]}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(url, f"expected a JSON object, got {type(payload).__name__}")

    return parse(payload, url=url)


def parse(payload: dict[str, Any], *, url: str = "<manifest>") -> Manifest:
    """Payload to `Manifest`, tolerantly.

    Unknown keys are ignored rather than rejected. The host agent is the one
    component a contributor installs once and may not update for months, so it
    has to keep working against a coordinator that has grown fields it has
    never heard of -- the alternative is that adding anything to the manifest
    breaks every stale installation in the fleet at once.
    """
    raw_runs = payload.get("runs") or []
    if not isinstance(raw_runs, list):
        raise ManifestError(url, f"'runs' should be a list, got {type(raw_runs).__name__}")

    runs = []
    for entry in raw_runs:
        if not isinstance(entry, dict):
            continue
        runs.append(
            Run(
                run_id=str(entry.get("run_id", "")),
                base_model=str(entry.get("base_model") or ""),
                base_precision=str(entry.get("base_precision") or ""),
                required_image=entry.get("required_image") or None,
                current_round=int(entry.get("current_round") or 0),
                target_rounds=int(entry.get("target_rounds") or 0),
            )
        )

    return Manifest(
        api_version=str(payload.get("api_version") or ""),
        runs=runs,
        heartbeat_interval_sec=payload.get("heartbeat_interval_sec"),
    )


def resolve(manifest: Manifest, config: HostConfig) -> ImageDecision:
    """Which image this host should be running.

    **When active runs disagree.** Two runs can name different images and only
    one container runs at a time, so something has to break the tie. This picks
    the image named by the most active runs, and breaks a tie between equal
    counts by taking the highest tag string.

    The alternative -- take the newest run's image -- was rejected because it
    inverts under exactly the condition that matters. A fleet steady on `v3`
    with one freshly-created `v4` run would have every contributor pull 2 GB and
    restart, abandoning work on the runs that are actually consuming the fleet,
    to chase the run with the least work in it. Majority means an operator rolls
    the fleet forward by migrating runs, which is a thing they can observe and
    stage, rather than by creating one run and watching every machine turn over
    at once.

    Preferring the higher tag on a tie is what makes the last step of that
    migration terminate: at 1-1 the fleet moves forward rather than oscillating
    on whatever order the coordinator happened to return rows in.
    """
    if config.image_tag:
        # An operator's explicit pin outranks the fleet. This is the switch for
        # holding a machine on a known image while something is investigated,
        # and a pin that the coordinator could override would not be one.
        image = config.image_tag
        if "/" not in image and ":" not in image:
            image = f"{config.image_repo}:{image}"
        return ImageDecision(image=image, reason="pinned by local config")

    named = [r.required_image for r in manifest.runs if r.required_image]
    if not named:
        if not manifest.runs:
            return ImageDecision(None, "no active runs", constrained=False)
        return ImageDecision(None, "no active run requires a specific image", constrained=False)

    counts: dict[str, int] = {}
    for image in named:
        counts[image] = counts.get(image, 0) + 1

    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    if len(counts) == 1:
        return ImageDecision(best, f"required by {counts[best]} active run(s)")
    losing = ", ".join(f"{img} x{n}" for img, n in sorted(counts.items()) if img != best)
    return ImageDecision(
        best,
        f"required by {counts[best]} of {len(named)} active runs (also wanted: {losing})",
    )


def _read_body(response: Any) -> str:
    try:
        raw = response.read()
    except Exception:  # noqa: BLE001 -- a body we cannot read is a body we report as empty
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
