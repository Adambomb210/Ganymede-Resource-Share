# Coordinator image (docs/02-architecture-v2.md 6.5).
#
# "Containerized from day one" is the point of this file: the same image runs
# unchanged in dev (docker compose up, localhost), bring-up (the same compose
# file on a real server), and production (this image behind a reverse proxy).
# Nothing about the coordinator's own behavior should depend on which stage
# it's running in -- that's what config.py's environment variables are for.

# Pin by digest (FROM python:3.11-slim@sha256:...) in production so a base-image
# republish can't silently change what ships; a tag is fine for bring-up, where
# reproducing the exact bring-up build is not yet the point.
FROM python:3.11-slim

WORKDIR /app

# Copy only what `pip install .` needs before the rest of the source, so a
# source-only change (docs, tests) doesn't invalidate the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY ganymede ./ganymede
COPY scripts ./scripts

# torch is a coordinator dependency (pyproject.toml's comment on why), which
# makes this a heavier-than-typical "slim" image; there is no way around that
# without splitting aggregation into its own service, which v1 doesn't do.
RUN pip install --no-cache-dir .

# Non-root: this is a public-facing HTTP service taking bearer tokens from
# volunteer machines on the open internet. It has no reason to run as root.
RUN useradd --create-home --uid 1000 ganymede \
    && mkdir -p /var/lib/ganymede \
    && chown -R ganymede:ganymede /var/lib/ganymede
USER ganymede

EXPOSE 8000

# No curl in the slim base; urllib avoids adding a package just for this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u, sys; \
        sys.exit(0 if u.urlopen('http://localhost:8000/healthz', timeout=3).status == 200 else 1)"

# app.py's `bootstrap` is a factory (it builds schema + storage, then the
# FastAPI app) -- `--factory` tells uvicorn to call it rather than import a
# module-level `app` object.
CMD ["uvicorn", "ganymede.coordinator.app:bootstrap", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
