# ganymede/worker-core -- the loop, the client, the probe (docs/02-architecture-v2.md 4.1).
#
# ~30 MB on top of torch-base, and that thinness is the point: this is the layer
# that changes, so it should be the layer that is cheap to rebuild and ship. It
# depends on **torch and the standard library only** -- no transformers, no peft,
# no HTTP library. See ganymede/worker/client.py for why urllib.
#
# FROM is pinned by digest, not tag. A tag can be republished under you; the
# whole guarantee of 4.1 is that every worker in a run has a bit-identical
# PyTorch, and a mutable tag cannot provide that.
#
#   docker build -f docker/worker-core.Dockerfile \
#     --build-arg TORCH_BASE=ganymede/torch-base@sha256:<digest> \
#     -t ganymede/worker-core:v1 .

ARG TORCH_BASE=ganymede/torch-base:cu124-t2.5
FROM ${TORCH_BASE}

ARG GANYMEDE_VERSION=dev
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata before source, so a code change does not invalidate the
# pip layer.
COPY pyproject.toml README.md LICENSE ./
COPY ganymede/__init__.py ganymede/device.py ./ganymede/
COPY ganymede/worker ./ganymede/worker
COPY ganymede/coordinator/__init__.py ganymede/coordinator/aggregate.py ./ganymede/coordinator/

# --no-deps for the package itself: torch is already here from the base, and
# resolving it again would pull several GB of wheels to arrive at what is
# already on disk. The two things worker-core genuinely needs are named
# explicitly instead.
#
# **numpy is not optional, and its absence is invisible until it is expensive.**
# `safetensors.torch.save` converts through numpy, so an image without it
# probes fine, registers fine, claims a task, downloads a base model, trains a
# full round -- and then dies on the one line that serializes the result. The
# whole round's compute is gone, and the traceback points at safetensors rather
# than at a missing dependency in an image built months earlier. It was omitted
# here exactly once; the container smoke test in CI now round-trips a tensor
# through safetensors precisely so that cannot recur.
RUN python -m pip install --no-cache-dir "safetensors>=0.4" "numpy>=1.24" \
    && python -m pip install --no-deps --no-cache-dir .

# 4.6: non-root, and the writable mounts declared explicitly. `transformers`,
# Triton and torch.compile all write caches, so a read-only rootfs needs these
# to exist and be writable or the failure is an unhelpful permission error deep
# in a library.
RUN useradd --create-home --uid 1000 ganymede \
    && mkdir -p /cache/hf /run/ganymede /var/lib/ganymede \
    && chown -R ganymede:ganymede /cache/hf /run/ganymede /var/lib/ganymede
USER ganymede

ENV HF_HOME=/cache/hf \
    GANYMEDE_CACHE_DIR=/cache/hf \
    GANYMEDE_STATE=/var/lib/ganymede \
    GANYMEDE_IMAGE_TAG=ganymede/worker-core:${GANYMEDE_VERSION}

VOLUME ["/cache/hf", "/var/lib/ganymede"]

LABEL org.opencontainers.image.title="ganymede/worker-core" \
      org.opencontainers.image.version="${GANYMEDE_VERSION}"

ENTRYPOINT ["python", "-m", "ganymede.worker.loop"]
