# ganymede/torch-base -- the heavy, rarely-rebuilt layer (docs/02-architecture-v2.md 4.1).
#
#   nvidia/cuda:12.4.1-runtime-ubuntu22.04
#     └─ ganymede/torch-base:cu124-t2.5    ~8 GB  torch + CUDA runtime libs
#          └─ ganymede/worker-core:vN      ~30 MB
#               └─ ganymede/worker-llm:vN  ~2 GB
#
# **Rebuild this deliberately and rarely.** Every worker in a run must have a
# bit-identical PyTorch: a silent version bump across a running swarm is a
# genuinely nasty class of bug, because adapters trained against subtly
# different kernels get averaged together and nothing errors. Downstream images
# therefore pin this one *by digest*, not by tag -- a tag can be republished, a
# digest cannot.
#
# After building:
#   docker buildx imagetools inspect ganymede/torch-base:cu124-t2.5
# and paste the digest into worker-core.Dockerfile's FROM line.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# The CUDA wheels rather than the PyPI default: the default build carries its own
# bundled CUDA libraries, which duplicates several GB already present in the base
# image and can disagree with the driver the host actually has.
RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# Fail the build rather than ship a torch that cannot see a GPU at runtime. This
# checks the build, not the host -- `torch.cuda.is_available()` is false in most
# build environments, and asserting it here would make the image unbuildable in CI.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

LABEL org.opencontainers.image.title="ganymede/torch-base" \
      org.opencontainers.image.description="Pinned PyTorch + CUDA runtime for Ganymede workers" \
      org.opencontainers.image.source="https://github.com/Adambomb210/Ganymede-Resource-Share"
