# ganymede/torch-base:cpu -- a CPU stand-in for the CUDA base (4.1).
#
# **Not a deployment artifact.** It exists so the container path itself can be
# tested: CI has no GPU, and "Developing on Windows" names the real risk that
# the Linux container path -- the primary deployment target for any third-party
# host -- gets the least day-to-day exercise and its bugs surface late, at
# exactly the moment you are asking someone else to install something.
#
# Everything above this layer (worker-core, worker-llm) is byte-identical
# whichever base it sits on, so a smoke test here exercises the real Dockerfiles,
# the real entrypoint, the read-only rootfs, and the declared writable mounts.
# What it cannot exercise is CUDA, which is what the GPU exit criteria are for.
#
# A separate file rather than build args on torch-base.Dockerfile because the
# two bases genuinely differ: ubuntu:22.04 ships Python 3.10 and has no 3.11
# without a third-party PPA, while python:3.11-slim has no apt-installed Python
# to manage at all. One file trying to be both would be less readable than two.

FROM python:3.11-slim

ARG TORCH_VERSION=2.5.1

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${TORCH_VERSION}" \
       --index-url https://download.pytorch.org/whl/cpu

RUN python -c "import torch; print('torch', torch.__version__, 'cpu build')"

LABEL org.opencontainers.image.title="ganymede/torch-base-cpu" \
      org.opencontainers.image.description="CPU PyTorch base -- for testing the container path, not for deployment"
