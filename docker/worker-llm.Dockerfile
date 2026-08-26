# ganymede/worker-llm -- worker-core plus the training stack (docs/02-architecture-v2.md 4.1).
#
# ~2 GB: transformers, peft, datasets, accelerate, bitsandbytes. This is the
# image a `llm_finetune` run's `required_image` names, and the one an
# `nf4` run needs -- bitsandbytes is CUDA-only, which is exactly what the
# capability probe reports and what 6.8's eligibility rules act on.
#
#   docker build -f docker/worker-llm.Dockerfile \
#     --build-arg WORKER_CORE=ganymede/worker-core@sha256:<digest> \
#     -t ganymede/worker-llm:v1 .

ARG WORKER_CORE=ganymede/worker-core:v1
FROM ${WORKER_CORE}

ARG GANYMEDE_VERSION=dev
USER root

WORKDIR /app

# --no-deps again for torch's sake, then the real dependency resolution for
# everything else. Pinning a range rather than an exact version here is
# deliberate: 4.1's bit-identical guarantee is about *torch*, and holding
# transformers to a single patch release would mean rebuilding this layer for
# every upstream bugfix.
RUN python -m pip install --no-cache-dir \
        "transformers>=4.45,<6" \
        "peft>=0.13" \
        "datasets>=3.0" \
        "accelerate>=1.0"

# bitsandbytes last and allowed to fail: it is CUDA-only and the build may be
# running on a machine or architecture where the wheel does not exist. A worker
# without it registers fine and simply reports no nf4 support (6.9), which is
# the designed outcome -- failing the whole image build over an optional
# quantization backend is not.
RUN python -m pip install --no-cache-dir "bitsandbytes>=0.43" \
    || echo "bitsandbytes unavailable for this platform; nf4 runs will not match this image"

COPY ganymede/trainer ./ganymede/trainer
COPY ganymede/coordinator ./ganymede/coordinator
COPY configs ./configs

RUN chown -R ganymede:ganymede /app
USER ganymede

ENV GANYMEDE_IMAGE_TAG=ganymede/worker-llm:${GANYMEDE_VERSION}

LABEL org.opencontainers.image.title="ganymede/worker-llm" \
      org.opencontainers.image.version="${GANYMEDE_VERSION}"

ENTRYPOINT ["python", "-m", "ganymede.worker.loop"]
