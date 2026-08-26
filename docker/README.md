# Worker images

The three-layer stack from architecture §4.1, plus a CPU base for testing it.

```
nvidia/cuda:12.4.1-runtime-ubuntu22.04
  └─ ganymede/torch-base:cu124-t2.5     ~8 GB    torch + CUDA runtime
       └─ ganymede/worker-core:vN        ~30 MB  loop, client, probe
            └─ ganymede/worker-llm:vN    ~2 GB   transformers, peft, datasets
```

## Pin by digest, not by tag

`torch-base` is the one layer that must not move under a running swarm. Every
worker in a run needs a bit-identical PyTorch, because adapters trained against
subtly different kernels get averaged together and **nothing errors** — you find
out at M4, as an unexplained gap against the baseline.

A tag can be republished. A digest cannot. So:

```sh
docker build -f docker/torch-base.Dockerfile -t ganymede/torch-base:cu124-t2.5 .
docker buildx imagetools inspect ganymede/torch-base:cu124-t2.5   # copy the digest

docker build -f docker/worker-core.Dockerfile \
  --build-arg TORCH_BASE=ganymede/torch-base@sha256:<digest> \
  --build-arg GANYMEDE_VERSION=v1 \
  -t ganymede/worker-core:v1 .

docker build -f docker/worker-llm.Dockerfile \
  --build-arg WORKER_CORE=ganymede/worker-core@sha256:<digest> \
  --build-arg GANYMEDE_VERSION=v1 \
  -t ganymede/worker-llm:v1 .
```

The pinned-digest guarantee applies **within a backend**. A CUDA container fleet
shares one PyTorch build; native installs necessarily won't, which §6.8 Tier 1
establishes is acceptable — but native installs should pin a version *range* and
report the resolved version in their `compute_profile`, so a genuinely
incompatible build is visible rather than silently averaged in.

## Running one

```sh
docker run --rm --gpus all \
  --user 1000:1000 --cap-drop=ALL --security-opt=no-new-privileges \
  --read-only \
  --tmpfs /tmp --tmpfs /run/ganymede \
  -v ganymede-hf:/cache/hf \
  -v ganymede-state:/var/lib/ganymede \
  --memory 16g --cpus 4 --pids-limit 256 \
  --stop-timeout 120 \
  -e GANYMEDE_COORDINATOR_URL=https://coordinator.example \
  -e GANYMEDE_KEY=... \
  ganymede/worker-llm:v1
```

`--read-only` is the flag most likely to surprise you, and the writable mounts
above are not optional: `transformers`, Triton and `torch.compile` all write
caches. Expect to discover one or two more paths on a new base image — add them
explicitly rather than reverting to a writable rootfs, which is the whole point
of the exercise.

`--stop-timeout 120` gives the worker room to abandon its lease cleanly on
SIGTERM. It is an optimization, not the correctness path: the worker also polls
a sentinel file (§4.4), because Windows has no SIGTERM equivalent and building
correctness on three different signal mechanisms means three ways to be flaky.

## Stopping and pausing

```sh
docker exec <container> touch /var/lib/ganymede/stop     # exit after the current round
docker exec <container> touch /var/lib/ganymede/pause    # stay up, take no new work
```

Both abandon the current lease rather than racing to finish. Docker's default
stop grace is 10 s, and a half-uploaded artifact is worse than none —
abandoning releases the shard for immediate re-lease.

## Testing the container path without a GPU

`torch-base-cpu.Dockerfile` is a stand-in so the container itself is testable in
CI. Everything above it is byte-identical, so a smoke test there exercises the
real `worker-core` Dockerfile, entrypoint, read-only rootfs and mounts:

```sh
docker build -f docker/torch-base-cpu.Dockerfile -t ganymede/torch-base:cpu .
docker build -f docker/worker-core.Dockerfile \
  --build-arg TORCH_BASE=ganymede/torch-base:cpu -t ganymede/worker-core:cpu .
docker run --rm ganymede/worker-core:cpu --probe-only --skip-bench
```

This is why CI builds and smoke-tests on Linux from M2 onward: the path a
Windows-based developer uses least is the primary deployment target, and its
bugs otherwise surface late.

## `--probe-only`

The first thing to ask a contributor for when they report getting no work:

```sh
docker run --rm --gpus all ganymede/worker-llm:v1 --probe-only
```

It prints the measured profile — allocation ceiling, precision support, bench
score — which is exactly what the coordinator's eligibility rules (§6.8) act on.
"No nf4" or "alloc_max_mb below the run's minimum" is a far better support
answer than silence.
