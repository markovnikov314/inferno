# P0 Official-Source Research Ledger

## uv

- `uv sync --frozen` installs from `uv.lock` without updating it; a missing or
  stale lockfile fails the command.
- Sources:
  - https://docs.astral.sh/uv/concepts/projects/sync/
  - https://docs.astral.sh/uv/reference/cli/

## Docker and NVIDIA GPU Containers

- P0 readiness requires a working NVIDIA driver, Docker, NVIDIA Container
  Toolkit/runtime support, and a containerized `nvidia-smi` check with
  `--gpus all`.
- Sources:
  - https://docs.docker.com/engine/containers/gpu/
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html
  - https://docs.docker.com/compose/how-tos/gpu-support/

## Lightning SSH

- Lightning exposes Studio SSH through copied SSH commands and CLI/IDE flows.
- Record only the command shape; redact literal SSH targets, usernames,
  hostnames, key paths when sensitive, private IPs, and tokens.
- Sources:
  - https://lightning.ai/docs/platform/build/ai-studio/ssh-access
  - https://lightning.ai/docs/platform/build/ai-studio/connect-local-ide
  - https://lightning.ai/docs/platform/developers/cli/studio
  - https://lightning.ai/docs/platform/security/security-features/secrets

## Later-Phase Engine Pins

- vLLM and SGLang pinning belongs to P1/P4, not P0.
- Later phases should pin image tags or build provenance, CUDA lane, and package
  versions; avoid `latest` and `nightly` for reproducible claims.
- Sources:
  - https://docs.vllm.ai/en/stable/deployment/docker/
  - https://docs.sglang.ai/get_started/install.html
