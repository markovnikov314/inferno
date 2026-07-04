# Project Inferno Agent Rules

Read this file before changing the repository.

## Scope Lock

Current authorized scope is P11: a local-only, GPU-aware browser dashboard
for existing Inferno evidence workflows plus an Ollama deployment-profile
extension. The dashboard may run remote SSH smoke tests for vLLM, SGLang,
Ollama, and TensorRT-LLM and display validated artifact-backed results.

Do not add a public live endpoint, cloud control plane, broad benchmark
runner, live router, autoscaler, or mocked product evidence. Strict engine
claims remain limited to validated vLLM/SGLang evidence. Ollama and llama.cpp
remain deployment-profile evidence. TensorRT-LLM remains engine-configuration
evidence.

## Required Start Checklist

- Read `INFERNO_MAP.md`.
- Read `.inferno/phase_state.yaml` and `.inferno/taskboard.yaml`.
- Read relevant policies in `.inferno/policies/`.
- Declare owned paths and forbidden paths before editing.

## Ownership

- Specialists write only their assigned paths.
- The integrator alone updates `INFERNO_MAP.md`, `.inferno/phase_state.yaml`,
  `.inferno/taskboard.yaml`, `.inferno/adr/`, and final phase status.
- Do not revert unrelated changes. Work with them or stop and report the
  conflict.

## Evidence

- Record literal commands, exit codes, and artifact paths in handoffs.
- Real-engine or GPU readiness claims require real host evidence.
- Mocks and fakes may test code paths only. They never satisfy readiness gates.

## Secrets

- Never commit tokens, SSH targets, hostnames, usernames, private IPs,
  unredacted traces, model weights, or private prompts.
- Commit `.env.example`; keep real values in ignored local environment files.
