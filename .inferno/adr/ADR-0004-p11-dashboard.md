# ADR-0004: P11 GPU-Aware Local Dashboard

## Status

Accepted.

## Context

P10 integrated TensorRT-LLM engine-configuration evidence and preserved the
existing strict vLLM/SGLang, llama.cpp deployment-profile, planner, router, and
hardening boundaries. The next operator need is a non-technical local UI that
can run smoke tests and show validated artifacts without turning Inferno into a
cloud control plane or live router.

## Decision

P11 authorizes a local browser dashboard served by `uv run python -m
inferno.cli dashboard`. The dashboard uses a Python API plus React/Vite
frontend, runs selected engines sequentially through existing artifact-producing
paths, detects the remote GPU before testing, and labels recommendations by
evidence scope.

Ollama is added as deployment-profile evidence using a pinned official Docker
image and default `qwen3.5:2b`. TensorRT-LLM remains engine-configuration
evidence. Tesla T4 selects the legacy TensorRT-LLM T4 profile, forces the
dashboard default `Qwen/Qwen3.5-2B`, and displays a warning that this
combination is unproven on T4 until validated by a real run.

## Consequences

- Dashboard state is local and in-memory; no SSH target is committed.
- Dashboard job outputs are written under ignored `artifacts/dashboard/`.
- Strict comparison gates continue to reject llama.cpp, Ollama, and
  TensorRT-LLM.
- Browser UI is allowed only as local operator tooling, not as a public service
  or cloud control plane.
