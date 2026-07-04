# ADR-0002: P9 Hardening Gate

## Status

Accepted.

## Context

P9 is authorized after P8 completed as an offline negative result. The master
plan allows hardening and extensions, but every extension needs an ADR and a
real-engine acceptance gate. The P8 evidence does not justify live routing,
additional adapters, dashboards, or a cloud control plane.

## Decision

Implement P9 as hardening only:

- clean-clone lockfile setup checks;
- CPU-only fake-based tests;
- GPU-gated real host/container smoke evidence using `INFERNO_GPU_SSH`;
- schema snapshot and report snapshot checks;
- static browser-preview generation for Markdown reports;
- image/model provenance checks;
- secret scan and offline dependency audit;
- P5/P6/P7/P8 regression baselines.

The acceptance gate is:

```powershell
$env:PYTHONPATH='src'; uv run python -m inferno.cli hardening-check --config configs/hardening/p9_hardening.yaml --include-gpu-smoke
```

The command must not persist the literal SSH target, username, hostname,
private IPs, tokens, raw private outputs, model weights, or private prompts.

## Consequences

P9 makes no new engine readiness claim. It preserves strict-mode llama.cpp
refusal and does not convert deployment-profile results into strict
engine-only claims. P10+ work still requires explicit authorization.
