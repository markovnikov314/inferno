# P9 Hardening

P9 adds repository hardening only. It does not introduce live routing, a
dashboard, a new adapter, a benchmark runner, or a serving control plane.

Run the local hardening checks:

```bash
make hardening-check HARDENING=configs/hardening/p9_hardening.yaml
```

Run the P9 real GPU acceptance gate with an operator-supplied environment
variable:

```bash
$env:PYTHONPATH='src'; uv run python -m inferno.cli hardening-check --config configs/hardening/p9_hardening.yaml --include-gpu-smoke
```

The GPU smoke gate uses `INFERNO_GPU_SSH` from the environment and writes only
redacted preflight evidence. It is a host/container smoke check, not a new
engine-readiness claim.

Outputs:

- `artifacts/hardening/p9_hardening/latest.json`
- `artifacts/hardening/p9_hardening/gpu_smoke.json`
- `artifacts/hardening/p9_hardening/report_preview.html`
- `artifacts/hardening/p9_hardening/report.md`

The hardening report covers clean-clone lockfile setup, CPU-only fake-test
coverage, schema snapshot drift, report snapshots, static browser preview
generation, image/model provenance, secret scanning, dependency audit, and
P5/P6/P7/P8 regression baselines.
