# Repository Audit

The repository audit checks clean-clone setup, schema drift, report snapshots,
static preview generation, image/model provenance, secret scanning, dependency
metadata, and regression baselines.

Run the local audit:

```bash
make hardening-check HARDENING=configs/hardening/repo_audit.yaml
```

Run the optional GPU smoke gate with a shell-provided target:

```bash
PYTHONPATH=src uv run python -m inferno.cli hardening-check --config configs/hardening/repo_audit.yaml --include-gpu-smoke
```

Outputs are written under `artifacts/hardening/repo_audit/`.
