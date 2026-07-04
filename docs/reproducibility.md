# Clean-Clone Reproduction Runbook

This runbook verifies a small real GPU smoke path from a clean checkout. Use an operator-supplied `INFERNO_GPU_SSH` value from the shell; do not write it into the repository.

```bash
uv sync --all-groups --frozen
make doctor
INFERNO_GPU_SSH='[operator supplied]' make doctor-gpu
INFERNO_GPU_SSH='[operator supplied]' make study ENGINE=vllm CONFIG=configs/study/research_smoke.yaml
INFERNO_GPU_SSH='[operator supplied]' make study ENGINE=sglang CONFIG=configs/study/research_smoke.yaml
make compare MODE=strict RUNS="artifacts/studies/research_smoke/latest.json"
```

Expected result: both engine studies produce three validated repeats, and strict compare writes a report only if the controlled comparison key matches.

To build the full redacted research package after all five research studies have validated runs:

```bash
make release-package MODE=redacted STUDY=configs/study/research_core.yaml
```

The command writes `artifacts/releases/research_core_v1/redacted/manifest.json` with hashes for package files.
