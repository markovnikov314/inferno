# Clean-Clone Reproduction Runbook

This runbook verifies a small real GPU smoke path from a clean checkout. Use an operator-supplied `INFERNO_GPU_SSH` value from the shell; do not write it into the repository.

```bash
uv sync --all-groups --frozen
make doctor
INFERNO_GPU_SSH='[operator supplied]' make doctor-gpu
INFERNO_GPU_SSH='[operator supplied]' make study ENGINE=vllm CONFIG=configs/study/p5_smoke.yaml
INFERNO_GPU_SSH='[operator supplied]' make study ENGINE=sglang CONFIG=configs/study/p5_smoke.yaml
make compare MODE=strict RUNS="artifacts/studies/p5_smoke/latest.json"
```

Expected result: both engine studies produce three validated repeats, and strict compare writes a report only if the controlled comparison key matches.

To build the full redacted v1.0 package after all five P5 studies have validated runs:

```bash
make release-package MODE=redacted STUDY=configs/study/p5_research_core.yaml
```

The command writes `artifacts/releases/p5_research_core_v1/redacted/manifest.json` with hashes for package files.
