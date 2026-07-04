# Project Inferno Map

## Current Phase

P11 - GPU-aware local browser dashboard.

## Phase Status

`IN_PROGRESS`

## Repository Scope

Project Inferno is a staged inference-evaluation repository. Completed phases
establish validated real-engine artifacts, strict vLLM/SGLang research reports,
the P6 llama.cpp deployment-profile extension, the P7 trace-grounded capacity
planner, the P8 offline serving-policy replay evaluator, the P9 hardening
gate, the P10 TensorRT-LLM engine-configuration extension, and the P11 local
operator dashboard. This file is a durable map of the repo state and evidence,
not a single-phase scratchpad.

## Target GPU Host

The GPU SSH target is supplied at runtime through `INFERNO_GPU_SSH`. The
literal target is not committed.

## P0 - Governance And GPU Preflight

P0 established the repository controls, durable state files, policy files,
handoff schema, scope lock, and redacted local/GPU preflight evidence. P0 did
not add serving-engine, benchmark, planner, router, or dashboard product scope.

Evidence:

- Preflight artifacts: `.inferno/evidence/preflight/doctor.json` and
  `.inferno/evidence/preflight/doctor-gpu.json`
- ADR: `.inferno/adr/ADR-0001-p0-scope-lock.md`
- Handoffs: `.inferno/handoffs/P0-INTEGRATE.yaml`,
  `.inferno/handoffs/P0-PREFLIGHT.yaml`, `.inferno/handoffs/P0-RESEARCH.yaml`,
  and `.inferno/handoffs/P0-VERIFY.yaml`

## P1 - Real vLLM Vertical Slice

P1 proved a pinned vLLM process on the GPU host with health, served-model
identity, raw streamed output, trace/log capture, and shutdown evidence. The
successful readiness evidence is the Qwen/Qwen3.5-2B run; earlier Gemma/T4
attempts remain historical limitations, not readiness evidence.

Evidence:

- vLLM engine config: `configs/engines/vllm.json`
- Successful P1 run:
  `artifacts/runs/p1-vllm-20260701T022756Z-3079abb0`
- P1 handoff: `.inferno/handoffs/P1-INTEGRATE.yaml`

## P2 - Artifact Contract And Validated Run Path

P2 established the manifest, raw SSE, request trace, telemetry, summary,
validation, checksum, and schema snapshot contract. The validator rejects
post-hoc manifests, non-monotonic traces, checksum mismatches, fake engine
evidence, incomplete provenance, and strict comparison key mismatches.

Evidence:

- Artifact contract code: `src/inferno/artifact_contract.py`
- Schema snapshot: `.inferno/contracts/artifacts/manifest.v1.schema.json`
- Contract tests: `tests/test_artifact_contract.py`
- Validated P2/P4-adjacent run evidence exists under `artifacts/runs/`
  including the final SGLang validation path used before P4 strict comparison.

No dedicated P2 handoff was found in the current repo scan.

## P3 - Reproducible vLLM Smoke Study

P3 produced three independent validated vLLM smoke repeats, retained
trace-complete artifacts, and rendered a static report with LOW_SAMPLE labels.
It did not add SGLang, llama.cpp, planner, router, or dashboard scope.

Evidence:

- Study config: `configs/study/vllm_smoke.yaml`
- Study record: `artifacts/studies/vllm_smoke/latest.json`
- Report: `artifacts/studies/vllm_smoke/report.md`
- P3 handoff: `.inferno/handoffs/P3-VLLM-SMOKE.yaml`

## P4 - Strict Interactive vLLM/SGLang Study

P4 added the SGLang path and strict vLLM/SGLang comparison. The accepted strict
interactive study has three validated repeats per engine, a matching controlled
comparison key, response evidence, LOW_SAMPLE labels, and a rendered strict
report. Failed or intermediate attempts remain excluded from gate evidence.

Evidence:

- SGLang engine config: `configs/engines/sglang.json`
- Strict study config: `configs/study/strict_interactive.yaml`
- Strict workload config: `configs/workloads/strict_interactive.yaml`
- Study record: `artifacts/studies/strict_interactive/latest.json`
- Strict report: `artifacts/compare/strict-20260701T132844Z/report.md`

No dedicated P4 handoff was found in the current repo scan.

## P5 - v1.0 Research Core Release

### Scope

P5 packages the existing strict vLLM-vs-SGLang research path into a
reproducible v1.0 Research Core release. The release covers five workload
families on the pinned Qwen/Qwen3.5-2B model and Tesla T4 environment:

- smoke
- interactive
- decode-heavy
- long-context-lite
- structured-output-lite

Out of scope for P5: llama.cpp, deployment-profile comparison, planner/router
work, dashboards, cloud control plane, and P6+ extensions.

### Acceptance Gates

```bash
uv sync --all-groups --frozen
make doctor
INFERNO_GPU_SSH=[operator supplied] make doctor-gpu
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=vllm CONFIG=configs/study/p5_smoke.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=sglang CONFIG=configs/study/p5_smoke.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=vllm CONFIG=configs/study/p5_interactive.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=sglang CONFIG=configs/study/p5_interactive.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=vllm CONFIG=configs/study/p5_decode_heavy.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=sglang CONFIG=configs/study/p5_decode_heavy.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=vllm CONFIG=configs/study/p5_long_context_lite.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=sglang CONFIG=configs/study/p5_long_context_lite.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=vllm CONFIG=configs/study/p5_structured_output_lite.yaml
INFERNO_GPU_SSH=[operator supplied] make study ENGINE=sglang CONFIG=configs/study/p5_structured_output_lite.yaml
make compare MODE=strict RUNS="artifacts/studies/<p5-study>/latest.json"
make release-package MODE=redacted STUDY=configs/study/p5_research_core.yaml
make test-contracts
make test
git diff --check
git diff --cached --check
```

### Evidence Index

- P5 study configs: `configs/study/p5_*.yaml`
- P5 workload configs: `configs/workloads/p5_*.yaml`
- P5 study records: `artifacts/studies/p5_*/latest.json`
- P5 strict reports:
  - `artifacts/compare/strict-20260701T142431Z/report.md`
  - `artifacts/compare/strict-20260701T150157Z/report.md`
  - `artifacts/compare/strict-20260701T153426Z/report.md`
  - `artifacts/compare/strict-20260701T160546Z/report.md`
  - `artifacts/compare/strict-20260701T163814Z/report.md`
- Redacted release package:
  `artifacts/releases/p5_research_core_v1/redacted/manifest.json`
- P5 handoff: `.inferno/handoffs/P5-RESEARCH-CORE.yaml`

### Result

P5 passed. The release package contains five strict workload families, three
validated vLLM repeats and three validated SGLang repeats per family, static
strict reports, redacted report inputs, redacted run summaries/traces/metadata,
methodology/limitations/reproducibility/redaction docs, and a package manifest
with hashes.

Strict reports include study type, controlled key status, controlled-variable
summary, run IDs, sample counts, LOW_SAMPLE labels, repeat-level uncertainty,
response equivalence or quality status, limitations, and value provenance.

The public claim remains limited to the tested vLLM/SGLang engines, pinned
Qwen/Qwen3.5-2B model/tokenizer revision, Tesla T4 environment, workload
families, and strict configuration envelope.

## P6 - v1.1 llama.cpp Deployment-Profile Extension

### Scope

P6 adds one llama.cpp GGUF deployment profile to the existing validated
artifact, study, report, and release-adjacent system. It is a deployment-profile
comparison, not a strict engine-only comparison.

Out of scope for P6: planner/router work, dashboards, cloud control plane, live
routing, P7/P8 decision modules, broad benchmark framework work, and strict
engine-only claims involving llama.cpp.

### Acceptance Gates

```bash
PYTHONPATH=src INFERNO_GPU_SSH=[operator supplied] uv run python -m inferno.cli study --engine llamacpp --config configs/study/profile_local.yaml
PYTHONPATH=src uv run python -m inferno.cli compare --mode deployment-profile --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/profile_local/latest.json"
PYTHONPATH=src uv run python -m inferno.cli compare --mode strict --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/profile_local/latest.json"
PYTHONPATH=src uv run python -m inferno.cli validate --run artifacts/runs/p6-llamacpp-profile-local-r01-20260701T180433Z-6e6969f5
PYTHONPATH=src uv run python -m inferno.cli validate --run artifacts/runs/p5-vllm-p5-smoke-r01-20260701T135244Z-049adeba
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run ruff check src tests
git diff --check
git diff --cached --check
```

The strict compare command is an expected refusal gate: it must fail with
`strict comparison rejects llamacpp; use MODE=deployment-profile`.

### Evidence Index

- llama.cpp engine config: `configs/engines/llamacpp.json`
- P6 study config: `configs/study/profile_local.yaml`
- P6 workload config: `configs/workloads/profile_local.yaml`
- Deployment-profile docs: `docs/deployment-profile.md`
- P6 run: `artifacts/runs/p6-llamacpp-profile-local-r01-20260701T180433Z-6e6969f5`
- P6 study record: `artifacts/studies/profile_local/latest.json`
- Deployment-profile report:
  `artifacts/compare/deployment-profile-20260701T181124Z/report.md`
- P6 handoff: `.inferno/handoffs/P6-DEPLOYMENT-PROFILE.yaml`

### Result

P6 passed. The llama.cpp profile launched on the GPU host with the pinned
`ghcr.io/ggml-org/llama.cpp:server-cuda` image, loaded
`lmstudio-community/Qwen3.5-2B-GGUF` at revision
`bb84e11355a036e28f080c7793fa6d22b7c4e344`, streamed real SSE output from
`Qwen3.5-2B-Q4_K_M.gguf`, and validated under the extended artifact contract.

The deployment-profile report includes the required banner:

`DEPLOYMENT PROFILE COMPARISON - engine and one or more controlled deployment variables differ. Do not attribute outcomes to engine alone.`

Existing P4/P5 strict artifacts remain valid under the backwards-compatible
contract extension. Strict mode rejects llama.cpp and remains limited to
vLLM/SGLang strict comparisons.

## P7 - Trace-Grounded Capacity Planner

### Scope

P7 adds an offline capacity planner that reads existing validated study
artifacts and returns only evidence-range-labeled results. It matches evidence
by environment fingerprint and declared comparison type, uses a transparent
nearest-neighbor distance with bounded range checks, and records nearest
evidence, observed range, holdout error, confidence, and cost availability.

Out of scope for P7: live routing, offline serving-policy routing, dashboards,
cloud control plane, new engine launches, new benchmark runners, P8+ decision
logs, and strict engine-only claims involving deployment-profile evidence.

### Acceptance Gates

```bash
$env:PYTHONPATH='src'; uv run python -m inferno.cli plan-capacity --config configs/planner/p7_capacity.yaml
$env:PYTHONPATH='src'; uv run python -m inferno.cli validate --run artifacts/runs/p5-vllm-p5-smoke-r01-20260701T135244Z-049adeba
$env:PYTHONPATH='src'; uv run python -m inferno.cli validate --run artifacts/runs/p6-llamacpp-profile-local-r01-20260701T180433Z-6e6969f5
$env:PYTHONPATH='src'; uv run python -m inferno.cli compare --mode strict --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/profile_local/latest.json"
uv run pytest -q
uv run ruff check src tests
git diff --check
git diff --cached --check
```

The strict compare command is an expected refusal gate: it must fail with
`strict comparison rejects llamacpp; use MODE=deployment-profile`.

### Evidence Index

- Planner code: `src/inferno/planner.py`
- CLI entrypoint: `src/inferno/cli.py`
- Planner config: `configs/planner/p7_capacity.yaml`
- Planner docs: `docs/capacity-planner.md`
- Planner tests: `tests/test_planner.py`
- Planner artifacts:
  - `artifacts/planner/p7_capacity_strict_t4_qwen35_2b/latest.json`
  - `artifacts/planner/p7_capacity_strict_t4_qwen35_2b/holdout_eval.json`
  - `artifacts/planner/p7_capacity_strict_t4_qwen35_2b/report.md`
- P7 handoff: `.inferno/handoffs/P7-CAPACITY-PLANNER.yaml`

### Result

P7 passed. The planner consumed 30 compatible P5 strict evidence runs from the
`p5_research_core_block_001` environment and rejected no configured P7 evidence
inputs. For the requested 8 rps, Poisson, 1024/2048 prompt-token, 256/512
output-token target, both strict candidates are labeled `EXTRAPOLATED` with low
confidence and no hard recommendation because the stored P5 evidence is
closed-loop, lower-throughput, and lower-output-token than the target.

The holdout evaluation uses leave-one-run-out nearest neighbor with a
same-candidate median baseline. The stored P7 result reports nearest-neighbor
TTFT MAE 293.949957 ms versus baseline 847.055783 ms, and nearest-neighbor E2E
MAE 285.532217 ms versus baseline 1696.915057 ms.

Cost remains `UNAVAILABLE` because the P7 config supplies a dated price source
but no per-candidate price values. Prompt token distances are marked
unavailable because current artifacts store prompt characters and prompt hashes,
not canonical prompt token counts.

## P8 - Offline Serving-Policy Router Replay

### Scope

P8 adds an offline serving-policy replay evaluator that reads existing
validated study artifacts and evaluates routing policies on held-out workload
blocks. It emits one decision-log row per replayed request and reports latency,
cost availability, failure risk, quality compatibility, selection distribution,
confidence, leakage checks, ablations, and the declared held-out objective.

Out of scope for P8: live routing, live endpoints, dashboards, cloud control
plane, new engine launches, benchmark runners, P9+ hardening, and strict
engine-only claims involving llama.cpp.

### Acceptance Gates

```bash
$env:PYTHONPATH='src'; uv run python -m inferno.cli replay-router --config configs/router/p8_offline_router.yaml
uv run pytest -q tests/test_router.py
uv run pytest -q
uv run ruff check src tests
$env:PYTHONPATH='src'; uv run python -m inferno.cli plan-capacity --config configs/planner/p7_capacity.yaml
$env:PYTHONPATH='src'; uv run python -m inferno.cli validate --run artifacts/runs/p5-vllm-p5-smoke-r01-20260701T135244Z-049adeba
$env:PYTHONPATH='src'; uv run python -m inferno.cli validate --run artifacts/runs/p6-llamacpp-profile-local-r01-20260701T180433Z-6e6969f5
$env:PYTHONPATH='src'; uv run python -m inferno.cli compare --mode strict --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/profile_local/latest.json"
git diff --check
git diff --cached --check
```

The strict compare command is an expected refusal gate: it must fail with
`strict comparison rejects llamacpp; use MODE=deployment-profile`.

### Evidence Index

- Router code: `src/inferno/router.py`
- CLI entrypoint: `src/inferno/cli.py`
- Router config: `configs/router/p8_offline_router.yaml`
- Router docs: `docs/offline-router.md`
- Router tests: `tests/test_router.py`
- Router artifacts:
  - `artifacts/router/p8_offline_policy_replay/latest.json`
  - `artifacts/router/p8_offline_policy_replay/decision_log.jsonl`
  - `artifacts/router/p8_offline_policy_replay/leakage_check.json`
  - `artifacts/router/p8_offline_policy_replay/ablation_eval.json`
  - `artifacts/router/p8_offline_policy_replay/report.md`
- P8 handoff: `.inferno/handoffs/P8-OFFLINE-ROUTER.yaml`

### Result

P8 completed as an honest negative result. The replay consumed 30 compatible
P5 strict evidence runs from `p5_research_core_block_001`, formed 15 held-out
replay requests, and passed leave-one-workload-block-out leakage checks and
ablation checks.

The declared objective is
`minimize_mean_e2e_ms_with_failure_and_quality_penalties`. Static SGLang is
the best static baseline with objective value 7904.385147. The proposed policy
also selected SGLang for all 15 replay requests and produced the same objective
value, so improvement over the best static baseline is 0.0% and the result is
reported as `NEGATIVE_RESULT`.

Static llama.cpp is omitted from the strict P8 replay because llama.cpp remains
deployment-profile only. The oracle lower bound is explicitly non-deployable.
Cost remains `UNAVAILABLE` because no dated per-candidate price values were
supplied. P8 makes no new real-engine readiness claim and required no new GPU
command.

## P9 - Hardening

### Scope

P9 adds repository hardening evidence after P8. It covers clean-clone lockfile
setup, CPU-only fake-based tests, a GPU-gated host/container smoke check,
schema snapshot drift, report snapshots, static browser-preview generation,
image/model provenance, secret scanning, offline dependency audit, and
P5/P6/P7/P8 regression baselines.

Out of scope for P9: live routing, dashboards, cloud control plane, new engine
adapters, benchmark runners, additional studies, and P10+ scope.

### Acceptance Gates

```bash
uv run pytest -q tests/test_hardening.py
uv run ruff check src/inferno/hardening.py src/inferno/cli.py tests/test_hardening.py
PYTHONPATH=src uv run python -m inferno.cli hardening-check --config configs/hardening/p9_hardening.yaml
INFERNO_GPU_SSH=[operator supplied] PYTHONPATH=src uv run python -m inferno.cli hardening-check --config configs/hardening/p9_hardening.yaml --include-gpu-smoke
uv sync --all-groups --frozen
PYTHONPATH=src uv run python -m inferno.cli doctor
PYTHONPATH=src uv run python -m inferno.cli validate --run artifacts/runs/p5-vllm-p5-smoke-r01-20260701T135244Z-049adeba
PYTHONPATH=src uv run python -m inferno.cli validate --run artifacts/runs/p6-llamacpp-profile-local-r01-20260701T180433Z-6e6969f5
PYTHONPATH=src uv run python -m inferno.cli plan-capacity --config configs/planner/p7_capacity.yaml
PYTHONPATH=src uv run python -m inferno.cli replay-router --config configs/router/p8_offline_router.yaml
PYTHONPATH=src uv run python -m inferno.cli compare --mode strict --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/profile_local/latest.json"
uv run pytest -q
uv run ruff check src tests
git diff --check
git diff --cached --check
```

The strict compare command is an expected refusal gate: it must fail with
`strict comparison rejects llamacpp; use MODE=deployment-profile`.

### Evidence Index

- Hardening code: `src/inferno/hardening.py`
- CLI entrypoint: `src/inferno/cli.py`
- Hardening config: `configs/hardening/p9_hardening.yaml`
- Hardening docs: `docs/hardening.md`
- Hardening tests: `tests/test_hardening.py`
- ADR: `.inferno/adr/ADR-0002-p9-hardening.md`
- Hardening artifacts:
  - `artifacts/hardening/p9_hardening/latest.json`
  - `artifacts/hardening/p9_hardening/gpu_smoke.json`
  - `artifacts/hardening/p9_hardening/report_preview.html`
  - `artifacts/hardening/p9_hardening/report.md`
- P9 handoff: `.inferno/handoffs/P9-HARDENING.yaml`

### Result

P9 passed. The hardening gate reports `PASS` across nine checks:
clean-clone lockfile setup, schema snapshot drift, report snapshots,
image/model provenance, secret scan, dependency audit, P5/P6/P7/P8 regression
baselines, static browser preview, and GPU-gated real host/container smoke.

The GPU smoke evidence used `INFERNO_GPU_SSH` from the shell and redacted the
target. It proved remote OS, Python, git, Docker, `nvidia-smi`, and an NVIDIA
CUDA container smoke on a Tesla T4 with driver 580.159.03 and CUDA 13.0. P9
makes no new engine readiness claim.

P5 Research Core remains valid, P6 llama.cpp remains deployment-profile only,
P7 capacity planner artifacts remain valid, and P8 remains an honest negative
offline-router result. Strict mode still rejects llama.cpp.

## Next Phase

## P10 - TensorRT-LLM And Expanded Workloads

### Scope

P10 adds TensorRT-LLM as engine-configuration evidence and extends workload
metadata without weakening P5 strict vLLM/SGLang claims or P6 deployment-profile
semantics.

Two TensorRT-LLM profiles exist:

- `tensorrtllm`: current NVIDIA TensorRT-LLM release container using
  `trtllm-serve serve` and the P5 `Qwen/Qwen3.5-2B` pin for supported newer
  GPU families.
- `tensorrtllm_t4`: legacy Tesla T4 profile using the official
  `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3` image, TensorRT-LLM
  0.11.0, the legacy `LLM` API, and
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

### Acceptance Gates

```bash
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff check src tests
PYTHONPATH=src INFERNO_GPU_SSH=[operator supplied] uv run python -m inferno.cli study --engine tensorrtllm_t4 --config configs/study/tensorrtllm_t4_smoke.yaml
PYTHONPATH=src uv run python -m inferno.cli validate --run artifacts/runs/p10-tensorrtllm_t4-smoke-r01-20260702T161929Z-9449e7f9
PYTHONPATH=src uv run python -m inferno.cli report --study configs/study/tensorrtllm_t4_smoke.yaml
PYTHONPATH=src uv run python -m inferno.cli compare --mode engine-configuration --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/tensorrtllm_t4_smoke/latest.json"
PYTHONPATH=src uv run python -m inferno.cli hardening-check --config configs/hardening/p9_hardening.yaml --include-gpu-smoke
git diff --check
git diff --cached --check
```

Strict comparison against the P6 profile remains an expected refusal.

### Evidence Index

- Current TensorRT-LLM config: `configs/engines/tensorrtllm.json`
- Legacy T4 TensorRT-LLM config: `configs/engines/tensorrtllm_t4.json`
- Legacy T4 study config: `configs/study/tensorrtllm_t4_smoke.yaml`
- Legacy T4 workload config: `configs/workloads/tensorrtllm_t4_smoke.yaml`
- Legacy T4 run:
  `artifacts/runs/p10-tensorrtllm_t4-smoke-r01-20260702T161929Z-9449e7f9`
- Legacy T4 study report: `artifacts/studies/tensorrtllm_t4_smoke/report.md`
- Engine-configuration report:
  `artifacts/compare/engine-configuration-20260702T162232Z/report.md`
- Current-release negative run:
  `artifacts/runs/p10-tensorrtllm-smoke-r01-20260702T055403Z-0472d193`
- P10 docs: `docs/tensorrt-llm.md`
- P10 ADR: `.inferno/adr/ADR-0003-p10-tensorrtllm-workloads.md`
- P10 handoff: `.inferno/handoffs/P10-TENSORRTLLM-WORKLOADS.yaml`

### Result

P10 passed as an engine-configuration extension. The legacy T4 TensorRT-LLM
profile built a real TensorRT-LLM engine on the Tesla T4 host, generated a
non-empty response, wrote trace/telemetry/summary/checksum artifacts, and
validated.

The current TensorRT-LLM release profile remains negative on T4: it exits
before health with `KeyError: 'sm_75'`. That negative artifact does not count
as current TensorRT-LLM readiness, Qwen support, or strict engine evidence.

P5 Research Core remains valid, P6 llama.cpp remains deployment-profile only,
P7 capacity planner artifacts remain valid, P8 remains an honest negative
offline-router result, and P9 hardening still passes. Strict mode still rejects
llama.cpp and TensorRT-LLM evidence.

## Next Phase

## P11 - GPU-Aware Local Browser Dashboard

### Scope

P11 adds a local-only browser dashboard for non-technical operator workflows.
It is a React/Vite frontend served by a Python FastAPI backend through
`uv run python -m inferno.cli dashboard`. The dashboard detects the remote GPU,
recommends safe engine profiles, runs selected engines sequentially through the
existing artifact-producing path, and displays validation-backed results.

P11 also adds Ollama as deployment-profile evidence using the official pinned
Docker image and default `qwen3.5:2b` model. Ollama is not strict engine
evidence. TensorRT-LLM remains engine-configuration evidence. On Tesla T4 /
SM75, the dashboard selects the legacy `tensorrtllm_t4` profile, forces the
dashboard default `Qwen/Qwen3.5-2B`, and labels it unproven on T4 until a real
run validates.

Out of scope for P11: public live endpoints, cloud control plane, live routing,
autoscaling, broad benchmark suites, committed SSH targets, model weights, and
strict claims involving llama.cpp, Ollama, or TensorRT-LLM.

### Acceptance Gates

```bash
uv sync --all-groups --frozen
uv run pytest -q
uv run ruff check src tests
npm install  # cwd: web/dashboard
npm run build  # cwd: web/dashboard
PYTHONPATH=src uv run python -m inferno.cli dashboard --smoke --no-open
PYTHONPATH=src uv run python -m inferno.cli compare --mode strict --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/profile_local/latest.json"
git diff --check
git diff --cached --check
```

The strict compare command remains an expected refusal gate. Strict mode must
continue to reject llama.cpp and must not accept Ollama or TensorRT-LLM.

GPU-gated real runs require an operator-supplied `INFERNO_GPU_SSH` or a
temporary dashboard SSH target. The literal target must not be committed.

### Evidence Index

- Dashboard backend: `src/inferno/dashboard/`
- Dashboard frontend: `web/dashboard/`
- Dashboard CLI: `src/inferno/cli.py`
- Ollama engine config: `configs/engines/ollama.json`
- Ollama workload config: `configs/workloads/ollama_smoke.yaml`
- Ollama study config: `configs/study/ollama_smoke.yaml`
- P11 ADR: `.inferno/adr/ADR-0004-p11-dashboard.md`

### Result

P11 local implementation gates pass. Real GPU smoke runs for the dashboard,
Ollama, and TensorRT profile selection remain pending until an operator supplies
an SSH target at runtime. The dashboard shows only `Validated`, `Needs SSH`,
`Unproven on this GPU`, `Legacy profile`, or `Failed` states. It never derives
readiness from config presence alone; validated claims require validated run
artifacts.
