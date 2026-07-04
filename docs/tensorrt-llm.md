# TensorRT-LLM P10 Profile

P10 adds TensorRT-LLM as an engine-configuration profile. It uses NVIDIA's
official TensorRT-LLM release container and the `trtllm-serve serve` OpenAI API
server path.

Sources used for the P10 config:

- https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/release
- https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v0.11.0/docs/source/reference/support-matrix.md

There are two P10 profiles:

- `tensorrtllm`: current NVIDIA TensorRT-LLM release container for newer
  supported GPU families. It keeps the P5 `Qwen/Qwen3.5-2B` model pin, but it
  does not pass on the Tesla T4 host.
- `tensorrtllm_t4`: legacy T4 profile using the official
  `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3` image, whose container
  reports TensorRT-LLM `0.11.0`. It uses the old TensorRT-LLM `LLM` API instead
  of `trtllm-serve`, and validates on the T4 host with
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Neither profile is a strict engine-only claim. TensorRT-LLM uses its own
container, backend, runtime knobs, build path, and in the T4 profile a different
model. Compare it with `MODE=engine-configuration` unless a future strict
comparison type proves all controlled variables match.

Run the P10 smoke study:

```bash
PYTHONPATH=src INFERNO_GPU_SSH=[operator supplied] uv run python -m inferno.cli study --engine tensorrtllm --config configs/study/tensorrtllm_smoke.yaml
PYTHONPATH=src INFERNO_GPU_SSH=[operator supplied] uv run python -m inferno.cli study --engine tensorrtllm_t4 --config configs/study/tensorrtllm_t4_smoke.yaml
```

Then validate the run and compare it against a compatible baseline with:

```bash
PYTHONPATH=src uv run python -m inferno.cli validate --run artifacts/runs/<p10-tensorrtllm-run>
PYTHONPATH=src uv run python -m inferno.cli compare --mode engine-configuration --runs "artifacts/studies/p5_smoke/latest.json artifacts/studies/tensorrtllm_t4_smoke/latest.json"
```

## GPU Gate Results

The current TensorRT-LLM release container was attempted on the configured GPU
host. The host GPU is a Tesla T4. NVIDIA's current support matrix lists
Blackwell, Grace Hopper, Hopper, Ada, and Ampere families, but not Turing/T4.
That artifact remains negative evidence: the container starts, then exits
before health because the runtime compilation path fails on `sm_75`.

Recorded negative evidence:

- run artifact: `artifacts/runs/p10-tensorrtllm-smoke-r01-20260702T055403Z-0472d193`
- study report: `artifacts/studies/tensorrtllm_smoke/report.md`
- failed gate reason: official TensorRT-LLM container exited before health with
  `KeyError: 'sm_75'`

Recorded positive T4 legacy evidence:

- run artifact: `artifacts/runs/p10-tensorrtllm_t4-smoke-r01-20260702T161929Z-9449e7f9`
- study report: `artifacts/studies/tensorrtllm_t4_smoke/report.md`
- engine-configuration report:
  `artifacts/compare/engine-configuration-20260702T162232Z/report.md`
- validation: `ok=true`

Do not treat the T4 legacy result as proof for current TensorRT-LLM, Qwen, or
OpenAI-compatible serving. Re-run `tensorrtllm` on an officially supported GPU
family before making those claims.

Strict mode remains limited to the established vLLM/SGLang envelope and must
continue to reject llama.cpp or TensorRT-LLM evidence.
