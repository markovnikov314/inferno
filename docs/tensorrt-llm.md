# TensorRT-LLM Profile

Inferno includes TensorRT-LLM as an engine-configuration profile. The current
profile uses NVIDIA's TensorRT-LLM release container and the `trtllm-serve`
OpenAI API path.

There are two profiles:

- `tensorrtllm`: current NVIDIA TensorRT-LLM release container for newer
  supported GPU families.
- `tensorrtllm_t4`: legacy Tesla T4 profile using
  `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3` and the older
  TensorRT-LLM `LLM` API.

Run smoke studies:

```bash
PYTHONPATH=src INFERNO_GPU_SSH="[operator supplied target]" uv run python -m inferno.cli study --engine tensorrtllm --config configs/study/tensorrtllm_smoke.yaml
PYTHONPATH=src INFERNO_GPU_SSH="[operator supplied target]" uv run python -m inferno.cli study --engine tensorrtllm_t4 --config configs/study/tensorrtllm_t4_smoke.yaml
```

Compare with:

```bash
PYTHONPATH=src uv run python -m inferno.cli compare --mode engine-configuration --runs "artifacts/studies/research_smoke/latest.json artifacts/studies/tensorrtllm_t4_smoke/latest.json"
```

Use `MODE=engine-configuration` for TensorRT-LLM reports.
