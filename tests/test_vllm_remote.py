from pathlib import Path

import pytest

from inferno import vllm_remote


def test_resolve_model_uses_pinned_config() -> None:
    config = {
        "engine": "vllm",
        "model": {
            "id": "Qwen/Qwen3.5-2B",
            "revision": "abc",
            "tokenizer_id": "Qwen/Qwen3.5-2B",
            "tokenizer_revision": "abc",
        }
    }

    model = vllm_remote.resolve_model("Qwen/Qwen3.5-2B@abc", config)

    assert model.model_id == "Qwen/Qwen3.5-2B"
    assert model.revision == "abc"
    assert model.tokenizer_revision == "abc"


def test_resolve_model_rejects_unpinned_revision() -> None:
    config = {
        "engine": "vllm",
        "model": {
            "id": "Qwen/Qwen3.5-2B",
            "revision": "abc",
            "tokenizer_id": "Qwen/Qwen3.5-2B",
            "tokenizer_revision": "abc",
        }
    }

    with pytest.raises(ValueError):
        vllm_remote.resolve_model("Qwen/Qwen3.5-2B@def", config)


def test_generated_text_parser_reads_chat_sse() -> None:
    raw = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"continuous "}}]}',
            'data: {"choices":[{"delta":{"content":"batching"}}]}',
            "data: [DONE]",
        ]
    )

    assert vllm_remote._generated_text(raw) == "continuous batching"


def test_config_is_pinned_to_qwen35_2b() -> None:
    config = vllm_remote._read_json(Path("configs/engines/vllm.json"))

    assert config["model"]["id"] == "Qwen/Qwen3.5-2B"
    assert config["served_model_name"] == "Qwen/Qwen3.5-2B"
    assert len(config["model"]["revision"]) == 40
    assert config["model"]["revision"] == config["model"]["tokenizer_revision"]
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in config["env"]
    assert "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0" in config["env"]
    assert config["launch_args"][config["launch_args"].index("--max-model-len") + 1] == "256"


def test_sglang_config_is_strict_compatible_with_vllm() -> None:
    vllm = vllm_remote._read_json(Path("configs/engines/vllm.json"))
    sglang = vllm_remote._read_json(Path("configs/engines/sglang.json"))
    model = vllm_remote.resolve_model(None, sglang)

    assert sglang["engine"] == "sglang"
    assert sglang["image"].startswith("lmsysorg/sglang@sha256:")
    assert sglang["model"] == vllm["model"]
    assert "--context-length" in sglang["launch_args"]
    assert sglang["launch_args"][sglang["launch_args"].index("--context-length") + 1] == (
        vllm["launch_args"][vllm["launch_args"].index("--max-model-len") + 1]
    )
    assert model.tokenizer_revision == model.revision


def test_llamacpp_config_declares_deployment_profile() -> None:
    config = vllm_remote._read_json(Path("configs/engines/llamacpp.json"))
    model = vllm_remote.resolve_model(None, config)
    launch_args = vllm_remote._launch_args(config, model)

    assert config["engine"] == "llamacpp"
    assert config["image"].startswith("ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:")
    assert config["model"]["weight_format"] == "gguf"
    assert config["model"]["quantization"] == "Q4_K_M"
    assert launch_args[launch_args.index("--hf-repo") + 1] == (
        "lmstudio-community/Qwen3.5-2B-GGUF:Q4_K_M"
    )
    assert launch_args[launch_args.index("--hf-file") + 1] == "Qwen3.5-2B-Q4_K_M.gguf"


def test_tensorrtllm_config_declares_engine_configuration_profile() -> None:
    config = vllm_remote._read_json(Path("configs/engines/tensorrtllm.json"))
    model = vllm_remote.resolve_model(None, config)
    launch_args = vllm_remote._launch_args(config, model)

    assert config["engine"] == "tensorrtllm"
    assert config["image"].startswith("nvcr.io/nvidia/tensorrt-llm/release:")
    assert "@sha256:" in config["image"]
    assert config["model"]["id"] == "Qwen/Qwen3.5-2B"
    assert config["model"]["weight_format"] == "huggingface"
    assert launch_args[0] == "Qwen/Qwen3.5-2B"
    assert launch_args[launch_args.index("--hf_revision") + 1] == config["model"]["revision"]
    assert launch_args[launch_args.index("--backend") + 1] == "pytorch"


def test_tensorrtllm_t4_config_declares_legacy_llm_api_profile() -> None:
    config = vllm_remote._read_json(Path("configs/engines/tensorrtllm_t4.json"))
    model = vllm_remote.resolve_model(None, config)
    launch_args = vllm_remote._launch_args(config, model)

    assert config["engine"] == "tensorrtllm_t4"
    assert config["run_mode"] == "llm_api_once"
    assert config["image"].startswith("nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3@sha256:")
    assert config["model"]["id"] == "Qwen/Qwen3.5-2B"
    assert config["model"]["weight_precision"] == "float16"
    assert launch_args[0] == "Qwen/Qwen3.5-2B"
    assert launch_args[launch_args.index("--revision") + 1] == config["model"]["revision"]
    assert launch_args[launch_args.index("--backend") + 1] == "llm_api"


def test_ollama_config_declares_deployment_profile() -> None:
    config = vllm_remote._read_json(Path("configs/engines/ollama.json"))
    model = vllm_remote.resolve_model(None, config)
    launch_args = vllm_remote._launch_args(config, model)
    up_script = vllm_remote._up_script(config, launch_args)

    assert config["engine"] == "ollama"
    assert config["image"].startswith("ollama/ollama@sha256:")
    assert config["served_model_name"] == "qwen3.5:2b"
    assert config["model"]["weight_format"] == "ollama"
    assert launch_args == []
    assert "docker exec inferno-dashboard-ollama ollama pull qwen3.5:2b" in up_script


def test_up_script_persists_compile_caches() -> None:
    config = vllm_remote._read_json(Path("configs/engines/vllm.json"))
    model = vllm_remote.resolve_model(None, config)
    script = vllm_remote._up_script(config, vllm_remote._launch_args(config, model))

    assert "$HOME/.cache/torch:/root/.cache/torch" in script
    assert "$HOME/.cache/triton:/root/.triton" in script
    assert "$HOME/.cache/vllm:/root/.cache/vllm" in script
    assert "-e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0" in script


def test_marked_json_parser_reads_llm_api_output() -> None:
    payload = vllm_remote._marked_json(
        'log\nINFERNO_JSON_START\n{"choices":[{"message":{"content":"ok"}}]}\nINFERNO_JSON_END\n'
    )

    assert vllm_remote._json_generated_text(payload) == "ok"


def test_dashboard_batch_requested_uses_batch_path() -> None:
    workload = {
        "prompt": "hello",
        "measurement": {"requests": 1},
        "warmup": {"requests": 0},
        "concurrency": 1,
        "runtime_knobs": {"dashboard_batch_prompts": ["hello"]},
    }

    assert not vllm_remote._dashboard_batch_requested(workload)

    workload["runtime_knobs"]["dashboard_requested_engine"] = "vllm"
    assert vllm_remote._dashboard_batch_requested(workload)

    workload["runtime_knobs"] = {"dashboard_batch_prompts": ["hello"]}
    workload["measurement"]["requests"] = 2
    assert vllm_remote._dashboard_batch_requested(workload)

    workload["measurement"]["requests"] = 1
    workload["runtime_knobs"]["dashboard_requested_engine"] = "ollama"
    assert vllm_remote._dashboard_batch_requested(workload)

    workload["runtime_knobs"]["dashboard_batch_prompts"] = ["hello", "second"]
    assert vllm_remote._dashboard_batch_requested(workload)


def test_batch_error_payload_fails_loudly_when_remote_returns_no_responses() -> None:
    payload = vllm_remote._json_or_batch_error('{"responses":[]}', 0)

    assert payload["responses"][0]["ok"] is False
    assert payload["responses"][0]["error"] == "batch command returned no responses"


def test_batch_remote_timeout_uses_dashboard_timeout() -> None:
    workload = {
        "timeout_seconds": 180,
        "measurement": {"requests": 1, "max_runtime_seconds": 180},
    }

    assert vllm_remote._batch_remote_timeout(workload) == 240


def test_dashboard_batch_script_handles_ollama_response_variants() -> None:
    script = vllm_remote._batch_script(
        {"port": 11434, "served_model_name": "qwen3.5:2b"},
        {
            "prompt": "hello",
            "sampling": {"max_tokens": 4},
            "measurement": {"requests": 1, "max_runtime_seconds": 30},
            "runtime_knobs": {"dashboard_batch_prompts": ["hello"]},
        },
    )

    assert "urllib.request" in script
    assert '"stream": True' in script
    assert '"Connection": "close"' in script
    assert "request_timeout = min(timeout_seconds, 180)" in script
    assert "sys.stdout.flush()" in script
    assert "os._exit(0)" in script
    assert "reasoning_content" in script
    assert "reasoning" in script
    assert "stream_events" in script
    assert "empty_generated_text" in script


def test_dashboard_compact_kv_overrides_are_runtime_only() -> None:
    config = {"engine": "sglang", "env": []}
    workload = {"runtime_knobs": {"dashboard_kv_cache_mode": "compact"}}

    updated, args = vllm_remote._dashboard_runtime_overrides(config, workload, ["--dtype", "half"])

    assert updated == config
    assert args[-3:] == ["--kv-cache-dtype", "fp8_e5m2", "--enable-cache-report"]
