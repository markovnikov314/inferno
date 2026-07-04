import json
from pathlib import Path

import pytest

from inferno import artifact_contract as contract


def test_valid_run_validates_and_schema_snapshots_match(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path)

    result = contract.validate_run(run_dir, write=False)

    assert result.ok, result.errors
    for name, expected in contract.schema_snapshots().items():
        assert (Path("schemas/artifacts") / name).read_text(encoding="utf-8") == expected


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("missing_tokenizer_revision", "invalid manifest"),
        ("posthoc_manifest", "manifest timestamp is after first request"),
        ("non_monotonic_trace", "non-monotonic"),
        ("corrupt_checksum", "checksum mismatch"),
        ("missing_raw_stream", "missing artifact"),
        ("empty_generated_text", "stream evidence did not produce text"),
        ("fake_engine", "engine evidence is not pinned real"),
        ("comparison_key_mismatch", "comparison key does not match controlled fields"),
    ],
)
def test_corruptions_fail_loudly(tmp_path: Path, corrupt: str, message: str) -> None:
    run_dir = _valid_run(tmp_path)
    _corrupt(run_dir, corrupt)

    result = contract.validate_run(run_dir, write=False)

    assert not result.ok
    assert message in "\n".join(result.errors)


def test_sglang_run_validates_with_native_stream_artifact(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path, engine="sglang", run_family="strict_comparison")

    result = contract.validate_run(run_dir, write=False)

    assert result.ok, result.errors
    manifest = contract.read_manifest(run_dir)
    assert manifest.artifacts.raw_sse == "raw_streams/sglang_native.sse"
    assert manifest.engine.name == "sglang"


def test_llamacpp_run_validates_with_gguf_profile_fields(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path, engine="llamacpp", run_family="deployment_profile")

    result = contract.validate_run(run_dir, write=False)

    assert result.ok, result.errors
    manifest = contract.read_manifest(run_dir)
    assert manifest.artifacts.raw_sse == "raw_streams/llamacpp_native.sse"
    assert manifest.model.weight_format == "gguf"
    assert manifest.model.quantization == "Q4_K_M"
    assert contract.comparison_key(manifest)["weights"]["format"] == "gguf"


def test_tensorrtllm_run_validates_with_expanded_workload_fields(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path, engine="tensorrtllm", run_family="engine_configuration")

    result = contract.validate_run(run_dir, write=False)

    assert result.ok, result.errors
    manifest = contract.read_manifest(run_dir)
    key = contract.comparison_key(manifest)
    assert manifest.artifacts.raw_sse == "raw_streams/tensorrtllm_native.sse"
    assert key["runtime"]["backend"] == "pytorch"
    assert key["runtime"]["tensor_parallel_size"] == 1
    assert key["workload"]["request_rate_rps"] == 0.1
    assert key["workload"]["streaming"] is True


def test_tensorrtllm_t4_run_validates_with_legacy_profile(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path, engine="tensorrtllm_t4", run_family="engine_configuration")

    result = contract.validate_run(run_dir, write=False)

    assert result.ok, result.errors
    manifest = contract.read_manifest(run_dir)
    key = contract.comparison_key(manifest)
    assert manifest.engine.name == "tensorrtllm_t4"
    assert manifest.artifacts.raw_sse == "raw_streams/tensorrtllm_native.sse"
    assert key["runtime"]["backend"] == "llm_api"
    assert key["model"]["id"] == "Qwen/Qwen3.5-2B"


def test_ollama_run_validates_as_deployment_profile(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path, engine="ollama", run_family="dashboard")

    result = contract.validate_run(run_dir, write=False)

    assert result.ok, result.errors
    manifest = contract.read_manifest(run_dir)
    key = contract.comparison_key(manifest)
    assert manifest.engine.name == "ollama"
    assert manifest.artifacts.raw_sse == "raw_streams/ollama_native.sse"
    assert key["weights"]["format"] == "ollama"


def test_strict_comparison_errors_name_mismatched_control(tmp_path: Path) -> None:
    vllm_manifest = contract.read_manifest(_valid_run(tmp_path / "left"))
    sglang_manifest = contract.read_manifest(
        _valid_run(tmp_path / "right", engine="sglang", run_family="strict_comparison", cache_state="warm_after_warmup")
    )

    errors = contract.strict_comparison_errors([vllm_manifest, sglang_manifest])

    assert errors
    assert "workload.cache_state" in errors[0]


def test_strict_comparison_errors_reject_llamacpp(tmp_path: Path) -> None:
    vllm_manifest = contract.read_manifest(_valid_run(tmp_path / "left"))
    llamacpp_manifest = contract.read_manifest(
        _valid_run(tmp_path / "right", engine="llamacpp", run_family="deployment_profile")
    )

    errors = contract.strict_comparison_errors([vllm_manifest, llamacpp_manifest])

    assert any("rejects llamacpp" in error for error in errors)


def test_strict_comparison_errors_reject_tensorrtllm(tmp_path: Path) -> None:
    vllm_manifest = contract.read_manifest(_valid_run(tmp_path / "left"))
    trt_manifest = contract.read_manifest(
        _valid_run(tmp_path / "right", engine="tensorrtllm", run_family="engine_configuration")
    )

    errors = contract.strict_comparison_errors([vllm_manifest, trt_manifest])

    assert any("supports only vllm and sglang" in error for error in errors)


def test_strict_comparison_errors_reject_tensorrtllm_t4(tmp_path: Path) -> None:
    vllm_manifest = contract.read_manifest(_valid_run(tmp_path / "left"))
    trt_manifest = contract.read_manifest(
        _valid_run(tmp_path / "right", engine="tensorrtllm_t4", run_family="engine_configuration")
    )

    errors = contract.strict_comparison_errors([vllm_manifest, trt_manifest])

    assert any("supports only vllm and sglang" in error for error in errors)


def test_strict_comparison_errors_reject_ollama(tmp_path: Path) -> None:
    vllm_manifest = contract.read_manifest(_valid_run(tmp_path / "left"))
    ollama_manifest = contract.read_manifest(
        _valid_run(tmp_path / "right", engine="ollama", run_family="dashboard")
    )

    errors = contract.strict_comparison_errors([vllm_manifest, ollama_manifest])

    assert any("rejects ollama" in error for error in errors)


def test_batch_summary_aggregates_multiple_requests(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path, run_family="dashboard")
    manifest = contract.read_manifest(run_dir)
    manifest = contract.RunManifest.model_validate(
        manifest.model_dump(mode="json")
        | {
            "workload": manifest.workload.model_dump(mode="json")
            | {
                "measurement_requests": 2,
                "concurrency": 2,
                "max_in_flight": 2,
            }
        }
    )
    raw = {
        "responses": [
            {
                "index": 0,
                "ok": True,
                "generated_text": "Alpha",
                "usage": {"completion_tokens": 2},
            },
            {
                "index": 1,
                "ok": False,
                "generated_text": "",
                "usage": None,
                "error": "timeout",
            },
        ]
    }
    _write_json(run_dir / manifest.artifacts.raw_sse, raw)
    rows = [
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="dashboard-1",
            event="request_start",
            perf_counter_ns=1000,
            request={"model": manifest.engine.served_model_name, "stream": False},
            request_sha256=contract.sha256_text("request-1"),
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="dashboard-2",
            event="request_start",
            perf_counter_ns=1200,
            request={"model": manifest.engine.served_model_name, "stream": False},
            request_sha256=contract.sha256_text("request-2"),
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="dashboard-1",
            event="stream_event",
            perf_counter_ns=1800,
            sse_index=0,
            delta_text_sha256=contract.sha256_text("Alpha"),
            delta_text_chars=5,
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="dashboard-1",
            event="request_end",
            perf_counter_ns=2500,
            ssh_exit_code=0,
            raw_sse=manifest.artifacts.raw_sse,
            stderr="command_logs/batch-requests.json",
            generated_text_sha256=contract.sha256_text("Alpha"),
            generated_text_chars=5,
            usage={"completion_tokens": 2},
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="dashboard-2",
            event="request_end",
            perf_counter_ns=3000,
            ssh_exit_code=1,
            raw_sse=manifest.artifacts.raw_sse,
            stderr="command_logs/batch-requests.json",
            generated_text_sha256=contract.sha256_text(""),
            generated_text_chars=0,
        ),
    ]
    _write_text(
        run_dir / manifest.artifacts.request_trace,
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
    )

    summary = contract.recompute_summary(run_dir, manifest)

    assert summary.request_count == 2
    assert summary.success_count == 1
    assert summary.error_count == 1
    assert summary.error_rate == 0.5
    assert summary.generated_text_chars == 5
    assert summary.completion_tokens == 2
    assert summary.stream_event_count == 2


def _valid_run(
    tmp_path: Path,
    *,
    engine: str = "vllm",
    run_family: str = "single_run",
    cache_state: str = "cold_per_run",
) -> Path:
    run_dir = tmp_path / f"{run_family.lower()}-{engine}-test"
    run_dir.mkdir(parents=True)
    artifacts = contract.default_artifacts(engine)
    if run_family in {"smoke_study", "strict_comparison", "research_core", "deployment_profile", "engine_configuration", "dashboard"}:
        artifacts = contract.ArtifactPaths.model_validate(
            artifacts.model_dump(mode="json")
            | {"request_trace_parquet": "request_trace.parquet", "telemetry_parquet": "telemetry.parquet"}
        )
    is_trt = engine in {"tensorrtllm", "tensorrtllm_t4"}
    is_dashboard = run_family == "dashboard"
    workload = contract.WorkloadInfo(
        workload_id="ollama_smoke" if engine == "ollama" else "smoke_real",
        prompt_template_id="dashboard-ollama-smoke-chat-v1" if engine == "ollama" else "smoke-chat-v1",
        seed=123,
        prompt_sha256=contract.sha256_text("hello"),
        prompt_chars=5,
        request_rate_rps=0.1 if is_trt or engine == "ollama" else None,
        arrival_model="closed_loop",
        concurrency=1 if is_trt or engine == "ollama" else None,
        max_in_flight=1 if is_trt or engine == "ollama" else None,
        prompt_target={"unit": "chars", "p50": 5, "p95": 5}
        if is_trt or engine == "ollama"
        else {},
        output_target={"unit": "tokens", "p50": 4, "p95": 4}
        if is_trt or engine == "ollama"
        else {},
        context_length=2048 if engine == "ollama" else 512 if is_trt else None,
        cache_state=cache_state,
        cache_policy="ollama_default" if engine == "ollama" else "no_prefix_reuse" if is_trt else None,
        warmup_requests=0,
        measurement_requests=1,
        max_runtime_seconds=300,
        streaming=True if engine == "ollama" else engine == "tensorrtllm" if is_trt else None,
        timeout_seconds=300 if is_trt or engine == "ollama" else None,
        retry_policy="none",
        failure_policy="fail_run",
        quality_requirement={"type": "non_empty_text"} if engine == "ollama" else {"type": "none"} if is_trt else {},
        cost_metadata={"status": "UNAVAILABLE"} if is_trt or engine == "ollama" else {},
        runtime_knobs={"deployment_profile": "ollama-docker-gpu"}
        if engine == "ollama"
        else {"backend": "llm_api" if engine == "tensorrtllm_t4" else "pytorch"}
        if is_trt
        else {},
        sampling={"temperature": 0, "max_tokens": 4},
    )
    manifest = contract.RunManifest(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        run_family=run_family,
        study_id=(
            "strict_interactive"
            if run_family == "strict_comparison"
            else "profile_local"
            if run_family == "deployment_profile"
            else "tensorrtllm_smoke"
            if run_family == "engine_configuration"
            else "ollama_smoke"
            if is_dashboard
            else None
        ),
        repeat_index=1 if run_family in {"strict_comparison", "deployment_profile", "engine_configuration", "dashboard"} else None,
        run_id=run_dir.name,
        status="SUCCEEDED",
        created_at="2026-07-01T00:00:00+00:00",
        updated_at="2026-07-01T00:00:01+00:00",
        manifest_created_before_request=True,
        manifest_written_perf_counter_ns=900,
        first_request_perf_counter_ns=1000,
        run_dir=str(run_dir),
        source=contract.SourceInfo(branch="workflow/vllm-vertical-slice", commit="abc", dirty=True),
        engine=contract.EngineInfo(
            name=engine,
            image=_image(engine),
            container_name=f"inferno-test-{engine}",
            port=11434 if engine == "ollama" else 8080 if engine == "llamacpp" else 8090 if engine == "tensorrtllm" else 8000,
            served_model_name=_served_model(engine),
            launch_args=_launch_args(engine),
        ),
        model=_model(engine),
        hardware=contract.HardwareInfo(
            gpu_model="Tesla T4",
            gpu_count=1,
            driver_version="580.159.03",
            cuda_version="13.0",
            memory_mib=15360,
        ),
        workload=workload,
        artifacts=artifacts,
        comparison_key={},
        health=contract.HealthEvidence(ok=True, model_ids=[_served_model(engine)]),
        stream=contract.StreamEvidence(
            ok=True,
            generated_text_sha256=contract.sha256_text("Hello"),
            generated_text_chars=5,
        ),
        shutdown=contract.ShutdownEvidence(ok=True),
    )
    manifest = contract.RunManifest.model_validate(
        manifest.model_dump(mode="json") | {"comparison_key": contract.comparison_key(manifest)}
    )
    _write_text(run_dir / artifacts.raw_sse, _raw_sse())
    contract.write_json(run_dir / artifacts.health, {"ok": True, "model_ids": [_served_model(engine)]})
    _write_text(run_dir / artifacts.engine_logs, f"{engine} log\n")
    contract.write_json(run_dir / "command_logs/up.json", {"exit_code": 0})
    _write_trace(run_dir, manifest)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    contract.write_trace_parquet(run_dir, manifest)
    if artifacts.telemetry_parquet:
        contract.write_telemetry_parquet(run_dir, manifest, [_telemetry_row()])
    contract.write_json(run_dir / artifacts.summary, contract.recompute_summary(run_dir, manifest))
    contract.write_checksums(run_dir, manifest)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    return run_dir


def _write_trace(run_dir: Path, manifest: contract.RunManifest) -> None:
    rows = [
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_start",
            perf_counter_ns=1000,
            request={"model": manifest.engine.served_model_name},
            request_sha256=contract.sha256_text("request"),
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="stream_event",
            perf_counter_ns=1100,
            sse_index=0,
            delta_text_sha256=contract.sha256_text(""),
            delta_text_chars=0,
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="stream_event",
            perf_counter_ns=2000,
            sse_index=1,
            delta_text_sha256=contract.sha256_text("Hello"),
            delta_text_chars=5,
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_end",
            perf_counter_ns=3000,
            ssh_exit_code=0,
            raw_sse=manifest.artifacts.raw_sse,
            stderr="command_logs/real-stream.stderr",
            generated_text_sha256=contract.sha256_text("Hello"),
            generated_text_chars=5,
        ),
    ]
    _write_text(
        run_dir / manifest.artifacts.request_trace,
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
    )


def _corrupt(run_dir: Path, name: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if name == "missing_tokenizer_revision":
        del manifest["model"]["tokenizer_revision"]
        _write_json(manifest_path, manifest)
    elif name == "posthoc_manifest":
        manifest["manifest_written_perf_counter_ns"] = 2000
        _write_json(manifest_path, manifest)
    elif name == "non_monotonic_trace":
        rows = (run_dir / "request_trace.jsonl").read_text(encoding="utf-8").splitlines()
        event = json.loads(rows[1])
        event["perf_counter_ns"] = 500
        rows[1] = json.dumps(event, sort_keys=True)
        _write_text(run_dir / "request_trace.jsonl", "\n".join(rows) + "\n")
    elif name == "corrupt_checksum":
        _write_text(run_dir / "engine_logs/vllm.log", "changed\n")
    elif name == "missing_raw_stream":
        (run_dir / "raw_streams/stream.sse").unlink()
    elif name == "empty_generated_text":
        manifest["stream"]["ok"] = False
        manifest["stream"]["generated_text_chars"] = 0
        _write_json(manifest_path, manifest)
    elif name == "fake_engine":
        manifest["engine"]["image"] = "fake-server:latest"
        _write_json(manifest_path, manifest)
    elif name == "comparison_key_mismatch":
        manifest["comparison_key"]["model"]["revision"] = "different"
        _write_json(manifest_path, manifest)
    else:
        raise AssertionError(name)


def _raw_sse() -> str:
    return "\n".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            "data: [DONE]",
            "",
        ]
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _launch_args(engine: str) -> list[str]:
    if engine == "tensorrtllm_t4":
        return [
            "Qwen/Qwen3.5-2B",
            "--revision",
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "--backend",
            "llm_api",
            "--dtype",
            "float16",
            "--max_batch_size",
            "1",
            "--max_seq_len",
            "160",
            "--tensor_parallel_size",
            "1",
        ]
    if engine == "tensorrtllm":
        return [
            "Qwen/Qwen3.5-2B",
            "--hf_revision",
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "--backend",
            "pytorch",
            "--max_batch_size",
            "1",
            "--max_seq_len",
            "512",
            "--tp_size",
            "1",
            "--pp_size",
            "1",
            "--kv_cache_free_gpu_memory_fraction",
            "0.80",
        ]
    if engine == "llamacpp":
        return [
            "--hf-repo",
            "lmstudio-community/Qwen3.5-2B-GGUF:Q4_K_M",
            "--hf-file",
            "Qwen3.5-2B-Q4_K_M.gguf",
            "--alias",
            "lmstudio-community/Qwen3.5-2B-GGUF:Q4_K_M",
            "--ctx-size",
            "512",
            "--n-gpu-layers",
            "all",
            "--parallel",
            "1",
        ]
    if engine == "ollama":
        return []
    if engine == "sglang":
        return [
            "--dtype",
            "half",
            "--context-length",
            "512",
            "--mem-fraction-static",
            "0.80",
            "--max-running-requests",
            "1",
            "--trust-remote-code",
        ]
    return [
        "--dtype",
        "half",
        "--max-model-len",
        "512",
        "--gpu-memory-utilization",
        "0.80",
        "--max-num-seqs",
        "1",
        "--trust-remote-code",
    ]


def _image(engine: str) -> str:
    if engine == "tensorrtllm_t4":
        return "nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3@sha256:" + "a" * 64
    if engine == "tensorrtllm":
        return "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:" + "a" * 64
    if engine == "sglang":
        return "lmsysorg/sglang@sha256:" + "a" * 64
    if engine == "llamacpp":
        return (
            "ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:"
            + "a" * 64
        )
    if engine == "ollama":
        return "ollama/ollama@sha256:" + "a" * 64
    return "vllm/vllm-openai@sha256:" + "a" * 64


def _served_model(engine: str) -> str:
    if engine == "tensorrtllm_t4":
        return "Qwen/Qwen3.5-2B"
    if engine == "tensorrtllm":
        return "Qwen/Qwen3.5-2B"
    if engine == "llamacpp":
        return "lmstudio-community/Qwen3.5-2B-GGUF:Q4_K_M"
    if engine == "ollama":
        return "qwen3.5:2b"
    return "Qwen/Qwen3.5-2B"


def _model(engine: str) -> contract.ModelInfo:
    if engine == "tensorrtllm_t4":
        revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
        return contract.ModelInfo(
            id="Qwen/Qwen3.5-2B",
            revision=revision,
            tokenizer_id="Qwen/Qwen3.5-2B",
            tokenizer_revision=revision,
            tokenizer_format="huggingface",
            weight_format="huggingface",
            weight_precision="float16",
        )
    if engine == "tensorrtllm":
        revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
        return contract.ModelInfo(
            id="Qwen/Qwen3.5-2B",
            revision=revision,
            tokenizer_id="Qwen/Qwen3.5-2B",
            tokenizer_revision=revision,
            tokenizer_format="huggingface",
            weight_format="huggingface",
            weight_precision="pytorch_backend_auto",
        )
    if engine == "llamacpp":
        revision = "bb84e11355a036e28f080c7793fa6d22b7c4e344"
        return contract.ModelInfo(
            id="lmstudio-community/Qwen3.5-2B-GGUF",
            revision=revision,
            tokenizer_id="gguf:lmstudio-community/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf",
            tokenizer_revision=revision,
            tokenizer_format="gguf_embedded",
            weight_format="gguf",
            weight_precision="quantized",
            quantization="Q4_K_M",
            gguf_repo="lmstudio-community/Qwen3.5-2B-GGUF",
            gguf_filename="Qwen3.5-2B-Q4_K_M.gguf",
            gguf_revision=revision,
        )
    if engine == "ollama":
        revision = "ollama-library-tag-qwen3.5-2b"
        return contract.ModelInfo(
            id="qwen3.5:2b",
            revision=revision,
            tokenizer_id="ollama:qwen3.5:2b",
            tokenizer_revision=revision,
            tokenizer_format="ollama_embedded",
            weight_format="ollama",
            weight_precision="ollama_auto",
        )
    return contract.ModelInfo(
        id="Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        tokenizer_id="Qwen/Qwen3.5-2B",
        tokenizer_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    )


def _telemetry_row() -> dict[str, object]:
    row = {"sample_index": 0, "collected_at_unix_ns": 1, "source": "test"}
    for metric in (
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_power_watts",
        "gpu_temperature_c",
        "host_cpu_percent",
        "host_rss_mib",
    ):
        row[metric] = None
        row[f"{metric}_missing_reason"] = "not_available_in_test"
    return row
