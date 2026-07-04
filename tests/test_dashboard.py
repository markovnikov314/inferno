import json
from pathlib import Path
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from inferno import artifact_contract as contract
from inferno import vllm_remote
from inferno.dashboard import jobs
from inferno.dashboard.compat import (
    GPU_ENV_VAR,
    STATUS_FAILED,
    STATUS_NEEDS_SSH,
    STATUS_UNPROVEN,
    STATUS_VALIDATED,
    GpuInfo,
    gpu_preflight,
    parse_gpu_probe,
    recommend_engines,
)
from inferno.dashboard.server import _is_loopback_host, _json_sse, create_app
from inferno.preflight import CommandResult


T4_PROBE = """INFERNO_GPU_QUERY
Tesla T4, 580.159.03, 15360, 7.5
INFERNO_CUDA_QUERY
13.0
INFERNO_DOCKER_GPU
0
"""


def test_preflight_without_ssh_returns_needs_ssh(tmp_path: Path) -> None:
    result = gpu_preflight(env={}, project_root=tmp_path)

    assert result["status"] == STATUS_NEEDS_SSH
    assert result["gpu"] is None
    assert {item["status"] for item in result["recommendations"]} == {STATUS_NEEDS_SSH}


def test_t4_recommendations_select_legacy_tensorrt_profile() -> None:
    gpu = parse_gpu_probe(T4_PROBE)

    assert gpu is not None
    recommendations = {item["engine"]: item for item in recommend_engines(gpu)}

    assert recommendations["vllm"]["status"] == STATUS_VALIDATED
    assert recommendations["sglang"]["status"] == STATUS_VALIDATED
    assert recommendations["tensorrtllm"]["status"] == STATUS_FAILED
    assert recommendations["tensorrtllm"]["selected_profile"] == "tensorrtllm_t4"
    assert any("legacy TensorRT-LLM 0.11.0" in warning for warning in recommendations["tensorrtllm"]["warnings"])
    assert any("known failed T4 path" in warning for warning in recommendations["tensorrtllm"]["warnings"])


def test_unknown_gpu_recommendations_mark_current_tensorrt_unproven() -> None:
    gpu = GpuInfo(
        name="NVIDIA L40S",
        driver_version="580.159.03",
        memory_mib=46068,
        compute_capability="8.9",
        cuda_version="13.0",
        docker_gpu_ok=True,
    )

    recommendations = {item["engine"]: item for item in recommend_engines(gpu)}

    assert recommendations["vllm"]["status"] == STATUS_UNPROVEN
    assert recommendations["tensorrtllm"]["status"] == STATUS_UNPROVEN
    assert recommendations["tensorrtllm"]["selected_profile"] == "tensorrtllm"


def test_gpu_preflight_redacts_supplied_ssh_target(tmp_path: Path) -> None:
    def fake_runner(args: object, timeout: int) -> CommandResult:
        command = list(args)[-1]  # type: ignore[arg-type]
        if "INFERNO_GPU_QUERY" in str(command):
            return CommandResult(0, T4_PROBE, "")
        return CommandResult(0, "ok", "")

    result = gpu_preflight(
        env={GPU_ENV_VAR: "operator@192.0.2.50"},
        project_root=tmp_path,
        runner=fake_runner,
    )

    payload = json.dumps(result, sort_keys=True)
    assert result["gpu"]["name"] == "Tesla T4"
    assert "operator" not in payload
    assert "192.0.2.50" not in payload


def test_dashboard_job_maps_tensorrt_to_t4_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_base_workload(tmp_path / "configs/workloads/tensorrtllm_t4_smoke.yaml")
    calls = []

    def fake_preflight(**kwargs: object) -> dict[str, object]:
        return {
            "gpu": {"name": "Tesla T4", "compute_capability": "7.5"},
            "recommendations": [
                {
                    "engine": "tensorrtllm",
                    "status": STATUS_UNPROVEN,
                    "selected_profile": "tensorrtllm_t4",
                    "warnings": [],
                }
            ],
        }

    def fake_runner(**kwargs: object) -> vllm_remote.RunRealResult:
        calls.append(kwargs)
        run_dir = tmp_path / "artifacts/runs/dashboard-dashboard-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        return vllm_remote.RunRealResult(0, run_dir.name, run_dir, True)

    monkeypatch.setattr(jobs, "gpu_preflight", fake_preflight)
    manager = jobs.DashboardJobManager(project_root=tmp_path, runner=fake_runner, env={})
    job = manager.create_job(
        engines=["tensorrtllm"],
        parameters={
            "model_profile": "qwen35-2b",
            "prompt": "hello",
            "batch_prompts": "hello\nsecond prompt",
            "total_requests": 3,
            "concurrent_users": 2,
            "request_rate_rps": 1.5,
            "warmup_requests": 1,
            "kv_cache_mode": "compact",
            "max_tokens": 1024,
            "temperature": 0,
            "top_p": 0.9,
            "timeout_seconds": 120,
        },
        ssh_target="operator@gpu-host",
    )

    for _ in range(100):
        snapshot = manager.get_job(job.job_id)
        if snapshot and snapshot.status not in {"queued", "running"}:
            break
        time.sleep(0.05)

    snapshot = manager.get_job(job.job_id)
    assert snapshot is not None
    assert snapshot.status == "succeeded"
    assert calls[0]["engine"] == "tensorrtllm_t4"
    assert calls[0]["run_family"] == "dashboard"
    assert calls[0]["env"]["INFERNO_ENGINE_HEALTH_TIMEOUT"] == "120"
    assert any(event["type"] == "engine_readiness" for event in snapshot.events)
    workload_path = tmp_path / calls[0]["config_path"]
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    assert workload["runtime_knobs"]["dashboard_requested_engine"] == "tensorrtllm"
    assert workload["runtime_knobs"]["dashboard_model_profile"] == "qwen35-2b"
    assert workload["runtime_knobs"]["dashboard_batch_prompts"] == ["hello", "second prompt"]
    assert workload["runtime_knobs"]["dashboard_requested_max_tokens"] == 1024
    assert workload["runtime_knobs"]["dashboard_effective_max_tokens"] == 24
    assert workload["measurement"]["requests"] == 3
    assert workload["warmup"]["requests"] == 1
    assert workload["cache_state"] == "warm_after_warmup"
    assert workload["concurrency"] == 2
    assert workload["max_in_flight"] == 2
    assert workload["request_rate_rps"] == 1.5
    assert workload["streaming"] is True
    assert workload["sampling"]["max_tokens"] == 24
    assert workload["sampling"]["top_p"] == 0.9
    assert workload["timeout_seconds"] == 120
    assert workload["runtime_knobs"]["dashboard_kv_cache_mode"] == "compact"


def test_dashboard_api_engine_catalog_and_preflight(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(GPU_ENV_VAR, raising=False)
    client = TestClient(create_app(project_root=tmp_path))

    engines = client.get("/api/engines")
    preflight = client.post("/api/preflight", json={})

    assert engines.status_code == 200
    assert {item["id"] for item in engines.json()["engines"]} >= {
        "vllm",
        "sglang",
        "ollama",
        "tensorrtllm",
    }
    assert preflight.status_code == 200
    assert preflight.json()["status"] == STATUS_NEEDS_SSH


def test_dashboard_api_rejects_job_without_ssh(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(GPU_ENV_VAR, raising=False)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post("/api/jobs", json={"engines": ["vllm"]})

    assert response.status_code == 400
    assert "SSH access is required" in response.json()["detail"]


def test_dashboard_sse_json_replaces_non_finite_values() -> None:
    payload = _json_sse({"ok": 1.0, "bad": float("nan"), "nested": [float("inf")]})

    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert json.loads(payload) == {"bad": None, "nested": [None], "ok": 1.0}


def test_dashboard_api_rejects_overlapping_jobs(tmp_path: Path, monkeypatch) -> None:
    def slow_preflight(**kwargs: object) -> dict[str, object]:
        time.sleep(0.5)
        return {"gpu": None, "recommendations": []}

    monkeypatch.setenv(GPU_ENV_VAR, "placeholder")
    monkeypatch.setattr(jobs, "gpu_preflight", slow_preflight)
    client = TestClient(create_app(project_root=tmp_path))

    first = client.post("/api/jobs", json={"engines": ["vllm"]})
    second = client.post("/api/jobs", json={"engines": ["ollama"]})

    assert first.status_code == 200
    assert second.status_code == 409
    assert "already running" in second.json()["detail"]


def test_dashboard_local_only_host_check() -> None:
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("localhost")
    assert _is_loopback_host("::1")
    assert not _is_loopback_host("0.0.0.0")
    assert not _is_loopback_host("192.0.2.20")


def test_failure_hint_reads_engine_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    log = run_dir / "engine_logs" / "tensorrtllm_t4.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "noise\nKeyError: 'Unsupported model architecture: Qwen3_5ForConditionalGeneration'\n",
        encoding="utf-8",
    )
    manifest = SimpleNamespace(
        artifacts=SimpleNamespace(engine_logs="engine_logs/tensorrtllm_t4.log")
    )

    assert "Unsupported model architecture" in (jobs._failure_hint(run_dir, manifest) or "")


def test_dashboard_metrics_derive_from_trace_and_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _dashboard_manifest(run_dir)
    trace_path = run_dir / manifest.artifacts.request_trace
    rows = [
        _trace(manifest, "r1", "request_start", 1_000_000_000),
        _trace(manifest, "r2", "request_start", 1_100_000_000),
        _trace(manifest, "r1", "stream_event", 1_200_000_000, delta_text_chars=3),
        _trace(manifest, "r2", "stream_event", 1_300_000_000, delta_text_chars=3),
        _trace(
            manifest,
            "r1",
            "request_end",
            2_000_000_000,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}},
        ),
        _trace(
            manifest,
            "r2",
            "request_end",
            2_300_000_000,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 6}},
        ),
    ]
    trace_path.write_text("".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows), encoding="utf-8")
    contract.write_telemetry_parquet(
        run_dir,
        manifest,
        [
            _telemetry_row(0, gpu=20, used=1000, total=2000),
            _telemetry_row(1, gpu=80, used=1500, total=2000),
        ],
    )
    summary = contract.RunSummary(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        run_id=manifest.run_id,
        parser_version=contract.PARSER_VERSION,
        event_granularity="sse_data_line",
        request_count=2,
        success_count=2,
        error_count=0,
        ttft_ms=200,
        e2e_ms=1100,
        generated_text_sha256=contract.sha256_text("abcdef"),
        generated_text_chars=6,
        completion_tokens=10,
        request_throughput_rps=2.0,
        output_token_throughput_tps=10.0,
        error_rate=0,
        stream_event_count=2,
        non_empty_stream_event_count=2,
        comparison_key={},
    )

    metrics = jobs._dashboard_metrics(run_dir, manifest, summary)

    assert metrics["ttft_ms"]["p50"] == 200
    assert metrics["e2e_ms"]["p95"] == 1200
    assert metrics["tpot_ms"]["p50"] == 200
    assert metrics["gpu_utilization_percent"]["p95"] == 80
    assert metrics["vram_used_mib"]["percent_of_total"]["p95"] == 75
    assert metrics["kv_cache_efficiency"]["value"] == 50
    assert metrics["continuous_batching_efficiency"]["kind"] == "proxy"
    assert metrics["scheduler_efficiency"]["value"] == 100


def _write_base_workload(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workload_id": "tensorrtllm_t4_smoke",
                "prompt_template_id": "engine_configuration-tensorrt-t4-smoke-chat-v1",
                "seed": 123,
                "prompt": "hello",
                "measurement": {"requests": 1, "max_runtime_seconds": 300},
                "sampling": {"temperature": 0, "max_tokens": 4},
                "runtime_knobs": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _dashboard_manifest(run_dir: Path) -> contract.RunManifest:
    artifacts = contract.ArtifactPaths.model_validate(
        contract.default_artifacts("vllm").model_dump(mode="json")
        | {"telemetry_parquet": "telemetry.parquet"}
    )
    manifest = contract.RunManifest(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        run_family="dashboard",
        run_id=run_dir.name,
        study_id="dashboard_vllm",
        repeat_index=1,
        status="SUCCEEDED",
        created_at="2026-07-01T00:00:00+00:00",
        updated_at="2026-07-01T00:00:01+00:00",
        manifest_created_before_request=True,
        manifest_written_perf_counter_ns=900,
        first_request_perf_counter_ns=1000,
        run_dir=str(run_dir),
        source=contract.SourceInfo(branch="test", commit="abc", dirty=True),
        engine=contract.EngineInfo(
            name="vllm",
            image="vllm/vllm-openai@sha256:" + "a" * 64,
            container_name="inferno-test-vllm",
            port=8000,
            served_model_name="Qwen/Qwen3.5-2B",
            launch_args=["--max-model-len", "256", "--gpu-memory-utilization", "0.8"],
        ),
        model=contract.ModelInfo(
            id="Qwen/Qwen3.5-2B",
            revision="a" * 40,
            tokenizer_id="Qwen/Qwen3.5-2B",
            tokenizer_revision="a" * 40,
        ),
        hardware=contract.HardwareInfo(
            gpu_model="Tesla T4",
            gpu_count=1,
            driver_version="580.159.03",
            cuda_version="13.0",
            memory_mib=15360,
        ),
        workload=contract.WorkloadInfo(
            workload_id="dashboard_vllm",
            prompt_template_id="dashboard-dashboard-chat-v1",
            seed=123,
            prompt_sha256=contract.sha256_text("hello"),
            prompt_chars=5,
            request_rate_rps=1.0,
            arrival_model="closed_loop",
            concurrency=2,
            max_in_flight=2,
            cache_state="warm_after_warmup",
            cache_policy="no_prefix_reuse",
            warmup_requests=1,
            measurement_requests=2,
            max_runtime_seconds=60,
            streaming=True,
            timeout_seconds=60,
            retry_policy="none",
            failure_policy="fail_run",
            runtime_knobs={"dashboard_kv_cache_mode": "compact"},
            sampling={"temperature": 0, "max_tokens": 8},
        ),
        artifacts=artifacts,
        comparison_key={},
    )
    return contract.RunManifest.model_validate(
        manifest.model_dump(mode="json") | {"comparison_key": contract.comparison_key(manifest)}
    )


def _trace(
    manifest: contract.RunManifest,
    request_id: str,
    event: str,
    ns: int,
    *,
    delta_text_chars: int | None = None,
    usage: dict[str, object] | None = None,
) -> contract.TraceEvent:
    return contract.TraceEvent(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        run_id=manifest.run_id,
        request_id=request_id,
        event=event,
        perf_counter_ns=ns,
        request={"stream": True} if event == "request_start" else None,
        request_sha256=contract.sha256_text(request_id) if event == "request_start" else None,
        sse_index=0 if event == "stream_event" else None,
        delta_text_sha256=contract.sha256_text("abc") if event == "stream_event" else None,
        delta_text_chars=delta_text_chars,
        ssh_exit_code=0 if event == "request_end" else None,
        generated_text_sha256=contract.sha256_text("abc") if event == "request_end" else None,
        generated_text_chars=3 if event == "request_end" else None,
        usage=usage,
    )


def _telemetry_row(index: int, *, gpu: float, used: float, total: float) -> dict[str, object]:
    return {
        "sample_index": index,
        "collected_at_unix_ns": 1_000_000_000 + index,
        "source": "test",
        "gpu_utilization_percent": gpu,
        "gpu_memory_used_mib": used,
        "gpu_memory_total_mib": total,
        "gpu_power_watts": None,
        "gpu_power_watts_missing_reason": "test",
        "gpu_temperature_c": None,
        "gpu_temperature_c_missing_reason": "test",
        "host_cpu_percent": None,
        "host_cpu_percent_missing_reason": "test",
        "host_rss_mib": None,
        "host_rss_mib_missing_reason": "test",
    }
