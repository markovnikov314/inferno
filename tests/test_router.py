import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferno import artifact_contract as contract
from inferno import router


def test_router_replay_writes_negative_result_leakage_and_ablations(tmp_path: Path) -> None:
    study_path = _strict_study(tmp_path)
    config_path = _router_config(tmp_path, [study_path])

    assert router.run_router_replay(config_path=config_path, project_root=tmp_path) == 0

    output_dir = tmp_path / "artifacts/router/p8"
    latest = json.loads((output_dir / "latest.json").read_text())
    decision_lines = (output_dir / "decision_log.jsonl").read_text().splitlines()

    assert latest["phase"] == "P8"
    assert latest["replay_request_count"] == 4
    assert len(decision_lines) == 4
    assert latest["leakage_check"]["status"] == "PASS"
    assert latest["ablation_evaluation"]["status"] == "PASS"
    assert latest["baseline_comparison"]["status"] == "NEGATIVE_RESULT"
    assert latest["baseline_comparison"]["best_static_policy"] == "static_sglang"
    assert latest["omitted_baselines"][0]["baseline"] == "static_llamacpp"
    assert str(tmp_path) not in json.dumps(latest)
    assert (output_dir / "report.md").exists()


def test_router_config_rejects_random_holdout(tmp_path: Path) -> None:
    config_path = _router_config(tmp_path, [], holdout_strategy="random")

    with pytest.raises(ValidationError):
        router.load_router_config(tmp_path / config_path)


def _router_config(
    tmp_path: Path,
    evidence_studies: list[Path],
    *,
    holdout_strategy: str = "leave_one_workload_block_out",
) -> Path:
    path = tmp_path / "configs/router/p8.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "replay_id": "p8-test",
            "candidate_scope": {
                "environment_fingerprint": "test-block",
                "comparison_type": "strict",
            },
            "objective": {
                "name": "minimize_mean_e2e_ms_with_failure_and_quality_penalties",
                "latency_metric": "e2e_ms",
                "failure_penalty_ms": 60000,
                "quality_incompatible_penalty_ms": 60000,
                "headroom_penalty_ms": 1000,
                "uncertainty_penalty_ms": 250,
            },
            "replay": {
                "holdout_strategy": holdout_strategy,
                "request_rate_rps": 0.1,
                "default_queue_depth": 0,
            },
            "pricing": {"source": "user_provided", "checked_at": "2026-07-02T00:00:00Z"},
            "evidence_studies": [str(path) for path in evidence_studies],
            "artifacts_dir": "artifacts/router/p8",
        },
    )
    return path.relative_to(tmp_path)


def _strict_study(tmp_path: Path) -> Path:
    runs = []
    for workload_id, prompt_chars, max_tokens in (
        ("p5_smoke", 44, 96),
        ("p5_decode_heavy", 122, 192),
    ):
        for repeat in (1, 2):
            for engine, e2e_ms in (("vllm", 20.0 + repeat), ("sglang", 10.0 + repeat)):
                run_dir = _valid_run(
                    tmp_path,
                    engine=engine,
                    workload_id=workload_id,
                    repeat_index=repeat,
                    prompt_chars=prompt_chars,
                    max_tokens=max_tokens,
                    ttft_ms=e2e_ms / 2,
                    e2e_ms=e2e_ms,
                )
                runs.append(
                    {
                        "engine": engine,
                        "repeat_index": repeat,
                        "run_id": run_dir.name,
                        "run_dir": str(run_dir),
                        "exit_code": 0,
                        "validation_ok": True,
                    }
                )
    path = tmp_path / "artifacts/studies/p5_test/latest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": "p5_test",
            "study_type": "strict_engine_comparison",
            "environment_block_id": "test-block",
            "runs": runs,
        },
    )
    return path.relative_to(tmp_path)


def _valid_run(
    tmp_path: Path,
    *,
    engine: str,
    workload_id: str,
    repeat_index: int,
    prompt_chars: int,
    max_tokens: int,
    ttft_ms: float,
    e2e_ms: float,
) -> Path:
    run_dir = tmp_path / f"artifacts/runs/p5-{engine}-{workload_id}-r{repeat_index:02d}"
    run_dir.mkdir(parents=True)
    artifacts = contract.ArtifactPaths.model_validate(
        contract.default_artifacts(engine).model_dump(mode="json")
        | {"request_trace_parquet": "request_trace.parquet", "telemetry_parquet": "telemetry.parquet"}
    )
    manifest = contract.RunManifest(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        phase="P5",
        run_id=run_dir.name,
        study_id="p5_test",
        repeat_index=repeat_index,
        status="SUCCEEDED",
        created_at="2026-07-01T00:00:00+00:00",
        updated_at="2026-07-01T00:00:01+00:00",
        manifest_created_before_request=True,
        manifest_written_perf_counter_ns=900,
        first_request_perf_counter_ns=1000,
        run_dir=str(run_dir),
        source=contract.SourceInfo(branch="test", commit="abc", dirty=True),
        engine=contract.EngineInfo(
            name=engine,
            image=_image(engine),
            container_name=f"inferno-test-{engine}",
            port=8000,
            served_model_name="Qwen/Qwen3.5-2B",
            launch_args=_launch_args(engine),
        ),
        model=contract.ModelInfo(
            id="Qwen/Qwen3.5-2B",
            revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
            tokenizer_id="Qwen/Qwen3.5-2B",
            tokenizer_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        ),
        hardware=contract.HardwareInfo(
            gpu_model="Tesla T4",
            gpu_count=1,
            driver_version="580.159.03",
            cuda_version="13.0",
            memory_mib=15360,
        ),
        workload=contract.WorkloadInfo(
            workload_id=workload_id,
            prompt_template_id=f"{workload_id}-v1",
            seed=123,
            prompt_sha256=contract.sha256_text(workload_id),
            prompt_chars=prompt_chars,
            arrival_model="closed_loop",
            cache_state="cold_per_run",
            warmup_requests=0,
            measurement_requests=1,
            max_runtime_seconds=300,
            retry_policy="none",
            failure_policy="fail_run",
            sampling={"temperature": 0, "top_p": 1, "max_tokens": max_tokens},
        ),
        artifacts=artifacts,
        comparison_key={},
        health=contract.HealthEvidence(ok=True, model_ids=["Qwen/Qwen3.5-2B"]),
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
    contract.write_json(run_dir / artifacts.health, {"ok": True, "model_ids": ["Qwen/Qwen3.5-2B"]})
    _write_text(run_dir / artifacts.engine_logs, f"{engine} log\n")
    contract.write_json(run_dir / "command_logs/up.json", {"exit_code": 0})
    _write_trace(run_dir, manifest, ttft_ms=ttft_ms, e2e_ms=e2e_ms)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    contract.write_trace_parquet(run_dir, manifest)
    contract.write_telemetry_parquet(run_dir, manifest, [_telemetry_row()])
    contract.write_json(run_dir / artifacts.summary, contract.recompute_summary(run_dir, manifest))
    contract.write_checksums(run_dir, manifest)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    assert contract.validate_run(run_dir, write=True).ok
    return run_dir


def _write_trace(
    run_dir: Path,
    manifest: contract.RunManifest,
    *,
    ttft_ms: float,
    e2e_ms: float,
) -> None:
    start = 1000
    first = start + int(ttft_ms * 1_000_000)
    end = start + int(e2e_ms * 1_000_000)
    rows = [
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="req-1",
            event="request_start",
            perf_counter_ns=start,
            request={"model": manifest.engine.served_model_name},
            request_sha256=contract.sha256_text("request"),
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="req-1",
            event="stream_event",
            perf_counter_ns=first,
            sse_index=0,
            delta_text_sha256=contract.sha256_text("Hello"),
            delta_text_chars=5,
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="req-1",
            event="request_end",
            perf_counter_ns=end,
            ssh_exit_code=0,
            generated_text_sha256=contract.sha256_text("Hello"),
            generated_text_chars=5,
            usage={"completion_tokens": 1},
        ),
    ]
    _write_text(
        run_dir / manifest.artifacts.request_trace,
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
    )


def _launch_args(engine: str) -> list[str]:
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
    if engine == "sglang":
        return "lmsysorg/sglang@sha256:" + "a" * 64
    return "vllm/vllm-openai@sha256:" + "a" * 64


def _raw_sse() -> str:
    return "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[],"usage":{"completion_tokens":1}}',
            "",
            "data: [DONE]",
            "",
        ]
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


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
