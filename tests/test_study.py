import json
from pathlib import Path

import pytest

from inferno import artifact_contract as contract
from inferno import study
from inferno import vllm_remote


def test_load_study_config_reads_json_compatible_yaml() -> None:
    config = study.load_study_config(Path("configs/study/vllm_smoke.yaml"))

    assert config.study_id == "vllm_smoke"
    assert config.engine == "vllm"
    assert config.repeats == 3


def test_load_strict_study_config_records_engine_order() -> None:
    config = study.load_study_config(Path("configs/study/strict_interactive.yaml"))

    assert config.study_id == "strict_interactive"
    assert config.allowed_engines() == ["vllm", "sglang"]
    assert config.run_phase() == "P4"
    assert [(item.engine, item.repeat_index) for item in config.run_order] == [
        ("sglang", 1),
        ("vllm", 1),
        ("vllm", 2),
        ("sglang", 2),
        ("sglang", 3),
        ("vllm", 3),
    ]


def test_load_p5_study_config_uses_p5_phase() -> None:
    config = study.load_study_config(Path("configs/study/p5_smoke.yaml"))

    assert config.study_id == "p5_smoke"
    assert config.run_phase() == "P5"
    assert config.low_sample_completed_requests == 3


def test_load_profile_study_config_uses_p6_phase() -> None:
    config = study.load_study_config(Path("configs/study/profile_local.yaml"))

    assert config.study_id == "profile_local"
    assert config.allowed_engines() == ["llamacpp"]
    assert config.run_phase() == "P6"


def test_load_tensorrtllm_study_config_uses_p10_phase() -> None:
    config = study.load_study_config(Path("configs/study/tensorrtllm_smoke.yaml"))

    assert config.study_id == "tensorrtllm_smoke"
    assert config.allowed_engines() == ["tensorrtllm"]
    assert config.run_phase() == "P10"


def test_load_tensorrtllm_t4_study_config_uses_p10_phase() -> None:
    config = study.load_study_config(Path("configs/study/tensorrtllm_t4_smoke.yaml"))

    assert config.study_id == "tensorrtllm_t4_smoke"
    assert config.allowed_engines() == ["tensorrtllm_t4"]
    assert config.run_phase() == "P10"


def test_load_ollama_study_config_uses_p11_phase() -> None:
    config = study.load_study_config(Path("configs/study/ollama_smoke.yaml"))

    assert config.study_id == "ollama_smoke"
    assert config.allowed_engines() == ["ollama"]
    assert config.run_phase() == "P11"


def test_run_study_invokes_three_p3_repeats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _study_config(tmp_path)
    calls = []

    def fake_run_real_result(**kwargs: object) -> vllm_remote.RunRealResult:
        calls.append(kwargs)
        repeat = int(kwargs["repeat_index"])
        run_dir = tmp_path / "artifacts" / "runs" / f"p3-vllm-smoke-r{repeat:02d}-test"
        return vllm_remote.RunRealResult(0, run_dir.name, run_dir, True)

    monkeypatch.setattr(vllm_remote, "run_real_result", fake_run_real_result)

    assert study.run_study(engine="vllm", config_path=config_path, project_root=tmp_path, env={}) == 0

    assert [call["repeat_index"] for call in calls] == [1, 2, 3]
    assert {call["phase"] for call in calls} == {"P3"}
    latest = json.loads((tmp_path / "artifacts/studies/vllm_smoke/latest.json").read_text())
    assert [run["run_id"] for run in latest["runs"]] == [
        "p3-vllm-smoke-r01-test",
        "p3-vllm-smoke-r02-test",
        "p3-vllm-smoke-r03-test",
    ]


def test_run_strict_study_uses_p4_and_declared_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _strict_study_config(tmp_path)
    calls = []

    def fake_run_real_result(**kwargs: object) -> vllm_remote.RunRealResult:
        calls.append(kwargs)
        repeat = int(kwargs["repeat_index"])
        engine = str(kwargs["engine"])
        run_dir = tmp_path / "artifacts" / "runs" / f"p4-{engine}-strict-r{repeat:02d}-test"
        return vllm_remote.RunRealResult(0, run_dir.name, run_dir, True)

    monkeypatch.setattr(vllm_remote, "run_real_result", fake_run_real_result)

    assert study.run_study(engine="sglang", config_path=config_path, project_root=tmp_path, env={}) == 0

    assert [call["phase"] for call in calls] == ["P4", "P4", "P4"]
    assert [call["repeat_index"] for call in calls] == [1, 2, 3]
    latest = json.loads((tmp_path / "artifacts/studies/strict_interactive/latest.json").read_text())
    assert [run["planned_order_index"] for run in latest["runs"]] == [1, 4, 5]
    assert {run["engine"] for run in latest["runs"]} == {"sglang"}


def test_run_deployment_profile_study_uses_p6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _profile_study_config(tmp_path)
    calls = []

    def fake_run_real_result(**kwargs: object) -> vllm_remote.RunRealResult:
        calls.append(kwargs)
        run_dir = tmp_path / "artifacts" / "runs" / "p6-llamacpp-profile-local-r01-test"
        return vllm_remote.RunRealResult(0, run_dir.name, run_dir, True)

    monkeypatch.setattr(vllm_remote, "run_real_result", fake_run_real_result)

    assert (
        study.run_study(engine="llamacpp", config_path=config_path, project_root=tmp_path, env={})
        == 0
    )

    assert [call["phase"] for call in calls] == ["P6"]
    latest = json.loads((tmp_path / "artifacts/studies/profile_local/latest.json").read_text())
    assert latest["runs"][0]["engine"] == "llamacpp"


def test_run_engine_configuration_study_uses_p10(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _tensorrt_study_config(tmp_path)
    calls = []

    def fake_run_real_result(**kwargs: object) -> vllm_remote.RunRealResult:
        calls.append(kwargs)
        run_dir = tmp_path / "artifacts" / "runs" / "p10-tensorrtllm-smoke-r01-test"
        return vllm_remote.RunRealResult(0, run_dir.name, run_dir, True)

    monkeypatch.setattr(vllm_remote, "run_real_result", fake_run_real_result)

    assert (
        study.run_study(
            engine="tensorrtllm", config_path=config_path, project_root=tmp_path, env={}
        )
        == 0
    )

    assert [call["phase"] for call in calls] == ["P10"]
    latest = json.loads((tmp_path / "artifacts/studies/tensorrtllm_smoke/latest.json").read_text())
    assert latest["runs"][0]["engine"] == "tensorrtllm"


def test_run_ollama_deployment_profile_study_uses_p11(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _ollama_study_config(tmp_path)
    calls = []

    def fake_run_real_result(**kwargs: object) -> vllm_remote.RunRealResult:
        calls.append(kwargs)
        run_dir = tmp_path / "artifacts" / "runs" / "p11-ollama-smoke-r01-test"
        return vllm_remote.RunRealResult(0, run_dir.name, run_dir, True)

    monkeypatch.setattr(vllm_remote, "run_real_result", fake_run_real_result)

    assert (
        study.run_study(engine="ollama", config_path=config_path, project_root=tmp_path, env={})
        == 0
    )

    assert [call["phase"] for call in calls] == ["P11"]
    latest = json.loads((tmp_path / "artifacts/studies/ollama_smoke/latest.json").read_text())
    assert latest["runs"][0]["engine"] == "ollama"


def test_report_writes_provenance_and_low_sample(tmp_path: Path) -> None:
    run_dir = _valid_p3_run(tmp_path)
    config_path = _study_config(tmp_path)
    artifacts_dir = tmp_path / "artifacts/studies/vllm_smoke"
    artifacts_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "study_id": "vllm_smoke",
        "engine": "vllm",
        "runs": [
            {
                "repeat_index": 1,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "exit_code": 0,
                "validation_ok": True,
            }
        ],
    }
    _write_json(artifacts_dir / "latest.json", payload)

    assert study.report_study(study_path=config_path, project_root=tmp_path) == 0

    report_inputs = json.loads((artifacts_dir / "report_inputs.json").read_text())
    run = report_inputs["runs"][0]
    assert run["labels"] == ["LOW_SAMPLE"]
    assert run["metrics"]["ttft_ms"]["p50"]["source_run_ids"] == [run_dir.name]
    assert run["artifacts"]["request_trace_parquet"].endswith("request_trace.parquet")
    assert (artifacts_dir / "report.md").exists()


def test_compare_strict_writes_report_for_matching_controls(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P4", study_id="strict_interactive", engine="vllm")
    sglang_run = _valid_p3_run(
        tmp_path,
        phase="P4",
        study_id="strict_interactive",
        engine="sglang",
        repeat_index=1,
    )

    assert (
        study.compare_runs(
            mode="strict",
            runs_arg=f"{vllm_run} {sglang_run}",
            project_root=tmp_path,
        )
        == 0
    )

    reports = list((tmp_path / "artifacts/compare").glob("strict-*/report_inputs.json"))
    assert len(reports) == 1
    report_inputs = json.loads(reports[0].read_text())
    assert report_inputs["study_type"] == "strict_engine_comparison"
    assert report_inputs["controlled_key_status"] == "MATCH"
    assert {run["engine"] for run in report_inputs["runs"]} == {"vllm", "sglang"}
    assert all("LOW_SAMPLE" in run["labels"] for run in report_inputs["runs"])


def test_compare_strict_refuses_control_mismatch(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P4", study_id="strict_interactive", engine="vllm")
    sglang_run = _valid_p3_run(
        tmp_path,
        phase="P4",
        study_id="strict_interactive",
        engine="sglang",
        cache_state="warm_after_warmup",
    )

    assert (
        study.compare_runs(
            mode="strict",
            runs_arg=f"{vllm_run} {sglang_run}",
            project_root=tmp_path,
        )
        == 1
    )
    assert not (tmp_path / "artifacts/compare").exists()


def test_compare_strict_refuses_llamacpp(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P4", study_id="strict_interactive", engine="vllm")
    llamacpp_run = _valid_p3_run(
        tmp_path,
        phase="P6",
        study_id="profile_local",
        engine="llamacpp",
    )
    _, errors = study.build_strict_compare_inputs([vllm_run, llamacpp_run], tmp_path)

    assert "strict comparison rejects llamacpp; use MODE=deployment-profile" in errors
    assert (
        study.compare_runs(
            mode="strict",
            runs_arg=f"{vllm_run} {llamacpp_run}",
            project_root=tmp_path,
        )
        == 1
    )
    assert not (tmp_path / "artifacts/compare").exists()


def test_compare_strict_refuses_ollama(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P4", study_id="strict_interactive", engine="vllm")
    ollama_run = _valid_p3_run(
        tmp_path,
        phase="P11",
        study_id="ollama_smoke",
        engine="ollama",
    )
    _, errors = study.build_strict_compare_inputs([vllm_run, ollama_run], tmp_path)

    assert "strict comparison rejects ollama; use MODE=deployment-profile" in errors
    assert (
        study.compare_runs(
            mode="strict",
            runs_arg=f"{vllm_run} {ollama_run}",
            project_root=tmp_path,
        )
        == 1
    )
    assert not (tmp_path / "artifacts/compare").exists()


def test_compare_deployment_profile_writes_banner_and_differences(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P5", study_id="p5_smoke", engine="vllm")
    sglang_run = _valid_p3_run(
        tmp_path,
        phase="P5",
        study_id="p5_smoke",
        engine="sglang",
    )
    llamacpp_run = _valid_p3_run(
        tmp_path,
        phase="P6",
        study_id="profile_local",
        engine="llamacpp",
    )

    assert (
        study.compare_runs(
            mode="deployment-profile",
            runs_arg=f"{vllm_run} {sglang_run} {llamacpp_run}",
            project_root=tmp_path,
        )
        == 0
    )

    reports = list((tmp_path / "artifacts/compare").glob("deployment-profile-*/report_inputs.json"))
    assert len(reports) == 1
    report_inputs = json.loads(reports[0].read_text())
    assert report_inputs["study_type"] == "deployment_profile_comparison"
    assert report_inputs["banner"] == study.DEPLOYMENT_PROFILE_BANNER
    fields = {item["field"] for item in report_inputs["non_matching_controls"]}
    assert "weights.format" in fields
    assert "model.tokenizer_format" in fields
    report = reports[0].with_name("report.md").read_text(encoding="utf-8")
    assert study.DEPLOYMENT_PROFILE_BANNER in report


def test_compare_deployment_profile_accepts_ollama(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P5", study_id="p5_smoke", engine="vllm")
    ollama_run = _valid_p3_run(
        tmp_path,
        phase="P11",
        study_id="ollama_smoke",
        engine="ollama",
    )

    assert (
        study.compare_runs(
            mode="deployment-profile",
            runs_arg=f"{vllm_run} {ollama_run}",
            project_root=tmp_path,
        )
        == 0
    )

    reports = list((tmp_path / "artifacts/compare").glob("deployment-profile-*/report_inputs.json"))
    assert len(reports) == 1
    report_inputs = json.loads(reports[0].read_text(encoding="utf-8"))
    assert {run["engine"] for run in report_inputs["runs"]} == {"vllm", "ollama"}
    assert study.validate_deployment_profile_report_inputs(report_inputs, tmp_path) == []


def test_compare_engine_configuration_writes_banner_and_differences(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P5", study_id="p5_smoke", engine="vllm")
    trt_run = _valid_p3_run(
        tmp_path,
        phase="P10",
        study_id="tensorrtllm_smoke",
        engine="tensorrtllm",
    )

    assert (
        study.compare_runs(
            mode="engine-configuration",
            runs_arg=f"{vllm_run} {trt_run}",
            project_root=tmp_path,
        )
        == 0
    )

    reports = list((tmp_path / "artifacts/compare").glob("engine-configuration-*/report_inputs.json"))
    assert len(reports) == 1
    report_inputs = json.loads(reports[0].read_text())
    assert report_inputs["study_type"] == "engine_configuration_comparison"
    assert report_inputs["banner"] == study.ENGINE_CONFIGURATION_BANNER
    assert {run["engine"] for run in report_inputs["runs"]} == {"vllm", "tensorrtllm"}
    fields = {item["field"] for item in report_inputs["non_matching_controls"]}
    assert "runtime.backend" in fields
    report = reports[0].with_name("report.md").read_text(encoding="utf-8")
    assert study.ENGINE_CONFIGURATION_BANNER in report
    assert "Engine Configuration Report" in report


def test_strict_report_validation_rejects_missing_low_sample_label(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P4", study_id="strict_interactive", engine="vllm")
    sglang_run = _valid_p3_run(
        tmp_path,
        phase="P4",
        study_id="strict_interactive",
        engine="sglang",
    )
    report_inputs, errors = study.build_strict_compare_inputs(
        [vllm_run, sglang_run],
        tmp_path,
        low_sample_completed_requests=3,
    )
    assert not errors
    report_inputs["runs"][0]["labels"] = []

    errors = study.validate_strict_report_inputs(report_inputs, tmp_path)

    assert any("missing LOW_SAMPLE" in error for error in errors)


def test_strict_report_validation_rejects_unresolved_artifact_path(tmp_path: Path) -> None:
    vllm_run = _valid_p3_run(tmp_path, phase="P4", study_id="strict_interactive", engine="vllm")
    sglang_run = _valid_p3_run(
        tmp_path,
        phase="P4",
        study_id="strict_interactive",
        engine="sglang",
    )
    report_inputs, errors = study.build_strict_compare_inputs([vllm_run, sglang_run], tmp_path)
    assert not errors
    report_inputs["runs"][0]["artifacts"]["summary"] = "missing-summary.json"

    errors = study.validate_strict_report_inputs(report_inputs, tmp_path)

    assert any("unresolved artifact path" in error for error in errors)


def test_usage_only_stream_chunks_are_safe() -> None:
    raw = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":" llama"}}]}',
            'data: {"choices":[],"usage":{"completion_tokens":2}}',
            "data: [DONE]",
        ]
    )

    assert contract.generated_text_events(raw) == [(0, "Hello"), (1, " llama"), (2, "")]
    assert contract.stream_usage(raw) == {"completion_tokens": 2}
    assert vllm_remote._generated_text(raw) == "Hello llama"


def test_telemetry_nulls_require_reasons(tmp_path: Path) -> None:
    run_dir, manifest = _p3_manifest(tmp_path)
    with pytest.raises(ValueError, match="missing reason"):
        contract.write_telemetry_parquet(
            run_dir,
            manifest,
            [
                {
                    "sample_index": 0,
                    "collected_at_unix_ns": 1,
                    "source": "test",
                    "gpu_utilization_percent": None,
                }
            ],
        )


def _valid_p3_run(
    tmp_path: Path,
    *,
    phase: str = "P3",
    study_id: str = "vllm_smoke",
    engine: str = "vllm",
    repeat_index: int = 1,
    cache_state: str = "cold_per_run",
) -> Path:
    run_dir, manifest = _p3_manifest(
        tmp_path,
        phase=phase,
        study_id=study_id,
        engine=engine,
        repeat_index=repeat_index,
        cache_state=cache_state,
    )
    artifacts = manifest.artifacts
    _write_text(run_dir / artifacts.raw_sse, _raw_sse())
    contract.write_json(run_dir / artifacts.health, {"ok": True, "model_ids": [_served_model(engine)]})
    _write_text(run_dir / artifacts.engine_logs, f"{engine} log\n")
    contract.write_json(run_dir / "command_logs/up.json", {"exit_code": 0})
    _write_trace(run_dir, manifest)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    contract.write_trace_parquet(run_dir, manifest)
    contract.write_telemetry_parquet(run_dir, manifest, [_telemetry_row()])
    contract.write_json(run_dir / artifacts.summary, contract.recompute_summary(run_dir, manifest))
    contract.write_checksums(run_dir, manifest)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    assert contract.validate_run(run_dir, write=True).ok
    return run_dir


def _p3_manifest(
    tmp_path: Path,
    *,
    phase: str = "P3",
    study_id: str = "vllm_smoke",
    engine: str = "vllm",
    repeat_index: int = 1,
    cache_state: str = "cold_per_run",
) -> tuple[Path, contract.RunManifest]:
    run_dir = tmp_path / f"artifacts/runs/{phase.lower()}-{engine}-{study_id}-r{repeat_index:02d}-test"
    run_dir.mkdir(parents=True)
    artifacts = contract.ArtifactPaths.model_validate(
        contract.default_artifacts(engine).model_dump(mode="json")
        | {"request_trace_parquet": "request_trace.parquet", "telemetry_parquet": "telemetry.parquet"}
    )
    manifest = contract.RunManifest(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        phase=phase,
        run_id=run_dir.name,
        study_id=study_id,
        repeat_index=repeat_index,
        status="SUCCEEDED",
        created_at="2026-07-01T00:00:00+00:00",
        updated_at="2026-07-01T00:00:01+00:00",
        manifest_created_before_request=True,
        manifest_written_perf_counter_ns=900_000_000,
        first_request_perf_counter_ns=1_000_000_000,
        run_dir=str(run_dir),
        source=contract.SourceInfo(branch="phase/03-vllm-smoke", commit="abc", dirty=True),
        engine=contract.EngineInfo(
            name=engine,
            image=_image(engine),
            container_name=f"inferno-test-{engine}",
            port=_port(engine),
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
        workload=contract.WorkloadInfo(
            workload_id="ollama_smoke" if engine == "ollama" else "smoke_real",
            prompt_template_id="p11-ollama-smoke-chat-v1" if engine == "ollama" else "smoke-chat-v1",
            seed=123,
            prompt_sha256=contract.sha256_text("hello"),
            prompt_chars=5,
            request_rate_rps=0.1 if engine in {"tensorrtllm", "ollama"} else None,
            arrival_model="closed_loop",
            concurrency=1 if engine in {"tensorrtllm", "ollama"} else None,
            max_in_flight=1 if engine in {"tensorrtllm", "ollama"} else None,
            prompt_target={"unit": "chars", "p50": 5, "p95": 5}
            if engine in {"tensorrtllm", "ollama"}
            else {},
            output_target={"unit": "tokens", "p50": 4, "p95": 4}
            if engine in {"tensorrtllm", "ollama"}
            else {},
            context_length=2048 if engine == "ollama" else 512 if engine == "tensorrtllm" else None,
            cache_state=cache_state,
            cache_policy="ollama_default" if engine == "ollama" else "no_prefix_reuse" if engine == "tensorrtllm" else None,
            warmup_requests=0,
            measurement_requests=1,
            max_runtime_seconds=300,
            streaming=True if engine in {"tensorrtllm", "ollama"} else None,
            timeout_seconds=300 if engine in {"tensorrtllm", "ollama"} else None,
            retry_policy="none",
            failure_policy="fail_run",
            quality_requirement={"type": "non_empty_text"}
            if engine == "ollama"
            else {"type": "none"}
            if engine == "tensorrtllm"
            else {},
            cost_metadata={"status": "UNAVAILABLE"} if engine in {"tensorrtllm", "ollama"} else {},
            runtime_knobs={"deployment_profile": "ollama-docker-gpu"}
            if engine == "ollama"
            else {"backend": "pytorch"}
            if engine == "tensorrtllm"
            else {},
            sampling={"temperature": 0, "max_tokens": 4},
        ),
        artifacts=artifacts,
        comparison_key={},
        health=contract.HealthEvidence(ok=True, model_ids=[_served_model(engine)]),
        stream=contract.StreamEvidence(
            ok=True,
            generated_text_sha256=contract.sha256_text("Hello there"),
            generated_text_chars=11,
        ),
        shutdown=contract.ShutdownEvidence(ok=True),
    )
    manifest = contract.RunManifest.model_validate(
        manifest.model_dump(mode="json") | {"comparison_key": contract.comparison_key(manifest)}
    )
    return run_dir, manifest


def _write_trace(run_dir: Path, manifest: contract.RunManifest) -> None:
    rows = [
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_start",
            perf_counter_ns=1_000_000_000,
            request={"model": manifest.engine.served_model_name},
            request_sha256=contract.sha256_text("request"),
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="stream_event",
            perf_counter_ns=1_100_000_000,
            sse_index=0,
            delta_text_sha256=contract.sha256_text("Hello"),
            delta_text_chars=5,
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="stream_event",
            perf_counter_ns=1_200_000_000,
            sse_index=1,
            delta_text_sha256=contract.sha256_text(" there"),
            delta_text_chars=6,
        ),
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_end",
            perf_counter_ns=1_500_000_000,
            ssh_exit_code=0,
            raw_sse=manifest.artifacts.raw_sse,
            stderr="command_logs/real-stream.stderr",
            generated_text_sha256=contract.sha256_text("Hello there"),
            generated_text_chars=11,
            usage={"completion_tokens": 2},
        ),
    ]
    _write_text(
        run_dir / manifest.artifacts.request_trace,
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
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


def _raw_sse() -> str:
    return "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            "",
            'data: {"choices":[],"usage":{"completion_tokens":2}}',
            "",
            "data: [DONE]",
            "",
        ]
    )


def _launch_args(engine: str) -> list[str]:
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
    if engine == "tensorrtllm":
        return "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@sha256:" + "a" * 64
    if engine == "sglang":
        return "lmsysorg/sglang@sha256:" + "a" * 64
    if engine == "llamacpp":
        return "ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:" + "a" * 64
    if engine == "ollama":
        return "ollama/ollama@sha256:" + "a" * 64
    return "vllm/vllm-openai@sha256:" + "a" * 64


def _port(engine: str) -> int:
    if engine == "ollama":
        return 11434
    if engine == "llamacpp":
        return 8080
    if engine == "tensorrtllm":
        return 8090
    return 8000


def _served_model(engine: str) -> str:
    if engine == "tensorrtllm":
        return "Qwen/Qwen3.5-2B"
    if engine == "llamacpp":
        return "lmstudio-community/Qwen3.5-2B-GGUF:Q4_K_M"
    if engine == "ollama":
        return "qwen3.5:2b"
    return "Qwen/Qwen3.5-2B"


def _model(engine: str) -> contract.ModelInfo:
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


def _study_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/study/vllm_smoke.yaml"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": "vllm_smoke",
            "engine": "vllm",
            "workload_config": "configs/workloads/smoke_real.yaml",
            "repeats": 3,
            "telemetry_cadence_seconds": 0.5,
            "low_sample_completed_requests": 50,
            "artifacts_dir": "artifacts/studies/vllm_smoke",
        },
    )
    return path.relative_to(tmp_path)


def _strict_study_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/study/strict_interactive.yaml"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": "strict_interactive",
            "study_type": "strict_engine_comparison",
            "engines": ["vllm", "sglang"],
            "environment_block_id": "strict_interactive_block_001",
            "run_order_seed": 4242,
            "run_order": [
                {"engine": "sglang", "repeat_index": 1},
                {"engine": "vllm", "repeat_index": 1},
                {"engine": "vllm", "repeat_index": 2},
                {"engine": "sglang", "repeat_index": 2},
                {"engine": "sglang", "repeat_index": 3},
                {"engine": "vllm", "repeat_index": 3},
            ],
            "workload_config": "configs/workloads/strict_interactive.yaml",
            "repeats": 3,
            "telemetry_cadence_seconds": 0.5,
            "low_sample_completed_requests": 3,
            "artifacts_dir": "artifacts/studies/strict_interactive",
        },
    )
    return path.relative_to(tmp_path)


def _profile_study_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/study/profile_local.yaml"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": "profile_local",
            "study_type": "deployment_profile_comparison",
            "engines": ["llamacpp"],
            "phase": "P6",
            "environment_block_id": "p6_block",
            "run_order_seed": 1,
            "run_order": [{"engine": "llamacpp", "repeat_index": 1}],
            "workload_config": "configs/workloads/profile_local.yaml",
            "repeats": 1,
            "telemetry_cadence_seconds": 0.5,
            "low_sample_completed_requests": 1,
            "artifacts_dir": "artifacts/studies/profile_local",
        },
    )
    return path.relative_to(tmp_path)


def _tensorrt_study_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/study/tensorrtllm_smoke.yaml"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": "tensorrtllm_smoke",
            "study_type": "engine_configuration_comparison",
            "engines": ["tensorrtllm"],
            "phase": "P10",
            "environment_block_id": "p10_block",
            "run_order_seed": 1,
            "run_order": [{"engine": "tensorrtllm", "repeat_index": 1}],
            "workload_config": "configs/workloads/tensorrtllm_smoke.yaml",
            "repeats": 1,
            "telemetry_cadence_seconds": 0.5,
            "low_sample_completed_requests": 1,
            "artifacts_dir": "artifacts/studies/tensorrtllm_smoke",
        },
    )
    return path.relative_to(tmp_path)


def _ollama_study_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/study/ollama_smoke.yaml"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": "ollama_smoke",
            "study_type": "deployment_profile_comparison",
            "engines": ["ollama"],
            "phase": "P11",
            "environment_block_id": "p11_block",
            "run_order_seed": 1,
            "run_order": [{"engine": "ollama", "repeat_index": 1}],
            "workload_config": "configs/workloads/ollama_smoke.yaml",
            "repeats": 1,
            "telemetry_cadence_seconds": 0.5,
            "low_sample_completed_requests": 1,
            "artifacts_dir": "artifacts/studies/ollama_smoke",
        },
    )
    return path.relative_to(tmp_path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
