import json
from pathlib import Path

from inferno import artifact_contract as contract
from inferno import release


FAMILIES = [
    ("p5_smoke", "smoke"),
    ("p5_interactive", "interactive"),
    ("p5_decode_heavy", "decode-heavy"),
    ("p5_long_context_lite", "long-context-lite"),
    ("p5_structured_output_lite", "structured-output-lite"),
]


def test_release_package_builds_redacted_manifest(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    studies = []
    for study_id, family in FAMILIES:
        config = _write_study_config(tmp_path, study_id, family)
        _write_latest(tmp_path, study_id, family)
        studies.append({"study_id": study_id, "config": str(config), "workload_family": family})
    release_config = tmp_path / "configs/study/p5_research_core.yaml"
    _write_json(
        release_config,
        {
            "schema_version": 1,
            "release_id": "p5_research_core_v1",
            "study_type": "p5_research_core_release",
            "mode": "redacted",
            "required_workload_families": [family for _, family in FAMILIES],
            "studies": studies,
            "docs": [
                "docs/methodology.md",
                "docs/limitations.md",
                "docs/reproducibility.md",
                "docs/privacy-and-redaction.md",
            ],
            "artifacts_dir": "artifacts/releases/p5_research_core_v1",
        },
    )

    assert (
        release.package_release(
            mode="redacted",
            study_path=release_config.relative_to(tmp_path),
            project_root=tmp_path,
        )
        == 0
    )

    package_dir = tmp_path / "artifacts/releases/p5_research_core_v1/redacted"
    manifest = json.loads((package_dir / "manifest.json").read_text())
    assert len(manifest["workload_families"]) == 5
    assert all(item["sha256"] for item in manifest["files"])
    report_inputs = json.loads(
        (package_dir / "reports/p5_smoke/report_inputs.redacted.json").read_text()
    )
    run = report_inputs["runs"][0]
    assert run["artifacts"]["raw_sse"].startswith("redacted:")
    assert Path(package_dir / run["artifacts"]["summary"]).exists()
    assert "C:\\Users" not in json.dumps(report_inputs)
    assert release.validate_release_package(package_dir) == []


def _write_study_config(tmp_path: Path, study_id: str, family: str) -> Path:
    path = tmp_path / f"configs/study/{study_id}.yaml"
    _write_json(
        path,
        {
            "schema_version": 1,
            "study_id": study_id,
            "study_type": "strict_engine_comparison",
            "engines": ["vllm", "sglang"],
            "phase": "P5",
            "environment_block_id": "test",
            "run_order_seed": 1,
            "run_order": [
                {"engine": "vllm", "repeat_index": 1},
                {"engine": "sglang", "repeat_index": 1},
                {"engine": "vllm", "repeat_index": 2},
                {"engine": "sglang", "repeat_index": 2},
                {"engine": "vllm", "repeat_index": 3},
                {"engine": "sglang", "repeat_index": 3},
            ],
            "workload_config": f"configs/workloads/{study_id}.yaml",
            "repeats": 3,
            "telemetry_cadence_seconds": 0.5,
            "low_sample_completed_requests": 3,
            "artifacts_dir": f"artifacts/studies/{study_id}",
        },
    )
    workload = tmp_path / f"configs/workloads/{study_id}.yaml"
    _write_json(workload, {"workload_family": family})
    return path.relative_to(tmp_path)


def _write_latest(tmp_path: Path, study_id: str, family: str) -> None:
    runs = []
    for repeat in range(1, 4):
        for engine in ("vllm", "sglang"):
            run_dir = _valid_run(tmp_path, study_id, family, engine, repeat)
            runs.append(
                {
                    "engine": engine,
                    "repeat_index": repeat,
                    "planned_order_index": len(runs) + 1,
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "exit_code": 0,
                    "validation_ok": True,
                }
            )
    _write_json(
        tmp_path / f"artifacts/studies/{study_id}/latest.json",
        {
            "schema_version": 1,
            "study_id": study_id,
            "study_type": "strict_engine_comparison",
            "runs": runs,
        },
    )


def _valid_run(tmp_path: Path, study_id: str, family: str, engine: str, repeat: int) -> Path:
    run_dir = tmp_path / f"artifacts/runs/p5-{engine}-{study_id}-r{repeat:02d}"
    run_dir.mkdir(parents=True)
    artifacts = contract.ArtifactPaths.model_validate(
        contract.default_artifacts(engine).model_dump(mode="json")
        | {"request_trace_parquet": "request_trace.parquet", "telemetry_parquet": "telemetry.parquet"}
    )
    text = '{"answer":"strict comparison holds controls fixed","controls":["model","hardware","sampling"]}'
    if family != "structured-output-lite":
        text = "Hello there"
    manifest = contract.RunManifest(
        schema_version=1,
        contract_version=contract.CONTRACT_VERSION,
        phase="P5",
        run_id=run_dir.name,
        study_id=study_id,
        repeat_index=repeat,
        status="SUCCEEDED",
        created_at="2026-07-01T00:00:00+00:00",
        updated_at="2026-07-01T00:00:01+00:00",
        manifest_created_before_request=True,
        manifest_written_perf_counter_ns=900_000_000,
        first_request_perf_counter_ns=1_000_000_000,
        run_dir=str(run_dir),
        source=contract.SourceInfo(branch="test", commit="abc", dirty=True),
        engine=contract.EngineInfo(
            name=engine,
            image=(
                "lmsysorg/sglang@sha256:" + "a" * 64
                if engine == "sglang"
                else "vllm/vllm-openai@sha256:" + "a" * 64
            ),
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
            workload_id=study_id,
            prompt_template_id=f"{study_id}-template",
            seed=123,
            prompt_sha256=contract.sha256_text("prompt"),
            prompt_chars=6,
            arrival_model="closed_loop",
            cache_state="cold_per_run",
            warmup_requests=0,
            measurement_requests=1,
            max_runtime_seconds=300,
            retry_policy="none",
            failure_policy="fail_run",
            sampling={"temperature": 0, "top_p": 1, "max_tokens": 32},
        ),
        artifacts=artifacts,
        comparison_key={},
        health=contract.HealthEvidence(ok=True, model_ids=["Qwen/Qwen3.5-2B"]),
        stream=contract.StreamEvidence(
            ok=True,
            generated_text_sha256=contract.sha256_text(text),
            generated_text_chars=len(text),
        ),
        shutdown=contract.ShutdownEvidence(ok=True),
    )
    manifest = contract.RunManifest.model_validate(
        manifest.model_dump(mode="json") | {"comparison_key": contract.comparison_key(manifest)}
    )
    _write_text(run_dir / artifacts.raw_sse, _raw_sse(text))
    contract.write_json(run_dir / artifacts.health, {"ok": True, "model_ids": ["Qwen/Qwen3.5-2B"]})
    _write_text(run_dir / artifacts.engine_logs, "log\n")
    contract.write_json(run_dir / "command_logs/up.json", {"exit_code": 0})
    _write_trace(run_dir, manifest, text)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    contract.write_trace_parquet(run_dir, manifest)
    contract.write_telemetry_parquet(run_dir, manifest, [_telemetry_row()])
    contract.write_json(run_dir / artifacts.summary, contract.recompute_summary(run_dir, manifest))
    contract.write_checksums(run_dir, manifest)
    contract.write_json(run_dir / artifacts.manifest, manifest)
    assert contract.validate_run(run_dir, write=True).ok
    return run_dir


def _write_trace(run_dir: Path, manifest: contract.RunManifest, text: str) -> None:
    rows = [
        contract.TraceEvent(
            schema_version=1,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_start",
            perf_counter_ns=1_000_000_000,
            request={"model": "Qwen/Qwen3.5-2B"},
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
            delta_text_sha256=contract.sha256_text(text),
            delta_text_chars=len(text),
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
            generated_text_sha256=contract.sha256_text(text),
            generated_text_chars=len(text),
            usage={"completion_tokens": 2},
        ),
    ]
    _write_text(
        run_dir / manifest.artifacts.request_trace,
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
    )


def _raw_sse(text: str) -> str:
    return "\n".join(
        [
            json.dumps({"choices": [{"delta": {"content": text}}]}).join(("data: ", "")),
            "",
            'data: {"choices":[],"usage":{"completion_tokens":2}}',
            "",
            "data: [DONE]",
            "",
        ]
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


def _write_docs(tmp_path: Path) -> None:
    for doc in (
        "methodology.md",
        "limitations.md",
        "reproducibility.md",
        "privacy-and-redaction.md",
    ):
        _write_text(tmp_path / "docs" / doc, f"# {doc}\n")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
