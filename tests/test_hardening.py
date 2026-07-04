import json
from pathlib import Path

from inferno import artifact_contract as contract
from inferno import hardening
from inferno import study


def test_hardening_check_writes_artifacts_with_fake_workspace(tmp_path: Path) -> None:
    _write_fake_workspace(tmp_path)
    config_path = _write_hardening_config(tmp_path)

    assert (
        hardening.run_hardening_check(
            config_path=config_path,
            project_root=tmp_path,
            include_gpu_smoke=False,
            env={},
        )
        == 0
    )

    output_dir = tmp_path / "artifacts/hardening/repository_audit"
    latest = json.loads((output_dir / "latest.json").read_text())
    assert latest["run_family"] == "repository_audit"
    assert latest["status"] == "PASS_WITH_WARNINGS"
    assert (output_dir / "report.md").exists()
    assert (output_dir / "report_preview.html").exists()
    assert any(check["name"] == "gpu_gated_real_smoke" for check in latest["checks"])


def test_secret_scan_fails_on_private_pattern(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    leak = "ssh" + "://" + "person" + "@example" + ".com\n"
    (tmp_path / "docs/leak.md").write_text(leak, encoding="utf-8")
    config = hardening.HardeningConfig.model_validate(
        {
            "schema_version": 1,
            "hardening_id": "repository_audit-test",
            "required_files": ["pyproject.toml"],
            "schema_snapshot_dir": ".inferno/contracts/artifacts",
            "secret_scan_paths": ["docs"],
            "regression": {
                "research_release_package": "release",
                "deployment_profile_report_inputs": "inputs.json",
                "capacity_planner_latest": "planner.json",
                "router_replay_latest": "router.json",
            },
            "artifacts_dir": "artifacts/hardening/repository_audit",
        }
    )

    result = hardening.check_secret_scan(config, tmp_path)

    assert result["status"] == "FAIL"
    assert result["details"]["findings"][0]["pattern"] == "ssh_url"


def test_gpu_smoke_writes_redacted_preflight(monkeypatch, tmp_path: Path) -> None:
    def fake_collect_preflight(**kwargs):
        assert kwargs["include_gpu"] is True
        return {
            "schema_version": 1,
            "status": "ok",
            "checks": [{"name": "remote_gpu", "status": "ok"}],
        }

    monkeypatch.setattr(hardening.preflight, "collect_preflight", fake_collect_preflight)
    output_dir = tmp_path / "artifacts/hardening/repository_audit"
    output_dir.mkdir(parents=True)

    target = "person" + "@example" + ".com"
    result = hardening.check_gpu_smoke(tmp_path, {"INFERNO_GPU_SSH": target}, output_dir)

    assert result["status"] == "PASS"
    assert json.loads((output_dir / "gpu_smoke.json").read_text())["status"] == "ok"


def _write_fake_workspace(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "pyproject.toml",
        """
[project]
dependencies = ["pydantic>=2,<3", "pyarrow>=16,<21"]

[dependency-groups]
dev = ["pytest>=8.3", "ruff>=0.8"]
""".lstrip(),
    )
    _write_text(
        tmp_path / "uv.lock",
        """
[[package]]
name = "pydantic"
[[package]]
name = "pyarrow"
[[package]]
name = "pytest"
[[package]]
name = "ruff"
""".lstrip(),
    )
    for name, content in contract.schema_snapshots().items():
        _write_text(tmp_path / ".inferno/contracts/artifacts" / name, content)
    _write_json(
        tmp_path / "configs/engines/vllm.json",
        _engine_config("vllm", "vllm/vllm-openai@sha256:" + "a" * 64),
    )
    _write_json(
        tmp_path / "configs/engines/llamacpp.json",
        _engine_config(
            "llamacpp",
            "ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:" + "a" * 64,
            gguf=True,
        ),
    )
    _write_text(tmp_path / "reports/research_core.md", "Study type: strict_engine_comparison.\nLOW_SAMPLE\n")
    _write_text(tmp_path / "reports/router_replay.md", "OFFLINE REPLAY ONLY\nNEGATIVE_RESULT\n")
    _write_json(
        tmp_path / "artifacts/releases/research_core/redacted/manifest.json",
        {
            "schema_version": 1,
            "release_id": "research_core-test",
            "mode": "redacted",
            "retention_class": "public_release",
            "required_workload_families": [],
            "workload_families": [],
            "studies": [],
            "files": [],
        },
    )
    _write_deployment_profile_inputs(tmp_path)
    _write_json(
        tmp_path / "artifacts/planner/capacity_planning/latest.json",
        {"schema_version": 1, "run_family": "capacity_planning", "compatible_evidence_count": 30},
    )
    _write_json(
        tmp_path / "artifacts/router/router_replay/latest.json",
        {
            "schema_version": 1,
            "run_family": "router_replay",
            "baseline_comparison": {"status": "NEGATIVE_RESULT"},
            "leakage_check": {"status": "PASS"},
        },
    )
    for path in ("docs", "src", "tests"):
        (tmp_path / path).mkdir(exist_ok=True)


def _write_hardening_config(tmp_path: Path) -> Path:
    path = tmp_path / "configs/hardening/repository_audit.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "hardening_id": "repository_audit-test",
            "required_files": ["pyproject.toml", "uv.lock"],
            "schema_snapshot_dir": ".inferno/contracts/artifacts",
            "report_snapshots": [
                {
                    "id": "research_core",
                    "path": "reports/research_core.md",
                    "must_contain": ["strict_engine_comparison", "LOW_SAMPLE"],
                },
                {
                    "id": "router_replay",
                    "path": "reports/router_replay.md",
                    "must_contain": ["OFFLINE REPLAY ONLY", "NEGATIVE_RESULT"],
                },
            ],
            "engine_configs": [
                "configs/engines/vllm.json",
                "configs/engines/llamacpp.json",
            ],
            "secret_scan_paths": ["docs", "src", "tests"],
            "allowed_runtime_dependencies": ["pydantic", "pyarrow"],
            "allowed_dev_dependencies": ["pytest", "ruff"],
            "regression": {
                "research_release_package": "artifacts/releases/research_core/redacted",
                "deployment_profile_report_inputs": "artifacts/compare/deployment/report_inputs.json",
                "capacity_planner_latest": "artifacts/planner/capacity_planning/latest.json",
                "router_replay_latest": "artifacts/router/router_replay/latest.json",
            },
            "artifacts_dir": "artifacts/hardening/repository_audit",
        },
    )
    return path.relative_to(tmp_path)


def _write_deployment_profile_inputs(tmp_path: Path) -> None:
    for name in ("summary.json", "request_trace.jsonl", "validation.json"):
        _write_text(tmp_path / "artifacts/compare/deployment" / name, "{}\n")
    _write_json(
        tmp_path / "artifacts/compare/deployment/report_inputs.json",
        {
            "schema_version": 1,
            "study_type": "deployment_profile_comparison",
            "mode": "deployment-profile",
            "banner": study.DEPLOYMENT_PROFILE_BANNER,
            "non_matching_controls": [{"field": "weights.format", "values": {}}],
            "runs": [
                {
                    "engine": "llamacpp",
                    "run_id": "run-1",
                    "validation_ok": True,
                    "labels": ["LOW_SAMPLE"],
                    "sample_counts": {"successes": 1},
                    "artifacts": {
                        "summary": "artifacts/compare/deployment/summary.json",
                        "request_trace": "artifacts/compare/deployment/request_trace.jsonl",
                        "validation": "artifacts/compare/deployment/validation.json",
                    },
                }
            ],
            "engine_summaries": [],
        },
    )


def _engine_config(engine: str, image: str, *, gguf: bool = False) -> dict[str, object]:
    model = {
        "id": "Qwen/Qwen3.5-2B",
        "revision": "a" * 40,
        "tokenizer_id": "Qwen/Qwen3.5-2B",
        "tokenizer_revision": "a" * 40,
    }
    if gguf:
        model.update(
            {
                "gguf_repo": "lmstudio-community/Qwen3.5-2B-GGUF",
                "gguf_filename": "Qwen3.5-2B-Q4_K_M.gguf",
                "gguf_revision": "b" * 40,
                "quantization": "Q4_K_M",
            }
        )
    return {"schema_version": 1, "engine": engine, "image": image, "model": model}


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
