"""P3 vLLM smoke study and static report."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from inferno import artifact_contract as contract
from inferno import vllm_remote

EngineName = Literal["vllm", "sglang", "llamacpp", "tensorrtllm", "tensorrtllm_t4", "ollama"]
DEPLOYMENT_PROFILE_BANNER = (
    "DEPLOYMENT PROFILE COMPARISON - engine and one or more controlled deployment variables "
    "differ. Do not attribute outcomes to engine alone."
)
ENGINE_CONFIGURATION_BANNER = (
    "ENGINE CONFIGURATION COMPARISON - engine and runtime configuration differ. "
    "Do not attribute outcomes to engine alone."
)


class RunOrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: EngineName
    repeat_index: int = Field(ge=1)


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    study_id: str = Field(min_length=1)
    study_type: str = "single_engine_smoke"
    engine: EngineName | None = None
    engines: list[EngineName] | None = None
    phase: Literal["P3", "P4", "P5", "P6", "P10", "P11"] | None = None
    environment_block_id: str | None = None
    run_order_seed: int | None = None
    run_order: list[RunOrderItem] = Field(default_factory=list)
    workload_config: str = Field(min_length=1)
    repeats: int = Field(ge=1)
    telemetry_cadence_seconds: float = Field(gt=0)
    low_sample_completed_requests: int = Field(ge=1)
    artifacts_dir: str = Field(min_length=1)

    def allowed_engines(self) -> list[EngineName]:
        if self.engines:
            return self.engines
        if self.engine:
            return [self.engine]
        return []

    def run_phase(self) -> Literal["P3", "P4", "P5", "P6", "P10", "P11"]:
        if self.phase:
            return self.phase
        if self.study_type == "engine_configuration_comparison":
            return "P10"
        if self.study_type == "deployment_profile_comparison":
            return "P6"
        return "P4" if self.study_type == "strict_engine_comparison" else "P3"


def load_study_config(path: Path) -> StudyConfig:
    return StudyConfig.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def run_study(
    *,
    engine: str,
    config_path: Path,
    project_root: Path,
    env: Mapping[str, str],
) -> int:
    config = load_study_config(project_root / config_path)
    if engine not in config.allowed_engines():
        raise ValueError(f"study config engines are {config.allowed_engines()}, got {engine}")

    artifacts_dir = _project_path(project_root, config.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        record
        for record in _load_study_records(artifacts_dir)
        if record.get("engine", config.engine) != engine
    ]
    records: list[dict[str, Any]] = []
    for planned_order_index, repeat_index in _repeat_order(config, engine):
        result = vllm_remote.run_real_result(
            engine=engine,
            config_path=Path(config.workload_config),
            project_root=project_root,
            env=env,
            phase=config.run_phase(),
            study_id=config.study_id,
            repeat_index=repeat_index,
            telemetry_cadence_seconds=config.telemetry_cadence_seconds,
        )
        records.append(
            {
                "engine": engine,
                "repeat_index": repeat_index,
                "planned_order_index": planned_order_index,
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "exit_code": result.returncode,
                "validation_ok": result.validation_ok,
            }
        )
        _write_study_runs(artifacts_dir, config, existing + records)
        print(result.run_dir)

    _write_study_runs(artifacts_dir, config, existing + records)
    return 0 if len(records) == config.repeats and all(item["validation_ok"] for item in records) else 1


def report_study(*, study_path: Path, project_root: Path) -> int:
    config = load_study_config(project_root / study_path)
    artifacts_dir = _project_path(project_root, config.artifacts_dir)
    runs_path = artifacts_dir / "latest.json"
    if not runs_path.exists():
        runs_path = artifacts_dir / "study_runs.json"
    study_runs = json.loads(runs_path.read_text(encoding="utf-8-sig"))
    if config.study_type == "strict_engine_comparison":
        run_dirs = [_project_path(project_root, str(run["run_dir"])) for run in study_runs["runs"]]
        report_inputs, errors = build_strict_compare_inputs(
            run_dirs,
            project_root,
            study_id=config.study_id,
            low_sample_completed_requests=config.low_sample_completed_requests,
        )
        errors.extend(validate_strict_report_inputs(report_inputs, project_root))
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
            return 1
        renderer = _render_strict_compare_report
    else:
        report_inputs = build_report_inputs(config, study_runs["runs"], project_root)
        renderer = _render_report
    report_path = artifacts_dir / "report.md"
    inputs_path = artifacts_dir / "report_inputs.json"
    _write_json(inputs_path, report_inputs)
    report_path.write_text(renderer(report_inputs), encoding="utf-8")
    print(report_path)
    return 0 if all(item["validation_ok"] for item in report_inputs["runs"]) else 1


def compare_runs(*, mode: str, runs_arg: str, project_root: Path) -> int:
    run_dirs = _resolve_run_dirs(runs_arg, project_root)
    if mode == "strict":
        report_inputs, errors = build_strict_compare_inputs(run_dirs, project_root)
        errors.extend(validate_strict_report_inputs(report_inputs, project_root))
        prefix = "strict"
        renderer = _render_strict_compare_report
    elif mode == "deployment-profile":
        report_inputs, errors = build_deployment_profile_inputs(run_dirs, project_root)
        errors.extend(validate_deployment_profile_report_inputs(report_inputs, project_root))
        prefix = "deployment-profile"
        renderer = _render_deployment_profile_report
    elif mode == "engine-configuration":
        report_inputs, errors = build_engine_configuration_inputs(run_dirs, project_root)
        errors.extend(validate_engine_configuration_report_inputs(report_inputs, project_root))
        prefix = "engine-configuration"
        renderer = _render_engine_configuration_report
    else:
        raise ValueError("supported compare modes: strict, deployment-profile, engine-configuration")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 1
    output_dir = project_root / "artifacts" / "compare" / f"{prefix}-{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "report_inputs.json"
    report_path = output_dir / "report.md"
    _write_json(inputs_path, report_inputs)
    report_path.write_text(renderer(report_inputs), encoding="utf-8")
    print(report_path)
    return 0


def build_strict_compare_inputs(
    run_dirs: list[Path],
    project_root: Path,
    *,
    study_id: str | None = None,
    workload_family: str | None = None,
    low_sample_completed_requests: int = 3,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifests: list[contract.RunManifest] = []
    runs: list[dict[str, Any]] = []
    summaries: dict[str, contract.RunSummary] = {}
    artifact_sources: dict[str, dict[str, str]] = {}

    for run_dir in run_dirs:
        try:
            manifest = contract.read_manifest(run_dir)
            validation = contract.validate_run(run_dir, write=False)
            summary = contract.RunSummary.model_validate_json(
                contract.artifact_path(run_dir, manifest.artifacts.summary).read_text(
                    encoding="utf-8-sig"
                )
            )
        except Exception as exc:
            errors.append(f"{run_dir}: {exc}")
            continue

        manifests.append(manifest)
        summaries[manifest.run_id] = summary
        artifacts = _artifact_sources(run_dir, manifest)
        artifact_sources[manifest.run_id] = artifacts
        if not validation.ok:
            errors.append(f"{manifest.run_id}: validation failed: {'; '.join(validation.errors)}")
        labels = ["LOW_SAMPLE"] if summary.success_count < low_sample_completed_requests else []
        runs.append(
            {
                "engine": manifest.engine.name,
                "repeat_index": manifest.repeat_index,
                "run_id": manifest.run_id,
                "validation_ok": validation.ok,
                "validation_errors": validation.errors,
                "labels": labels,
                "sample_counts": _sample_counts(summary, manifest, run_dir),
                "metrics": _run_metrics(summary, manifest.run_id, artifacts),
                "response_evidence": {
                    "generated_text_sha256": summary.generated_text_sha256,
                    "generated_text_chars": summary.generated_text_chars,
                    "completion_tokens": summary.completion_tokens,
                    "completion_tokens_missing_reason": summary.completion_tokens_missing_reason,
                },
                "quality_evidence": _quality_evidence(run_dir, manifest),
                "artifacts": artifacts,
            }
        )

    engine_names = {manifest.engine.name for manifest in manifests}
    if "llamacpp" in engine_names:
        errors.append("strict comparison rejects llamacpp; use MODE=deployment-profile")
    if "ollama" in engine_names:
        errors.append("strict comparison rejects ollama; use MODE=deployment-profile")
    if engine_names != {"vllm", "sglang"}:
        errors.append(f"strict comparison requires vllm and sglang runs, got {sorted(engine_names)}")
    if not errors:
        errors.extend(contract.strict_comparison_errors(manifests))

    engine_summaries = []
    for engine in sorted(engine_names):
        engine_runs = [run for run in runs if run["engine"] == engine]
        engine_summaries.append(
            _engine_summary(
                engine,
                engine_runs,
                [summaries[run["run_id"]] for run in engine_runs],
                artifact_sources,
            )
        )

    return (
        {
            "schema_version": 1,
            "study_id": study_id,
            "study_type": "strict_engine_comparison",
            "workload_family": workload_family,
            "low_sample_completed_requests": low_sample_completed_requests,
            "mode": "strict",
            "generated_at": datetime.now(UTC).isoformat(),
            "run_ids": [run["run_id"] for run in runs],
            "controlled_comparison_key": manifests[0].comparison_key if manifests else None,
            "controlled_key_status": "MATCH" if not errors else "REFUSED",
            "runs": runs,
            "engine_summaries": engine_summaries,
            "response_equivalence": _response_equivalence(runs),
            "limitations": [
                "Strict conclusions apply only to the validated hardware, model, workload, and controls above.",
                "LOW_SAMPLE marks runs with fewer than three completed requests.",
                "No universal engine winner is claimed.",
            ],
        },
        errors,
    )


def build_deployment_profile_inputs(
    run_dirs: list[Path],
    project_root: Path,
    *,
    study_id: str | None = None,
    workload_family: str | None = None,
    low_sample_completed_requests: int = 3,
    study_type: str = "deployment_profile_comparison",
    mode: str = "deployment-profile",
    banner: str = DEPLOYMENT_PROFILE_BANNER,
    required_engine: str | tuple[str, ...] = ("llamacpp", "ollama"),
    limitations: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifests: list[contract.RunManifest] = []
    runs: list[dict[str, Any]] = []
    summaries: dict[str, contract.RunSummary] = {}
    artifact_sources: dict[str, dict[str, str]] = {}

    for run_dir in run_dirs:
        try:
            manifest = contract.read_manifest(run_dir)
            validation = contract.validate_run(run_dir, write=False)
            summary = contract.RunSummary.model_validate_json(
                contract.artifact_path(run_dir, manifest.artifacts.summary).read_text(
                    encoding="utf-8-sig"
                )
            )
        except Exception as exc:
            errors.append(f"{run_dir}: {exc}")
            continue

        manifests.append(manifest)
        summaries[manifest.run_id] = summary
        artifacts = _artifact_sources(run_dir, manifest)
        artifact_sources[manifest.run_id] = artifacts
        if not validation.ok:
            errors.append(f"{manifest.run_id}: validation failed: {'; '.join(validation.errors)}")
        labels = ["LOW_SAMPLE"] if summary.success_count < low_sample_completed_requests else []
        runs.append(
            {
                "engine": manifest.engine.name,
                "repeat_index": manifest.repeat_index,
                "run_id": manifest.run_id,
                "validation_ok": validation.ok,
                "validation_errors": validation.errors,
                "labels": labels,
                "profile": _deployment_profile(manifest),
                "sample_counts": _sample_counts(summary, manifest, run_dir),
                "metrics": _run_metrics(summary, manifest.run_id, artifacts),
                "response_evidence": {
                    "generated_text_sha256": summary.generated_text_sha256,
                    "generated_text_chars": summary.generated_text_chars,
                    "completion_tokens": summary.completion_tokens,
                    "completion_tokens_missing_reason": summary.completion_tokens_missing_reason,
                },
                "quality_evidence": _quality_evidence(run_dir, manifest),
                "artifacts": artifacts,
            }
        )

    engine_names = {manifest.engine.name for manifest in manifests}
    if len(engine_names) < 2:
        errors.append(f"{mode} comparison requires at least two engines")
    required_engines = set(required_engine) if isinstance(required_engine, (list, tuple, set)) else {required_engine}
    if engine_names.isdisjoint(required_engines):
        label = " or ".join(sorted(required_engines))
        errors.append(f"{mode} comparison requires at least one {label} run")

    non_matching_controls = _non_matching_controls(manifests)
    engine_summaries = []
    for engine in sorted(engine_names):
        engine_runs = [run for run in runs if run["engine"] == engine]
        engine_summaries.append(
            _engine_summary(
                engine,
                engine_runs,
                [summaries[run["run_id"]] for run in engine_runs],
                artifact_sources,
            )
        )

    return (
        {
            "schema_version": 1,
            "study_id": study_id,
            "study_type": study_type,
            "workload_family": workload_family,
            "low_sample_completed_requests": low_sample_completed_requests,
            "mode": mode,
            "banner": banner,
            "generated_at": datetime.now(UTC).isoformat(),
            "run_ids": [run["run_id"] for run in runs],
            "non_matching_controls": non_matching_controls,
            "runs": runs,
            "engine_summaries": engine_summaries,
            "response_equivalence": _response_equivalence(runs),
            "limitations": limitations or [
                "This is not a strict engine comparison.",
                "Engine, weight format, precision, quantization, or build controls may differ.",
                "Do not attribute outcomes to engine identity alone.",
            ],
        },
        errors,
    )


def build_engine_configuration_inputs(
    run_dirs: list[Path],
    project_root: Path,
    *,
    study_id: str | None = None,
    workload_family: str | None = None,
    low_sample_completed_requests: int = 3,
) -> tuple[dict[str, Any], list[str]]:
    return build_deployment_profile_inputs(
        run_dirs,
        project_root,
        study_id=study_id,
        workload_family=workload_family,
        low_sample_completed_requests=low_sample_completed_requests,
        study_type="engine_configuration_comparison",
        mode="engine-configuration",
        banner=ENGINE_CONFIGURATION_BANNER,
        required_engine=("tensorrtllm", "tensorrtllm_t4"),
        limitations=[
            "This is not a strict engine-only claim.",
            "TensorRT-LLM runtime, backend, and build controls may differ from other engines.",
            "Do not attribute outcomes to engine identity alone.",
        ],
    )


def validate_strict_report_inputs(
    report_inputs: Mapping[str, Any],
    project_root: Path,
) -> list[str]:
    errors: list[str] = []
    if report_inputs.get("study_type") != "strict_engine_comparison":
        errors.append("report study_type is not strict_engine_comparison")
    if report_inputs.get("controlled_key_status") != "MATCH":
        errors.append("strict controlled key status is not MATCH")

    threshold = int(report_inputs.get("low_sample_completed_requests") or 3)
    for run in report_inputs.get("runs", []):
        run_id = run.get("run_id", "<unknown>")
        if not run.get("validation_ok"):
            errors.append(f"{run_id}: validation_ok is false")
        successes = int((run.get("sample_counts") or {}).get("successes") or 0)
        if successes < threshold and "LOW_SAMPLE" not in set(run.get("labels") or []):
            errors.append(f"{run_id}: missing LOW_SAMPLE label")
        for name, path in (run.get("artifacts") or {}).items():
            if path in (None, "", "[redacted]"):
                continue
            if name == "raw_sse" and str(path).startswith("redacted:"):
                continue
            if not _path_exists(project_root, str(path)):
                errors.append(f"{run_id}: unresolved artifact path: {path}")

    for path in _source_paths(report_inputs.get("engine_summaries", [])):
        if not _path_exists(project_root, path):
            errors.append(f"unresolved metric source path: {path}")
    for path in _source_paths(report_inputs.get("runs", [])):
        if not _path_exists(project_root, path):
            errors.append(f"unresolved metric source path: {path}")
    return sorted(set(errors))


def validate_deployment_profile_report_inputs(
    report_inputs: Mapping[str, Any],
    project_root: Path,
) -> list[str]:
    errors: list[str] = []
    if report_inputs.get("study_type") != "deployment_profile_comparison":
        errors.append("report study_type is not deployment_profile_comparison")
    if report_inputs.get("mode") != "deployment-profile":
        errors.append("report mode is not deployment-profile")
    if report_inputs.get("banner") != DEPLOYMENT_PROFILE_BANNER:
        errors.append("deployment-profile banner is missing or changed")
    if not report_inputs.get("non_matching_controls"):
        errors.append("deployment-profile report must list non-matching controls")
    engines = {run.get("engine") for run in report_inputs.get("runs", [])}
    if engines.isdisjoint({"llamacpp", "ollama"}):
        errors.append("deployment-profile report requires a llamacpp or ollama run")

    threshold = int(report_inputs.get("low_sample_completed_requests") or 3)
    for run in report_inputs.get("runs", []):
        run_id = run.get("run_id", "<unknown>")
        if not run.get("validation_ok"):
            errors.append(f"{run_id}: validation_ok is false")
        successes = int((run.get("sample_counts") or {}).get("successes") or 0)
        if successes < threshold and "LOW_SAMPLE" not in set(run.get("labels") or []):
            errors.append(f"{run_id}: missing LOW_SAMPLE label")
        for name, path in (run.get("artifacts") or {}).items():
            if path in (None, "", "[redacted]"):
                continue
            if name == "raw_sse" and str(path).startswith("redacted:"):
                continue
            if not _path_exists(project_root, str(path)):
                errors.append(f"{run_id}: unresolved artifact path: {path}")

    for path in _source_paths(report_inputs.get("engine_summaries", [])):
        if not _path_exists(project_root, path):
            errors.append(f"unresolved metric source path: {path}")
    for path in _source_paths(report_inputs.get("runs", [])):
        if not _path_exists(project_root, path):
            errors.append(f"unresolved metric source path: {path}")
    return sorted(set(errors))


def validate_engine_configuration_report_inputs(
    report_inputs: Mapping[str, Any],
    project_root: Path,
) -> list[str]:
    errors: list[str] = []
    if report_inputs.get("study_type") != "engine_configuration_comparison":
        errors.append("report study_type is not engine_configuration_comparison")
    if report_inputs.get("mode") != "engine-configuration":
        errors.append("report mode is not engine-configuration")
    if report_inputs.get("banner") != ENGINE_CONFIGURATION_BANNER:
        errors.append("engine-configuration banner is missing or changed")
    if not report_inputs.get("non_matching_controls"):
        errors.append("engine-configuration report must list non-matching controls")
    engines = {run.get("engine") for run in report_inputs.get("runs", [])}
    if engines.isdisjoint({"tensorrtllm", "tensorrtllm_t4"}):
        errors.append("engine-configuration report requires a TensorRT-LLM run")

    threshold = int(report_inputs.get("low_sample_completed_requests") or 3)
    for run in report_inputs.get("runs", []):
        run_id = run.get("run_id", "<unknown>")
        if not run.get("validation_ok"):
            errors.append(f"{run_id}: validation_ok is false")
        successes = int((run.get("sample_counts") or {}).get("successes") or 0)
        if successes < threshold and "LOW_SAMPLE" not in set(run.get("labels") or []):
            errors.append(f"{run_id}: missing LOW_SAMPLE label")
        for name, path in (run.get("artifacts") or {}).items():
            if path in (None, "", "[redacted]"):
                continue
            if name == "raw_sse" and str(path).startswith("redacted:"):
                continue
            if not _path_exists(project_root, str(path)):
                errors.append(f"{run_id}: unresolved artifact path: {path}")

    for path in _source_paths(report_inputs.get("engine_summaries", [])):
        if not _path_exists(project_root, path):
            errors.append(f"unresolved metric source path: {path}")
    for path in _source_paths(report_inputs.get("runs", [])):
        if not _path_exists(project_root, path):
            errors.append(f"unresolved metric source path: {path}")
    return sorted(set(errors))


def build_report_inputs(
    config: StudyConfig,
    run_records: list[Mapping[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    runs = []
    for record in run_records:
        run_dir = _project_path(project_root, str(record["run_dir"]))
        manifest = contract.read_manifest(run_dir)
        validation = contract.validate_run(run_dir, write=False)
        summary = contract.RunSummary.model_validate_json(
            contract.artifact_path(run_dir, manifest.artifacts.summary).read_text(encoding="utf-8-sig")
        )
        artifacts = _artifact_sources(run_dir, manifest)
        success_count = summary.success_count
        labels = ["LOW_SAMPLE"] if success_count < config.low_sample_completed_requests else []
        telemetry_samples = (
            pq.read_table(contract.artifact_path(run_dir, manifest.artifacts.telemetry_parquet)).num_rows
            if manifest.artifacts.telemetry_parquet
            else 0
        )
        runs.append(
            {
                "repeat_index": manifest.repeat_index,
                "run_id": manifest.run_id,
                "validation_ok": validation.ok,
                "validation_errors": validation.errors,
                "labels": labels,
                "sample_counts": {
                    "requests": summary.request_count,
                    "successes": success_count,
                    "errors": summary.error_count,
                    "itl_samples": len(summary.itl_ms),
                    "telemetry_samples": telemetry_samples,
                },
                "metrics": {
                    "ttft_ms": _metric_percentiles([summary.ttft_ms], manifest.run_id, artifacts["summary"]),
                    "e2e_ms": _metric_percentiles([summary.e2e_ms], manifest.run_id, artifacts["summary"]),
                    "itl_ms": _metric_percentiles(summary.itl_ms, manifest.run_id, artifacts["request_trace"]),
                    "request_throughput_rps": _metric_value(
                        summary.request_throughput_rps, manifest.run_id, artifacts["summary"]
                    ),
                    "output_token_throughput_tps": _metric_value(
                        summary.output_token_throughput_tps,
                        manifest.run_id,
                        artifacts["summary"],
                        summary.output_token_throughput_missing_reason,
                    ),
                    "error_rate": _metric_value(summary.error_rate, manifest.run_id, artifacts["summary"]),
                },
                "artifacts": artifacts,
            }
        )

    return {
        "schema_version": 1,
        "study_id": config.study_id,
        "study_type": config.study_type,
        "generated_at": datetime.now(UTC).isoformat(),
        "study_config": config.model_dump(mode="json"),
        "run_ids": [run["run_id"] for run in runs],
        "runs": runs,
        "limitations": [
            "Single-engine report only; no cross-engine comparison or winner claim.",
            "LOW_SAMPLE appears when completed request count is below the configured threshold.",
        ],
    }


def _write_study_runs(
    artifacts_dir: Path,
    config: StudyConfig,
    records: list[Mapping[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "study_id": config.study_id,
        "study_type": config.study_type,
        "engine": config.engine,
        "engines": config.allowed_engines(),
        "environment_block_id": config.environment_block_id,
        "run_order_seed": config.run_order_seed,
        "planned_run_order": [item.model_dump(mode="json") for item in config.run_order],
        "study_config": config.model_dump(mode="json"),
        "updated_at": datetime.now(UTC).isoformat(),
        "runs": sorted(
            list(records),
            key=lambda item: (
                int(item.get("planned_order_index") or 9999),
                str(item.get("engine") or ""),
                int(item.get("repeat_index") or 0),
            ),
        ),
    }
    _write_json(artifacts_dir / "study_runs.json", payload)
    _write_json(artifacts_dir / "latest.json", payload)


def _load_study_records(artifacts_dir: Path) -> list[dict[str, Any]]:
    latest = artifacts_dir / "latest.json"
    if not latest.exists():
        return []
    return list(json.loads(latest.read_text(encoding="utf-8-sig")).get("runs", []))


def _repeat_order(config: StudyConfig, engine: str) -> list[tuple[int | None, int]]:
    if not config.run_order:
        return [(None, index) for index in range(1, config.repeats + 1)]
    pairs = [
        (index, item.repeat_index)
        for index, item in enumerate(config.run_order, start=1)
        if item.engine == engine
    ]
    if len(pairs) != config.repeats:
        raise ValueError(f"run_order has {len(pairs)} entries for {engine}, expected {config.repeats}")
    return pairs


def _render_report(report_inputs: Mapping[str, Any]) -> str:
    lines = [
        f"# {report_inputs['study_id']} Single-Engine Report",
        "",
        f"Study type: {report_inputs['study_type']}. No cross-engine comparison or winner claim.",
        "",
        "| Repeat | Run ID | Valid | Labels | TTFT p50/p95/p99 ms | E2E p50/p95/p99 ms | ITL p50/p95/p99 ms | Req/s | Output tok/s | Error rate | Samples |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report_inputs["runs"]:
        metrics = run["metrics"]
        lines.append(
            "| {repeat} | `{run_id}` | {valid} | {labels} | {ttft} | {e2e} | {itl} | {rps} | {otps} | {error} | {samples} |".format(
                repeat=run["repeat_index"],
                run_id=run["run_id"],
                valid="yes" if run["validation_ok"] else "no",
                labels=", ".join(run["labels"]) or "-",
                ttft=_fmt_percentiles(metrics["ttft_ms"]),
                e2e=_fmt_percentiles(metrics["e2e_ms"]),
                itl=_fmt_percentiles(metrics["itl_ms"]),
                rps=_fmt_metric(metrics["request_throughput_rps"]),
                otps=_fmt_metric(metrics["output_token_throughput_tps"]),
                error=_fmt_metric(metrics["error_rate"]),
                samples=run["sample_counts"]["successes"],
            )
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Every value above resolves through `report_inputs.json` to validated run IDs and artifact paths.",
        ]
    )
    for run in report_inputs["runs"]:
        lines.append(f"- `{run['run_id']}`: validation={run['artifacts']['validation']}, telemetry={run['artifacts']['telemetry']}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Single-engine report only; not a strict engine comparison.",
            "- `LOW_SAMPLE` is expected for the one-request smoke workload.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_strict_compare_report(report_inputs: Mapping[str, Any]) -> str:
    study_name = report_inputs.get("study_id") or "Strict vLLM vs SGLang"
    response = report_inputs["response_equivalence"]
    lines = [
        f"# {study_name} Strict vLLM vs SGLang Report",
        "",
        "Study type: strict_engine_comparison.",
        "",
        f"Controlled key status: {report_inputs['controlled_key_status']}.",
        (
            "Response equivalence / quality: E1 configuration equivalence passed; "
            f"{response['e2_status']}; {response['quality_status']}."
        ),
        "",
        "## Controlled Variables",
        "",
        "| Field | Value |",
        "|---|---|",
        *_controlled_key_rows(report_inputs.get("controlled_comparison_key") or {}),
        "",
        "## Engine Summary",
        "",
        "| Engine | Runs | Labels | TTFT p50/p95/p99 ms | E2E p50/p95/p99 ms | Req/s p50/p95/p99 | Output tok/s p50/p95/p99 | Error p50/p95/p99 | Samples |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report_inputs["engine_summaries"]:
        metrics = item["metrics"]
        lines.append(
            "| {engine} | {runs} | {labels} | {ttft} | {e2e} | {rps} | {otps} | {error} | {samples} |".format(
                engine=item["engine"],
                runs=", ".join(f"`{run_id}`" for run_id in item["run_ids"]),
                labels=", ".join(item["labels"]) or "-",
                ttft=_fmt_percentiles(metrics["ttft_ms"]),
                e2e=_fmt_percentiles(metrics["e2e_ms"]),
                rps=_fmt_percentiles(metrics["request_throughput_rps"]),
                otps=_fmt_percentiles(metrics["output_token_throughput_tps"]),
                error=_fmt_percentiles(metrics["error_rate"]),
                samples=item["sample_counts"]["successes"],
            )
        )

    lines.extend(
        [
            "",
            "## Repeat-Level Uncertainty",
            "",
            "| Engine | Metric | Repeats | Mean | 95% CI half-width | Min | Max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report_inputs["engine_summaries"]:
        for metric_name, uncertainty in item.get("uncertainty", {}).items():
            lines.append(
                "| {engine} | {metric} | {n} | {mean} | {ci} | {minv} | {maxv} |".format(
                    engine=item["engine"],
                    metric=metric_name,
                    n=uncertainty["n"],
                    mean=_fmt_optional_number(uncertainty["mean"]),
                    ci=_fmt_optional_number(uncertainty["ci95_half_width"]),
                    minv=_fmt_optional_number(uncertainty["min"]),
                    maxv=_fmt_optional_number(uncertainty["max"]),
                )
            )

    lines.extend(
        [
            "",
            "## Repeat Runs",
            "",
            "| Engine | Repeat | Run ID | Valid | Labels | Successes | TTFT ms | E2E ms | Output SHA256 | Completion Tokens | Quality |",
            "|---|---:|---|---|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for run in report_inputs["runs"]:
        lines.append(
            "| {engine} | {repeat} | `{run_id}` | {valid} | {labels} | {successes} | {ttft} | {e2e} | `{sha}` | {tokens} | {quality} |".format(
                engine=run["engine"],
                repeat=run["repeat_index"],
                run_id=run["run_id"],
                valid="yes" if run["validation_ok"] else "no",
                labels=", ".join(run["labels"]) or "-",
                successes=run["sample_counts"]["successes"],
                ttft=_fmt_metric(run["metrics"]["ttft_ms"]["p50"]),
                e2e=_fmt_metric(run["metrics"]["e2e_ms"]["p50"]),
                sha=run["response_evidence"]["generated_text_sha256"],
                tokens=run["response_evidence"]["completion_tokens"],
                quality=(run.get("quality_evidence") or {}).get("status", "unknown"),
            )
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Every value above resolves through `report_inputs.json` to validated run IDs and artifact paths.",
        ]
    )
    for run in report_inputs["runs"]:
        raw = run["artifacts"].get("raw_sse", "redacted:not_in_public_report")
        lines.append(
            f"- `{run['run_id']}`: validation={run['artifacts']['validation']}, "
            f"raw={raw}, trace={run['artifacts']['request_trace']}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Strict conclusions apply only to this validated hardware, model, workload, and control envelope.",
            "- `LOW_SAMPLE` is visible where request counts are below the strict reporting threshold.",
            "- This report does not declare a universal winner.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_deployment_profile_report(report_inputs: Mapping[str, Any]) -> str:
    study_name = report_inputs.get("study_id") or (
        "P10" if report_inputs.get("mode") == "engine-configuration" else "P6"
    )
    response = report_inputs["response_equivalence"]
    lines = [
        f"# {study_name} Deployment Profile Report",
        "",
        "Study type: deployment_profile_comparison.",
        "",
        report_inputs["banner"],
        "",
        (
            "Response / quality: deployment profiles may produce different outputs; "
            f"{response['e2_status']}; {response['quality_status']}."
        ),
        "",
        "## Non-Matching Controls",
        "",
        "| Field | Values |",
        "|---|---|",
    ]
    for item in report_inputs["non_matching_controls"]:
        values = ", ".join(
            f"`{run_id}`={json.dumps(value, sort_keys=True)}"
            for run_id, value in item["values"].items()
        )
        lines.append(f"| {item['field']} | {values} |")

    lines.extend(
        [
            "",
            "## Engine Summary",
            "",
            "| Engine | Runs | Labels | TTFT p50/p95/p99 ms | E2E p50/p95/p99 ms | Req/s p50/p95/p99 | Output tok/s p50/p95/p99 | Error p50/p95/p99 | Samples |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report_inputs["engine_summaries"]:
        metrics = item["metrics"]
        lines.append(
            "| {engine} | {runs} | {labels} | {ttft} | {e2e} | {rps} | {otps} | {error} | {samples} |".format(
                engine=item["engine"],
                runs=", ".join(f"`{run_id}`" for run_id in item["run_ids"]),
                labels=", ".join(item["labels"]) or "-",
                ttft=_fmt_percentiles(metrics["ttft_ms"]),
                e2e=_fmt_percentiles(metrics["e2e_ms"]),
                rps=_fmt_percentiles(metrics["request_throughput_rps"]),
                otps=_fmt_percentiles(metrics["output_token_throughput_tps"]),
                error=_fmt_percentiles(metrics["error_rate"]),
                samples=item["sample_counts"]["successes"],
            )
        )

    lines.extend(
        [
            "",
            "## Repeat Runs",
            "",
            "| Engine | Repeat | Run ID | Valid | Profile | Labels | Successes | TTFT ms | E2E ms | Completion Tokens | Quality |",
            "|---|---:|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for run in report_inputs["runs"]:
        profile = run["profile"]
        profile_text = (
            f"{profile['weight_format']}/{profile['precision']}/"
            f"{profile['quantization'] or 'none'}"
        )
        lines.append(
            "| {engine} | {repeat} | `{run_id}` | {valid} | {profile} | {labels} | {successes} | {ttft} | {e2e} | {tokens} | {quality} |".format(
                engine=run["engine"],
                repeat=run["repeat_index"],
                run_id=run["run_id"],
                valid="yes" if run["validation_ok"] else "no",
                profile=profile_text,
                labels=", ".join(run["labels"]) or "-",
                successes=run["sample_counts"]["successes"],
                ttft=_fmt_metric(run["metrics"]["ttft_ms"]["p50"]),
                e2e=_fmt_metric(run["metrics"]["e2e_ms"]["p50"]),
                tokens=run["response_evidence"]["completion_tokens"],
                quality=(run.get("quality_evidence") or {}).get("status", "unknown"),
            )
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Every value above resolves through `report_inputs.json` to validated run IDs and artifact paths.",
        ]
    )
    for run in report_inputs["runs"]:
        raw = run["artifacts"].get("raw_sse", "redacted:not_in_public_report")
        lines.append(
            f"- `{run['run_id']}`: validation={run['artifacts']['validation']}, "
            f"raw={raw}, trace={run['artifacts']['request_trace']}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This report is a deployment-profile comparison, not a strict engine-only claim.",
            "- Non-matching controls above are part of the result interpretation.",
            "- This report does not declare a universal winner.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_engine_configuration_report(report_inputs: Mapping[str, Any]) -> str:
    return (
        _render_deployment_profile_report(report_inputs)
        .replace("Deployment Profile Report", "Engine Configuration Report", 1)
        .replace(
            "deployment profiles may produce different outputs",
            "engine configurations may produce different outputs",
            1,
        )
        .replace(
            "Study type: deployment_profile_comparison.",
            "Study type: engine_configuration_comparison.",
            1,
        )
        .replace(
            "This report is a deployment-profile comparison",
            "This report is an engine-configuration comparison",
        )
    )


def _artifact_sources(run_dir: Path, manifest: contract.RunManifest) -> dict[str, str]:
    return {
        "summary": str(contract.artifact_path(run_dir, manifest.artifacts.summary)),
        "request_trace": str(contract.artifact_path(run_dir, manifest.artifacts.request_trace)),
        "request_trace_parquet": str(
            contract.artifact_path(run_dir, manifest.artifacts.request_trace_parquet or "")
        ),
        "raw_sse": str(contract.artifact_path(run_dir, manifest.artifacts.raw_sse)),
        "telemetry": str(contract.artifact_path(run_dir, manifest.artifacts.telemetry_parquet or "")),
        "validation": str(contract.artifact_path(run_dir, manifest.artifacts.validation)),
    }


def _resolve_run_dirs(runs_arg: str, project_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for token in runs_arg.split():
        candidate = _project_path(project_root, token)
        if candidate.is_dir() and (candidate / "latest.json").exists():
            payload = json.loads((candidate / "latest.json").read_text(encoding="utf-8-sig"))
            run_dirs.extend(_project_path(project_root, str(run["run_dir"])) for run in payload["runs"])
        elif candidate.is_file() and candidate.name == "latest.json":
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
            run_dirs.extend(_project_path(project_root, str(run["run_dir"])) for run in payload["runs"])
        elif candidate.is_dir():
            run_dirs.append(candidate)
        else:
            run_dirs.append(project_root / "artifacts" / "runs" / token)
    return run_dirs


def _sample_counts(
    summary: contract.RunSummary,
    manifest: contract.RunManifest,
    run_dir: Path,
) -> dict[str, int]:
    telemetry_samples = (
        pq.read_table(contract.artifact_path(run_dir, manifest.artifacts.telemetry_parquet)).num_rows
        if manifest.artifacts.telemetry_parquet
        else 0
    )
    return {
        "requests": summary.request_count,
        "successes": summary.success_count,
        "errors": summary.error_count,
        "itl_samples": len(summary.itl_ms),
        "telemetry_samples": telemetry_samples,
    }


def _run_metrics(
    summary: contract.RunSummary,
    run_id: str,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "ttft_ms": _metric_percentiles([summary.ttft_ms], run_id, artifacts["summary"]),
        "e2e_ms": _metric_percentiles([summary.e2e_ms], run_id, artifacts["summary"]),
        "itl_ms": _metric_percentiles(summary.itl_ms, run_id, artifacts["request_trace"]),
        "request_throughput_rps": _metric_value(
            summary.request_throughput_rps,
            run_id,
            artifacts["summary"],
        ),
        "output_token_throughput_tps": _metric_value(
            summary.output_token_throughput_tps,
            run_id,
            artifacts["summary"],
            summary.output_token_throughput_missing_reason,
        ),
        "error_rate": _metric_value(summary.error_rate, run_id, artifacts["summary"]),
    }


def _engine_summary(
    engine: str,
    runs: list[Mapping[str, Any]],
    summaries: list[contract.RunSummary],
    artifact_sources: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    run_ids = [run["run_id"] for run in runs]
    summary_paths = [artifact_sources[run_id]["summary"] for run_id in run_ids]
    trace_paths = [artifact_sources[run_id]["request_trace"] for run_id in run_ids]
    return {
        "engine": engine,
        "run_ids": run_ids,
        "labels": sorted({label for run in runs for label in run["labels"]}),
        "sample_counts": {
            "runs": len(runs),
            "successes": sum(summary.success_count for summary in summaries),
            "requests": sum(summary.request_count for summary in summaries),
        },
        "metrics": {
            "ttft_ms": _metric_percentiles_many(
                [summary.ttft_ms for summary in summaries], run_ids, summary_paths
            ),
            "e2e_ms": _metric_percentiles_many(
                [summary.e2e_ms for summary in summaries], run_ids, summary_paths
            ),
            "itl_ms": _metric_percentiles_many(
                [value for summary in summaries for value in summary.itl_ms], run_ids, trace_paths
            ),
            "request_throughput_rps": _metric_percentiles_many(
                [summary.request_throughput_rps for summary in summaries], run_ids, summary_paths
            ),
            "output_token_throughput_tps": _metric_percentiles_many(
                [summary.output_token_throughput_tps for summary in summaries],
                run_ids,
                summary_paths,
                missing_reason="output_token_throughput_missing_in_one_or_more_runs"
                if any(summary.output_token_throughput_tps is None for summary in summaries)
                else None,
            ),
            "error_rate": _metric_percentiles_many(
                [summary.error_rate for summary in summaries], run_ids, summary_paths
            ),
        },
        "uncertainty": {
            "ttft_ms": _repeat_uncertainty(
                [summary.ttft_ms for summary in summaries], run_ids, summary_paths
            ),
            "e2e_ms": _repeat_uncertainty(
                [summary.e2e_ms for summary in summaries], run_ids, summary_paths
            ),
            "itl_ms": _repeat_uncertainty(
                [_percentile(summary.itl_ms, 50) for summary in summaries],
                run_ids,
                trace_paths,
            ),
            "request_throughput_rps": _repeat_uncertainty(
                [summary.request_throughput_rps for summary in summaries], run_ids, summary_paths
            ),
            "output_token_throughput_tps": _repeat_uncertainty(
                [summary.output_token_throughput_tps for summary in summaries],
                run_ids,
                summary_paths,
            ),
            "error_rate": _repeat_uncertainty(
                [summary.error_rate for summary in summaries], run_ids, summary_paths
            ),
        },
    }


def _deployment_profile(manifest: contract.RunManifest) -> dict[str, Any]:
    key = contract.comparison_key(manifest)
    weights = key.get("weights") or {}
    model = key.get("model") or {}
    return {
        "engine": manifest.engine.name,
        "engine_image": manifest.engine.image,
        "model": f"{manifest.model.id}@{manifest.model.revision}",
        "tokenizer": f"{manifest.model.tokenizer_id}@{manifest.model.tokenizer_revision}",
        "tokenizer_format": model.get("tokenizer_format"),
        "weight_format": weights.get("format"),
        "precision": weights.get("precision"),
        "quantization": weights.get("quantization"),
        "gguf_repo": weights.get("gguf_repo"),
        "gguf_filename": weights.get("gguf_filename"),
        "gguf_revision": weights.get("gguf_revision"),
    }


def _non_matching_controls(manifests: list[contract.RunManifest]) -> list[dict[str, Any]]:
    if len(manifests) < 2:
        return []
    keys = {manifest.run_id: contract.comparison_key(manifest) for manifest in manifests}
    baseline = next(iter(keys.values()))
    fields = sorted(
        {
            path
            for key in keys.values()
            for path in _diff_paths(baseline, key)
            if path != "schema_version"
        }
    )
    return [
        {
            "field": field,
            "values": {run_id: _lookup_path(key, field) for run_id, key in keys.items()},
        }
        for field in fields
    ]


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        fields: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                fields.append(child)
            else:
                fields.extend(_diff_paths(left[key], right[key], child))
        return fields
    return [] if left == right else [prefix or "<root>"]


def _lookup_path(value: Mapping[str, Any], path: str) -> Any:
    cursor: Any = value
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _response_equivalence(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    fingerprints = {
        run["run_id"]: run["response_evidence"]["generated_text_sha256"] for run in runs
    }
    token_counts = {
        run["run_id"]: run["response_evidence"]["completion_tokens"] for run in runs
    }
    present_fingerprints = {value for value in fingerprints.values() if value}
    if not present_fingerprints:
        e2_status = "E2_UNAVAILABLE"
    elif len(present_fingerprints) == 1:
        e2_status = "E2_MATCH"
    else:
        e2_status = "E2_DIFFERENT"
    return {
        "e1_configuration_equivalence": True,
        "e2_status": e2_status,
        "e2_output_fingerprints": fingerprints,
        "e2_completion_tokens": token_counts,
        "quality_status": _combined_quality_status(runs),
        "quality_by_run": {
            run["run_id"]: run.get("quality_evidence", {"status": "E3_NOT_APPLICABLE"})
            for run in runs
        },
    }


def _quality_evidence(run_dir: Path, manifest: contract.RunManifest) -> dict[str, Any]:
    if "structured" not in manifest.workload.workload_id:
        return {"tier": "E2", "status": "E3_NOT_APPLICABLE", "detail": "no task scorer declared"}
    try:
        text = "".join(
            value
            for _, value in contract.generated_text_events(
                contract.artifact_path(run_dir, manifest.artifacts.raw_sse).read_text(
                    encoding="utf-8-sig"
                )
            )
        ).strip()
        payload = json.loads(text)
    except Exception as exc:
        return {
            "tier": "E3",
            "status": "E3_JSON_INVALID",
            "detail": type(exc).__name__,
        }
    return {
        "tier": "E3",
        "status": "E3_JSON_VALID" if isinstance(payload, dict) else "E3_JSON_INVALID",
        "detail": "json_object" if isinstance(payload, dict) else "json_non_object",
    }


def _combined_quality_status(runs: list[Mapping[str, Any]]) -> str:
    statuses = {str((run.get("quality_evidence") or {}).get("status")) for run in runs}
    if "E3_JSON_INVALID" in statuses:
        return "E3_JSON_INVALID"
    if "E3_JSON_VALID" in statuses:
        return "E3_JSON_VALID" if statuses <= {"E3_JSON_VALID"} else "E3_MIXED"
    return "E3_NOT_APPLICABLE"


def _repeat_uncertainty(
    values: list[float | None],
    run_ids: list[str],
    sources: list[str],
) -> dict[str, Any]:
    pairs = [(run_id, value, source) for run_id, value, source in zip(run_ids, values, sources) if value is not None]
    present = [float(value) for _, value, _ in pairs]
    if not present:
        return {
            "n": 0,
            "repeat_values": [],
            "mean": None,
            "min": None,
            "max": None,
            "ci95_half_width": None,
            "source_run_ids": [],
            "source_paths": [],
        }
    mean = sum(present) / len(present)
    half_width = None
    if len(present) > 1:
        variance = sum((value - mean) ** 2 for value in present) / (len(present) - 1)
        half_width = _t95(len(present) - 1) * math.sqrt(variance) / math.sqrt(len(present))
    return {
        "n": len(present),
        "repeat_values": [
            {"run_id": run_id, "value": value, "source_path": source}
            for run_id, value, source in pairs
        ],
        "mean": round(mean, 6),
        "min": min(present),
        "max": max(present),
        "ci95_half_width": round(half_width, 6) if half_width is not None else None,
        "source_run_ids": [run_id for run_id, _, _ in pairs],
        "source_paths": [source for _, _, source in pairs],
    }


def _metric_percentiles_many(
    values: list[float | None],
    run_ids: list[str],
    sources: list[str],
    missing_reason: str | None = None,
) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return {
        "p50": _metric_value_many(_percentile(present, 50), run_ids, sources, missing_reason),
        "p95": _metric_value_many(_percentile(present, 95), run_ids, sources, missing_reason),
        "p99": _metric_value_many(_percentile(present, 99), run_ids, sources, missing_reason),
    }


def _metric_value_many(
    value: float | None,
    run_ids: list[str],
    sources: list[str],
    missing_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "missing_reason": missing_reason if value is None else None,
        "source_run_ids": run_ids,
        "source_paths": sources,
    }


def _metric_percentiles(values: list[float | None], run_id: str, source: str) -> dict[str, Any]:
    present = [value for value in values if value is not None]
    return {
        "p50": _metric_value(_percentile(present, 50), run_id, source),
        "p95": _metric_value(_percentile(present, 95), run_id, source),
        "p99": _metric_value(_percentile(present, 99), run_id, source),
    }


def _metric_value(
    value: float | None,
    run_id: str,
    source: str,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "value": value,
        "missing_reason": missing_reason if value is None else None,
        "source_run_ids": [run_id],
        "source_paths": [source],
    }


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _t95(degrees_of_freedom: int) -> float:
    return {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(degrees_of_freedom, 1.96)


def _fmt_percentiles(metric: Mapping[str, Mapping[str, Any]]) -> str:
    return "/".join(_fmt_metric(metric[key]) for key in ("p50", "p95", "p99"))


def _fmt_metric(metric: Mapping[str, Any]) -> str:
    value = metric.get("value")
    if value is None:
        return "null"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _fmt_optional_number(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.3f}" if isinstance(value, int | float) else str(value)


def _controlled_key_rows(key: Mapping[str, Any]) -> list[str]:
    model = key.get("model") or {}
    hardware = key.get("hardware") or {}
    workload = key.get("workload") or {}
    weights = key.get("weights") or {}
    runtime = key.get("runtime") or {}
    sampling = key.get("sampling") or {}
    rows = [
        ("Model", f"{model.get('id')}@{model.get('revision')}"),
        ("Tokenizer", f"{model.get('tokenizer_id')}@{model.get('tokenizer_revision')}"),
        (
            "Weights",
            (
                f"format={weights.get('format')}, precision={weights.get('precision')}, "
                f"quantization={weights.get('quantization')}"
            ),
        ),
        (
            "Hardware",
            (
                f"{hardware.get('gpu_count')}x {hardware.get('gpu_model')}, "
                f"driver={hardware.get('driver_version')}, cuda={hardware.get('cuda_version')}"
            ),
        ),
        (
            "Runtime",
            (
                f"context={runtime.get('context_limit')}, "
                f"max_concurrent={runtime.get('max_concurrent_requests')}, "
                f"memory_fraction={runtime.get('memory_fraction')}, "
                f"tp={runtime.get('tensor_parallel_size')}, "
                f"pp={runtime.get('pipeline_parallel_size')}, "
                f"backend={runtime.get('backend')}"
            ),
        ),
        (
            "Workload",
            (
                f"{workload.get('workload_id')}, template={workload.get('prompt_template_id')}, "
                f"seed={workload.get('seed')}, cache={workload.get('cache_state')}, "
                f"rate={workload.get('request_rate_rps')}, "
                f"concurrency={workload.get('concurrency') or workload.get('max_in_flight')}, "
                f"context={workload.get('context_length')}, streaming={workload.get('streaming')}"
            ),
        ),
        (
            "Policies",
            (
                f"arrival={workload.get('arrival_model')}, warmup={workload.get('warmup_requests')}, "
                f"retry={workload.get('retry_policy')}, failure={workload.get('failure_policy')}"
            ),
        ),
        ("Sampling", json.dumps(sampling, sort_keys=True)),
    ]
    return [f"| {name} | {value} |" for name, value in rows]


def _project_path(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _path_exists(project_root: Path, path: str) -> bool:
    if not path or path == "[redacted]":
        return False
    return _project_path(project_root, path).exists()


def _source_paths(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        paths = [str(path) for path in value.get("source_paths", []) if path]
        for item in value.values():
            paths.extend(_source_paths(item))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            paths.extend(_source_paths(item))
        return paths
    return []


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
