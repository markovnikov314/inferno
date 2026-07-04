"""P5 redacted release package builder."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from inferno import artifact_contract as contract
from inferno import study


class ReleaseStudyRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(min_length=1)
    config: str = Field(min_length=1)
    workload_family: str = Field(min_length=1)


class ReleaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    release_id: str = Field(min_length=1)
    study_type: Literal["p5_research_core_release"]
    mode: Literal["redacted"] = "redacted"
    required_workload_families: list[str] = Field(default_factory=list)
    studies: list[ReleaseStudyRef] = Field(min_length=1)
    docs: list[str] = Field(default_factory=list)
    artifacts_dir: str = Field(min_length=1)


def package_release(*, mode: str, study_path: Path, project_root: Path) -> int:
    if mode != "redacted":
        raise ValueError("P5 release packaging supports MODE=redacted only")
    payload = json.loads((project_root / study_path).read_text(encoding="utf-8-sig"))
    if payload.get("study_type") == "p5_research_core_release":
        config = ReleaseConfig.model_validate(payload)
    else:
        config = _single_study_release_config(study_path, payload)

    errors = _validate_release_config(config)
    reports = []
    run_maps: dict[str, dict[str, str]] = {}
    report_inputs_by_study: dict[str, dict[str, Any]] = {}
    for item in config.studies:
        study_config = study.load_study_config(project_root / item.config)
        latest_path = project_root / study_config.artifacts_dir / "latest.json"
        if study_config.study_type != "strict_engine_comparison":
            errors.append(f"{item.study_id}: not a strict_engine_comparison study")
            continue
        if study_config.repeats < 3:
            errors.append(f"{item.study_id}: repeats must be at least 3")
        if not latest_path.exists():
            errors.append(
                f"{item.study_id}: missing study latest.json at {_display_path(project_root, latest_path)}"
            )
            continue

        latest = json.loads(latest_path.read_text(encoding="utf-8-sig"))
        run_dirs = [study._project_path(project_root, str(run["run_dir"])) for run in latest["runs"]]
        report_inputs, compare_errors = study.build_strict_compare_inputs(
            run_dirs,
            project_root,
            study_id=study_config.study_id,
            workload_family=item.workload_family,
            low_sample_completed_requests=study_config.low_sample_completed_requests,
        )
        errors.extend(f"{item.study_id}: {error}" for error in compare_errors)
        errors.extend(
            f"{item.study_id}: {error}"
            for error in study.validate_strict_report_inputs(report_inputs, project_root)
        )
        for engine in ("vllm", "sglang"):
            count = sum(1 for run in report_inputs["runs"] if run["engine"] == engine)
            if count < study_config.repeats:
                errors.append(f"{item.study_id}: {engine} has {count} repeats")
        report_inputs_by_study[item.study_id] = report_inputs
        reports.append(
            {
                "study_id": item.study_id,
                "workload_family": item.workload_family,
                "config": item.config,
                "run_ids": report_inputs["run_ids"],
            }
        )

    if errors:
        print(json.dumps({"ok": False, "errors": sorted(set(errors))}, indent=2, sort_keys=True))
        return 1

    output_dir = project_root / config.artifacts_dir / mode
    _prepare_output_dir(project_root, output_dir)
    for item in config.studies:
        report_inputs = report_inputs_by_study[item.study_id]
        for run in report_inputs["runs"]:
            if run["run_id"] not in run_maps:
                run_maps[run["run_id"]] = _write_redacted_run(
                    output_dir,
                    Path(run["artifacts"]["summary"]).parent,
                )
        redacted_inputs = _redacted_report_inputs(report_inputs, run_maps)
        report_dir = output_dir / "reports" / item.study_id
        _write_json(report_dir / "report_inputs.redacted.json", redacted_inputs)
        (report_dir / "report.md").write_text(
            study._render_strict_compare_report(redacted_inputs),
            encoding="utf-8",
        )

    copied_docs = _copy_docs(project_root, output_dir, config.docs)
    manifest = {
        "schema_version": 1,
        "release_id": config.release_id,
        "study_type": config.study_type,
        "mode": mode,
        "retention_class": "public_release",
        "generated_at": datetime.now(UTC).isoformat(),
        "required_workload_families": config.required_workload_families,
        "workload_families": [item.workload_family for item in config.studies],
        "studies": [
            entry
            | {
                "report": f"reports/{entry['study_id']}/report.md",
                "report_inputs": f"reports/{entry['study_id']}/report_inputs.redacted.json",
            }
            for entry in reports
        ],
        "docs": copied_docs,
        "files": _hashed_files(output_dir),
    }
    _write_json(output_dir / "manifest.json", manifest)
    validation_errors = validate_release_package(output_dir)
    if validation_errors:
        print(json.dumps({"ok": False, "errors": validation_errors}, indent=2, sort_keys=True))
        return 1
    print(output_dir)
    return 0


def validate_release_package(package_dir: Path) -> list[str]:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing release manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    if manifest.get("mode") != "redacted":
        errors.append("release package mode is not redacted")
    if manifest.get("retention_class") != "public_release":
        errors.append("release retention_class is not public_release")
    if len(manifest.get("required_workload_families") or []) >= 5 and len(
        set(manifest.get("workload_families") or [])
    ) < 5:
        errors.append("release package has fewer than five workload families")

    recorded = {item["path"]: item for item in manifest.get("files", [])}
    for relative, item in recorded.items():
        path = package_dir / relative
        if not path.exists():
            errors.append(f"missing package file: {relative}")
            continue
        if _sha256(path) != item.get("sha256"):
            errors.append(f"package hash mismatch: {relative}")

    for entry in manifest.get("studies", []):
        report_inputs_path = package_dir / entry["report_inputs"]
        if not report_inputs_path.exists():
            errors.append(f"{entry.get('study_id')}: missing report inputs")
            continue
        report_inputs = json.loads(report_inputs_path.read_text(encoding="utf-8-sig"))
        errors.extend(
            f"{entry.get('study_id')}: {error}"
            for error in study.validate_strict_report_inputs(report_inputs, package_dir)
        )

    errors.extend(_scan_for_private_material(package_dir))
    return sorted(set(errors))


def _single_study_release_config(study_path: Path, payload: Mapping[str, Any]) -> ReleaseConfig:
    study_id = str(payload["study_id"])
    family = study_id.removeprefix("p5_") or study_id
    return ReleaseConfig(
        schema_version=1,
        release_id=f"{study_id}_redacted",
        study_type="p5_research_core_release",
        required_workload_families=[family],
        studies=[ReleaseStudyRef(study_id=study_id, config=str(study_path), workload_family=family)],
        artifacts_dir=f"artifacts/releases/{study_id}",
    )


def _validate_release_config(config: ReleaseConfig) -> list[str]:
    families = {item.workload_family for item in config.studies}
    required = set(config.required_workload_families)
    errors = []
    missing = sorted(required - families)
    if missing:
        errors.append(f"missing required workload families: {', '.join(missing)}")
    if len(required) >= 5 and len(families) < 5:
        errors.append("at least five workload families are required")
    return errors


def _prepare_output_dir(project_root: Path, output_dir: Path) -> None:
    resolved = output_dir.resolve()
    allowed = (project_root / "artifacts" / "releases").resolve()
    if allowed not in (resolved, *resolved.parents):
        raise ValueError(f"release output must stay under {allowed}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _write_redacted_run(output_dir: Path, run_dir: Path) -> dict[str, str]:
    manifest = contract.read_manifest(run_dir)
    target = output_dir / "runs" / manifest.run_id
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "summary.json"
    validation_path = target / "validation.json"
    trace_path = target / "request_trace.redacted.jsonl"
    metadata_path = target / "run_metadata.json"
    shutil.copy2(contract.artifact_path(run_dir, manifest.artifacts.summary), summary_path)
    shutil.copy2(contract.artifact_path(run_dir, manifest.artifacts.validation), validation_path)
    _write_redacted_trace(run_dir, manifest, trace_path)
    telemetry = contract.artifact_path(run_dir, manifest.artifacts.telemetry_parquet or "")
    telemetry_rel = None
    if manifest.artifacts.telemetry_parquet and telemetry.exists():
        telemetry_target = target / "telemetry.parquet"
        shutil.copy2(telemetry, telemetry_target)
        telemetry_rel = _rel(output_dir, telemetry_target)
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "retention_class": "redacted_shareable",
            "engine": manifest.engine.model_dump(mode="json"),
            "model": manifest.model.model_dump(mode="json"),
            "hardware": manifest.hardware.model_dump(mode="json"),
            "workload": manifest.workload.model_dump(mode="json"),
            "source": manifest.source.model_dump(mode="json"),
            "validation_ok": True,
            "included_artifacts": {
                "summary": _rel(output_dir, summary_path),
                "validation": _rel(output_dir, validation_path),
                "request_trace": _rel(output_dir, trace_path),
                "telemetry": telemetry_rel,
            },
            "source_artifact_hashes": _source_artifact_hashes(run_dir, manifest),
        },
    )
    artifacts = {
        "summary": _rel(output_dir, summary_path),
        "request_trace": _rel(output_dir, trace_path),
        "validation": _rel(output_dir, validation_path),
        "raw_sse": "redacted:hash_in_run_metadata",
    }
    if telemetry_rel:
        artifacts["telemetry"] = telemetry_rel
    return artifacts


def _write_redacted_trace(
    run_dir: Path,
    manifest: contract.RunManifest,
    target: Path,
) -> None:
    rows = []
    for event in contract.read_trace(run_dir, manifest):
        row = event.model_dump(mode="json", exclude_none=True)
        row.pop("raw_sse", None)
        row.pop("stderr", None)
        if "request" in row:
            row["request"] = {"redacted": True, "request_sha256": row.get("request_sha256")}
        rows.append(row)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _source_artifact_hashes(run_dir: Path, manifest: contract.RunManifest) -> list[dict[str, str]]:
    hashes = []
    for relative in contract.checksum_paths(manifest):
        path = contract.artifact_path(run_dir, relative)
        if path.exists():
            hashes.append({"path": relative, "sha256": _sha256(path)})
    return hashes


def _redacted_report_inputs(
    report_inputs: Mapping[str, Any],
    run_maps: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(report_inputs))
    payload["retention_class"] = "redacted_shareable"
    replacements = {}
    for run in payload["runs"]:
        run_id = run["run_id"]
        original = run["artifacts"]
        redacted = dict(run_maps[run_id])
        replacements.update(
            {
                original["summary"]: redacted["summary"],
                original["request_trace"]: redacted["request_trace"],
                original["validation"]: redacted["validation"],
            }
        )
        if original.get("telemetry") and redacted.get("telemetry"):
            replacements[original["telemetry"]] = redacted["telemetry"]
        run["artifacts"] = redacted
    _replace_source_paths(payload, replacements)
    return payload


def _replace_source_paths(value: Any, replacements: Mapping[str, str]) -> None:
    if isinstance(value, dict):
        if "source_paths" in value:
            value["source_paths"] = [replacements.get(path, path) for path in value["source_paths"]]
        if "source_path" in value:
            value["source_path"] = replacements.get(value["source_path"], value["source_path"])
        for item in value.values():
            _replace_source_paths(item, replacements)
    elif isinstance(value, list):
        for item in value:
            _replace_source_paths(item, replacements)


def _copy_docs(project_root: Path, output_dir: Path, docs: list[str]) -> list[str]:
    copied = []
    for doc in docs:
        source = project_root / doc
        if not source.exists():
            raise ValueError(f"missing release doc: {doc}")
        target = output_dir / doc
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(doc.replace("\\", "/"))
    return copied


def _hashed_files(output_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = _rel(output_dir, path)
        if relative == "manifest.json":
            continue
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return files


def _scan_for_private_material(package_dir: Path) -> list[str]:
    private = re.compile(r"ssh\.lightning\.ai|C:\\\\Users\\\\|/home/[^/\s]+")
    errors = []
    for path in package_dir.rglob("*"):
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".sha256"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if private.search(text):
            errors.append(f"private material pattern found in {_rel(package_dir, path)}")
    return errors


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _display_path(root: Path, path: Path) -> str:
    try:
        return _rel(root, path)
    except ValueError:
        return path.name
