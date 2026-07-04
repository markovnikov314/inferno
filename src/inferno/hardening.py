"""Repository audit checks."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import html
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from inferno import artifact_contract as contract
from inferno import preflight
from inferno import release
from inferno import study


class HardeningModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportSnapshot(HardeningModel):
    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    must_contain: list[str] = Field(default_factory=list)


class RegressionConfig(HardeningModel):
    research_release_package: str
    deployment_profile_report_inputs: str
    capacity_planner_latest: str
    router_replay_latest: str
    strict_refusal_runs: list[str] = Field(default_factory=list)


class HardeningConfig(HardeningModel):
    schema_version: Literal[1]
    hardening_id: str = Field(min_length=1)
    required_files: list[str] = Field(min_length=1)
    schema_snapshot_dir: str = Field(min_length=1)
    report_snapshots: list[ReportSnapshot] = Field(default_factory=list)
    engine_configs: list[str] = Field(default_factory=list)
    real_artifact_runs: list[str] = Field(default_factory=list)
    secret_scan_paths: list[str] = Field(default_factory=list)
    allowed_runtime_dependencies: list[str] = Field(default_factory=list)
    allowed_dev_dependencies: list[str] = Field(default_factory=list)
    regression: RegressionConfig
    artifacts_dir: str = Field(min_length=1)


SECRET_PATTERNS = {
    "ssh_url": re.compile(r"\bssh://\S+"),
    "ssh_at_host": re.compile(r"\b[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}\b"),
    "private_ipv4": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3})\b"
    ),
    "absolute_user_path": re.compile(r"(?:[A-Za-z]:\\Users\\|/home/)[^\\/\s]+"),
    "token_like": re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|(?:ghp|hf|xox[baprs])_[A-Za-z0-9_=-]{16,})\b"),
}


def load_hardening_config(path: Path) -> HardeningConfig:
    return HardeningConfig.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def run_hardening_check(
    *,
    config_path: Path,
    project_root: Path,
    include_gpu_smoke: bool = False,
    env: Mapping[str, str] | None = None,
) -> int:
    config = load_hardening_config(_project_path(project_root, str(config_path)))
    output_dir = _project_path(project_root, config.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_clean_clone(config, project_root),
        check_schema_snapshots(config, project_root),
        check_report_snapshots(config, project_root),
        check_model_and_image_provenance(config, project_root),
        check_secret_scan(config, project_root),
        check_dependency_audit(config, project_root),
        check_regressions(config, project_root),
    ]
    preview = write_browser_preview(config, project_root, output_dir)
    checks.append(preview)

    gpu_smoke = (
        check_gpu_smoke(project_root, env or {}, output_dir)
        if include_gpu_smoke
        else _check("gpu_gated_real_smoke", "SKIPPED", "pass --include-gpu-smoke to require GPU preflight")
    )
    checks.append(gpu_smoke)

    result = {
        "schema_version": 1,
        "run_family": "repository_audit",
        "hardening_id": config.hardening_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": _overall_status(checks),
        "checks": checks,
        "artifact_paths": {
            "latest": f"{config.artifacts_dir}/latest.json",
            "report": f"{config.artifacts_dir}/report.md",
            "browser_preview": f"{config.artifacts_dir}/report_preview.html",
            "gpu_smoke": f"{config.artifacts_dir}/gpu_smoke.json",
        },
        "limitations": [
            "Repository audit only; no live router, dashboard, adapter, or benchmark runner.",
            "Dependency audit is offline and lockfile-based; it is not a vulnerability database scan.",
            "GPU smoke is a gated host/container preflight, not a new engine readiness claim.",
            "Local private run and compare inputs remain outside public secret-scan scope.",
        ],
    }
    _write_json(output_dir / "latest.json", result)
    (output_dir / "report.md").write_text(render_hardening_report(result), encoding="utf-8")
    print(output_dir / "report.md")
    return 0 if result["status"] != "FAIL" else 1


def check_clean_clone(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    missing = [path for path in config.required_files if not _project_path(project_root, path).exists()]
    lock = _project_path(project_root, "uv.lock")
    pyproject = _project_path(project_root, "pyproject.toml")
    details = {
        "required_file_count": len(config.required_files),
        "missing": missing,
        "lockfile_sha256": _sha256(lock) if lock.exists() else None,
        "pyproject_sha256": _sha256(pyproject) if pyproject.exists() else None,
    }
    return _check(
        "clean_clone_lockfile_setup",
        "PASS" if not missing else "FAIL",
        "required clean-clone files are present" if not missing else "required files are missing",
        details,
    )


def check_schema_snapshots(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    snapshot_dir = _project_path(project_root, config.schema_snapshot_dir)
    mismatches = []
    for name, expected in contract.schema_snapshots().items():
        path = snapshot_dir / name
        actual = path.read_text(encoding="utf-8-sig") if path.exists() else None
        if actual != expected:
            mismatches.append(name)
    return _check(
        "schema_migration_and_snapshot",
        "PASS" if not mismatches else "FAIL",
        "artifact schema snapshots match; migration_required=false"
        if not mismatches
        else "artifact schema snapshots drifted",
        {"migration_required": False, "mismatched_snapshots": mismatches},
    )


def check_report_snapshots(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    rows = []
    failures = []
    for item in config.report_snapshots:
        path = _project_path(project_root, item.path)
        text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        missing = [marker for marker in item.must_contain if marker not in text]
        if not path.exists() or missing:
            failures.append(item.id)
        rows.append(
            {
                "id": item.id,
                "path": _display_path(project_root, path),
                "sha256": _sha256(path) if path.exists() else None,
                "missing_markers": missing,
            }
        )
    return _check(
        "report_snapshot_tests",
        "PASS" if not failures else "FAIL",
        "configured report snapshots contain required markers"
        if not failures
        else "one or more report snapshots changed",
        {"snapshots": rows},
    )


def check_model_and_image_provenance(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    engine_rows = []
    run_rows = []
    failures = []
    for ref in config.engine_configs:
        path = _project_path(project_root, ref)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        model = payload.get("model") or {}
        engine = str(payload.get("engine") or path.stem)
        errors = []
        if "@sha256:" not in str(payload.get("image", "")):
            errors.append("image is not pinned by digest")
        for key in ("id", "revision", "tokenizer_id", "tokenizer_revision"):
            if not model.get(key):
                errors.append(f"missing model.{key}")
        if engine == "llamacpp":
            for key in ("gguf_repo", "gguf_filename", "gguf_revision", "quantization"):
                if not model.get(key):
                    errors.append(f"missing model.{key}")
        if engine in {"tensorrtllm", "tensorrtllm_t4"}:
            for key in ("weight_format", "weight_precision"):
                if not model.get(key):
                    errors.append(f"missing model.{key}")
            if "--backend" not in payload.get("launch_args", []):
                errors.append("missing TensorRT-LLM --backend launch arg")
        if errors:
            failures.append(ref)
        engine_rows.append(
            {
                "path": _display_path(project_root, path),
                "engine": engine,
                "image_pinned": "@sha256:" in str(payload.get("image", "")),
                "model_id": model.get("id"),
                "revision": model.get("revision"),
                "errors": errors,
            }
        )
    for ref in config.real_artifact_runs:
        run_dir = _project_path(project_root, ref)
        validation = contract.validate_run(run_dir, write=False)
        manifest = contract.read_manifest(run_dir)
        if not validation.ok:
            failures.append(ref)
        run_rows.append(
            {
                "run_dir": _display_path(project_root, run_dir),
                "run_id": manifest.run_id,
                "engine": manifest.engine.name,
                "validation_ok": validation.ok,
                "model_id": manifest.model.id,
                "revision": manifest.model.revision,
                "weight_format": manifest.model.weight_format
                or (manifest.comparison_key.get("weights") or {}).get("format"),
                "quantization": manifest.model.quantization,
            }
        )
    return _check(
        "image_and_model_provenance",
        "PASS" if not failures else "FAIL",
        "engine configs are pinned and spot-check real artifacts validate"
        if not failures
        else "provenance check failed",
        {"engine_configs": engine_rows, "real_artifact_spot_checks": run_rows},
    )


def check_secret_scan(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    findings = []
    scanned_files = 0
    for ref in config.secret_scan_paths:
        path = _project_path(project_root, ref)
        candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for candidate in candidates:
            if _skip_scan_file(candidate):
                continue
            scanned_files += 1
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            for name, pattern in SECRET_PATTERNS.items():
                matches = [
                    match
                    for match in pattern.finditer(text)
                    if not _scanner_rule_line(text, match.start())
                ]
                if matches:
                    findings.append(
                        {
                            "path": _display_path(project_root, candidate),
                            "pattern": name,
                            "match_count": len(matches),
                        }
                    )
    return _check(
        "secret_scan",
        "PASS" if not findings else "FAIL",
        "no private material patterns found in configured public/durable paths"
        if not findings
        else "private material pattern found",
        {
            "scanned_files": scanned_files,
            "findings": findings,
            "excluded_scope": [
                "artifacts/runs raw streams and engine logs",
                "local-private compare inputs",
                ".inferno/evidence/commands historical scratch logs",
            ],
        },
    )


def check_dependency_audit(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    pyproject = tomllib.loads(_project_path(project_root, "pyproject.toml").read_text(encoding="utf-8"))
    lock_text = _project_path(project_root, "uv.lock").read_text(encoding="utf-8")
    runtime = [_dependency_name(dep) for dep in pyproject.get("project", {}).get("dependencies", [])]
    dev = [
        _dependency_name(dep)
        for dep in pyproject.get("dependency-groups", {}).get("dev", [])
    ]
    runtime_extra = sorted(set(runtime) - set(config.allowed_runtime_dependencies))
    dev_extra = sorted(set(dev) - set(config.allowed_dev_dependencies))
    missing_from_lock = sorted(
        name for name in set(runtime + dev) if not re.search(rf'name = "{re.escape(name)}"', lock_text)
    )
    failures = runtime_extra or dev_extra or missing_from_lock
    return _check(
        "dependency_audit",
        "PASS" if not failures else "FAIL",
        "pyproject dependencies are allowed and present in uv.lock"
        if not failures
        else "dependency audit failed",
        {
            "runtime_dependencies": runtime,
            "dev_dependencies": dev,
            "unexpected_runtime_dependencies": runtime_extra,
            "unexpected_dev_dependencies": dev_extra,
            "missing_from_lock": missing_from_lock,
            "audit_mode": "offline_lockfile_allowlist",
        },
    )


def check_regressions(config: HardeningConfig, project_root: Path) -> dict[str, Any]:
    errors = []
    release_errors = release.validate_release_package(
        _project_path(project_root, config.regression.research_release_package)
    )
    errors.extend(f"research_release: {error}" for error in release_errors)

    deployment_inputs = json.loads(
        _project_path(project_root, config.regression.deployment_profile_report_inputs).read_text(
            encoding="utf-8-sig"
        )
    )
    deployment_errors = study.validate_deployment_profile_report_inputs(
        deployment_inputs,
        project_root,
    )
    errors.extend(f"deployment_profile: {error}" for error in deployment_errors)

    planner_latest = json.loads(
        _project_path(project_root, config.regression.capacity_planner_latest).read_text(
            encoding="utf-8-sig"
        )
    )
    router_latest = json.loads(
        _project_path(project_root, config.regression.router_replay_latest).read_text(
            encoding="utf-8-sig"
        )
    )
    if planner_latest.get("run_family") != "capacity_planning" or planner_latest.get("compatible_evidence_count") != 30:
        errors.append("capacity_planner: unexpected run_family or evidence count")
    if (router_latest.get("baseline_comparison") or {}).get("status") != "NEGATIVE_RESULT":
        errors.append("router_replay: negative result is not preserved")
    if (router_latest.get("leakage_check") or {}).get("status") != "PASS":
        errors.append("router_replay: leakage check is not PASS")

    if len(config.regression.strict_refusal_runs) >= 2:
        manifests = [
            contract.read_manifest(_project_path(project_root, path))
            for path in config.regression.strict_refusal_runs
        ]
        refusal_errors = contract.strict_comparison_errors(manifests)
        if not any("strict comparison rejects llamacpp" in error for error in refusal_errors):
            errors.append("strict_refusal: llamacpp was not rejected")

    return _check(
        "research_regression_baselines",
        "PASS" if not errors else "FAIL",
        "research, deployment-profile, capacity-planner, and router-replay baselines hold"
        if not errors
        else "one or more regression baselines failed",
        {
            "errors": errors,
            "capacity_compatible_evidence_count": planner_latest.get("compatible_evidence_count"),
            "router_replay_status": (router_latest.get("baseline_comparison") or {}).get("status"),
        },
    )


def write_browser_preview(
    config: HardeningConfig,
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sections = []
    failures = []
    for snapshot in config.report_snapshots:
        path = _project_path(project_root, snapshot.path)
        if not path.exists():
            failures.append(snapshot.id)
            continue
        text = path.read_text(encoding="utf-8-sig")
        sections.append(
            "<section>"
            f"<h2>{html.escape(snapshot.id)}</h2>"
            f"<p>{html.escape(_display_path(project_root, path))}</p>"
            f"<pre>{html.escape(text[:6000])}</pre>"
            "</section>"
        )
    page = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Project Inferno Repository Audit Preview</title>"
        "<style>body{font-family:Arial,sans-serif;margin:2rem;line-height:1.4}"
        "pre{white-space:pre-wrap;border:1px solid #ccc;padding:1rem}</style>"
        "</head><body><h1>Project Inferno Repository Audit Preview</h1>"
        + "".join(sections)
        + "</body></html>\n"
    )
    target = output_dir / "report_preview.html"
    target.write_text(page, encoding="utf-8")
    return _check(
        "browser_rendering_check",
        "PASS" if not failures else "FAIL",
        "static HTML preview was generated for report rendering checks"
        if not failures
        else "missing report for browser preview",
        {
            "preview_path": _display_path(project_root, target),
            "source_report_count": len(sections),
            "missing_reports": failures,
            "method": "stdlib escaped HTML preview; no browser automation required for markdown-only reports",
        },
    )


def check_gpu_smoke(
    project_root: Path,
    env: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    result = preflight.collect_preflight(include_gpu=True, env=env, project_root=project_root)
    _write_json(output_dir / "gpu_smoke.json", result)
    return _check(
        "gpu_gated_real_smoke",
        "PASS" if result.get("status") == "ok" else "FAIL",
        "remote GPU/container preflight passed using INFERNO_GPU_SSH"
        if result.get("status") == "ok"
        else "remote GPU/container preflight failed or was skipped",
        {
            "preflight_status": result.get("status"),
            "artifact": _display_path(project_root, output_dir / "gpu_smoke.json"),
            "check_names": [check.get("name") for check in result.get("checks", [])],
        },
    )


def render_hardening_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Repository Audit Evidence",
        "",
        "REPOSITORY AUDIT ONLY - no live router, dashboard, new adapter, or benchmark runner is introduced.",
        "",
        f"Hardening ID: `{result['hardening_id']}`",
        f"Status: `{result['status']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Message |",
        "|---|---|---|",
    ]
    for check in result["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['message']} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in result["artifact_paths"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def _check(
    name: str,
    status: Literal["PASS", "WARN", "FAIL", "SKIPPED"],
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "details": dict(details or {})}


def _overall_status(checks: list[Mapping[str, Any]]) -> str:
    statuses = {check.get("status") for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses - {"PASS"}:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\[]", spec, maxsplit=1)[0].strip().lower()


def _skip_scan_file(path: Path) -> bool:
    parts = set(path.parts)
    if parts & {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"}:
        return True
    return path.suffix.lower() in {".pyc", ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}


def _scanner_rule_line(text: str, offset: int) -> bool:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    line = text[start:] if end == -1 else text[start:end]
    return "re.compile" in line or "SECRET_PATTERNS" in line


def _project_path(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
