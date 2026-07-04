"""Trace-grounded capacity planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inferno import artifact_contract as contract

PlannerStatus = Literal["IN_RANGE", "INTERPOLATED", "EXTRAPOLATED", "UNAVAILABLE"]

COMPARISON_TYPES = {
    "strict": "strict_engine_comparison",
    "engine_configuration": "engine_configuration_comparison",
    "deployment_profile": "deployment_profile_comparison",
}


class PlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenDistribution(PlannerModel):
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)


class TrafficSpec(PlannerModel):
    request_rate_rps: float = Field(gt=0)
    arrival_model: str = Field(min_length=1)
    prompt_tokens: TokenDistribution
    output_tokens: TokenDistribution
    shared_prefix_pct: float = Field(ge=0, le=100)
    burst_factor: float = Field(ge=0)


class SloSpec(PlannerModel):
    percentile: Literal["p50", "p95", "p99"]
    ttft_ms: float = Field(gt=0)
    e2e_ms: float = Field(gt=0)


class CandidateScope(PlannerModel):
    environment_fingerprint: str = Field(min_length=1)
    comparison_type: Literal["strict", "engine_configuration", "deployment_profile"]


class PricingSpec(PlannerModel):
    source: Literal["user_provided"]
    checked_at: str = Field(min_length=1)
    hourly_usd_by_candidate: dict[str, float] = Field(default_factory=dict)

    @field_validator("checked_at")
    @classmethod
    def checked_at_is_iso8601(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class CapacityPlannerConfig(PlannerModel):
    schema_version: Literal[1]
    plan_id: str = Field(min_length=1)
    traffic: TrafficSpec
    slo: SloSpec
    candidate_scope: CandidateScope
    pricing: PricingSpec
    evidence_studies: list[str] = Field(min_length=1)
    artifacts_dir: str = Field(min_length=1)


@dataclass(frozen=True)
class EvidencePoint:
    candidate_id: str
    run_id: str
    run_dir: Path
    study_id: str
    engine: str
    comparison_type: str
    environment_fingerprint: str
    validation_ok: bool
    features: dict[str, Any]
    metrics: dict[str, float | None]
    profile: dict[str, Any]
    source_paths: dict[str, str]


def load_plan_config(path: Path) -> CapacityPlannerConfig:
    return CapacityPlannerConfig.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def run_capacity_plan(*, config_path: Path, project_root: Path) -> int:
    config = load_plan_config(_project_path(project_root, str(config_path)))
    output_dir = _project_path(project_root, config.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence, rejected = _load_evidence(config, project_root)
    holdout = evaluate_holdout(evidence)
    result = build_capacity_plan(config, evidence, rejected, holdout, project_root)

    _write_json(output_dir / "latest.json", result)
    _write_json(output_dir / "holdout_eval.json", holdout)
    (output_dir / "report.md").write_text(render_capacity_report(result), encoding="utf-8")
    print(output_dir / "report.md")
    return 0 if evidence else 1


def build_capacity_plan(
    config: CapacityPlannerConfig,
    evidence: list[EvidencePoint],
    rejected_evidence: list[dict[str, Any]],
    holdout: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    candidates = []
    for candidate_id in sorted({point.candidate_id for point in evidence}):
        points = [point for point in evidence if point.candidate_id == candidate_id]
        candidates.append(_candidate_result(config, candidate_id, points, holdout, project_root))

    return {
        "schema_version": 1,
        "run_family": "capacity_planning",
        "plan_id": config.plan_id,
        "planner": "nearest_neighbor_bounded_v1",
        "comparison_type": config.candidate_scope.comparison_type,
        "environment_fingerprint": config.candidate_scope.environment_fingerprint,
        "traffic": config.traffic.model_dump(mode="json"),
        "slo": config.slo.model_dump(mode="json"),
        "pricing": {
            "source": config.pricing.source,
            "checked_at": config.pricing.checked_at,
            "cost_status": "AVAILABLE"
            if config.pricing.hourly_usd_by_candidate
            else "UNAVAILABLE",
            "unavailable_reason": None
            if config.pricing.hourly_usd_by_candidate
            else "cost unavailable without user-provided dated candidate price metadata",
        },
        "evidence_studies": config.evidence_studies,
        "compatible_evidence_count": len(evidence),
        "rejected_evidence": rejected_evidence,
        "candidates": candidates,
        "holdout_evaluation": holdout,
        "limitations": [
            "Planner results are bounded by validated stored artifacts only.",
            "EXTRAPOLATED is not measured capacity and is not a hard recommendation.",
            "Prompt token counts are unavailable unless stored by source artifacts.",
            "Cost is unavailable without dated user-provided candidate price metadata.",
        ],
    }


def evaluate_holdout(evidence: list[EvidencePoint]) -> dict[str, Any]:
    rows = []
    for candidate_id in sorted({point.candidate_id for point in evidence}):
        points = [point for point in evidence if point.candidate_id == candidate_id]
        if len(points) < 2:
            continue
        for point in points:
            train = [item for item in points if item.run_id != point.run_id]
            nearest = min(train, key=lambda item: _point_distance(point, item))
            row = {
                "candidate_id": candidate_id,
                "held_out_run_id": point.run_id,
                "nearest_run_id": nearest.run_id,
                "nearest_distance": round(_point_distance(point, nearest), 6),
                "ttft_ms": _holdout_metric(point, nearest, train, "ttft_ms"),
                "e2e_ms": _holdout_metric(point, nearest, train, "e2e_ms"),
            }
            rows.append(row)

    return {
        "schema_version": 1,
        "baseline": "same-candidate median of remaining validated runs",
        "method": "leave_one_run_out_nearest_neighbor",
        "sample_count": len(rows),
        "rows": rows,
        "aggregate": _holdout_aggregate(rows),
    }


def render_capacity_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Trace-Grounded Capacity Planner",
        "",
        "TRACE-GROUNDED PLANNER - recommendations are bounded by validated stored artifacts. "
        "EXTRAPOLATED is not measured capacity.",
        "",
        f"Plan ID: `{result['plan_id']}`",
        f"Comparison type: `{result['comparison_type']}`",
        f"Environment fingerprint: `{result['environment_fingerprint']}`",
        "",
        "## Target Traffic",
        "",
        (
            f"- Request rate: {result['traffic']['request_rate_rps']} rps; "
            f"arrival: {result['traffic']['arrival_model']}; "
            f"burst factor: {result['traffic']['burst_factor']}"
        ),
        (
            f"- Prompt tokens p50/p95: {result['traffic']['prompt_tokens']['p50']}/"
            f"{result['traffic']['prompt_tokens']['p95']}; output tokens p50/p95: "
            f"{result['traffic']['output_tokens']['p50']}/"
            f"{result['traffic']['output_tokens']['p95']}"
        ),
        f"- SLO {result['slo']['percentile']}: TTFT <= {result['slo']['ttft_ms']} ms, "
        f"E2E <= {result['slo']['e2e_ms']} ms",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Status | Confidence | Nearest run | Distance | SLO basis | Cost |",
        "|---|---|---|---|---:|---|---|",
    ]
    for item in result["candidates"]:
        nearest = item.get("nearest_evidence") or {}
        lines.append(
            "| {candidate} | {status} | {confidence} | `{run}` | {distance} | {slo} | {cost} |".format(
                candidate=item["candidate_id"],
                status=item["status"],
                confidence=item["confidence"],
                run=nearest.get("run_id", "-"),
                distance=_fmt(item.get("distance", {}).get("score")),
                slo=item["slo_assessment"]["status"],
                cost=item["cost"]["status"],
            )
        )

    holdout = result["holdout_evaluation"]
    aggregate = holdout.get("aggregate") or {}
    lines.extend(
        [
            "",
            "## Holdout Evaluation",
            "",
            f"Baseline: {holdout['baseline']}.",
            (
                f"Nearest-neighbor TTFT MAE: {_fmt(aggregate.get('nearest_neighbor_ttft_mae_ms'))} ms; "
                f"baseline TTFT MAE: {_fmt(aggregate.get('baseline_ttft_mae_ms'))} ms; "
                f"improvement: {_fmt(aggregate.get('ttft_improvement_over_baseline_pct'))}%."
            ),
            (
                f"Nearest-neighbor E2E MAE: {_fmt(aggregate.get('nearest_neighbor_e2e_mae_ms'))} ms; "
                f"baseline E2E MAE: {_fmt(aggregate.get('baseline_e2e_mae_ms'))} ms; "
                f"improvement: {_fmt(aggregate.get('e2e_improvement_over_baseline_pct'))}%."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def _load_evidence(
    config: CapacityPlannerConfig,
    project_root: Path,
) -> tuple[list[EvidencePoint], list[dict[str, Any]]]:
    points: list[EvidencePoint] = []
    rejected: list[dict[str, Any]] = []
    expected_study_type = COMPARISON_TYPES[config.candidate_scope.comparison_type]
    for study_ref in config.evidence_studies:
        latest_path = _project_path(project_root, study_ref)
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            rejected.append({"path": study_ref, "reason": f"unreadable study: {exc}"})
            continue

        if latest.get("study_type") != expected_study_type:
            rejected.append(
                {
                    "path": study_ref,
                    "study_type": latest.get("study_type"),
                    "reason": f"comparison type is not {expected_study_type}",
                }
            )
            continue
        if latest.get("environment_block_id") != config.candidate_scope.environment_fingerprint:
            rejected.append(
                {
                    "path": study_ref,
                    "environment_fingerprint": latest.get("environment_block_id"),
                    "reason": "environment fingerprint mismatch",
                }
            )
            continue

        for record in latest.get("runs", []):
            run_dir = _project_path(project_root, str(record.get("run_dir", "")))
            try:
                manifest = contract.read_manifest(run_dir)
                validation = contract.validate_run(run_dir, write=False)
                summary = contract.RunSummary.model_validate_json(
                    contract.artifact_path(run_dir, manifest.artifacts.summary).read_text(
                        encoding="utf-8-sig"
                    )
                )
            except Exception as exc:
                rejected.append(
                    {
                        "path": _display_path(project_root, run_dir),
                        "reason": f"invalid run artifact: {exc}",
                    }
                )
                continue
            if not validation.ok:
                rejected.append(
                    {
                        "path": _display_path(project_root, run_dir),
                        "run_id": manifest.run_id,
                        "reason": "validation failed",
                        "errors": validation.errors,
                    }
                )
                continue
            points.append(
                _evidence_point(
                    project_root=project_root,
                    latest=latest,
                    latest_path=latest_path,
                    run_dir=run_dir,
                    manifest=manifest,
                    summary=summary,
                    validation_ok=validation.ok,
                    comparison_type=config.candidate_scope.comparison_type,
                )
            )
    return points, rejected


def _evidence_point(
    *,
    project_root: Path,
    latest: Mapping[str, Any],
    latest_path: Path,
    run_dir: Path,
    manifest: contract.RunManifest,
    summary: contract.RunSummary,
    validation_ok: bool,
    comparison_type: str,
) -> EvidencePoint:
    candidate_id = _candidate_id(manifest, comparison_type)
    artifacts = manifest.artifacts
    return EvidencePoint(
        candidate_id=candidate_id,
        run_id=manifest.run_id,
        run_dir=run_dir,
        study_id=str(latest.get("study_id") or manifest.study_id),
        engine=manifest.engine.name,
        comparison_type=comparison_type,
        environment_fingerprint=str(latest.get("environment_block_id")),
        validation_ok=validation_ok,
        features={
            "arrival_model": manifest.workload.arrival_model,
            "request_rate_rps": manifest.workload.request_rate_rps or summary.request_throughput_rps,
            "prompt_chars": manifest.workload.prompt_chars,
            "prompt_tokens_p50": _target_value(manifest.workload.prompt_target, "p50"),
            "prompt_tokens_p95": _target_value(manifest.workload.prompt_target, "p95"),
            "output_tokens_p50": _target_value(
                manifest.workload.output_target, "p50", summary.completion_tokens
            ),
            "output_tokens_p95": _target_value(
                manifest.workload.output_target, "p95", summary.completion_tokens
            ),
            "shared_prefix_pct": _shared_prefix_pct(
                manifest.workload.cache_policy or manifest.workload.cache_state
            ),
            "burst_factor": manifest.workload.runtime_knobs.get("burst_factor")
            or (1.0 if manifest.workload.arrival_model != "burst" else None),
            "max_tokens": manifest.workload.sampling.get("max_tokens"),
            "context_length": manifest.workload.context_length,
            "streaming": manifest.workload.streaming,
            "max_in_flight": manifest.workload.max_in_flight or manifest.workload.concurrency,
            "success_count": summary.success_count,
        },
        metrics={
            "ttft_ms": summary.ttft_ms,
            "e2e_ms": summary.e2e_ms,
            "request_throughput_rps": summary.request_throughput_rps,
            "error_rate": summary.error_rate,
        },
        profile={
            "engine": manifest.engine.name,
            "model": f"{manifest.model.id}@{manifest.model.revision}",
            "weight_format": (manifest.comparison_key.get("weights") or {}).get("format"),
            "precision": (manifest.comparison_key.get("weights") or {}).get("precision"),
            "quantization": (manifest.comparison_key.get("weights") or {}).get("quantization"),
            "runtime": manifest.comparison_key.get("runtime"),
        },
        source_paths={
            "study": _display_path(project_root, latest_path),
            "run_dir": _display_path(project_root, run_dir),
            "manifest": _display_path(project_root, run_dir / artifacts.manifest),
            "summary": _display_path(project_root, run_dir / artifacts.summary),
            "validation": _display_path(project_root, run_dir / artifacts.validation),
        },
    )


def _candidate_result(
    config: CapacityPlannerConfig,
    candidate_id: str,
    points: list[EvidencePoint],
    holdout: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    distances = [(_target_distance(config.traffic, point), point) for point in points]
    nearest_distance, nearest = min(distances, key=lambda item: item[0]["score"] or math.inf)
    status = _status(config.traffic, points, nearest_distance)
    return {
        "candidate_id": candidate_id,
        "status": status,
        "confidence": _confidence(status, points, nearest_distance),
        "nearest_evidence": _point_payload(nearest, project_root),
        "distance": nearest_distance,
        "observed_range": _observed_range(points),
        "slo_assessment": _slo_assessment(config, nearest, status),
        "holdout_error": _candidate_holdout(candidate_id, holdout),
        "cost": _cost(config, candidate_id, nearest),
        "recommendation": _recommendation(status),
    }


def _target_distance(traffic: TrafficSpec, point: EvidencePoint) -> dict[str, Any]:
    components = []
    missing = []

    def add(name: str, target: float, observed: float | None) -> None:
        if observed is None:
            missing.append(name)
            return
        denominator = max(abs(target), abs(observed), 1.0)
        components.append({"name": name, "distance": abs(target - observed) / denominator})

    add("request_rate_rps", traffic.request_rate_rps, point.features["request_rate_rps"])
    add("output_tokens.p50", traffic.output_tokens.p50, point.features["output_tokens_p50"])
    add("output_tokens.p95", traffic.output_tokens.p95, point.features["output_tokens_p95"])
    add("prompt_tokens.p50", traffic.prompt_tokens.p50, point.features["prompt_tokens_p50"])
    add("prompt_tokens.p95", traffic.prompt_tokens.p95, point.features["prompt_tokens_p95"])
    add("shared_prefix_pct", traffic.shared_prefix_pct, point.features["shared_prefix_pct"])
    add("burst_factor", traffic.burst_factor, point.features["burst_factor"])
    components.append(
        {
            "name": "arrival_model",
            "distance": 0.0 if traffic.arrival_model == point.features["arrival_model"] else 1.0,
        }
    )
    score = sum(item["distance"] for item in components) / len(components) if components else None
    return {
        "score": round(score, 6) if score is not None else None,
        "components": [
            {"name": item["name"], "distance": round(item["distance"], 6)}
            for item in components
        ],
        "missing_features": missing,
    }


def _status(
    traffic: TrafficSpec,
    points: list[EvidencePoint],
    nearest_distance: Mapping[str, Any],
) -> PlannerStatus:
    if not points:
        return "UNAVAILABLE"
    rates = _present(point.features["request_rate_rps"] for point in points)
    outputs = _present(point.features["output_tokens_p95"] for point in points)
    arrivals = {point.features["arrival_model"] for point in points}
    if traffic.arrival_model not in arrivals:
        return "EXTRAPOLATED"
    if rates and traffic.request_rate_rps > max(rates):
        return "EXTRAPOLATED"
    if outputs and traffic.output_tokens.p95 > max(outputs):
        return "EXTRAPOLATED"
    if nearest_distance.get("missing_features"):
        return "UNAVAILABLE"
    if rates and outputs and traffic.request_rate_rps >= min(rates) and traffic.output_tokens.p95 >= min(outputs):
        return "IN_RANGE"
    return "INTERPOLATED"


def _confidence(
    status: PlannerStatus,
    points: list[EvidencePoint],
    nearest_distance: Mapping[str, Any],
) -> Literal["low", "medium"]:
    if status != "IN_RANGE" or nearest_distance.get("missing_features"):
        return "low"
    if min(point.features["success_count"] for point in points) < 3:
        return "low"
    return "medium"


def _observed_range(points: list[EvidencePoint]) -> dict[str, Any]:
    return {
        "arrival_models": sorted({point.features["arrival_model"] for point in points}),
        "request_rate_rps": _range(point.features["request_rate_rps"] for point in points),
        "prompt_chars": _range(point.features["prompt_chars"] for point in points),
        "prompt_tokens": _feature_range_or_unavailable(
            (point.features["prompt_tokens_p50"] for point in points),
            "not stored in source artifacts",
        ),
        "output_tokens": _range(point.features["output_tokens_p50"] for point in points),
        "ttft_ms": _range(point.metrics["ttft_ms"] for point in points),
        "e2e_ms": _range(point.metrics["e2e_ms"] for point in points),
    }


def _slo_assessment(
    config: CapacityPlannerConfig,
    nearest: EvidencePoint,
    status: PlannerStatus,
) -> dict[str, Any]:
    ttft_ok = nearest.metrics["ttft_ms"] is not None and nearest.metrics["ttft_ms"] <= config.slo.ttft_ms
    e2e_ok = nearest.metrics["e2e_ms"] is not None and nearest.metrics["e2e_ms"] <= config.slo.e2e_ms
    if status != "IN_RANGE":
        label = f"NO_HARD_RECOMMENDATION_{status}"
    elif ttft_ok and e2e_ok:
        label = "MEETS_SLO_ON_NEAREST_EVIDENCE"
    else:
        label = "MISSES_SLO_ON_NEAREST_EVIDENCE"
    return {
        "status": label,
        "nearest_ttft_ms": nearest.metrics["ttft_ms"],
        "nearest_e2e_ms": nearest.metrics["e2e_ms"],
        "ttft_slo_ms": config.slo.ttft_ms,
        "e2e_slo_ms": config.slo.e2e_ms,
    }


def _cost(config: CapacityPlannerConfig, candidate_id: str, nearest: EvidencePoint) -> dict[str, Any]:
    hourly = config.pricing.hourly_usd_by_candidate.get(candidate_id)
    if hourly is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "cost unavailable without user-provided dated candidate price metadata",
        }
    rps = nearest.metrics["request_throughput_rps"]
    if not rps:
        return {"status": "UNAVAILABLE", "reason": "request throughput unavailable"}
    return {
        "status": "AVAILABLE",
        "checked_at": config.pricing.checked_at,
        "usd_per_1000_requests_nearest": round(hourly / (float(rps) * 3600) * 1000, 6),
    }


def _recommendation(status: PlannerStatus) -> str:
    if status == "IN_RANGE":
        return "BOUNDED_BY_NEAREST_EVIDENCE"
    if status == "INTERPOLATED":
        return "INTERPOLATED_REVIEW_REQUIRED"
    if status == "EXTRAPOLATED":
        return "NO_HARD_RECOMMENDATION_EXTRAPOLATED"
    return "NO_RECOMMENDATION_UNAVAILABLE"


def _point_payload(point: EvidencePoint, project_root: Path) -> dict[str, Any]:
    del project_root
    return {
        "candidate_id": point.candidate_id,
        "run_id": point.run_id,
        "study_id": point.study_id,
        "engine": point.engine,
        "features": point.features,
        "metrics": point.metrics,
        "profile": point.profile,
        "source_paths": point.source_paths,
    }


def _candidate_holdout(candidate_id: str, holdout: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in holdout.get("rows", []) if row.get("candidate_id") == candidate_id]
    if not rows:
        return {"status": "UNAVAILABLE", "reason": "fewer than two compatible runs"}
    return {
        "status": "AVAILABLE",
        "sample_count": len(rows),
        "nearest_neighbor_ttft_mae_ms": _mae(row["ttft_ms"]["nearest_abs_error"] for row in rows),
        "baseline_ttft_mae_ms": _mae(row["ttft_ms"]["baseline_abs_error"] for row in rows),
        "nearest_neighbor_e2e_mae_ms": _mae(row["e2e_ms"]["nearest_abs_error"] for row in rows),
        "baseline_e2e_mae_ms": _mae(row["e2e_ms"]["baseline_abs_error"] for row in rows),
    }


def _holdout_metric(
    point: EvidencePoint,
    nearest: EvidencePoint,
    train: list[EvidencePoint],
    metric: str,
) -> dict[str, float | None]:
    actual = point.metrics[metric]
    nearest_prediction = nearest.metrics[metric]
    baseline_prediction = _median([item.metrics[metric] for item in train])
    return {
        "actual": actual,
        "nearest_prediction": nearest_prediction,
        "baseline_prediction": baseline_prediction,
        "nearest_abs_error": _abs_error(actual, nearest_prediction),
        "baseline_abs_error": _abs_error(actual, baseline_prediction),
    }


def _holdout_aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    nearest_ttft = _mae(row["ttft_ms"]["nearest_abs_error"] for row in rows)
    baseline_ttft = _mae(row["ttft_ms"]["baseline_abs_error"] for row in rows)
    nearest_e2e = _mae(row["e2e_ms"]["nearest_abs_error"] for row in rows)
    baseline_e2e = _mae(row["e2e_ms"]["baseline_abs_error"] for row in rows)
    return {
        "nearest_neighbor_ttft_mae_ms": nearest_ttft,
        "baseline_ttft_mae_ms": baseline_ttft,
        "ttft_improvement_over_baseline_pct": _improvement(baseline_ttft, nearest_ttft),
        "nearest_neighbor_e2e_mae_ms": nearest_e2e,
        "baseline_e2e_mae_ms": baseline_e2e,
        "e2e_improvement_over_baseline_pct": _improvement(baseline_e2e, nearest_e2e),
    }


def _point_distance(left: EvidencePoint, right: EvidencePoint) -> float:
    components = []
    for name in ("request_rate_rps", "prompt_chars", "output_tokens_p50", "shared_prefix_pct"):
        left_value = left.features.get(name)
        right_value = right.features.get(name)
        if left_value is None or right_value is None:
            continue
        denominator = max(abs(float(left_value)), abs(float(right_value)), 1.0)
        components.append(abs(float(left_value) - float(right_value)) / denominator)
    components.append(0.0 if left.features["arrival_model"] == right.features["arrival_model"] else 1.0)
    return sum(components) / len(components) if components else math.inf


def _candidate_id(manifest: contract.RunManifest, comparison_type: str) -> str:
    if comparison_type != "deployment_profile":
        return manifest.engine.name
    weights = manifest.comparison_key.get("weights") or {}
    return "/".join(
        str(part)
        for part in (
            manifest.engine.name,
            weights.get("format"),
            weights.get("precision"),
            weights.get("quantization") or "none",
        )
    )


def _shared_prefix_pct(cache_state: str) -> float | None:
    if cache_state in {"cold_per_run", "warm_after_warmup"}:
        return 0.0
    if "shared_prefix" in cache_state:
        return None
    return None


def _target_value(target: Mapping[str, Any], key: str, fallback: Any = None) -> float | None:
    value = target.get(key, fallback)
    return float(value) if value is not None else None


def _feature_range_or_unavailable(values: Any, reason: str) -> dict[str, Any]:
    present = _present(values)
    if not present:
        return {"status": "UNAVAILABLE", "reason": reason}
    return {"min": min(present), "max": max(present)}


def _range(values: Any) -> dict[str, float | None]:
    present = _present(values)
    return {"min": min(present) if present else None, "max": max(present) if present else None}


def _present(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None]


def _median(values: list[float | None]) -> float | None:
    present = _present(values)
    return float(median(present)) if present else None


def _abs_error(actual: float | None, predicted: float | None) -> float | None:
    if actual is None or predicted is None:
        return None
    return abs(actual - predicted)


def _mae(values: Any) -> float | None:
    present = _present(values)
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def _improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return round((baseline - candidate) / baseline * 100, 6)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.3f}" if isinstance(value, int | float) else str(value)


def _project_path(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
