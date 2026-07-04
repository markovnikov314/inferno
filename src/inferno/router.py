"""P8 offline serving-policy replay evaluator."""

from __future__ import annotations

from collections import Counter
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

COMPARISON_TYPES = {
    "strict": "strict_engine_comparison",
    "engine_configuration": "engine_configuration_comparison",
    "deployment_profile": "deployment_profile_comparison",
}


class RouterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateScope(RouterModel):
    environment_fingerprint: str = Field(min_length=1)
    comparison_type: Literal["strict", "engine_configuration", "deployment_profile"]


class ObjectiveSpec(RouterModel):
    name: str = Field(min_length=1)
    latency_metric: Literal["e2e_ms", "ttft_ms"]
    failure_penalty_ms: float = Field(ge=0)
    quality_incompatible_penalty_ms: float = Field(ge=0)
    headroom_penalty_ms: float = Field(ge=0)
    uncertainty_penalty_ms: float = Field(ge=0)


class ReplaySpec(RouterModel):
    holdout_strategy: Literal["leave_one_workload_block_out"]
    request_rate_rps: float = Field(gt=0)
    default_queue_depth: int = Field(ge=0)
    queue_depth_by_workload: dict[str, int] = Field(default_factory=dict)


class QualitySpec(RouterModel):
    require_json_for_workloads: list[str] = Field(default_factory=list)


class PricingSpec(RouterModel):
    source: Literal["user_provided"]
    checked_at: str = Field(min_length=1)
    hourly_usd_by_candidate: dict[str, float] = Field(default_factory=dict)

    @field_validator("checked_at")
    @classmethod
    def checked_at_is_iso8601(cls, value: str) -> str:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class RouterReplayConfig(RouterModel):
    schema_version: Literal[1]
    replay_id: str = Field(min_length=1)
    candidate_scope: CandidateScope
    objective: ObjectiveSpec
    replay: ReplaySpec
    quality: QualitySpec = Field(default_factory=QualitySpec)
    pricing: PricingSpec
    evidence_studies: list[str] = Field(min_length=1)
    artifacts_dir: str = Field(min_length=1)


@dataclass(frozen=True)
class CandidateOutcome:
    candidate_id: str
    run_id: str
    run_dir: Path
    study_id: str
    block_id: str
    repeat_index: int
    engine: str
    profile: dict[str, Any]
    features: dict[str, Any]
    metrics: dict[str, float | None]
    quality: dict[str, Any]
    source_paths: dict[str, str]


@dataclass(frozen=True)
class ReplayRequest:
    request_id: str
    block_id: str
    repeat_index: int
    features: dict[str, Any]
    outcomes: dict[str, CandidateOutcome]


def load_router_config(path: Path) -> RouterReplayConfig:
    return RouterReplayConfig.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def run_router_replay(*, config_path: Path, project_root: Path) -> int:
    config = load_router_config(_project_path(project_root, str(config_path)))
    output_dir = _project_path(project_root, config.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes, rejected = _load_evidence(config, project_root)
    requests, skipped_requests = _build_replay_requests(outcomes, config)
    result = build_router_replay(config, outcomes, requests, rejected, skipped_requests, project_root)

    _write_json(output_dir / "latest.json", result)
    _write_json(output_dir / "leakage_check.json", result["leakage_check"])
    _write_json(output_dir / "ablation_eval.json", result["ablation_evaluation"])
    _write_jsonl(output_dir / "decision_log.jsonl", result["decision_log"])
    (output_dir / "report.md").write_text(render_router_report(result), encoding="utf-8")
    print(output_dir / "report.md")
    return 0 if result["replay_request_count"] else 1


def build_router_replay(
    config: RouterReplayConfig,
    outcomes: list[CandidateOutcome],
    requests: list[ReplayRequest],
    rejected_evidence: list[dict[str, Any]],
    skipped_requests: list[dict[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    del project_root
    policies = _policy_names(requests, config)
    decision_log = [
        _decision_row(config, request, outcomes, policies, "proposed_policy") for request in requests
    ]
    policy_results = {
        policy: _evaluate_policy(policy, decision_log, requests, config) for policy in policies
    }
    leakage_check = _leakage_check(decision_log)
    ablations = {
        name: _evaluate_policy(
            name,
            [
                _decision_row(config, request, outcomes, [name], variant)
                for request in requests
            ],
            requests,
            config,
        )
        for name, variant in {
            "proposed_no_quality_filter": "no_quality_filter",
            "proposed_no_headroom_penalty": "no_headroom_penalty",
            "proposed_no_uncertainty_penalty": "no_uncertainty_penalty",
        }.items()
    }
    baseline = _best_static_baseline(policy_results)
    proposed = policy_results.get("proposed_policy")
    comparison = _proposed_comparison(proposed, baseline)
    return {
        "schema_version": 1,
        "phase": "P8",
        "replay_id": config.replay_id,
        "evaluator": "offline_serving_policy_replay_v1",
        "comparison_type": config.candidate_scope.comparison_type,
        "environment_fingerprint": config.candidate_scope.environment_fingerprint,
        "holdout_strategy": config.replay.holdout_strategy,
        "objective": config.objective.model_dump(mode="json"),
        "pricing": _pricing_summary(config),
        "evidence_studies": config.evidence_studies,
        "compatible_evidence_count": len(outcomes),
        "replay_request_count": len(requests),
        "candidate_ids": sorted({outcome.candidate_id for outcome in outcomes}),
        "omitted_baselines": _omitted_baselines(requests, config),
        "rejected_evidence": rejected_evidence,
        "skipped_replay_requests": skipped_requests,
        "leakage_check": leakage_check,
        "policy_results": policy_results,
        "baseline_comparison": comparison,
        "ablation_evaluation": {
            "status": "PASS" if ablations else "UNAVAILABLE",
            "variants": ablations,
        },
        "decision_log": decision_log,
        "limitations": [
            "P8 is offline replay only; no live endpoint or runtime router is introduced.",
            "Replay decisions are bounded by existing validated artifacts and held out by workload block.",
            "Prompt token counts are unavailable in source artifacts, so canonical prompt length uses prompt characters.",
            "Cost is unavailable unless dated user-provided candidate prices are supplied.",
            "A non-positive proposed-policy improvement is reported as a negative result, not a deployment claim.",
        ],
    }


def render_router_report(result: Mapping[str, Any]) -> str:
    comparison = result["baseline_comparison"]
    lines = [
        "# P8 Offline Serving-Policy Router Replay",
        "",
        "OFFLINE REPLAY ONLY - no live router, endpoint, or serving control plane is introduced.",
        "",
        f"Replay ID: `{result['replay_id']}`",
        f"Comparison type: `{result['comparison_type']}`",
        f"Environment fingerprint: `{result['environment_fingerprint']}`",
        f"Holdout strategy: `{result['holdout_strategy']}`",
        f"Objective: `{result['objective']['name']}` using `{result['objective']['latency_metric']}`",
        "",
        "## Result",
        "",
        (
            f"Proposed policy status: **{comparison['status']}**. "
            f"Improvement over best static baseline: {_fmt(comparison.get('improvement_pct'))}%."
        ),
        "",
        "## Policy Evaluation",
        "",
        "| Policy | Objective | Mean TTFT ms | Mean E2E ms | Fail risk | Quality misses | Selection | Confidence | Cost |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for name, item in result["policy_results"].items():
        lines.append(
            "| {policy} | {objective} | {ttft} | {e2e} | {risk} | {quality} | {selection} | {confidence} | {cost} |".format(
                policy=name,
                objective=_fmt(item["objective_value"]),
                ttft=_fmt(item["latency"]["mean_ttft_ms"]),
                e2e=_fmt(item["latency"]["mean_e2e_ms"]),
                risk=_fmt(item["failure_risk"]["mean_selected_error_rate"]),
                quality=item["quality_compatibility"]["incompatible_selection_count"],
                selection=_fmt_distribution(item["selection_distribution"]),
                confidence=_fmt_distribution(item["confidence_distribution"]),
                cost=item["cost"]["status"],
            )
        )

    lines.extend(
        [
            "",
            "## Baselines",
            "",
        ]
    )
    if result["omitted_baselines"]:
        lines.extend(
            f"- `{item['baseline']}` omitted: {item['reason']}"
            for item in result["omitted_baselines"]
        )
        lines.append("")
    lines.extend(
        [
            "The oracle lower bound is explicitly non-deployable and uses held-out outcomes.",
            "",
            "## Leakage And Ablations",
            "",
            f"- Leakage check: `{result['leakage_check']['status']}`.",
            f"- Ablation check: `{result['ablation_evaluation']['status']}`.",
            "",
            "## Artifacts",
            "",
            "- `latest.json`",
            "- `decision_log.jsonl`",
            "- `leakage_check.json`",
            "- `ablation_eval.json`",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def _load_evidence(
    config: RouterReplayConfig,
    project_root: Path,
) -> tuple[list[CandidateOutcome], list[dict[str, Any]]]:
    outcomes: list[CandidateOutcome] = []
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
            outcomes.append(
                _candidate_outcome(
                    config=config,
                    project_root=project_root,
                    latest=latest,
                    latest_path=latest_path,
                    run_dir=run_dir,
                    manifest=manifest,
                    summary=summary,
                )
            )
    return outcomes, rejected


def _candidate_outcome(
    *,
    config: RouterReplayConfig,
    project_root: Path,
    latest: Mapping[str, Any],
    latest_path: Path,
    run_dir: Path,
    manifest: contract.RunManifest,
    summary: contract.RunSummary,
) -> CandidateOutcome:
    repeat_index = manifest.repeat_index or 1
    workload_id = manifest.workload.workload_id
    cache_state = manifest.workload.cache_state
    required_quality = _required_quality(config, workload_id)
    quality = _quality_compatibility(run_dir, manifest, required_quality)
    artifacts = manifest.artifacts
    return CandidateOutcome(
        candidate_id=_candidate_id(manifest, config.candidate_scope.comparison_type),
        run_id=manifest.run_id,
        run_dir=run_dir,
        study_id=str(latest.get("study_id") or manifest.study_id),
        block_id=workload_id,
        repeat_index=repeat_index,
        engine=manifest.engine.name,
        profile=_profile(manifest),
        features={
            "canonical_prompt_length": {
                "unit": "tokens"
                if manifest.workload.prompt_target.get("p50") is not None
                else "chars",
                "value": manifest.workload.prompt_target.get("p50")
                or manifest.workload.prompt_chars,
                "prompt_tokens": manifest.workload.prompt_target.get("p50"),
                "prompt_tokens_missing_reason": None
                if manifest.workload.prompt_target.get("p50") is not None
                else "not stored in source artifacts",
            },
            "requested_output_tokens": manifest.workload.output_target.get("p50")
            or manifest.workload.sampling.get("max_tokens"),
            "prefix_cache_policy": {
                "cache_state": cache_state,
                "cache_policy": manifest.workload.cache_policy,
                "shared_prefix_pct": _shared_prefix_pct(
                    manifest.workload.cache_policy or cache_state
                ),
            },
            "replay_queue_depth": _queue_depth(config, workload_id),
            "replay_request_rate_rps": manifest.workload.request_rate_rps
            or config.replay.request_rate_rps,
            "quality_requirement": manifest.workload.quality_requirement.get(
                "type", required_quality
            )
            or required_quality,
            "arrival_model": manifest.workload.arrival_model,
            "measurement_requests": manifest.workload.measurement_requests,
            "context_length": manifest.workload.context_length,
            "streaming": manifest.workload.streaming,
            "max_in_flight": manifest.workload.max_in_flight or manifest.workload.concurrency,
        },
        metrics={
            "ttft_ms": summary.ttft_ms,
            "e2e_ms": summary.e2e_ms,
            "request_throughput_rps": summary.request_throughput_rps,
            "error_rate": summary.error_rate,
            "completion_tokens": float(summary.completion_tokens)
            if summary.completion_tokens is not None
            else None,
        },
        quality=quality,
        source_paths={
            "study": _display_path(project_root, latest_path),
            "run_dir": _display_path(project_root, run_dir),
            "manifest": _display_path(project_root, run_dir / artifacts.manifest),
            "summary": _display_path(project_root, run_dir / artifacts.summary),
            "validation": _display_path(project_root, run_dir / artifacts.validation),
        },
    )


def _build_replay_requests(
    outcomes: list[CandidateOutcome],
    config: RouterReplayConfig,
) -> tuple[list[ReplayRequest], list[dict[str, Any]]]:
    del config
    grouped: dict[tuple[str, int], list[CandidateOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault((outcome.block_id, outcome.repeat_index), []).append(outcome)

    requests: list[ReplayRequest] = []
    skipped: list[dict[str, Any]] = []
    for (block_id, repeat_index), group in sorted(grouped.items()):
        by_candidate = {outcome.candidate_id: outcome for outcome in group}
        if len(by_candidate) < 2:
            skipped.append(
                {
                    "block_id": block_id,
                    "repeat_index": repeat_index,
                    "reason": "fewer than two candidate outcomes for replay request",
                    "candidate_ids": sorted(by_candidate),
                }
            )
            continue
        first = group[0]
        requests.append(
            ReplayRequest(
                request_id=f"{block_id}-r{repeat_index:02d}",
                block_id=block_id,
                repeat_index=repeat_index,
                features=first.features,
                outcomes=by_candidate,
            )
        )
    return requests, skipped


def _decision_row(
    config: RouterReplayConfig,
    request: ReplayRequest,
    all_outcomes: list[CandidateOutcome],
    policies: list[str],
    variant: str,
) -> dict[str, Any]:
    heldout_run_ids = sorted(outcome.run_id for outcome in request.outcomes.values())
    train = [outcome for outcome in all_outcomes if outcome.block_id != request.block_id]
    decisions = {
        policy: _select_candidate(policy, request, train, config, variant) for policy in policies
    }
    return {
        "schema_version": 1,
        "request_id": request.request_id,
        "heldout_block_id": request.block_id,
        "repeat_index": request.repeat_index,
        "features": request.features,
        "heldout_run_ids": heldout_run_ids,
        "train_block_ids": sorted({outcome.block_id for outcome in train}),
        "train_run_ids": sorted(outcome.run_id for outcome in train),
        "candidate_profiles": {
            candidate_id: outcome.profile for candidate_id, outcome in request.outcomes.items()
        },
        "candidate_actuals": {
            candidate_id: _actual_payload(outcome) for candidate_id, outcome in request.outcomes.items()
        },
        "policy_decisions": decisions,
    }


def _select_candidate(
    policy: str,
    request: ReplayRequest,
    train: list[CandidateOutcome],
    config: RouterReplayConfig,
    variant: str,
) -> dict[str, Any]:
    if policy.startswith("static_"):
        candidate_id = policy.removeprefix("static_")
        return _static_decision(candidate_id, request, train, config)
    if policy == "oracle_lower_bound":
        return _oracle_decision(request, config)
    if policy == "transparent_heuristic":
        return _heuristic_decision(request, train, config)
    return _proposed_decision(request, train, config, variant)


def _static_decision(
    candidate_id: str,
    request: ReplayRequest,
    train: list[CandidateOutcome],
    config: RouterReplayConfig,
) -> dict[str, Any]:
    if candidate_id not in request.outcomes:
        return _unavailable_decision(candidate_id, "candidate not present in held-out request")
    train_candidate = [outcome for outcome in train if outcome.candidate_id == candidate_id]
    quality_supported = _training_quality_supported(
        train_candidate, request.features["quality_requirement"]
    )
    return {
        "selected_candidate_id": candidate_id,
        "reason": "static candidate baseline",
        "deployable": True,
        "confidence": _confidence(train_candidate, 0.0, True, quality_supported),
        "score": None,
        "estimated": _estimate_payload(candidate_id, train_candidate, request, config, 0.0),
    }


def _oracle_decision(request: ReplayRequest, config: RouterReplayConfig) -> dict[str, Any]:
    metric = config.objective.latency_metric
    candidate_id, outcome = min(
        request.outcomes.items(),
        key=lambda item: item[1].metrics[metric] if item[1].metrics[metric] is not None else math.inf,
    )
    return {
        "selected_candidate_id": candidate_id,
        "reason": "oracle lower bound uses held-out actuals; non-deployable",
        "deployable": False,
        "confidence": "oracle",
        "score": outcome.metrics[metric],
        "estimated": {"uses_heldout_actuals": True},
    }


def _heuristic_decision(
    request: ReplayRequest,
    train: list[CandidateOutcome],
    config: RouterReplayConfig,
) -> dict[str, Any]:
    candidates = []
    for candidate_id in request.outcomes:
        train_candidate = [outcome for outcome in train if outcome.candidate_id == candidate_id]
        quality_ok = _training_quality_ok(train_candidate, request.features["quality_requirement"])
        if not quality_ok:
            continue
        estimate = _median(outcome.metrics[config.objective.latency_metric] for outcome in train_candidate)
        candidates.append((estimate if estimate is not None else math.inf, candidate_id, train_candidate))
    if not candidates:
        return _unavailable_decision(None, "no quality-compatible training candidates")
    score, candidate_id, train_candidate = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "selected_candidate_id": candidate_id,
        "reason": "transparent heuristic: lowest training median latency after quality filter",
        "deployable": True,
        "confidence": _confidence(
            train_candidate,
            0.0,
            True,
            _training_quality_supported(train_candidate, request.features["quality_requirement"]),
        ),
        "score": score if score != math.inf else None,
        "estimated": _estimate_payload(candidate_id, train_candidate, request, config, 0.0),
    }


def _proposed_decision(
    request: ReplayRequest,
    train: list[CandidateOutcome],
    config: RouterReplayConfig,
    variant: str,
) -> dict[str, Any]:
    scored = []
    for candidate_id in request.outcomes:
        train_candidate = [outcome for outcome in train if outcome.candidate_id == candidate_id]
        if not train_candidate:
            continue
        quality_ok = variant == "no_quality_filter" or _training_quality_ok(
            train_candidate, request.features["quality_requirement"]
        )
        quality_supported = variant == "no_quality_filter" or _training_quality_supported(
            train_candidate, request.features["quality_requirement"]
        )
        distance, nearest = _nearest_training(request, train_candidate)
        latency = nearest.metrics[config.objective.latency_metric] if nearest else None
        if latency is None:
            continue
        failure_risk = _mean(outcome.metrics["error_rate"] for outcome in train_candidate) or 0.0
        headroom = _headroom(train_candidate, config)
        uncertainty = _uncertainty_penalty(train_candidate, config, variant)
        score = latency + config.objective.failure_penalty_ms * failure_risk
        if not quality_ok:
            score += config.objective.quality_incompatible_penalty_ms
        if variant != "no_headroom_penalty" and headroom is not None and headroom < 0:
            score += abs(headroom) / config.replay.request_rate_rps * config.objective.headroom_penalty_ms
        score += uncertainty
        scored.append(
            {
                "candidate_id": candidate_id,
                "score": round(score, 6),
                "distance": round(distance, 6) if distance is not None else None,
                "quality_ok": quality_ok,
                "quality_supported": quality_supported,
                "train_candidate": train_candidate,
            }
        )
    if not scored:
        return _unavailable_decision(None, "no trainable candidates")
    best = min(scored, key=lambda item: (item["score"], item["candidate_id"]))
    return {
        "selected_candidate_id": best["candidate_id"],
        "reason": "nearest-neighbor latency with headroom, failure-risk, quality, and uncertainty terms",
        "deployable": True,
        "confidence": _confidence(
            best["train_candidate"],
            best["distance"] or math.inf,
            bool(best["quality_ok"]),
            bool(best["quality_supported"]),
        ),
        "score": best["score"],
        "estimated": _estimate_payload(
            best["candidate_id"],
            best["train_candidate"],
            request,
            config,
            best["distance"],
        ),
    }


def _evaluate_policy(
    policy: str,
    decision_log: list[Mapping[str, Any]],
    requests: list[ReplayRequest],
    config: RouterReplayConfig,
) -> dict[str, Any]:
    by_request = {request.request_id: request for request in requests}
    selected_rows = []
    for row in decision_log:
        decision = (row.get("policy_decisions") or {}).get(policy)
        if not decision:
            continue
        candidate_id = decision.get("selected_candidate_id")
        request = by_request[row["request_id"]]
        outcome = request.outcomes.get(candidate_id) if candidate_id else None
        if outcome is None:
            continue
        selected_rows.append((decision, outcome))
    objective_values = [
        _objective_actual(outcome, config) for _, outcome in selected_rows
    ]
    objective_value = _mean(objective_values)
    selection = Counter(outcome.candidate_id for _, outcome in selected_rows)
    confidence = Counter(str(decision.get("confidence")) for decision, _ in selected_rows)
    return {
        "policy": policy,
        "deployable": policy != "oracle_lower_bound",
        "decision_count": len(selected_rows),
        "objective_name": config.objective.name,
        "objective_value": round(objective_value, 6) if objective_value is not None else None,
        "latency": {
            "mean_ttft_ms": _rounded_mean(outcome.metrics["ttft_ms"] for _, outcome in selected_rows),
            "mean_e2e_ms": _rounded_mean(outcome.metrics["e2e_ms"] for _, outcome in selected_rows),
            "p95_ttft_ms": _percentile([outcome.metrics["ttft_ms"] for _, outcome in selected_rows], 95),
            "p95_e2e_ms": _percentile([outcome.metrics["e2e_ms"] for _, outcome in selected_rows], 95),
        },
        "cost": _cost_summary(config, [outcome for _, outcome in selected_rows]),
        "failure_risk": {
            "mean_selected_error_rate": _rounded_mean(
                outcome.metrics["error_rate"] for _, outcome in selected_rows
            ),
        },
        "quality_compatibility": {
            "incompatible_selection_count": sum(
                1 for _, outcome in selected_rows if not outcome.quality["compatible"]
            ),
            "statuses": dict(Counter(outcome.quality["status"] for _, outcome in selected_rows)),
        },
        "selection_distribution": _distribution(selection, len(selected_rows)),
        "confidence_distribution": _distribution(confidence, len(selected_rows)),
    }


def _leakage_check(decision_log: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    errors = []
    for row in decision_log:
        train_blocks = set(row["train_block_ids"])
        heldout_block = row["heldout_block_id"]
        train_runs = set(row["train_run_ids"])
        heldout_runs = set(row["heldout_run_ids"])
        overlap = sorted(train_runs & heldout_runs)
        block_leak = heldout_block in train_blocks
        if block_leak:
            errors.append(f"{row['request_id']}: held-out block present in train blocks")
        if overlap:
            errors.append(f"{row['request_id']}: held-out run present in train runs")
        rows.append(
            {
                "request_id": row["request_id"],
                "heldout_block_id": heldout_block,
                "train_block_count": len(train_blocks),
                "heldout_run_count": len(heldout_runs),
                "train_run_count": len(train_runs),
                "run_overlap_count": len(overlap),
                "block_leak": block_leak,
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "method": "leave-one-workload-block-out; no random request split",
        "errors": errors,
        "rows": rows,
    }


def _policy_names(requests: list[ReplayRequest], config: RouterReplayConfig) -> list[str]:
    candidate_ids = sorted({candidate for request in requests for candidate in request.outcomes})
    names = [f"static_{candidate_id}" for candidate_id in candidate_ids]
    names.extend(["transparent_heuristic", "oracle_lower_bound", "proposed_policy"])
    return names


def _omitted_baselines(
    requests: list[ReplayRequest],
    config: RouterReplayConfig,
) -> list[dict[str, str]]:
    del requests
    if config.candidate_scope.comparison_type != "strict":
        return []
    return [
        {
            "baseline": "static_llamacpp",
            "reason": "llama.cpp is not appropriate for strict engine comparison; use deployment-profile scope",
        }
    ]


def _best_static_baseline(policy_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    static_items = [
        item
        for name, item in policy_results.items()
        if name.startswith("static_") and item.get("objective_value") is not None
    ]
    return min(static_items, key=lambda item: (item["objective_value"], item["policy"])) if static_items else None


def _proposed_comparison(
    proposed: Mapping[str, Any] | None,
    best_static: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not proposed or proposed.get("objective_value") is None or not best_static:
        return {
            "status": "NEGATIVE_RESULT",
            "reason": "proposed policy or static baseline unavailable",
            "best_static_policy": None,
            "improvement_pct": None,
        }
    improvement = _improvement(best_static["objective_value"], proposed["objective_value"])
    return {
        "status": "IMPROVED"
        if improvement is not None and improvement > 0
        else "NEGATIVE_RESULT",
        "reason": "proposed policy beat the best static baseline"
        if improvement is not None and improvement > 0
        else "proposed policy did not improve the declared held-out objective over the best static baseline",
        "best_static_policy": best_static["policy"],
        "best_static_objective_value": best_static["objective_value"],
        "proposed_objective_value": proposed["objective_value"],
        "improvement_pct": improvement,
    }


def _actual_payload(outcome: CandidateOutcome) -> dict[str, Any]:
    return {
        "run_id": outcome.run_id,
        "metrics": outcome.metrics,
        "quality": outcome.quality,
        "source_paths": outcome.source_paths,
    }


def _estimate_payload(
    candidate_id: str,
    train_candidate: list[CandidateOutcome],
    request: ReplayRequest,
    config: RouterReplayConfig,
    distance: float | None,
) -> dict[str, Any]:
    required_quality = request.features["quality_requirement"]
    return {
        "candidate_id": candidate_id,
        "train_sample_count": len(train_candidate),
        "nearest_distance": round(distance, 6) if distance is not None else None,
        "predicted_latency_ms": _rounded_median(
            outcome.metrics[config.objective.latency_metric] for outcome in train_candidate
        ),
        "observed_headroom_rps": _headroom(train_candidate, config),
        "failure_risk": _rounded_mean(outcome.metrics["error_rate"] for outcome in train_candidate),
        "quality_compatible_in_training": _training_quality_ok(
            train_candidate, required_quality
        ),
        "quality_supported_in_training": _training_quality_supported(
            train_candidate, required_quality
        ),
    }


def _unavailable_decision(candidate_id: str | None, reason: str) -> dict[str, Any]:
    return {
        "selected_candidate_id": candidate_id,
        "reason": reason,
        "deployable": False,
        "confidence": "unavailable",
        "score": None,
        "estimated": {},
    }


def _nearest_training(
    request: ReplayRequest,
    train_candidate: list[CandidateOutcome],
) -> tuple[float | None, CandidateOutcome | None]:
    if not train_candidate:
        return None, None
    pairs = [(_distance(request.features, outcome.features), outcome) for outcome in train_candidate]
    return min(pairs, key=lambda item: item[0])


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    components = []

    def add(name: str, left_value: float | None, right_value: float | None) -> None:
        if left_value is None or right_value is None:
            return
        denominator = max(abs(float(left_value)), abs(float(right_value)), 1.0)
        components.append(abs(float(left_value) - float(right_value)) / denominator)

    add(
        "prompt",
        (left.get("canonical_prompt_length") or {}).get("value"),
        (right.get("canonical_prompt_length") or {}).get("value"),
    )
    add("output", left.get("requested_output_tokens"), right.get("requested_output_tokens"))
    add("queue", left.get("replay_queue_depth"), right.get("replay_queue_depth"))
    add(
        "request_rate",
        left.get("replay_request_rate_rps"),
        right.get("replay_request_rate_rps"),
    )
    components.append(
        0.0
        if (left.get("prefix_cache_policy") or {}).get("cache_state")
        == (right.get("prefix_cache_policy") or {}).get("cache_state")
        else 1.0
    )
    return sum(components) / len(components) if components else math.inf


def _objective_actual(outcome: CandidateOutcome, config: RouterReplayConfig) -> float | None:
    latency = outcome.metrics[config.objective.latency_metric]
    if latency is None:
        return None
    value = latency + config.objective.failure_penalty_ms * float(outcome.metrics["error_rate"] or 0)
    if not outcome.quality["compatible"]:
        value += config.objective.quality_incompatible_penalty_ms
    return value


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


def _profile(manifest: contract.RunManifest) -> dict[str, Any]:
    weights = manifest.comparison_key.get("weights") or {}
    model = manifest.comparison_key.get("model") or {}
    return {
        "engine": manifest.engine.name,
        "model": f"{model.get('id')}@{model.get('revision')}",
        "weight_format": weights.get("format"),
        "precision": weights.get("precision"),
        "quantization": weights.get("quantization"),
        "runtime": manifest.comparison_key.get("runtime"),
    }


def _required_quality(config: RouterReplayConfig, workload_id: str) -> str:
    return "json_object" if workload_id in set(config.quality.require_json_for_workloads) else "none"


def _quality_compatibility(
    run_dir: Path,
    manifest: contract.RunManifest,
    required_quality: str,
) -> dict[str, Any]:
    if required_quality == "none":
        return {"required": "none", "compatible": True, "status": "E3_NOT_APPLICABLE"}
    try:
        raw_sse = contract.artifact_path(run_dir, manifest.artifacts.raw_sse).read_text(
            encoding="utf-8-sig"
        )
        generated = "".join(text for _, text in contract.generated_text_events(raw_sse)).strip()
        payload = json.loads(generated)
    except Exception as exc:
        return {
            "required": required_quality,
            "compatible": False,
            "status": "E3_JSON_INVALID",
            "detail": type(exc).__name__,
        }
    return {
        "required": required_quality,
        "compatible": isinstance(payload, dict),
        "status": "E3_JSON_VALID" if isinstance(payload, dict) else "E3_JSON_INVALID",
        "detail": "json_object" if isinstance(payload, dict) else "json_non_object",
    }


def _training_quality_ok(train_candidate: list[CandidateOutcome], required_quality: str) -> bool:
    if required_quality == "none":
        return True
    relevant = [
        outcome for outcome in train_candidate if outcome.quality.get("required") == required_quality
    ]
    return all(outcome.quality["compatible"] for outcome in relevant)


def _training_quality_supported(
    train_candidate: list[CandidateOutcome],
    required_quality: str,
) -> bool:
    if required_quality == "none":
        return True
    return any(outcome.quality.get("required") == required_quality for outcome in train_candidate)


def _headroom(train_candidate: list[CandidateOutcome], config: RouterReplayConfig) -> float | None:
    throughput = _median(outcome.metrics["request_throughput_rps"] for outcome in train_candidate)
    if throughput is None:
        return None
    return round(throughput - config.replay.request_rate_rps, 6)


def _uncertainty_penalty(
    train_candidate: list[CandidateOutcome],
    config: RouterReplayConfig,
    variant: str,
) -> float:
    if variant == "no_uncertainty_penalty":
        return 0.0
    if len(train_candidate) < 3:
        return config.objective.uncertainty_penalty_ms
    values = _present(outcome.metrics[config.objective.latency_metric] for outcome in train_candidate)
    if len(values) < 2:
        return config.objective.uncertainty_penalty_ms
    spread = max(values) - min(values)
    center = max(_median(values) or 1.0, 1.0)
    return min(spread / center, 1.0) * config.objective.uncertainty_penalty_ms


def _confidence(
    train_candidate: list[CandidateOutcome],
    distance: float,
    quality_ok: bool,
    quality_supported: bool,
) -> Literal["low", "medium"]:
    if not quality_ok or not quality_supported or len(train_candidate) < 3 or distance > 0.50:
        return "low"
    return "medium"


def _pricing_summary(config: RouterReplayConfig) -> dict[str, Any]:
    return {
        "source": config.pricing.source,
        "checked_at": config.pricing.checked_at,
        "cost_status": "AVAILABLE"
        if config.pricing.hourly_usd_by_candidate
        else "UNAVAILABLE",
        "unavailable_reason": None
        if config.pricing.hourly_usd_by_candidate
        else "cost unavailable without user-provided dated candidate price metadata",
    }


def _cost_summary(config: RouterReplayConfig, selected: list[CandidateOutcome]) -> dict[str, Any]:
    if not config.pricing.hourly_usd_by_candidate:
        return {
            "status": "UNAVAILABLE",
            "reason": "cost unavailable without user-provided dated candidate price metadata",
        }
    values = []
    for outcome in selected:
        hourly = config.pricing.hourly_usd_by_candidate.get(outcome.candidate_id)
        rps = outcome.metrics["request_throughput_rps"]
        if hourly is None or not rps:
            continue
        values.append(hourly / (float(rps) * 3600) * 1000)
    if not values:
        return {"status": "UNAVAILABLE", "reason": "candidate price or throughput missing"}
    return {
        "status": "AVAILABLE",
        "checked_at": config.pricing.checked_at,
        "mean_usd_per_1000_requests": round(sum(values) / len(values), 6),
    }


def _shared_prefix_pct(cache_state: str) -> float | None:
    if cache_state in {"cold_per_run", "warm_after_warmup"}:
        return 0.0
    if "shared_prefix" in cache_state:
        return None
    return None


def _queue_depth(config: RouterReplayConfig, workload_id: str) -> int:
    return config.replay.queue_depth_by_workload.get(workload_id, config.replay.default_queue_depth)


def _distribution(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    if total <= 0:
        return {}
    return {
        key: {"count": count, "pct": round(count / total * 100, 6)}
        for key, count in sorted(counter.items())
    }


def _rounded_mean(values: Any) -> float | None:
    value = _mean(values)
    return round(value, 6) if value is not None else None


def _rounded_median(values: Any) -> float | None:
    value = _median(values)
    return round(value, 6) if value is not None else None


def _mean(values: Any) -> float | None:
    present = _present(values)
    if not present:
        return None
    return sum(present) / len(present)


def _median(values: Any) -> float | None:
    present = _present(values)
    return float(median(present)) if present else None


def _present(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None]


def _percentile(values: list[float | None], percentile: int) -> float | None:
    present = sorted(_present(values))
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    rank = (len(present) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return present[lower]
    return round(present[lower] + (present[upper] - present[lower]) * (rank - lower), 6)


def _improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    return round((baseline - candidate) / baseline * 100, 6)


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    return f"{float(value):.3f}" if isinstance(value, int | float) else str(value)


def _fmt_distribution(value: Mapping[str, Mapping[str, Any]]) -> str:
    if not value:
        return "-"
    return ", ".join(f"{key}:{item['count']}" for key, item in value.items())


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


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
