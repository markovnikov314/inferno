"""Artifact contract and validator."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
CONTRACT_VERSION = "inferno.artifacts.v1"
PARSER_VERSION = "inferno.sse.v1"
STUDY_SUMMARY_FIELDS = {
    "itl_ms",
    "completion_tokens",
    "completion_tokens_missing_reason",
    "request_throughput_rps",
    "output_token_throughput_tps",
    "output_token_throughput_missing_reason",
    "error_rate",
}

RunStatus = Literal["PLANNED", "RUNNING", "SUCCEEDED", "FAILED", "INVALID", "SKIPPED_NO_GPU"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactPaths(ContractModel):
    manifest: str
    request_trace: str
    request_trace_parquet: str | None = None
    raw_sse: str
    engine_logs: str
    health: str
    telemetry_parquet: str | None = None
    engine_native_metrics: str | None = None
    summary: str
    validation: str
    checksums: str
    launch_provenance: str | None = None


class SourceInfo(ContractModel):
    branch: str
    commit: str
    dirty: bool


class EngineInfo(ContractModel):
    name: Literal["vllm", "sglang", "llamacpp", "tensorrtllm", "tensorrtllm_t4", "ollama"]
    image: str = Field(min_length=1)
    container_name: str = Field(min_length=1)
    port: int
    served_model_name: str = Field(min_length=1)
    launch_args: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)


class ModelInfo(ContractModel):
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    tokenizer_format: str | None = None
    weight_format: str | None = None
    weight_precision: str | None = None
    quantization: str | None = None
    gguf_repo: str | None = None
    gguf_filename: str | None = None
    gguf_revision: str | None = None


class HardwareInfo(ContractModel):
    gpu_model: str = Field(min_length=1)
    gpu_count: int = Field(ge=1)
    driver_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    memory_mib: int | None = None


class WorkloadInfo(ContractModel):
    workload_id: str = Field(min_length=1)
    prompt_template_id: str = Field(min_length=1)
    seed: int
    prompt_sha256: str = Field(min_length=64, max_length=64)
    prompt_chars: int = Field(ge=1)
    request_rate_rps: float | None = Field(default=None, gt=0)
    arrival_model: str = Field(min_length=1)
    concurrency: int | None = Field(default=None, ge=1)
    max_in_flight: int | None = Field(default=None, ge=1)
    prompt_target: dict[str, Any] = Field(default_factory=dict)
    output_target: dict[str, Any] = Field(default_factory=dict)
    context_length: int | None = Field(default=None, ge=1)
    cache_state: str = Field(min_length=1)
    cache_policy: str | None = Field(default=None, min_length=1)
    warmup_requests: int = Field(ge=0)
    measurement_requests: int = Field(ge=1)
    max_runtime_seconds: int = Field(ge=1)
    streaming: bool | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)
    retry_policy: str = Field(min_length=1)
    failure_policy: str = Field(min_length=1)
    quality_requirement: dict[str, Any] = Field(default_factory=dict)
    cost_metadata: dict[str, Any] = Field(default_factory=dict)
    runtime_knobs: dict[str, Any] = Field(default_factory=dict)
    sampling: dict[str, Any]


class HealthEvidence(ContractModel):
    ok: bool
    model_ids: list[str]


class StreamEvidence(ContractModel):
    ok: bool
    generated_text_sha256: str = Field(min_length=64, max_length=64)
    generated_text_chars: int = Field(ge=0)


class ShutdownEvidence(ContractModel):
    ok: bool


class RunManifest(ContractModel):
    schema_version: Literal[1]
    contract_version: Literal["inferno.artifacts.v1"]
    run_family: Literal["single_run", "smoke_study", "strict_comparison", "research_core", "deployment_profile", "engine_configuration", "dashboard"]
    run_id: str = Field(min_length=1)
    study_id: str | None = None
    repeat_index: int | None = Field(default=None, ge=1)
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    manifest_created_before_request: bool
    manifest_written_perf_counter_ns: int = Field(ge=1)
    first_request_perf_counter_ns: int | None = None
    run_dir: str
    source: SourceInfo
    engine: EngineInfo
    model: ModelInfo
    hardware: HardwareInfo
    workload: WorkloadInfo
    artifacts: ArtifactPaths
    comparison_key: dict[str, Any]
    health: HealthEvidence | None = None
    stream: StreamEvidence | None = None
    shutdown: ShutdownEvidence | None = None


class TraceEvent(ContractModel):
    schema_version: Literal[1]
    contract_version: Literal["inferno.artifacts.v1"]
    run_id: str
    request_id: str
    event: Literal["request_start", "stream_event", "request_end"]
    perf_counter_ns: int = Field(ge=1)
    request: dict[str, Any] | None = None
    request_sha256: str | None = None
    raw_sse: str | None = None
    stderr: str | None = None
    ssh_exit_code: int | None = None
    generated_text_sha256: str | None = None
    generated_text_chars: int | None = None
    usage: dict[str, Any] | None = None
    sse_index: int | None = None
    delta_text_sha256: str | None = None
    delta_text_chars: int | None = None


class TelemetrySample(ContractModel):
    schema_version: Literal[1]
    contract_version: Literal["inferno.artifacts.v1"]
    run_id: str
    sample_index: int = Field(ge=0)
    collected_at_unix_ns: int = Field(ge=1)
    source: str = Field(min_length=1)
    gpu_utilization_percent: float | None = None
    gpu_utilization_percent_missing_reason: str | None = None
    gpu_memory_used_mib: float | None = None
    gpu_memory_used_mib_missing_reason: str | None = None
    gpu_memory_total_mib: float | None = None
    gpu_memory_total_mib_missing_reason: str | None = None
    gpu_power_watts: float | None = None
    gpu_power_watts_missing_reason: str | None = None
    gpu_temperature_c: float | None = None
    gpu_temperature_c_missing_reason: str | None = None
    host_cpu_percent: float | None = None
    host_cpu_percent_missing_reason: str | None = None
    host_rss_mib: float | None = None
    host_rss_mib_missing_reason: str | None = None


class RunSummary(ContractModel):
    schema_version: Literal[1]
    contract_version: Literal["inferno.artifacts.v1"]
    run_id: str
    parser_version: Literal["inferno.sse.v1"]
    event_granularity: Literal["sse_data_line"]
    request_count: int
    success_count: int
    error_count: int
    ttft_ms: float | None
    e2e_ms: float
    itl_ms: list[float] = Field(default_factory=list)
    generated_text_sha256: str = Field(min_length=64, max_length=64)
    generated_text_chars: int
    completion_tokens: int | None = None
    completion_tokens_missing_reason: str | None = None
    request_throughput_rps: float = 0.0
    output_token_throughput_tps: float | None = None
    output_token_throughput_missing_reason: str | None = None
    error_rate: float = 0.0
    stream_event_count: int
    non_empty_stream_event_count: int
    comparison_key: dict[str, Any]


class ValidationResult(ContractModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    contract_version: Literal["inferno.artifacts.v1"] = CONTRACT_VERSION
    run_id: str
    ok: bool
    errors: list[str] = Field(default_factory=list)


def default_artifacts(engine: str = "vllm") -> ArtifactPaths:
    return ArtifactPaths(
        manifest="manifest.json",
        request_trace="request_trace.jsonl",
        raw_sse=(
            "raw_streams/sglang_native.sse"
            if engine == "sglang"
            else "raw_streams/llamacpp_native.sse"
            if engine == "llamacpp"
            else "raw_streams/ollama_native.sse"
            if engine == "ollama"
            else "raw_streams/tensorrtllm_native.sse"
            if engine in {"tensorrtllm", "tensorrtllm_t4"}
            else "raw_streams/stream.sse"
        ),
        engine_logs=f"engine_logs/{engine}.log",
        health="health.json",
        summary="summary.json",
        validation="validation.json",
        checksums="checksums.sha256",
        launch_provenance="command_logs/up.json",
    )


def artifact_path(run_dir: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else run_dir / candidate


def write_json(path: Path, payload: BaseModel | Mapping[str, Any]) -> None:
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8-sig"))


def read_trace(run_dir: Path, manifest: RunManifest) -> list[TraceEvent]:
    path = artifact_path(run_dir, manifest.artifacts.request_trace)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(TraceEvent.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"invalid trace row {line_number}: {exc}") from exc
    return rows


def comparison_key(manifest: RunManifest) -> dict[str, Any]:
    args = manifest.engine.launch_args
    weight_format = _weight_format(manifest, args)
    model_key = {
        "id": manifest.model.id,
        "revision": manifest.model.revision,
        "tokenizer_id": manifest.model.tokenizer_id,
        "tokenizer_revision": manifest.model.tokenizer_revision,
    }
    if manifest.model.tokenizer_format or weight_format == "gguf":
        model_key["tokenizer_format"] = manifest.model.tokenizer_format or _tokenizer_format(
            weight_format
        )
    weights_key = {
        "format": weight_format,
        "precision": manifest.model.weight_precision or _launch_arg(args, "--dtype"),
        "state_precision": _state_precision(args),
        "conv_state_precision": _conv_state_precision(manifest.engine.env, args),
        "quantization": manifest.model.quantization or _launch_arg(args, "--quantization"),
        "trust_remote_code": "--trust-remote-code" in args,
    }
    if (
        weight_format == "gguf"
        or manifest.model.gguf_repo
        or manifest.model.gguf_filename
        or manifest.model.gguf_revision
    ):
        weights_key.update(
            {
                "gguf_repo": manifest.model.gguf_repo,
                "gguf_filename": manifest.model.gguf_filename,
                "gguf_revision": manifest.model.gguf_revision,
            }
        )
    runtime_key: dict[str, Any] = {
        "context_limit": _int_or_none(
            _launch_arg(args, "--max-model-len")
            or _launch_arg(args, "--context-length")
            or _launch_arg(args, "--ctx-size")
            or _launch_arg(args, "-c")
            or _launch_arg(args, "--max_seq_len")
        ),
        "language_only": "--language-model-only" in args or "--language-only" in args,
        "cuda_graph_decode": _cuda_graph_decode(args),
        "max_concurrent_requests": _int_or_none(
            _launch_arg(args, "--max-num-seqs")
            or _launch_arg(args, "--max-running-requests")
            or _launch_arg(args, "--parallel")
            or _launch_arg(args, "-np")
            or _launch_arg(args, "--max_batch_size")
        ),
        "memory_fraction": _float_or_none(
            _launch_arg(args, "--gpu-memory-utilization")
            or _launch_arg(args, "--mem-fraction-static")
            or _launch_arg(args, "--free_gpu_memory_fraction")
            or _launch_arg(args, "--kv_cache_free_gpu_memory_fraction")
        ),
    }
    for name, value in {
        "tensor_parallel_size": _int_or_none(
            _launch_arg(args, "--tensor_parallel_size") or _launch_arg(args, "--tp_size")
        ),
        "pipeline_parallel_size": _int_or_none(
            _launch_arg(args, "--pipeline_parallel_size") or _launch_arg(args, "--pp_size")
        ),
        "kv_cache_dtype": _launch_arg(args, "--kv_cache_dtype"),
        "backend": _launch_arg(args, "--backend"),
    }.items():
        if value is not None:
            runtime_key[name] = value
    hf_revision = _launch_arg(args, "--hf_revision") or _launch_arg(args, "--revision")
    if hf_revision is not None and (
        manifest.engine.name in {"tensorrtllm", "tensorrtllm_t4"}
        or "hf_revision" in (manifest.comparison_key.get("runtime") or {})
    ):
        runtime_key["hf_revision"] = hf_revision
    if manifest.workload.runtime_knobs:
        runtime_key["engine_runtime_knobs"] = manifest.workload.runtime_knobs

    workload_key: dict[str, Any] = {
        "workload_id": manifest.workload.workload_id,
        "prompt_template_id": manifest.workload.prompt_template_id,
        "prompt_sha256": manifest.workload.prompt_sha256,
        "prompt_chars": manifest.workload.prompt_chars,
        "seed": manifest.workload.seed,
        "arrival_model": manifest.workload.arrival_model,
        "cache_state": manifest.workload.cache_state,
        "warmup_requests": manifest.workload.warmup_requests,
        "measurement_requests": manifest.workload.measurement_requests,
        "max_runtime_seconds": manifest.workload.max_runtime_seconds,
        "retry_policy": manifest.workload.retry_policy,
        "failure_policy": manifest.workload.failure_policy,
    }
    for name, value in {
        "request_rate_rps": manifest.workload.request_rate_rps,
        "concurrency": manifest.workload.concurrency,
        "max_in_flight": manifest.workload.max_in_flight,
        "prompt_target": manifest.workload.prompt_target,
        "output_target": manifest.workload.output_target,
        "context_length": manifest.workload.context_length,
        "cache_policy": manifest.workload.cache_policy,
        "streaming": manifest.workload.streaming,
        "timeout_seconds": manifest.workload.timeout_seconds,
        "quality_requirement": manifest.workload.quality_requirement,
    }.items():
        if value not in (None, {}, []):
            workload_key[name] = value

    key = {
        "schema_version": SCHEMA_VERSION,
        "model": model_key,
        "weights": weights_key,
        "hardware": manifest.hardware.model_dump(mode="json"),
        "runtime": runtime_key,
        "workload": workload_key,
        "sampling": manifest.workload.sampling,
    }
    if manifest.workload.cost_metadata:
        key["cost_metadata"] = manifest.workload.cost_metadata
    return key


def recompute_summary(run_dir: Path, manifest: RunManifest | None = None) -> RunSummary:
    manifest = read_manifest(run_dir) if manifest is None else manifest
    trace = read_trace(run_dir, manifest)
    assert_monotonic_trace(trace)
    raw_sse = artifact_path(run_dir, manifest.artifacts.raw_sse).read_text(encoding="utf-8-sig")
    generated_events = generated_text_events(raw_sse)
    generated = "".join(text for _, text in generated_events)
    starts = [event for event in trace if event.event == "request_start"]
    ends = [event for event in trace if event.event == "request_end"]
    if not starts or not ends:
        raise ValueError("trace missing request_start or request_end events")
    end_by_id = {event.request_id: event for event in ends}
    text_trace_events = _text_trace_events(trace)
    itl_ms = [
        round((right.perf_counter_ns - left.perf_counter_ns) / 1_000_000, 6)
        for left, right in zip(text_trace_events, text_trace_events[1:])
    ]
    ttfts = []
    e2es = []
    successes = 0
    token_values = []
    for start in starts:
        end = end_by_id.get(start.request_id)
        if not end:
            continue
        first_text_event = next(
            (
                event
                for event in text_trace_events
                if event.request_id == start.request_id
            ),
            None,
        )
        if first_text_event:
            ttfts.append(round((first_text_event.perf_counter_ns - start.perf_counter_ns) / 1_000_000, 6))
        e2es.append(round((end.perf_counter_ns - start.perf_counter_ns) / 1_000_000, 6))
        if end.ssh_exit_code == 0 and (end.generated_text_chars or 0) > 0:
            successes += 1
        usage = end.usage or {}
        tokens = _int_or_none_any(usage.get("completion_tokens") or usage.get("output_tokens"))
        if tokens is not None:
            token_values.append(tokens)
    raw_usage = stream_usage(raw_sse) or {}
    completion_tokens = sum(token_values) if token_values else _int_or_none_any(
        raw_usage.get("completion_tokens") or raw_usage.get("output_tokens")
    )
    output_tps = None
    output_tps_reason = None
    if completion_tokens is None:
        output_tps_reason = "stream_usage_missing"
    elif not text_trace_events:
        output_tps_reason = "first_text_missing"
    else:
        generation_seconds = (max(end.perf_counter_ns for end in ends) - min(event.perf_counter_ns for event in text_trace_events)) / 1_000_000_000
        if generation_seconds <= 0:
            output_tps_reason = "non_positive_generation_duration"
        else:
            output_tps = round(completion_tokens / generation_seconds, 6)
    request_seconds = (max(end.perf_counter_ns for end in ends) - min(start.perf_counter_ns for start in starts)) / 1_000_000_000
    request_count = len(starts)
    error_count = request_count - successes
    return RunSummary(
        schema_version=SCHEMA_VERSION,
        contract_version=CONTRACT_VERSION,
        run_id=manifest.run_id,
        parser_version=PARSER_VERSION,
        event_granularity="sse_data_line",
        request_count=request_count,
        success_count=successes,
        error_count=error_count,
        ttft_ms=round(sum(ttfts) / len(ttfts), 6) if ttfts else None,
        e2e_ms=round(sum(e2es) / len(e2es), 6) if e2es else 0.0,
        itl_ms=itl_ms,
        generated_text_sha256=sha256_text(generated),
        generated_text_chars=len(generated),
        completion_tokens=completion_tokens,
        completion_tokens_missing_reason="stream_usage_missing" if completion_tokens is None else None,
        request_throughput_rps=round(successes / request_seconds, 6)
        if request_seconds > 0
        else 0.0,
        output_token_throughput_tps=output_tps,
        output_token_throughput_missing_reason=output_tps_reason,
        error_rate=round(error_count / request_count, 6) if request_count else 1.0,
        stream_event_count=len(generated_events),
        non_empty_stream_event_count=sum(1 for _, text in generated_events if text),
        comparison_key=comparison_key(manifest),
    )


def validate_run(run_dir: Path, *, write: bool = True) -> ValidationResult:
    errors: list[str] = []
    run_id = run_dir.name
    manifest: RunManifest | None = None
    try:
        manifest = read_manifest(run_dir)
        run_id = manifest.run_id
    except Exception as exc:
        errors.append(f"invalid manifest: {exc}")

    if manifest:
        errors.extend(_validate_manifest(run_dir, manifest))
        try:
            summary_text = artifact_path(run_dir, manifest.artifacts.summary).read_text(
                encoding="utf-8-sig"
            )
            summary_payload = json.loads(summary_text)
            summary = RunSummary.model_validate(summary_payload)
            recomputed = recompute_summary(run_dir, manifest)
            summary_dump = summary.model_dump(mode="json")
            recomputed_dump = recomputed.model_dump(mode="json")
            if manifest.run_family == "single_run":
                for field in STUDY_SUMMARY_FIELDS - set(summary_payload):
                    recomputed_dump[field] = summary_dump[field]
            if summary_dump != recomputed_dump:
                errors.append("summary does not recompute from raw artifacts")
        except Exception as exc:
            errors.append(f"invalid summary: {exc}")
        errors.extend(_validate_study_artifacts(run_dir, manifest))
        errors.extend(_validate_checksums(run_dir, manifest))

    result = ValidationResult(run_id=run_id, ok=not errors, errors=errors)
    if write:
        target = (
            artifact_path(run_dir, manifest.artifacts.validation)
            if manifest
            else run_dir / "validation.json"
        )
        write_json(target, result)
    return result


def strict_comparison_errors(manifests: list[RunManifest]) -> list[str]:
    if len(manifests) < 2:
        return ["strict comparison requires at least two runs"]
    errors: list[str] = []
    engine_names = {manifest.engine.name for manifest in manifests}
    if "llamacpp" in engine_names:
        errors.append("strict comparison rejects llamacpp; use MODE=deployment-profile")
    if "ollama" in engine_names:
        errors.append("strict comparison rejects ollama; use MODE=deployment-profile")
    if engine_names - {"vllm", "sglang"}:
        errors.append(f"strict comparison supports only vllm and sglang, got {sorted(engine_names)}")
    baseline = comparison_key(manifests[0])
    for manifest in manifests:
        actual = comparison_key(manifest)
        if manifest.comparison_key != actual:
            errors.append(f"{manifest.run_id}: comparison key does not match controlled fields")
            continue
        if actual != baseline:
            fields = ", ".join(_diff_paths(baseline, actual))
            errors.append(f"{manifest.run_id}: controlled key mismatch: {fields}")
    return errors


def write_checksums(run_dir: Path, manifest: RunManifest) -> None:
    lines = []
    for path in checksum_paths(manifest):
        target = artifact_path(run_dir, path)
        lines.append(f"{sha256_bytes(target.read_bytes())}  {path}")
    artifact_path(run_dir, manifest.artifacts.checksums).write_text(
        "\n".join(sorted(lines)) + "\n", encoding="utf-8"
    )


def checksum_paths(manifest: RunManifest) -> list[str]:
    excluded = {"manifest", "validation", "checksums"}
    paths = [
        value
        for key, value in manifest.artifacts.model_dump(mode="json").items()
        if key not in excluded and value
    ]
    return sorted(str(path) for path in paths)


def write_trace_parquet(run_dir: Path, manifest: RunManifest) -> None:
    if not manifest.artifacts.request_trace_parquet:
        return
    rows = [event.model_dump(mode="json") for event in read_trace(run_dir, manifest)]
    _write_parquet(artifact_path(run_dir, manifest.artifacts.request_trace_parquet), rows)


def write_telemetry_parquet(
    run_dir: Path,
    manifest: RunManifest,
    rows: list[Mapping[str, Any]],
) -> None:
    if not manifest.artifacts.telemetry_parquet:
        return
    samples = [
        TelemetrySample.model_validate(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_version": CONTRACT_VERSION,
                "run_id": manifest.run_id,
                **dict(row),
            }
        ).model_dump(mode="json")
        for row in rows
    ]
    _validate_telemetry_rows(samples)
    _write_parquet(artifact_path(run_dir, manifest.artifacts.telemetry_parquet), samples)


def assert_monotonic_trace(trace: list[TraceEvent]) -> None:
    previous = 0
    for event in trace:
        if event.perf_counter_ns < previous:
            raise ValueError("trace timestamps are non-monotonic")
        previous = event.perf_counter_ns


def generated_text_events(raw_sse: str) -> list[tuple[int, str]]:
    if "data: " not in raw_sse:
        try:
            payload = json.loads(raw_sse)
        except json.JSONDecodeError:
            return []
        if isinstance(payload.get("responses"), list):
            return [
                (index, str(item.get("generated_text") or ""))
                for index, item in enumerate(payload["responses"])
            ]
        text = _generated_text_from_json(raw_sse)
        return [(0, text)] if text else []
    events = []
    for line in raw_sse.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        payload = json.loads(data)
        choice = (payload.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        text = delta.get("content") or delta.get("reasoning_content") or choice.get("text") or ""
        events.append((len(events), text))
    return events


def stream_usage(raw_sse: str) -> dict[str, Any] | None:
    if "data: " not in raw_sse:
        return _usage_from_json(raw_sse)
    usage = None
    for line in raw_sse.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        payload = json.loads(data)
        if payload.get("usage") is not None:
            usage = dict(payload["usage"])
    return usage


def _generated_text_from_json(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload.get("responses"), list):
        return "".join(str(item.get("generated_text") or "") for item in payload["responses"])
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content") or choice.get("text") or ""


def _usage_from_json(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(payload.get("responses"), list):
        total = 0
        seen = False
        for item in payload["responses"]:
            usage = item.get("usage") or {}
            value = _int_or_none_any(usage.get("completion_tokens") or usage.get("output_tokens"))
            if value is not None:
                total += value
                seen = True
        return {"completion_tokens": total} if seen else None
    usage = payload.get("usage")
    return dict(usage) if usage else None


def schema_snapshots() -> dict[str, str]:
    models: dict[str, type[BaseModel]] = {
        "manifest.v1.schema.json": RunManifest,
        "request_trace_event.v1.schema.json": TraceEvent,
        "telemetry_sample.v1.schema.json": TelemetrySample,
        "summary.v1.schema.json": RunSummary,
        "validation.v1.schema.json": ValidationResult,
    }
    return {
        name: json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        for name, model in models.items()
    }


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_manifest(run_dir: Path, manifest: RunManifest) -> list[str]:
    errors = []
    if manifest.status != "SUCCEEDED":
        errors.append(f"run status is {manifest.status}, not SUCCEEDED")
    if not manifest.manifest_created_before_request:
        errors.append("manifest was not declared before request")
    if manifest.first_request_perf_counter_ns is None:
        errors.append("first request timestamp is missing")
    elif manifest.manifest_written_perf_counter_ns > manifest.first_request_perf_counter_ns:
        errors.append("manifest timestamp is after first request")
    if not _is_pinned_real_engine(manifest.engine):
        errors.append("engine evidence is not pinned real engine evidence")
    if "unknown" in {
        manifest.hardware.gpu_model,
        manifest.hardware.driver_version,
        manifest.hardware.cuda_version,
    }:
        errors.append("hardware provenance is incomplete")
    if not manifest.health or not manifest.health.ok:
        errors.append("health evidence did not pass")
    elif manifest.engine.served_model_name not in manifest.health.model_ids:
        errors.append("model identity check failed")
    if not manifest.stream or not manifest.stream.ok or manifest.stream.generated_text_chars <= 0:
        errors.append("stream evidence did not produce text")
    if not manifest.shutdown or not manifest.shutdown.ok:
        errors.append("shutdown evidence did not pass")
    if manifest.comparison_key != comparison_key(manifest):
        errors.append("comparison key does not match controlled fields")
    for name, path in manifest.artifacts.model_dump(mode="json").items():
        if name == "validation" or path is None:
            continue
        if not artifact_path(run_dir, path).exists():
            errors.append(f"missing artifact: {path}")
    return errors


def _validate_study_artifacts(run_dir: Path, manifest: RunManifest) -> list[str]:
    if manifest.run_family not in {"smoke_study", "strict_comparison", "research_core", "deployment_profile", "engine_configuration", "dashboard"}:
        return []
    errors = []
    if not manifest.study_id:
        errors.append(f"{manifest.run_family} manifest missing study_id")
    if manifest.repeat_index is None:
        errors.append(f"{manifest.run_family} manifest missing repeat_index")
    if not manifest.artifacts.request_trace_parquet:
        errors.append(f"{manifest.run_family} manifest missing request_trace.parquet path")
    if not manifest.artifacts.telemetry_parquet:
        errors.append(f"{manifest.run_family} manifest missing telemetry.parquet path")
    if manifest.artifacts.request_trace_parquet:
        try:
            table = pq.read_table(artifact_path(run_dir, manifest.artifacts.request_trace_parquet))
            if table.num_rows != len(read_trace(run_dir, manifest)):
                errors.append("request_trace.parquet row count does not match JSONL trace")
        except Exception as exc:
            errors.append(f"invalid request_trace.parquet: {exc}")
    if manifest.artifacts.telemetry_parquet:
        try:
            rows = pq.read_table(artifact_path(run_dir, manifest.artifacts.telemetry_parquet)).to_pylist()
            _validate_telemetry_rows(rows)
        except Exception as exc:
            errors.append(f"invalid telemetry.parquet: {exc}")
    return errors


def _validate_checksums(run_dir: Path, manifest: RunManifest) -> list[str]:
    errors = []
    checksum_file = artifact_path(run_dir, manifest.artifacts.checksums)
    if not checksum_file.exists():
        return ["missing checksums.sha256"]
    recorded = {}
    for line in checksum_file.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        recorded[path] = digest
    for path in checksum_paths(manifest):
        target = artifact_path(run_dir, path)
        if not target.exists():
            errors.append(f"missing checksum target: {path}")
        elif recorded.get(path) != sha256_bytes(target.read_bytes()):
            errors.append(f"checksum mismatch: {path}")
    return errors


def _write_parquet(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows) or [{"schema_version": SCHEMA_VERSION}])
    pq.write_table(table, path)


def _validate_telemetry_rows(rows: list[Mapping[str, Any]]) -> None:
    metrics = (
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_power_watts",
        "gpu_temperature_c",
        "host_cpu_percent",
        "host_rss_mib",
    )
    for row_number, row in enumerate(rows, start=1):
        TelemetrySample.model_validate(row)
        for metric in metrics:
            reason = row.get(f"{metric}_missing_reason")
            if row.get(metric) is None and not reason:
                raise ValueError(f"telemetry row {row_number} missing reason for {metric}")
            if row.get(metric) is not None and reason:
                raise ValueError(f"telemetry row {row_number} has value and reason for {metric}")


def _one(trace: list[TraceEvent], event_name: str) -> TraceEvent:
    matches = [event for event in trace if event.event == event_name]
    if len(matches) != 1:
        raise ValueError(f"expected one {event_name} event, found {len(matches)}")
    return matches[0]


def _text_trace_events(trace: list[TraceEvent]) -> list[TraceEvent]:
    return [
        event
        for event in trace
        if event.event == "stream_event" and event.delta_text_chars and event.delta_text_chars > 0
    ]


def _is_pinned_real_engine(engine: EngineInfo) -> bool:
    if "@sha256:" not in engine.image:
        return False
    if engine.name == "vllm":
        return engine.image.startswith("vllm/vllm-openai@sha256:")
    if engine.name == "sglang":
        return engine.image.startswith("lmsysorg/sglang@sha256:")
    if engine.name == "llamacpp":
        return engine.image.startswith("ghcr.io/ggml-org/llama.cpp:")
    if engine.name == "tensorrtllm":
        return engine.image.startswith("nvcr.io/nvidia/tensorrt-llm/release:")
    if engine.name == "tensorrtllm_t4":
        return engine.image.startswith(
            ("nvcr.io/nvidia/tensorrt-llm/release:", "nvcr.io/nvidia/tritonserver:")
        )
    if engine.name == "ollama":
        return engine.image.startswith("ollama/ollama@sha256:")
    return False


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


def _launch_arg(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for arg in args:
        if arg.startswith(prefix):
            return arg.removeprefix(prefix)
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _weight_format(manifest: RunManifest, args: list[str]) -> str:
    if manifest.model.weight_format:
        return manifest.model.weight_format
    if _launch_arg(args, "--hf-repo") or _launch_arg(args, "-hf") or _launch_arg(args, "-hfr"):
        return "gguf"
    model_path = _launch_arg(args, "--model") or _launch_arg(args, "-m")
    if model_path and model_path.lower().endswith(".gguf"):
        return "gguf"
    return _launch_arg(args, "--load-format") or "huggingface"


def _tokenizer_format(weight_format: str) -> str:
    return "gguf_embedded" if weight_format == "gguf" else "huggingface"


def _cuda_graph_decode(args: list[str]) -> str:
    if "--enforce-eager" in args:
        return "disabled"
    if _launch_arg(args, "--cuda-graph-backend-decode") == "disabled":
        return "disabled"
    return "enabled_or_default"


def _state_precision(args: list[str]) -> str | None:
    value = _launch_arg(args, "--mamba-ssm-dtype") or _launch_arg(args, "--dtype")
    return {"float16": "half", "fp16": "half"}.get(value or "", value)


def _conv_state_precision(env: list[str], args: list[str]) -> str | None:
    value = _env_arg(env, "SGLANG_MAMBA_CONV_DTYPE") or _state_precision(args)
    return {"float16": "half", "fp16": "half"}.get(value or "", value)


def _env_arg(env: list[str], name: str) -> str | None:
    prefix = name + "="
    for item in env:
        if item.startswith(prefix):
            return item.removeprefix(prefix)
    return None


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value is not None else None


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value is not None else None


def _int_or_none_any(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
