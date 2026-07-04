"""In-memory dashboard job runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any

import pyarrow.parquet as pq

from inferno import artifact_contract as contract
from inferno import vllm_remote
from inferno.dashboard.compat import GPU_ENV_VAR, gpu_preflight


Runner = Callable[..., vllm_remote.RunRealResult]
ACTIVE_STATES = {"queued", "running"}


@dataclass
class DashboardJob:
    job_id: str
    requested_engines: list[str]
    parameters: dict[str, Any]
    created_at: str
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    gpu: dict[str, Any] | None = None
    cancel_requested: bool = False
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "requested_engines": self.requested_engines,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "status": self.status,
            "events": self.events,
            "results": self.results,
            "gpu": self.gpu,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
        }


class DashboardJobManager:
    def __init__(
        self,
        *,
        project_root: Path,
        runner: Runner | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.project_root = project_root
        self.runner = runner or vllm_remote.run_real_result
        self.env = dict(os.environ if env is None else env)
        self._jobs: dict[str, DashboardJob] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        *,
        engines: list[str],
        parameters: Mapping[str, Any],
        ssh_target: str | None = None,
    ) -> DashboardJob:
        job = DashboardJob(
            job_id=f"p11-dashboard-{uuid.uuid4().hex[:8]}",
            requested_engines=engines,
            parameters=dict(parameters),
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            active = next((item for item in self._jobs.values() if item.status in ACTIVE_STATES), None)
            if active:
                raise RuntimeError(f"Job {active.job_id} is already running")
            self._jobs[job.job_id] = job
        thread = threading.Thread(
            target=self._run_job,
            args=(job.job_id, ssh_target),
            name=f"inferno-dashboard-{job.job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> DashboardJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> DashboardJob | None:
        job = self.get_job(job_id)
        if not job:
            return None
        job.cancel_requested = True
        self._event(job, "cancel_requested", "Cancel requested; current remote command may finish first.")
        return job

    def _run_job(self, job_id: str, ssh_target: str | None) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        env = dict(self.env)
        if ssh_target:
            env[GPU_ENV_VAR] = ssh_target
        env["INFERNO_ENGINE_HEALTH_TIMEOUT"] = str(_dashboard_health_timeout(job.parameters))
        try:
            job.status = "running"
            self._event(job, "job_started", "Dashboard job started.")
            preflight = gpu_preflight(env=env, project_root=self.project_root)
            job.gpu = preflight.get("gpu")
            recommendations = {
                item["engine"]: item for item in preflight.get("recommendations", [])
            }
            self._event(job, "gpu_preflight", "GPU compatibility checked.", {"preflight": preflight})
            if not env.get(GPU_ENV_VAR):
                raise RuntimeError("INFERNO_GPU_SSH is required to run dashboard jobs")

            for requested in job.requested_engines:
                if job.cancel_requested:
                    job.status = "cancelled"
                    self._event(job, "job_cancelled", "Job cancelled before next engine.")
                    return
                engine = self._selected_engine(requested, recommendations)
                workload_path = self._write_workload(job, requested, engine)
                self._event(
                    job,
                    "engine_started",
                    f"Running {engine}.",
                    {"requested_engine": requested, "engine": engine, "workload": str(workload_path)},
                )
                self._event(
                    job,
                    "engine_readiness",
                    f"Waiting up to {env['INFERNO_ENGINE_HEALTH_TIMEOUT']}s for {engine} API readiness.",
                    {"engine": engine, "health_timeout_seconds": int(env["INFERNO_ENGINE_HEALTH_TIMEOUT"])},
                )
                result = self.runner(
                    engine=engine,
                    config_path=workload_path.relative_to(self.project_root),
                    project_root=self.project_root,
                    env=env,
                    phase="P11",
                    study_id=f"dashboard_{requested}",
                    repeat_index=1,
                    telemetry_cadence_seconds=0.5,
                )
                result_payload = self._result_payload(requested, engine, result)
                job.results.append(result_payload)
                error_hint = result_payload.get("failure_hint") or (result_payload.get("validation_errors") or [None])[0]
                message = f"{engine} finished with exit code {result.returncode}."
                if error_hint:
                    message = f"{message} {error_hint}"
                self._event(
                    job,
                    "engine_finished",
                    message,
                    {"engine": engine, "run_dir": str(result.run_dir), "validation_ok": result.validation_ok},
                )
            job.status = "succeeded" if all(item["validation_ok"] for item in job.results) else "failed"
            self._event(job, "job_finished", f"Dashboard job {job.status}.")
        except Exception as exc:  # pragma: no cover - defensive top-level job guard
            job.status = "failed"
            job.error = str(exc)
            self._event(job, "job_failed", str(exc))

    def _selected_engine(self, requested: str, recommendations: Mapping[str, Mapping[str, Any]]) -> str:
        if requested == "tensorrtllm":
            profile = recommendations.get("tensorrtllm", {}).get("selected_profile")
            return str(profile or "tensorrtllm")
        return requested

    def _write_workload(self, job: DashboardJob, requested: str, engine: str) -> Path:
        source = self._base_workload(engine)
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        prompt = str(job.parameters.get("prompt") or payload["prompt"])
        batch_prompts = [
            line.strip()
            for line in str(job.parameters.get("batch_prompts") or "").splitlines()
            if line.strip()
        ] or [prompt]
        requested_max_tokens = _int_range(
            job.parameters.get("max_tokens"),
            1,
            1024,
            payload["sampling"].get("max_tokens") or 64,
        )
        max_tokens = min(requested_max_tokens, _dashboard_max_tokens(engine))
        total_requests = _int_range(job.parameters.get("total_requests"), 1, 128, payload["measurement"].get("requests") or 1)
        concurrent_users = _int_range(job.parameters.get("concurrent_users"), 1, 16, payload.get("concurrency") or 1)
        warmup_requests = _int_range(job.parameters.get("warmup_requests"), 0, 32, payload.get("warmup", {}).get("requests") or 1)
        temperature = _float_range(job.parameters.get("temperature"), 0, 2, payload["sampling"].get("temperature") or 0)
        top_p = _float_range(job.parameters.get("top_p"), 0, 1, payload["sampling"].get("top_p") or 1)
        request_rate_rps = _float_range(job.parameters.get("request_rate_rps"), 0.01, 100, payload.get("request_rate_rps") or 0.1)
        timeout_seconds = _int_range(
            job.parameters.get("timeout_seconds"),
            30,
            3600,
            payload["measurement"].get("max_runtime_seconds") or 600,
        )
        model_profile = str(job.parameters.get("model_profile") or "qwen35-2b")
        kv_cache_mode = str(job.parameters.get("kv_cache_mode") or "engine_default")
        payload["workload_id"] = f"dashboard_{requested}"
        payload["prompt_template_id"] = "p11-dashboard-chat-v1"
        payload["prompt"] = "\n---\n".join(batch_prompts)
        payload["cache_state"] = "warm_after_warmup" if warmup_requests else "cold_per_run"
        payload["request_rate_rps"] = request_rate_rps
        payload["concurrency"] = concurrent_users
        payload["max_in_flight"] = concurrent_users
        payload["sampling"]["max_tokens"] = max_tokens
        payload["sampling"]["top_p"] = top_p
        payload["sampling"]["temperature"] = temperature
        payload["streaming"] = True
        payload.setdefault("warmup", {})["requests"] = warmup_requests
        payload["measurement"]["requests"] = total_requests
        payload["measurement"]["max_runtime_seconds"] = timeout_seconds
        payload["measurement"]["timeout_seconds"] = timeout_seconds
        payload["timeout_seconds"] = timeout_seconds
        payload.setdefault("runtime_knobs", {})["dashboard_requested_engine"] = requested
        payload["runtime_knobs"]["dashboard_model_profile"] = model_profile
        payload["runtime_knobs"]["dashboard_batch_prompts"] = batch_prompts
        payload["runtime_knobs"]["dashboard_requested_max_tokens"] = requested_max_tokens
        payload["runtime_knobs"]["dashboard_effective_max_tokens"] = max_tokens
        payload["runtime_knobs"]["dashboard_kv_cache_mode"] = kv_cache_mode
        path = self.project_root / "artifacts" / "dashboard" / "jobs" / job.job_id / "workloads" / f"{engine}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _base_workload(self, engine: str) -> Path:
        if engine == "ollama":
            return self.project_root / "configs" / "workloads" / "ollama_smoke.yaml"
        if engine == "tensorrtllm":
            return self.project_root / "configs" / "workloads" / "tensorrtllm_smoke.yaml"
        if engine == "tensorrtllm_t4":
            return self.project_root / "configs" / "workloads" / "tensorrtllm_t4_smoke.yaml"
        return self.project_root / "configs" / "workloads" / "smoke_real.yaml"

    def _result_payload(
        self,
        requested: str,
        engine: str,
        result: vllm_remote.RunRealResult,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_engine": requested,
            "engine": engine,
            "exit_code": result.returncode,
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "validation_ok": result.validation_ok,
            "artifacts": {},
            "summary": None,
            "validation_errors": [],
            "failure_hint": None,
        }
        try:
            manifest = contract.read_manifest(result.run_dir)
            validation = contract.validate_run(result.run_dir, write=False)
            summary = contract.RunSummary.model_validate_json(
                contract.artifact_path(result.run_dir, manifest.artifacts.summary).read_text(
                    encoding="utf-8-sig"
                )
            )
            payload["validation_errors"] = validation.errors
            payload["artifacts"] = {
                "manifest": str(contract.artifact_path(result.run_dir, manifest.artifacts.manifest)),
                "summary": str(contract.artifact_path(result.run_dir, manifest.artifacts.summary)),
                "validation": str(contract.artifact_path(result.run_dir, manifest.artifacts.validation)),
                "request_trace": str(contract.artifact_path(result.run_dir, manifest.artifacts.request_trace)),
                "raw_sse": str(contract.artifact_path(result.run_dir, manifest.artifacts.raw_sse)),
            }
            payload["summary"] = {
                "ttft_ms": summary.ttft_ms,
                "e2e_ms": summary.e2e_ms,
                "generated_text_chars": summary.generated_text_chars,
                "completion_tokens": summary.completion_tokens,
                "request_count": summary.request_count,
                "success_count": summary.success_count,
                "error_count": summary.error_count,
                "request_throughput_rps": summary.request_throughput_rps,
                "output_token_throughput_tps": summary.output_token_throughput_tps,
                "error_rate": summary.error_rate,
            }
            payload["metrics"] = _dashboard_metrics(result.run_dir, manifest, summary)
            if not result.validation_ok:
                payload["failure_hint"] = _failure_hint(result.run_dir, manifest)
        except Exception as exc:
            payload["validation_errors"] = [str(exc)]
        return payload

    def _event(
        self,
        job: DashboardJob,
        event_type: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        job.events.append(
            {
                "index": len(job.events),
                "time": datetime.now(UTC).isoformat(),
                "type": event_type,
                "message": message,
                "details": dict(details or {}),
            }
        )


def _int_range(value: object, low: int, high: int, default: object) -> int:
    return max(low, min(high, int(value if value not in (None, "") else default)))


def _float_range(value: object, low: float, high: float, default: object) -> float:
    return max(low, min(high, float(value if value not in (None, "") else default)))


def _dashboard_metrics(
    run_dir: Path,
    manifest: contract.RunManifest,
    summary: contract.RunSummary,
) -> dict[str, Any]:
    trace = contract.read_trace(run_dir, manifest)
    requests = _request_metrics(trace)
    telemetry = _telemetry_metrics(run_dir, manifest)
    return {
        "output_tps": _value_metric(
            summary.output_token_throughput_tps,
            summary.output_token_throughput_missing_reason,
            source="summary",
            unit="tokens/s",
        ),
        "ttft_ms": _stats(
            (item["ttft_ms"] for item in requests if item.get("ttft_ms") is not None),
            unit="ms",
            reason="first_text_missing",
            source="request_trace",
        ),
        "tpot_ms": _stats(
            _tpot_values(requests),
            unit="ms/token",
            reason="completion_tokens_or_ttft_missing",
            source="request_trace",
        ),
        "e2e_ms": _stats(
            (item["e2e_ms"] for item in requests if item.get("e2e_ms") is not None),
            unit="ms",
            reason="request_timing_missing",
            source="request_trace",
        ),
        "concurrency": {
            "configured": manifest.workload.concurrency,
            "max_in_flight": manifest.workload.max_in_flight,
            "observed_max_overlap": _observed_max_overlap(requests),
            "source": "manifest+request_trace",
        },
        "gpu_utilization_percent": telemetry["gpu_utilization_percent"],
        "vram_used_mib": telemetry["vram_used_mib"],
        "kv_cache_efficiency": _kv_cache_efficiency(requests),
        "continuous_batching_efficiency": _continuous_batching_efficiency(
            requests,
            manifest.workload.concurrency,
        ),
        "scheduler_efficiency": _scheduler_efficiency(summary, manifest),
        "cache": {
            "state": manifest.workload.cache_state,
            "policy": manifest.workload.cache_policy,
            "warmup_requests": manifest.workload.warmup_requests,
            "kv_cache_mode": (manifest.workload.runtime_knobs or {}).get(
                "dashboard_kv_cache_mode",
                "engine_default",
            ),
        },
    }


def _request_metrics(trace: list[contract.TraceEvent]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in trace:
        item = grouped.setdefault(event.request_id, {"stream_events": []})
        if event.event == "request_start":
            item["start"] = event
        elif event.event == "request_end":
            item["end"] = event
        elif event.event == "stream_event":
            item["stream_events"].append(event)

    rows = []
    for request_id, item in sorted(grouped.items()):
        start = item.get("start")
        end = item.get("end")
        if not start or not end:
            continue
        first_text = next(
            (
                event
                for event in item["stream_events"]
                if event.delta_text_chars and event.delta_text_chars > 0
            ),
            None,
        )
        usage = end.usage or {}
        e2e_ms = (end.perf_counter_ns - start.perf_counter_ns) / 1_000_000
        ttft_ms = (
            (first_text.perf_counter_ns - start.perf_counter_ns) / 1_000_000
            if first_text
            else None
        )
        rows.append(
            {
                "request_id": request_id,
                "start_ns": start.perf_counter_ns,
                "end_ns": end.perf_counter_ns,
                "e2e_ms": e2e_ms,
                "ttft_ms": ttft_ms,
                "completion_tokens": _int_or_none(
                    usage.get("completion_tokens") or usage.get("output_tokens")
                ),
                "prompt_tokens": _int_or_none(
                    usage.get("prompt_tokens") or usage.get("input_tokens")
                ),
                "cached_prompt_tokens": _cached_prompt_tokens(usage),
                "usage": usage,
            }
        )
    return rows


def _telemetry_metrics(run_dir: Path, manifest: contract.RunManifest) -> dict[str, Any]:
    if not manifest.artifacts.telemetry_parquet:
        unavailable = _stats((), unit="%", reason="telemetry_missing", source="telemetry")
        return {
            "gpu_utilization_percent": unavailable,
            "vram_used_mib": _stats((), unit="MiB", reason="telemetry_missing", source="telemetry"),
        }
    try:
        rows = pq.read_table(
            contract.artifact_path(run_dir, manifest.artifacts.telemetry_parquet)
        ).to_pylist()
    except Exception:
        return {
            "gpu_utilization_percent": _stats(
                (),
                unit="%",
                reason="telemetry_unreadable",
                source="telemetry",
            ),
            "vram_used_mib": _stats(
                (),
                unit="MiB",
                reason="telemetry_unreadable",
                source="telemetry",
            ),
        }

    gpu = _stats(
        (row.get("gpu_utilization_percent") for row in rows),
        unit="%",
        reason="gpu_utilization_unavailable",
        source="telemetry",
    )
    used = _stats(
        (row.get("gpu_memory_used_mib") for row in rows),
        unit="MiB",
        reason="gpu_memory_unavailable",
        source="telemetry",
    )
    percent_values = []
    total_values = []
    for row in rows:
        used_mib = _float_or_none(row.get("gpu_memory_used_mib"))
        total_mib = _float_or_none(row.get("gpu_memory_total_mib"))
        if total_mib is not None:
            total_values.append(total_mib)
        if used_mib is not None and total_mib and total_mib > 0:
            percent_values.append(used_mib / total_mib * 100)
    used["percent_of_total"] = _stats(
        percent_values,
        unit="%",
        reason="gpu_memory_total_unavailable",
        source="telemetry",
    )
    used["total_mib"] = round(max(total_values), 6) if total_values else None
    return {"gpu_utilization_percent": gpu, "vram_used_mib": used}


def _tpot_values(requests: list[dict[str, Any]]) -> list[float]:
    values = []
    for item in requests:
        e2e = _float_or_none(item.get("e2e_ms"))
        ttft = _float_or_none(item.get("ttft_ms"))
        tokens = _int_or_none(item.get("completion_tokens"))
        if e2e is None or ttft is None or tokens is None or tokens <= 1:
            continue
        values.append((e2e - ttft) / (tokens - 1))
    return values


def _kv_cache_efficiency(requests: list[dict[str, Any]]) -> dict[str, Any]:
    cached = 0
    prompt = 0
    seen_native_cache_field = False
    for item in requests:
        cached_tokens = _int_or_none(item.get("cached_prompt_tokens"))
        prompt_tokens = _int_or_none(item.get("prompt_tokens"))
        if cached_tokens is None:
            continue
        seen_native_cache_field = True
        cached += cached_tokens
        prompt += prompt_tokens or 0
    if not seen_native_cache_field or prompt <= 0:
        return {
            "status": "unavailable",
            "reason": "native_cached_token_usage_missing",
            "source": "usage.prompt_tokens_details",
        }
    return {
        "status": "ok",
        "value": round(cached / prompt * 100, 6),
        "unit": "%",
        "cached_prompt_tokens": cached,
        "prompt_tokens": prompt,
        "source": "usage.prompt_tokens_details",
    }


def _continuous_batching_efficiency(
    requests: list[dict[str, Any]],
    configured_concurrency: int | None,
) -> dict[str, Any]:
    complete = [item for item in requests if item.get("start_ns") and item.get("end_ns")]
    configured = max(1, int(configured_concurrency or 1))
    if len(complete) < 2 or configured <= 1:
        return {
            "status": "unavailable",
            "reason": "requires_multiple_requests_with_concurrency",
            "source": "request_trace_proxy",
        }
    wall = max(item["end_ns"] for item in complete) - min(item["start_ns"] for item in complete)
    work = sum(item["end_ns"] - item["start_ns"] for item in complete)
    if wall <= 0:
        return {
            "status": "unavailable",
            "reason": "non_positive_wall_time",
            "source": "request_trace_proxy",
        }
    observed_parallelism = work / wall
    return {
        "status": "ok",
        "kind": "proxy",
        "value": round(min(100.0, observed_parallelism / configured * 100), 6),
        "unit": "%",
        "observed_parallelism": round(observed_parallelism, 6),
        "source": "request_trace_proxy",
    }


def _scheduler_efficiency(
    summary: contract.RunSummary,
    manifest: contract.RunManifest,
) -> dict[str, Any]:
    requested = _float_or_none(manifest.workload.request_rate_rps)
    achieved = _float_or_none(summary.request_throughput_rps)
    if requested is None or requested <= 0 or achieved is None:
        return {
            "status": "unavailable",
            "reason": "requested_or_achieved_rps_missing",
            "source": "summary_proxy",
        }
    return {
        "status": "ok",
        "kind": "proxy",
        "value": round(min(100.0, achieved / requested * 100), 6),
        "unit": "%",
        "requested_rps": requested,
        "achieved_rps": achieved,
        "source": "summary_proxy",
    }


def _observed_max_overlap(requests: list[dict[str, Any]]) -> int:
    points = []
    for item in requests:
        if item.get("start_ns") and item.get("end_ns"):
            points.append((int(item["start_ns"]), 1))
            points.append((int(item["end_ns"]), -1))
    active = 0
    highest = 0
    for _, delta in sorted(points, key=lambda item: (item[0], -item[1])):
        active += delta
        highest = max(highest, active)
    return highest


def _stats(
    values: Any,
    *,
    unit: str,
    reason: str,
    source: str,
) -> dict[str, Any]:
    clean = sorted(value for value in (_float_or_none(item) for item in values) if value is not None)
    if not clean:
        return {"status": "unavailable", "reason": reason, "unit": unit, "source": source, "count": 0}
    return {
        "status": "ok",
        "unit": unit,
        "source": source,
        "count": len(clean),
        "mean": round(sum(clean) / len(clean), 6),
        "p50": round(_percentile(clean, 50), 6),
        "p95": round(_percentile(clean, 95), 6),
        "p99": round(_percentile(clean, 99), 6),
        "min": round(clean[0], 6),
        "max": round(clean[-1], 6),
    }


def _value_metric(value: object, reason: str | None, *, source: str, unit: str) -> dict[str, Any]:
    number = _float_or_none(value)
    if number is None:
        return {"status": "unavailable", "reason": reason or "value_missing", "source": source, "unit": unit}
    return {"status": "ok", "value": round(number, 6), "source": source, "unit": unit}


def _percentile(values: list[float], percentile: int) -> float:
    index = max(0, min(len(values) - 1, int((percentile / 100) * len(values) + 0.999999) - 1))
    return values[index]


def _cached_prompt_tokens(usage: Mapping[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details") or usage.get("input_token_details")
    if isinstance(details, Mapping):
        if "cached_tokens" in details:
            return _int_or_none(details.get("cached_tokens"))
        if "cached_prompt_tokens" in details:
            return _int_or_none(details.get("cached_prompt_tokens"))
    return _int_or_none(usage.get("cached_tokens"))


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    if number is None or not (number == number) or number in {float("inf"), float("-inf")}:
        return None
    return number


def _dashboard_max_tokens(engine: str) -> int:
    if engine in {"vllm", "sglang"}:
        return 96
    if engine == "tensorrtllm_t4":
        return 24
    return 64


def _dashboard_health_timeout(parameters: Mapping[str, Any]) -> int:
    requested_timeout = _int_range(parameters.get("timeout_seconds"), 30, 3600, 180)
    return max(90, min(240, requested_timeout))


def _failure_hint(run_dir: Path, manifest: contract.RunManifest) -> str | None:
    log_paths = [
        contract.artifact_path(run_dir, manifest.artifacts.engine_logs),
        run_dir / "command_logs" / "llm-api-once.json",
        run_dir / "command_logs" / "batch-requests.json",
    ]
    needles = ("Unsupported model architecture", "Operation too slow", "KeyError:", "RuntimeError:", "Traceback")
    for path in log_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            clean = line.strip().strip('"')
            if any(needle in clean for needle in needles):
                return clean[:500]
    return None
