"""Remote OpenAI-compatible engine launcher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any

from inferno import artifact_contract as contract
from inferno.preflight import GPU_ENV_VAR, SSH_OPTIONS, CommandResult, secret_values
from inferno.redaction import redact

CONFIG_PATHS = {
    "vllm": Path("configs/engines/vllm.json"),
    "sglang": Path("configs/engines/sglang.json"),
    "llamacpp": Path("configs/engines/llamacpp.json"),
    "tensorrtllm": Path("configs/engines/tensorrtllm.json"),
    "tensorrtllm_t4": Path("configs/engines/tensorrtllm_t4.json"),
    "ollama": Path("configs/engines/ollama.json"),
}
RUNS_DIR = Path("artifacts/runs")


@dataclass(frozen=True)
class ModelPin:
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str


@dataclass(frozen=True)
class StreamCapture:
    returncode: int
    generated_text: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class RunRealResult:
    returncode: int
    run_id: str
    run_dir: Path
    validation_ok: bool


def run_real(
    *,
    engine: str,
    config_path: Path,
    project_root: Path,
    env: Mapping[str, str],
) -> int:
    result = run_real_result(
        engine=engine,
        config_path=config_path,
        project_root=project_root,
        env=env,
    )
    print(result.run_dir)
    return result.returncode


def run_real_result(
    *,
    engine: str,
    config_path: Path,
    project_root: Path,
    env: Mapping[str, str],
    phase: str = "P2",
    study_id: str | None = None,
    repeat_index: int | None = None,
    telemetry_cadence_seconds: float | None = None,
) -> RunRealResult:
    config = _config(project_root, engine)
    workload = _load_workload(project_root / config_path)
    model = resolve_model(None, config)
    if phase in {"P3", "P4", "P5", "P6", "P10", "P11"}:
        slug = (study_id or workload["workload_id"]).removeprefix(f"{engine}_").replace("_", "-")
        repeat = f"r{repeat_index or 1:02d}"
        run_id = f"{phase.lower()}-{engine}-{slug}-{repeat}-{_stamp()}-{uuid.uuid4().hex[:8]}"
    else:
        run_id = f"p2-{engine}-{_stamp()}-{uuid.uuid4().hex[:8]}"
    run_dir = project_root / RUNS_DIR / run_id
    _mkdirs(run_dir)

    launch_args = _launch_args(config, model)
    config, launch_args = _dashboard_runtime_overrides(config, workload, launch_args)
    manifest = _p2_manifest(
        run_id=run_id,
        run_dir=run_dir,
        phase=phase,
        study_id=study_id,
        repeat_index=repeat_index,
        config=config,
        model=model,
        workload=workload,
        launch_args=launch_args,
        project_root=project_root,
    )
    contract.write_json(run_dir / manifest.artifacts.manifest, manifest)
    if config.get("run_mode") == "llm_api_once":
        return _run_llm_api_once_result(
            config=config,
            workload=workload,
            manifest=manifest,
            run_id=run_id,
            run_dir=run_dir,
            env=env,
            telemetry_cadence_seconds=telemetry_cadence_seconds,
        )

    up_result = _remote("up", _up_script(config, launch_args), timeout=1800, run_dir=run_dir, env=env)
    manifest = _manifest_with(
        manifest,
        status="RUNNING" if up_result.returncode == 0 else "FAILED",
        hardware=_hardware_from_launch(up_result.stdout).model_dump(mode="json"),
        updated_at=_iso_now(),
    )
    manifest = _manifest_with(manifest, comparison_key=contract.comparison_key(manifest))
    contract.write_json(run_dir / manifest.artifacts.manifest, manifest)
    if up_result.returncode != 0:
        return RunRealResult(1, run_id, run_dir, validation_ok=False)

    health = _wait_health(
        run_dir=run_dir,
        config=config,
        served_model_name=config["served_model_name"],
        env=env,
    )
    contract.write_json(
        run_dir / manifest.artifacts.health,
        {"ok": health.ok, "model_ids": health.model_ids},
    )
    manifest = _manifest_with(manifest, health=health.model_dump(mode="json"), updated_at=_iso_now())

    if _dashboard_batch_requested(workload):
        stream = _remote_batch_trace(
            config=config,
            workload=workload,
            run_dir=run_dir,
            manifest=manifest,
            env=env,
            telemetry_cadence_seconds=telemetry_cadence_seconds,
            telemetry_max_seconds=int(workload["measurement"]["max_runtime_seconds"]) + 60,
        )
    else:
        body = _request_body(config["served_model_name"], workload, config)
        stream = _remote_stream_trace(
            body,
            run_dir=run_dir,
            manifest=manifest,
            env=env,
            telemetry_cadence_seconds=telemetry_cadence_seconds,
            telemetry_max_seconds=int(workload["measurement"]["max_runtime_seconds"]) + 60,
        )
    stream_evidence = contract.StreamEvidence(
        ok=stream.returncode == 0 and bool(stream.generated_text.strip()),
        generated_text_sha256=contract.sha256_text(stream.generated_text),
        generated_text_chars=len(stream.generated_text),
    )
    manifest = _manifest_with(
        manifest,
        first_request_perf_counter_ns=stream.start_ns,
        stream=stream_evidence.model_dump(mode="json"),
        updated_at=_iso_now(),
    )

    _capture_engine_metrics(run_dir=run_dir, manifest=manifest, env=env)
    logs = _capture_logs(config=config, run_dir=run_dir, env=env)
    down_result = _remote("down", _down_script(config), timeout=120, run_dir=run_dir, env=env)
    manifest = _manifest_with(
        manifest,
        shutdown={"ok": down_result.returncode == 0},
        status="SUCCEEDED"
        if health.ok and stream_evidence.ok and down_result.returncode == 0
        else "FAILED",
        updated_at=_iso_now(),
    )
    if logs != run_dir / manifest.artifacts.engine_logs:
        raise RuntimeError(f"unexpected engine log path: {logs}")

    contract.write_trace_parquet(run_dir, manifest)
    summary = contract.recompute_summary(run_dir, manifest)
    contract.write_json(run_dir / manifest.artifacts.summary, summary)
    contract.write_checksums(run_dir, manifest)
    contract.write_json(run_dir / manifest.artifacts.manifest, manifest)
    validation = contract.validate_run(run_dir, write=True)
    return RunRealResult(0 if validation.ok else 1, run_id, run_dir, validation.ok)


def _run_llm_api_once_result(
    *,
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    manifest: contract.RunManifest,
    run_id: str,
    run_dir: Path,
    env: Mapping[str, str],
    telemetry_cadence_seconds: float | None,
) -> RunRealResult:
    body = _request_body(config["served_model_name"], workload, config)
    telemetry = _start_telemetry(
        manifest=manifest,
        cadence_seconds=telemetry_cadence_seconds,
        max_seconds=int(workload["measurement"]["max_runtime_seconds"]) + 900,
        env=env,
    )
    start_ns = time.perf_counter_ns()
    result = _remote(
        "llm-api-once",
        _llm_api_once_script(config, workload),
        timeout=int(workload["measurement"]["max_runtime_seconds"]) + 1200,
        run_dir=run_dir,
        env=env,
    )
    end_ns = time.perf_counter_ns()
    _finish_telemetry(telemetry, run_dir=run_dir, manifest=manifest, env=env)
    contract.write_json(
        run_dir / manifest.artifacts.launch_provenance,
        redact(
            {
                "source": "llm_api_once",
                "exit_code": result.returncode,
                "command_log": "command_logs/llm-api-once.json",
            },
            secrets=secret_values(env),
        ),
    )

    payload = _marked_json(result.stdout)
    generated_text = _json_generated_text(payload)
    raw_path = contract.artifact_path(run_dir, manifest.artifacts.raw_sse)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    engine_log = contract.artifact_path(run_dir, manifest.artifacts.engine_logs)
    engine_log.parent.mkdir(parents=True, exist_ok=True)
    engine_log.write_text(
        redact(result.stdout + "\n" + result.stderr, secrets=secret_values(env)),
        encoding="utf-8",
    )
    contract.write_json(
        contract.artifact_path(run_dir, manifest.artifacts.engine_native_metrics),
        {
            "ok": result.returncode == 0,
            "source": "tensorrt_llm_llm_api",
            "engine_version": payload.get("inferno", {}).get("engine_version"),
            "build_config": payload.get("inferno", {}).get("build_config"),
        },
    )

    health = contract.HealthEvidence(
        ok=result.returncode == 0 and bool(generated_text.strip()),
        model_ids=[config["served_model_name"]] if generated_text.strip() else [],
    )
    contract.write_json(run_dir / manifest.artifacts.health, health)
    _write_json_trace(
        run_dir=run_dir,
        manifest=manifest,
        body=body,
        start_ns=start_ns,
        end_ns=end_ns,
        returncode=result.returncode,
        generated_text=generated_text,
    )
    stream_evidence = contract.StreamEvidence(
        ok=result.returncode == 0 and bool(generated_text.strip()),
        generated_text_sha256=contract.sha256_text(generated_text),
        generated_text_chars=len(generated_text),
    )
    manifest = _manifest_with(
        manifest,
        status="SUCCEEDED" if health.ok and stream_evidence.ok else "FAILED",
        hardware=_hardware_from_launch(result.stdout).model_dump(mode="json"),
        health=health.model_dump(mode="json"),
        stream=stream_evidence.model_dump(mode="json"),
        shutdown={"ok": True},
        first_request_perf_counter_ns=start_ns,
        updated_at=_iso_now(),
    )
    manifest = _manifest_with(manifest, comparison_key=contract.comparison_key(manifest))
    contract.write_trace_parquet(run_dir, manifest)
    contract.write_json(run_dir / manifest.artifacts.summary, contract.recompute_summary(run_dir, manifest))
    contract.write_checksums(run_dir, manifest)
    contract.write_json(run_dir / manifest.artifacts.manifest, manifest)
    validation = contract.validate_run(run_dir, write=True)
    return RunRealResult(0 if validation.ok else 1, run_id, run_dir, validation.ok)


def up(*, engine: str, model_arg: str | None, project_root: Path, env: Mapping[str, str]) -> int:
    config = _config(project_root, engine)
    model = resolve_model(model_arg, config)
    run_id = f"p1-{engine}-{_stamp()}-{uuid.uuid4().hex[:8]}"
    run_dir = project_root / RUNS_DIR / run_id
    _mkdirs(run_dir)

    manifest = _base_manifest(run_id, run_dir, config, model, project_root)
    _write_json(run_dir / "manifest.json", manifest | {"status": "PLANNED"})

    launch_args = _launch_args(config, model)
    manifest["engine"]["launch_args"] = launch_args
    manifest["engine"]["env"] = list(config.get("env", []))
    manifest["engine"]["launch_command"] = _launch_command(config, launch_args)
    _write_json(run_dir / "manifest.json", manifest | {"status": "PLANNED"})

    script = _up_script(config, launch_args)
    result = _remote("up", script, timeout=1800, run_dir=run_dir, env=env)
    manifest["updated_at"] = _iso_now()
    manifest["artifacts"]["launch_provenance"] = _rel(run_dir / "command_logs" / "up.json")
    manifest["remote_provenance"] = _split_output(result.stdout)
    manifest["remote_errors"] = _split_output(result.stderr)
    manifest["status"] = "RUNNING" if result.returncode == 0 else "FAILED"
    _write_json(run_dir / "manifest.json", redact(manifest, secrets=secret_values(env)))
    _write_json(
        project_root / _current_state(engine),
        {
            "engine": engine,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "served_model_name": config["served_model_name"],
            "container_name": config["container_name"],
            "port": config["port"],
        },
    )
    print(run_dir)
    return 0 if result.returncode == 0 else 1


def engine_health(*, engine: str, project_root: Path, env: Mapping[str, str]) -> int:
    config = _config(project_root, engine)
    state = _state(project_root, engine)
    run_dir = Path(state["run_dir"])
    manifest = _read_json(run_dir / "manifest.json")
    timeout = int(env.get("INFERNO_ENGINE_HEALTH_TIMEOUT", "900"))
    deadline = time.monotonic() + timeout
    last: CommandResult | None = None

    while time.monotonic() < deadline:
        result = _remote(
            "engine-health",
            _health_script(
                state["container_name"],
                int(state["port"]),
                config.get("health_paths", ["/health", "/v1/models"]),
            ),
            timeout=30,
            run_dir=run_dir,
            env=env,
        )
        last = result
        if result.returncode == 0:
            payloads = _health_payloads(result.stdout)
            model_ids = _served_model_ids(payloads)
            if state["served_model_name"] in model_ids:
                health_path = run_dir / "health.json"
                _write_json(health_path, {"payloads": payloads, "model_ids": model_ids})
                manifest["health"] = {"ok": True, "model_ids": model_ids}
                manifest["artifacts"]["health"] = _rel(health_path)
                manifest["updated_at"] = _iso_now()
                _write_json(run_dir / "manifest.json", manifest)
                print(json.dumps({"status": "ok", "model_ids": model_ids}, indent=2))
                return 0
            result = CommandResult(2, result.stdout, f"model identity mismatch: {model_ids}")
        if "container_not_running" in result.stderr:
            break
        time.sleep(5)

    _capture_logs(config=_config(project_root, engine), run_dir=run_dir, env=env)
    manifest["status"] = "FAILED"
    manifest["health"] = {
        "ok": False,
        "stderr": _split_output(last.stderr if last else "health timed out"),
    }
    manifest["updated_at"] = _iso_now()
    _write_json(run_dir / "manifest.json", redact(manifest, secrets=secret_values(env)))
    print(f"{engine} health failed; see artifacts in {run_dir}", file=sys.stderr)
    return 1


def _wait_health(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    served_model_name: str,
    env: Mapping[str, str],
) -> contract.HealthEvidence:
    timeout = int(env.get("INFERNO_ENGINE_HEALTH_TIMEOUT", "900"))
    deadline = time.monotonic() + timeout
    last: CommandResult | None = None

    while time.monotonic() < deadline:
        result = _remote(
            "engine-health",
            _health_script(
                config["container_name"],
                config["port"],
                config.get("health_paths", ["/health", "/v1/models"]),
            ),
            timeout=30,
            run_dir=run_dir,
            env=env,
        )
        last = result
        if result.returncode == 0:
            payloads = _health_payloads(result.stdout)
            model_ids = _served_model_ids(payloads)
            if served_model_name in model_ids:
                return contract.HealthEvidence(ok=True, model_ids=model_ids)
        if "container_not_running" in result.stderr:
            break
        time.sleep(5)

    _write_json(
        run_dir / "health.json",
        {
            "ok": False,
            "model_ids": [],
            "stderr": _split_output(last.stderr if last else "health timed out"),
        },
    )
    return contract.HealthEvidence(ok=False, model_ids=[])


def real_stream(
    *,
    engine: str,
    prompt: str,
    project_root: Path,
    env: Mapping[str, str],
) -> int:
    state = _state(project_root, engine)
    run_dir = Path(state["run_dir"])
    manifest = _read_json(run_dir / "manifest.json")
    body = {
        "model": state["served_model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 96,
        "stream": True,
    }
    if state.get("engine") != "sglang":
        body["stream_options"] = {"include_usage": True}
    stream_path = run_dir / "raw_streams" / "stream.sse"
    trace_path = run_dir / "request_trace.jsonl"
    stderr_path = run_dir / "command_logs" / "real-stream.stderr"
    start_ns = time.perf_counter_ns()
    result = _remote_stream(
        _stream_script(body, int(state.get("port", 8000))),
        run_dir=run_dir,
        env=env,
    )
    end_ns = time.perf_counter_ns()
    stream_path.write_bytes(result.stdout)
    stderr_path.write_bytes(redact(result.stderr.decode("utf-8", "replace"), secrets=secret_values(env)).encode())

    generated = _generated_text(result.stdout.decode("utf-8", "replace"))
    trace_rows = [
        {
            "event": "request_start",
            "perf_counter_ns": start_ns,
            "request": body,
            "request_sha256": _sha256(json.dumps(body, sort_keys=True).encode()),
        },
        {
            "event": "request_end",
            "perf_counter_ns": end_ns,
            "ssh_exit_code": result.returncode,
            "raw_sse": _rel(stream_path),
            "stderr": _rel(stderr_path),
            "generated_text_sha256": _sha256(generated.encode()),
            "generated_text_chars": len(generated),
        },
    ]
    trace_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trace_rows), encoding="utf-8")

    ok = result.returncode == 0 and bool(generated.strip())
    manifest["stream"] = {
        "ok": ok,
        "generated_text_sha256": _sha256(generated.encode()),
        "generated_text_chars": len(generated),
    }
    manifest["artifacts"]["raw_sse"] = _rel(stream_path)
    manifest["artifacts"]["request_trace"] = _rel(trace_path)
    manifest["updated_at"] = _iso_now()
    if not ok:
        manifest["status"] = "FAILED"
    _write_json(run_dir / "manifest.json", manifest)
    print(generated.strip())
    return 0 if ok else 1


def down(*, engine: str, project_root: Path, env: Mapping[str, str]) -> int:
    config = _config(project_root, engine)
    state_path = project_root / _current_state(engine)
    state = _read_json(state_path) if state_path.exists() else {}
    run_dir = Path(state["run_dir"]) if state else project_root / RUNS_DIR / "manual-down"
    _mkdirs(run_dir)
    logs = _capture_logs(config=config, run_dir=run_dir, env=env)
    result = _remote("down", _down_script(config), timeout=120, run_dir=run_dir, env=env)

    if state:
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        manifest["shutdown"] = {"ok": result.returncode == 0}
        manifest["artifacts"]["engine_logs"] = _rel(logs)
        manifest["status"] = (
            "SUCCEEDED" if result.returncode == 0 and manifest.get("stream", {}).get("ok") else "FAILED"
        )
        manifest["updated_at"] = _iso_now()
        _write_json(manifest_path, redact(manifest, secrets=secret_values(env)))
        state_path.unlink(missing_ok=True)
    print(f"{engine} stopped")
    return 0 if result.returncode == 0 else 1


def resolve_model(model_arg: str | None, config: Mapping[str, Any]) -> ModelPin:
    model_config = config["model"]
    raw = (model_arg or model_config["id"]).strip()
    model_id, sep, revision = raw.rpartition("@")
    if not sep:
        model_id, revision = raw, model_config["revision"]
    if model_id != model_config["id"]:
        raise ValueError(f"unsupported model for {config['engine']}: {model_id}")
    if revision != model_config["revision"]:
        raise ValueError(f"model revision mismatch for {model_id}: {revision}")
    if config["engine"] == "sglang" and model_config["tokenizer_revision"] != revision:
        raise ValueError("SGLang config requires tokenizer_revision to match model revision")
    return ModelPin(
        model_id=model_id,
        revision=revision,
        tokenizer_id=model_config["tokenizer_id"],
        tokenizer_revision=model_config["tokenizer_revision"],
    )


def _p2_manifest(
    *,
    run_id: str,
    run_dir: Path,
    phase: str,
    study_id: str | None,
    repeat_index: int | None,
    config: Mapping[str, Any],
    model: ModelPin,
    workload: Mapping[str, Any],
    launch_args: Sequence[str],
    project_root: Path,
) -> contract.RunManifest:
    artifacts = contract.default_artifacts(str(config["engine"]))
    if phase in {"P3", "P4", "P5", "P6", "P10", "P11"}:
        artifacts = contract.ArtifactPaths.model_validate(
            artifacts.model_dump(mode="json")
            | {
                "request_trace_parquet": "request_trace.parquet",
                "telemetry_parquet": "telemetry.parquet",
                "engine_native_metrics": "engine_metrics.json"
                if config["engine"] in {"tensorrtllm", "tensorrtllm_t4"}
                else artifacts.engine_native_metrics,
            }
        )
    manifest = contract.RunManifest(
        schema_version=contract.SCHEMA_VERSION,
        contract_version=contract.CONTRACT_VERSION,
        phase=phase,
        run_id=run_id,
        study_id=study_id,
        repeat_index=repeat_index,
        status="PLANNED",
        created_at=_iso_now(),
        updated_at=_iso_now(),
        manifest_created_before_request=True,
        manifest_written_perf_counter_ns=time.perf_counter_ns(),
        first_request_perf_counter_ns=None,
        run_dir=str(run_dir),
        source=contract.SourceInfo.model_validate(_source(project_root)),
        engine=contract.EngineInfo(
            name=config["engine"],
            image=config["image"],
            container_name=config["container_name"],
            port=config["port"],
            served_model_name=config["served_model_name"],
            launch_args=list(launch_args),
            env=list(config.get("env", [])),
        ),
        model=contract.ModelInfo(
            id=model.model_id,
            revision=model.revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            tokenizer_format=config["model"].get("tokenizer_format"),
            weight_format=config["model"].get("weight_format"),
            weight_precision=config["model"].get("weight_precision"),
            quantization=config["model"].get("quantization"),
            gguf_repo=config["model"].get("gguf_repo"),
            gguf_filename=config["model"].get("gguf_filename"),
            gguf_revision=config["model"].get("gguf_revision"),
        ),
        hardware=contract.HardwareInfo(
            gpu_model="unknown",
            gpu_count=1,
            driver_version="unknown",
            cuda_version="unknown",
        ),
        workload=_workload_info(workload),
        artifacts=artifacts,
        comparison_key={},
    )
    return _manifest_with(manifest, comparison_key=contract.comparison_key(manifest))


def _manifest_with(manifest: contract.RunManifest, **updates: Any) -> contract.RunManifest:
    payload = manifest.model_dump(mode="json")
    payload.update(updates)
    return contract.RunManifest.model_validate(payload)


def _load_workload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _workload_info(workload: Mapping[str, Any]) -> contract.WorkloadInfo:
    measurement = workload["measurement"]
    warmup = workload.get("warmup", {})
    prompt = workload["prompt"]
    targets = workload.get("targets", {})
    return contract.WorkloadInfo(
        workload_id=workload["workload_id"],
        prompt_template_id=workload["prompt_template_id"],
        seed=workload["seed"],
        prompt_sha256=contract.sha256_text(prompt),
        prompt_chars=len(prompt),
        request_rate_rps=workload.get("request_rate_rps"),
        arrival_model=workload["arrival_model"],
        concurrency=workload.get("concurrency"),
        max_in_flight=workload.get("max_in_flight"),
        prompt_target=dict(workload.get("prompt_target") or targets.get("prompt") or {}),
        output_target=dict(workload.get("output_target") or targets.get("output") or {}),
        context_length=workload.get("context_length"),
        cache_state=workload["cache_state"],
        cache_policy=workload.get("cache_policy"),
        warmup_requests=int(warmup.get("requests", 0)),
        measurement_requests=int(measurement["requests"]),
        max_runtime_seconds=int(measurement["max_runtime_seconds"]),
        streaming=workload.get("streaming"),
        timeout_seconds=workload.get("timeout_seconds") or measurement.get("timeout_seconds"),
        retry_policy=workload["retry_policy"],
        failure_policy=workload["failure_policy"],
        quality_requirement=dict(workload.get("quality_requirement") or {}),
        cost_metadata=dict(workload.get("cost_metadata") or {}),
        runtime_knobs=dict(workload.get("runtime_knobs") or {}),
        sampling=dict(workload["sampling"]),
    )


def _request_body(
    served_model_name: str,
    workload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    prompt: str | None = None,
    stream: bool | None = None,
) -> dict[str, Any]:
    body = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt if prompt is not None else workload["prompt"]}],
        "stream": bool(workload.get("streaming", True) if stream is None else stream),
        **dict(workload["sampling"]),
    }
    if body["stream"] and config.get("request_stream_options", True):
        body["stream_options"] = {"include_usage": True}
    return body


def _dashboard_batch_requested(workload: Mapping[str, Any]) -> bool:
    knobs = dict(workload.get("runtime_knobs") or {})
    prompts = knobs.get("dashboard_batch_prompts") or []
    measurement = dict(workload.get("measurement") or {})
    return (
        bool(knobs.get("dashboard_requested_engine"))
        or int(measurement.get("requests") or 1) > 1
        or int(workload.get("concurrency") or 1) > 1
        or len(prompts) > 1
    )


def _dashboard_runtime_overrides(
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    launch_args: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    updated_config = dict(config)
    updated_args = list(launch_args)
    knobs = dict(workload.get("runtime_knobs") or {})
    if knobs.get("dashboard_kv_cache_mode") != "compact":
        return updated_config, updated_args

    engine = str(config["engine"])
    if engine == "vllm":
        updated_args = _with_launch_arg(updated_args, "--kv-cache-dtype", "fp8")
        updated_args = _with_flag(updated_args, "--enable-prefix-caching")
    elif engine == "sglang":
        updated_args = _with_launch_arg(updated_args, "--kv-cache-dtype", "fp8_e5m2")
        updated_args = _with_flag(updated_args, "--enable-cache-report")
    elif engine == "ollama":
        env = list(config.get("env", []))
        env = _with_env(env, "OLLAMA_FLASH_ATTENTION", "1")
        env = _with_env(env, "OLLAMA_KV_CACHE_TYPE", "q8_0")
        updated_config["env"] = env
    return updated_config, updated_args


def _with_launch_arg(args: list[str], name: str, value: str) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for index, item in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if item == name:
            skip_next = index + 1 < len(args)
            continue
        if item.startswith(name + "="):
            continue
        cleaned.append(item)
    return [*cleaned, name, value]


def _with_flag(args: list[str], name: str) -> list[str]:
    return args if name in args else [*args, name]


def _with_env(env: list[str], name: str, value: str) -> list[str]:
    prefix = name + "="
    return [item for item in env if not item.startswith(prefix)] + [prefix + value]


def _config(project_root: Path, engine: str) -> dict[str, Any]:
    if engine not in CONFIG_PATHS:
        raise ValueError(
            "supported engines: vllm, sglang, llamacpp, tensorrtllm, tensorrtllm_t4, ollama"
        )
    config = _read_json(project_root / CONFIG_PATHS[engine])
    if config.get("engine") != engine:
        raise ValueError(f"invalid {engine} config")
    return config


def _state(project_root: Path, engine: str) -> dict[str, Any]:
    state_path = project_root / _current_state(engine)
    if not state_path.exists():
        raise RuntimeError(f"no current {engine} run; run `make up ENGINE={engine}` first")
    return _read_json(state_path)


def _base_manifest(
    run_id: str,
    run_dir: Path,
    config: Mapping[str, Any],
    model: ModelPin,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "P1",
        "run_id": run_id,
        "status": "PLANNED",
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
        "manifest_created_before_request": True,
        "run_dir": str(run_dir),
        "source": _source(project_root),
        "engine": {
            "name": config["engine"],
            "image": config["image"],
            "image_tag_hint": config.get("image_tag_hint"),
            "container_name": config["container_name"],
            "port": config["port"],
            "served_model_name": config["served_model_name"],
        },
        "model": {
            "id": model.model_id,
            "revision": model.revision,
            "tokenizer_id": model.tokenizer_id,
            "tokenizer_revision": model.tokenizer_revision,
        },
        "artifacts": {
            "manifest": _rel(run_dir / "manifest.json"),
        },
    }


def _launch_args(config: Mapping[str, Any], model: ModelPin) -> list[str]:
    if config["engine"] == "vllm":
        return [
            "--model",
            model.model_id,
            "--revision",
            model.revision,
            "--tokenizer",
            model.tokenizer_id,
            "--tokenizer-revision",
            model.tokenizer_revision,
            "--served-model-name",
            config["served_model_name"],
            *config["launch_args"],
        ]
    if config["engine"] == "llamacpp":
        gguf = config["gguf"]
        return [
            "--hf-repo",
            f"{gguf['repo_id']}:{gguf['quantization']}",
            "--hf-file",
            gguf["filename"],
            "--alias",
            config["served_model_name"],
            "--host",
            "0.0.0.0",
            "--port",
            str(config["port"]),
            *config["launch_args"],
        ]
    if config["engine"] == "tensorrtllm":
        return [
            model.model_id,
            "--hf_revision",
            model.revision,
            "--tokenizer",
            model.tokenizer_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(config["port"]),
            "--served_model_name",
            config["served_model_name"],
            *config["launch_args"],
        ]
    if config["engine"] == "tensorrtllm_t4":
        return [
            model.model_id,
            "--revision",
            model.revision,
            "--tokenizer",
            model.tokenizer_id,
            "--backend",
            "llm_api",
            *config["launch_args"],
        ]
    if config["engine"] == "ollama":
        return [*config.get("launch_args", [])]
    return [
        "--model-path",
        model.model_id,
        "--revision",
        model.revision,
        "--tokenizer-path",
        model.tokenizer_id,
        "--served-model-name",
        config["served_model_name"],
        *config["launch_args"],
    ]


def _up_script(config: Mapping[str, Any], launch_args: Sequence[str]) -> str:
    command = [*config.get("command_prefix", []), *launch_args]
    args = " ".join(shlex.quote(item) for item in command)
    name = shlex.quote(config["container_name"])
    image = shlex.quote(config["image"])
    port = shlex.quote(str(config["port"]))
    env_args = " ".join(f"-e {shlex.quote(item)}" for item in config.get("env", []))
    docker_args = " ".join(shlex.quote(item) for item in config.get("docker_args", []))
    mount_args = (
        '-v "$HOME/.cache/huggingface:/root/.cache/huggingface" '
        '-v "$HOME/.cache/llama.cpp:/root/.cache/llama.cpp" '
        '-v "$HOME/.cache/torch:/root/.cache/torch" '
        '-v "$HOME/.cache/triton:/root/.triton" '
        '-v "$HOME/.cache/vllm:/root/.cache/vllm" '
        '-v "$HOME/.ollama:/root/.ollama"'
    )
    run_prefix = f"docker run -d --gpus all --name {name} --ipc=host {docker_args}"
    port_args = f"-p 127.0.0.1:{port}:{port}"
    version_command = _container_version_command(config)
    return f"""set -eu
echo '== gpu =='
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | sed -n '1,4p'
echo '== docker =='
docker --version
echo '== image_pull =='
docker pull {image}
echo '== image_inspect =='
docker image inspect {image} --format 'id={{{{.Id}}}} created={{{{.Created}}}} digests={{{{json .RepoDigests}}}}'
echo '== cleanup =='
docker rm -f {name} >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/llama.cpp" "$HOME/.cache/torch" "$HOME/.cache/triton" "$HOME/.cache/vllm" "$HOME/.ollama"
echo '== launch =='
{run_prefix} {port_args} {env_args} {mount_args} {image} {args}
echo '== container =='
docker ps --filter name={name} --format '{{{{json .}}}}'
{_post_launch_script(config)}
echo '== engine_version =='
docker exec {name} {version_command} || true
docker inspect -f '{{{{.State.Running}}}}' {name} | grep true
"""


def _health_script(container_name: str, port: int, paths: Sequence[str]) -> str:
    name = shlex.quote(container_name)
    endpoint = f"http://127.0.0.1:{port}"
    paths_json = json.dumps(list(paths))
    return (
        f"state=\"$(docker inspect -f '{{{{.State.Running}}}} {{{{.State.ExitCode}}}}' {name} 2>/dev/null || true)\"\n"
        "case \"$state\" in true\\ *) ;; *) echo \"container_not_running $state\" >&2; exit 3;; esac\n"
        + _python_script(
        f"""
import json
import urllib.request

for path in json.loads({paths_json!r}):
    with urllib.request.urlopen({endpoint!r} + path, timeout=10) as response:
        print(json.dumps({{
            "path": path,
            "status": response.status,
            "body": response.read().decode("utf-8", "replace"),
        }}))
""".strip()
    )
    )


def _stream_script(body: Mapping[str, Any], port: int, timeout_seconds: int = 300) -> str:
    body_json = json.dumps(body)
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    return _python_script(
        f"""
import json
import sys
import urllib.request

body = json.loads({body_json!r})
request = urllib.request.Request(
    {endpoint!r},
    data=json.dumps(body).encode("utf-8"),
    headers={{"Content-Type": "application/json", "Connection": "close"}},
)
with urllib.request.urlopen(request, timeout={timeout_seconds!r}) as response:
    for chunk in response:
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
""".strip()
    )


def _batch_script(config: Mapping[str, Any], workload: Mapping[str, Any]) -> str:
    endpoint = f"http://127.0.0.1:{config['port']}/v1/chat/completions"
    knobs = dict(workload.get("runtime_knobs") or {})
    prompts = knobs.get("dashboard_batch_prompts") or [workload["prompt"]]
    prompts = [str(item) for item in prompts if str(item).strip()] or [str(workload["prompt"])]
    sampling = dict(workload["sampling"])
    requests = int(workload["measurement"]["requests"])
    warmups = int((workload.get("warmup") or {}).get("requests") or 0)
    concurrency = int(workload.get("concurrency") or 1)
    request_rate = float(workload.get("request_rate_rps") or 0)
    timeout_seconds = int(workload.get("timeout_seconds") or workload["measurement"]["max_runtime_seconds"])
    include_usage = bool(config.get("request_stream_options", True))
    return _python_script(
        f"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
import urllib.request

endpoint = {endpoint!r}
model = {config["served_model_name"]!r}
prompts = {prompts!r}
sampling = {sampling!r}
requests = {requests!r}
warmups = {warmups!r}
concurrency = {concurrency!r}
request_rate = {request_rate!r}
timeout_seconds = {timeout_seconds!r}
request_timeout = min(timeout_seconds, 180)
include_usage = {include_usage!r}
script_start_ns = time.perf_counter_ns()

def text_from(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text = ""
        for part in value:
            if isinstance(part, dict):
                text += str(part.get("text") or part.get("content") or "")
            else:
                text += str(part)
        return text
    return ""

def run_one(index, measured):
    if measured and request_rate > 0:
        target = script_start_ns + int(index / request_rate * 1_000_000_000)
        delay = (target - time.perf_counter_ns()) / 1_000_000_000
        if delay > 0:
            time.sleep(delay)
    prompt = prompts[index % len(prompts)]
    body = {{
        "model": model,
        "messages": [{{"role": "user", "content": prompt}}],
        "stream": True,
        **sampling,
    }}
    if include_usage:
        body["stream_options"] = {{"include_usage": True}}
    started = time.perf_counter_ns()
    events = []
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={{"Content-Type": "application/json", "Connection": "close"}},
        )
        generated_parts = []
        usage = None
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                if payload.get("usage") is not None:
                    usage = payload.get("usage")
                choice = (payload.get("choices") or [{{}}])[0]
                delta = choice.get("delta") or {{}}
                text = (
                    text_from(delta.get("content"))
                    or text_from(delta.get("reasoning_content"))
                    or text_from(delta.get("reasoning"))
                    or text_from(delta.get("thinking"))
                    or text_from(choice.get("text"))
                    or text_from(choice.get("content"))
                    or text_from(payload.get("response"))
                )
                events.append({{
                    "delta_text": text,
                    "delta_ns": time.perf_counter_ns() - script_start_ns,
                }})
                generated_parts.append(text)
        generated = "".join(generated_parts)
        ok = bool(str(generated).strip())
        error = None if ok else (
            "empty_generated_text; "
            + "stream_events=" + str(len(events))
        )
    except Exception as exc:
        generated = ""
        ok = False
        usage = None
        error = type(exc).__name__ + ": " + str(exc)
    ended = time.perf_counter_ns()
    return {{
        "index": index,
        "ok": ok,
        "prompt": prompt,
        "generated_text": generated,
        "usage": usage,
        "stream_events": events,
        "start_delta_ns": started - script_start_ns,
        "end_delta_ns": ended - script_start_ns,
        "error": error,
    }}

for warmup in range(warmups):
    run_one(warmup, False)

responses = []
with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
    futures = [pool.submit(run_one, index, True) for index in range(requests)]
    for future in as_completed(futures):
        responses.append(future.result())
responses.sort(key=lambda item: item["index"])
print(json.dumps({{"responses": responses}}, sort_keys=True))
sys.stdout.flush()
os._exit(0)
""".strip()
    )


def _llm_api_once_script(config: Mapping[str, Any], workload: Mapping[str, Any]) -> str:
    name = shlex.quote(config["container_name"])
    image = shlex.quote(config["image"])
    docker_args = " ".join(shlex.quote(item) for item in config.get("docker_args", []))
    script_path = f"/tmp/inferno_{re.sub(r'[^A-Za-z0-9_.-]', '_', config['container_name'])}.py"
    build_config = config.get("build_config", {})
    prompt = workload["prompt"]
    sampling = dict(workload["sampling"])
    python = f"""
import json
import time

import tensorrt_llm
from tensorrt_llm import BuildConfig, LLM, SamplingParams

model = {config["model"]["id"]!r}
revision = {config["model"]["revision"]!r}
tokenizer = {config["model"]["tokenizer_id"]!r}
tokenizer_revision = {config["model"]["tokenizer_revision"]!r}
prompt = {prompt!r}
served_model = {config["served_model_name"]!r}
build_config_data = {build_config!r}
sampling_data = {sampling!r}

started_ns = time.perf_counter_ns()
llm = LLM(
    model=model,
    tokenizer=tokenizer,
    revision=revision,
    tokenizer_revision=tokenizer_revision,
    dtype={config.get("dtype", "float16")!r},
    build_config=BuildConfig(**build_config_data),
    enable_tqdm=False,
)
outputs = llm.generate(
    [prompt],
    SamplingParams(
        max_new_tokens=int(sampling_data.get("max_tokens", 32)),
        temperature=float(sampling_data.get("temperature", 0)),
        top_p=float(sampling_data.get("top_p", 1)),
    ),
)
generated = outputs[0].outputs[0].text
finished_ns = time.perf_counter_ns()
llm.shutdown()
payload = {{
    "id": "inferno-llm-api-once",
    "object": "chat.completion",
    "model": served_model,
    "choices": [
        {{
            "index": 0,
            "message": {{"role": "assistant", "content": generated}},
            "finish_reason": None,
        }}
    ],
    "usage": None,
    "inferno": {{
        "engine_version": getattr(tensorrt_llm, "__version__", "unknown"),
        "build_config": build_config_data,
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": finished_ns,
    }},
}}
print("INFERNO_JSON_START")
print(json.dumps(payload, sort_keys=True))
print("INFERNO_JSON_END")
""".strip()
    return f"""set -eu
echo '== gpu =='
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | sed -n '1,4p'
echo '== docker =='
docker --version
echo '== image_pull =='
docker pull {image}
echo '== image_inspect =='
docker image inspect {image} --format 'id={{{{.Id}}}} created={{{{.Created}}}} digests={{{{json .RepoDigests}}}}'
echo '== cleanup =='
docker rm -f {name} >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/huggingface"
cat > {shlex.quote(script_path)} <<'PY'
{python}
PY
echo '== launch =='
docker run --rm --gpus all --name {name} --ipc=host {docker_args} -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v {shlex.quote(script_path)}:{shlex.quote(script_path)}:ro --entrypoint bash {image} -lc {shlex.quote("python3 " + script_path)}
rm -f {shlex.quote(script_path)}
"""


def _down_script(config: Mapping[str, Any]) -> str:
    return f"docker rm -f {shlex.quote(config['container_name'])} >/dev/null 2>&1 || true"


def _python_script(script: str) -> str:
    return "python3 - <<'PY'\n" + script + "\nPY"


def _launch_command(config: Mapping[str, Any], launch_args: Sequence[str]) -> str:
    command = [*config.get("command_prefix", []), *launch_args]
    return " ".join(
        [
            "docker",
            "run",
            "-d",
            "--gpus",
            "all",
            "--name",
            config["container_name"],
            "--ipc=host",
            *config.get("docker_args", []),
            "-p",
            f"127.0.0.1:{config['port']}:{config['port']}",
            *(item for env_item in config.get("env", []) for item in ("-e", env_item)),
            "-v",
            "$HOME/.cache/huggingface:/root/.cache/huggingface",
            "-v",
            "$HOME/.cache/llama.cpp:/root/.cache/llama.cpp",
            "-v",
            "$HOME/.cache/torch:/root/.cache/torch",
            "-v",
            "$HOME/.cache/triton:/root/.triton",
            "-v",
            "$HOME/.cache/vllm:/root/.cache/vllm",
            "-v",
            "$HOME/.ollama:/root/.ollama",
            config["image"],
            *(shlex.quote(item) for item in command),
        ]
    )


def _post_launch_script(config: Mapping[str, Any]) -> str:
    if config["engine"] != "ollama":
        return ""
    name = shlex.quote(config["container_name"])
    model = shlex.quote(config["served_model_name"])
    return f"""echo '== ollama_pull =='
sleep 3
docker exec {name} ollama pull {model}
"""


def _container_version_command(config: Mapping[str, Any]) -> str:
    if config.get("version_command"):
        return " ".join(shlex.quote(item) for item in config["version_command"])
    version_module = json.dumps(str(config.get("version_module", config["engine"])))
    return (
        "python3 -c "
        + shlex.quote(
            "import importlib; "
            f"m=importlib.import_module({version_module}); "
            'print(getattr(m, "__version__", "unknown"))'
        )
    )


def _remote(
    name: str,
    script: str,
    *,
    timeout: int,
    run_dir: Path,
    env: Mapping[str, str],
) -> CommandResult:
    args = _ssh_args(script, env)
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        result = CommandResult(completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or f"{name} timed out after {timeout} seconds"
        result = CommandResult(124, str(stdout), str(stderr))
    _log_command(name, args, result, run_dir, env)
    return result


def _remote_stream(script: str, *, run_dir: Path, env: Mapping[str, str]) -> CommandResult:
    args = _ssh_args(script, env)
    try:
        completed = subprocess.run(args, capture_output=True, check=False, timeout=360)
        result = CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = _as_bytes(exc.stdout or b"")
        stderr = _as_bytes(exc.stderr or b"real-stream timed out after 360 seconds")
        result = CommandResult(
            124,
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
        )
    _write_json(
        run_dir / "command_logs" / "real-stream.json",
        redact(
            {
                "command": "ssh [REDACTED] <remote stream request>",
                "exit_code": result.returncode,
                "stdout": ["[raw SSE stored separately]"],
                "stderr": _split_output(result.stderr),
            },
            secrets=secret_values(env),
        ),
    )
    return CommandResult(result.returncode, stdout, stderr)


def _remote_stream_trace(
    body: Mapping[str, Any],
    *,
    run_dir: Path,
    manifest: contract.RunManifest,
    env: Mapping[str, str],
    telemetry_cadence_seconds: float | None = None,
    telemetry_max_seconds: int = 360,
) -> StreamCapture:
    timeout_seconds = manifest.workload.timeout_seconds or manifest.workload.max_runtime_seconds
    args = _ssh_args(_stream_script(body, manifest.engine.port, timeout_seconds), env)
    raw_path = run_dir / manifest.artifacts.raw_sse
    trace_path = run_dir / manifest.artifacts.request_trace
    stderr_path = run_dir / "command_logs" / "real-stream.stderr"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    request_id = "smoke-1"
    request_sha = contract.canonical_json_sha256(body)
    start_ns = time.perf_counter_ns()
    rows: list[contract.TraceEvent] = [
        contract.TraceEvent(
            schema_version=contract.SCHEMA_VERSION,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id=request_id,
            event="request_start",
            perf_counter_ns=start_ns,
            request=dict(body),
            request_sha256=request_sha,
        )
    ]

    telemetry_process = _start_telemetry(
        manifest=manifest,
        cadence_seconds=telemetry_cadence_seconds,
        max_seconds=telemetry_max_seconds,
        env=env,
    )
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to capture stream pipes")

    sse_index = 0
    with raw_path.open("wb") as raw_file:
        for line in process.stdout:
            raw_file.write(line)
            raw_file.flush()
            text = line.decode("utf-8", "replace")
            if not line.startswith(b"data: "):
                continue
            data = text.removeprefix("data: ").strip()
            if data == "[DONE]":
                continue
            delta = _delta_text(data)
            rows.append(
                contract.TraceEvent(
                    schema_version=contract.SCHEMA_VERSION,
                    contract_version=contract.CONTRACT_VERSION,
                    run_id=manifest.run_id,
                    request_id=request_id,
                    event="stream_event",
                    perf_counter_ns=time.perf_counter_ns(),
                    sse_index=sse_index,
                    delta_text_sha256=contract.sha256_text(delta),
                    delta_text_chars=len(delta),
                )
            )
            sse_index += 1

    stderr = process.stderr.read().decode("utf-8", "replace")
    returncode = process.wait(timeout=30)
    end_ns = time.perf_counter_ns()
    stderr_path.write_text(redact(stderr, secrets=secret_values(env)), encoding="utf-8")
    generated_text = _generated_text(raw_path.read_text(encoding="utf-8", errors="replace"))
    usage = contract.stream_usage(raw_path.read_text(encoding="utf-8", errors="replace"))
    rows.append(
        contract.TraceEvent(
            schema_version=contract.SCHEMA_VERSION,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id=request_id,
            event="request_end",
            perf_counter_ns=end_ns,
            ssh_exit_code=returncode,
            raw_sse=manifest.artifacts.raw_sse,
            stderr="command_logs/real-stream.stderr",
            generated_text_sha256=contract.sha256_text(generated_text),
            generated_text_chars=len(generated_text),
            usage=usage,
        )
    )
    trace_path.write_text(
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _finish_telemetry(telemetry_process, run_dir=run_dir, manifest=manifest, env=env)
    _write_json(
        run_dir / "command_logs" / "real-stream.json",
        redact(
            {
                "command": "ssh [REDACTED] <remote stream request>",
                "exit_code": returncode,
                "stdout": ["[raw SSE stored separately]"],
                "stderr": _split_output(stderr),
            },
            secrets=secret_values(env),
        ),
    )
    return StreamCapture(returncode, generated_text, start_ns, end_ns)


def _remote_batch_trace(
    *,
    config: Mapping[str, Any],
    workload: Mapping[str, Any],
    run_dir: Path,
    manifest: contract.RunManifest,
    env: Mapping[str, str],
    telemetry_cadence_seconds: float | None = None,
    telemetry_max_seconds: int = 360,
) -> StreamCapture:
    raw_path = run_dir / manifest.artifacts.raw_sse
    trace_path = run_dir / manifest.artifacts.request_trace
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    local_base_ns = time.perf_counter_ns()
    telemetry_process = _start_telemetry(
        manifest=manifest,
        cadence_seconds=telemetry_cadence_seconds,
        max_seconds=telemetry_max_seconds,
        env=env,
    )
    result = _remote(
        "batch-requests",
        _batch_script(config, workload),
        timeout=_batch_remote_timeout(workload),
        run_dir=run_dir,
        env=env,
    )
    local_end_ns = time.perf_counter_ns()
    _finish_telemetry(telemetry_process, run_dir=run_dir, manifest=manifest, env=env)
    payload = _json_or_batch_error(result.stdout, result.returncode)
    responses = list(payload.get("responses") or [])
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[contract.TraceEvent] = []
    for item in responses:
        index = int(item.get("index") or 0)
        request_id = f"dashboard-{index + 1}"
        prompt = str(item.get("prompt") or "")
        body = _request_body(config["served_model_name"], workload, config, prompt=prompt, stream=True)
        start_ns = local_base_ns + int(item.get("start_delta_ns") or 0)
        end_ns = local_base_ns + int(item.get("end_delta_ns") or (local_end_ns - local_base_ns))
        generated = str(item.get("generated_text") or "")
        rows.append(
            contract.TraceEvent(
                schema_version=contract.SCHEMA_VERSION,
                contract_version=contract.CONTRACT_VERSION,
                run_id=manifest.run_id,
                request_id=request_id,
                event="request_start",
                perf_counter_ns=start_ns,
                request=body,
                request_sha256=contract.canonical_json_sha256(body),
            )
        )
        for sse_index, event in enumerate(item.get("stream_events") or []):
            delta = str(event.get("delta_text") or "")
            rows.append(
                contract.TraceEvent(
                    schema_version=contract.SCHEMA_VERSION,
                    contract_version=contract.CONTRACT_VERSION,
                    run_id=manifest.run_id,
                    request_id=request_id,
                    event="stream_event",
                    perf_counter_ns=local_base_ns + int(event.get("delta_ns") or 0),
                    sse_index=sse_index,
                    delta_text_sha256=contract.sha256_text(delta),
                    delta_text_chars=len(delta),
                )
            )
        rows.append(
            contract.TraceEvent(
                schema_version=contract.SCHEMA_VERSION,
                contract_version=contract.CONTRACT_VERSION,
                run_id=manifest.run_id,
                request_id=request_id,
                event="request_end",
                perf_counter_ns=end_ns,
                ssh_exit_code=0 if item.get("ok") else 1,
                raw_sse=manifest.artifacts.raw_sse,
                stderr="command_logs/batch-requests.json",
                generated_text_sha256=contract.sha256_text(generated),
                generated_text_chars=len(generated),
                usage=item.get("usage") or None,
            )
        )
    rows.sort(key=lambda row: row.perf_counter_ns)
    trace_path.write_text(
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    generated_text = "".join(str(item.get("generated_text") or "") for item in responses)
    batch_ok = bool(responses) and all(item.get("ok") for item in responses)
    return StreamCapture(0 if batch_ok else 1, generated_text, local_base_ns, local_end_ns)


def _json_or_batch_error(stdout: str, returncode: int) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
        if isinstance(payload.get("responses"), list) and payload["responses"]:
            return payload
    except json.JSONDecodeError:
        pass
    message = (
        f"batch command failed with exit code {returncode}"
        if returncode
        else "batch command returned no responses"
    )
    return {
        "responses": [
            {
                "index": 0,
                "ok": False,
                "prompt": "",
                "generated_text": "",
                "usage": None,
                "start_delta_ns": 0,
                "end_delta_ns": 1,
                "error": message,
            }
        ]
    }


def _batch_remote_timeout(workload: Mapping[str, Any]) -> int:
    measurement = dict(workload.get("measurement") or {})
    timeout_seconds = int(
        workload.get("timeout_seconds")
        or measurement.get("timeout_seconds")
        or measurement.get("max_runtime_seconds")
        or 300
    )
    return max(60, timeout_seconds + 60)


def _marked_json(stdout: str) -> dict[str, Any]:
    try:
        text = stdout.split("INFERNO_JSON_START", 1)[1].split("INFERNO_JSON_END", 1)[0]
        return dict(json.loads(text.strip()))
    except (IndexError, json.JSONDecodeError, TypeError, ValueError):
        return {
            "id": "inferno-llm-api-once",
            "object": "chat.completion",
            "model": "unknown",
            "choices": [],
            "usage": None,
            "inferno": {"parse_error": "missing_marked_json"},
        }


def _json_generated_text(payload: Mapping[str, Any]) -> str:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return str(message.get("content") or choice.get("text") or "")


def _write_json_trace(
    *,
    run_dir: Path,
    manifest: contract.RunManifest,
    body: Mapping[str, Any],
    start_ns: int,
    end_ns: int,
    returncode: int,
    generated_text: str,
) -> None:
    event_ns = max(start_ns, end_ns - 1)
    rows = [
        contract.TraceEvent(
            schema_version=contract.SCHEMA_VERSION,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_start",
            perf_counter_ns=start_ns,
            request=dict(body),
            request_sha256=contract.canonical_json_sha256(body),
        ),
        contract.TraceEvent(
            schema_version=contract.SCHEMA_VERSION,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="stream_event",
            perf_counter_ns=event_ns,
            sse_index=0,
            delta_text_sha256=contract.sha256_text(generated_text),
            delta_text_chars=len(generated_text),
        ),
        contract.TraceEvent(
            schema_version=contract.SCHEMA_VERSION,
            contract_version=contract.CONTRACT_VERSION,
            run_id=manifest.run_id,
            request_id="smoke-1",
            event="request_end",
            perf_counter_ns=end_ns,
            ssh_exit_code=returncode,
            raw_sse=manifest.artifacts.raw_sse,
            stderr="command_logs/llm-api-once.json",
            generated_text_sha256=contract.sha256_text(generated_text),
            generated_text_chars=len(generated_text),
        ),
    ]
    trace_path = contract.artifact_path(run_dir, manifest.artifacts.request_trace)
    trace_path.write_text(
        "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _delta_text(data: str) -> str:
    payload = json.loads(data)
    choice = (payload.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    return delta.get("content") or delta.get("reasoning_content") or choice.get("text") or ""


def _start_telemetry(
    *,
    manifest: contract.RunManifest,
    cadence_seconds: float | None,
    max_seconds: int,
    env: Mapping[str, str],
) -> subprocess.Popen[str] | None:
    if not manifest.artifacts.telemetry_parquet:
        return None
    cadence = 0.5 if cadence_seconds is None else cadence_seconds
    return subprocess.Popen(
        _ssh_args(_telemetry_script(manifest.engine.container_name, cadence, max_seconds), env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish_telemetry(
    process: subprocess.Popen[str] | None,
    *,
    run_dir: Path,
    manifest: contract.RunManifest,
    env: Mapping[str, str],
) -> None:
    if not manifest.artifacts.telemetry_parquet:
        return
    stdout = ""
    stderr = ""
    returncode: int | None = None
    if process is not None:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=10)
        returncode = process.returncode
    rows = []
    for line in stdout.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        rows = [_missing_telemetry_row("telemetry_collector_no_samples")]
    contract.write_telemetry_parquet(run_dir, manifest, rows)
    _write_json(
        run_dir / "command_logs" / "telemetry.json",
        redact(
            {
                "command": "ssh [REDACTED] <remote telemetry collector>",
                "exit_code": returncode,
                "samples": len(rows),
                "stderr": _split_output(stderr),
            },
            secrets=secret_values(env),
        ),
    )


def _capture_engine_metrics(
    *,
    run_dir: Path,
    manifest: contract.RunManifest,
    env: Mapping[str, str],
) -> None:
    if not manifest.artifacts.engine_native_metrics:
        return
    result = _remote(
        "engine-metrics",
        _metrics_script(manifest.engine.container_name, manifest.engine.port),
        timeout=30,
        run_dir=run_dir,
        env=env,
    )
    contract.artifact_path(run_dir, manifest.artifacts.engine_native_metrics).write_text(
        json.dumps(
            redact(
                {
                    "ok": result.returncode == 0,
                    "endpoint": "/metrics",
                    "exit_code": result.returncode,
                    "stdout": _split_output(result.stdout),
                    "stderr": _split_output(result.stderr),
                },
                secrets=secret_values(env),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _missing_telemetry_row(reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_index": 0,
        "collected_at_unix_ns": time.time_ns(),
        "source": "telemetry_fallback",
    }
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
        row[f"{metric}_missing_reason"] = reason
    return row


def _metrics_script(container_name: str, port: int) -> str:
    name = shlex.quote(container_name)
    endpoint = f"http://127.0.0.1:{port}/metrics"
    return (
        f"state=\"$(docker inspect -f '{{{{.State.Running}}}} {{{{.State.ExitCode}}}}' {name} 2>/dev/null || true)\"\n"
        "case \"$state\" in true\\ *) ;; *) echo \"container_not_running $state\" >&2; exit 3;; esac\n"
        + _python_script(
            f"""
import urllib.request

with urllib.request.urlopen({endpoint!r}, timeout=10) as response:
    print(response.read().decode("utf-8", "replace"))
""".strip()
        )
    )


def _telemetry_script(container_name: str, cadence_seconds: float, max_seconds: int) -> str:
    name = shlex.quote(container_name)
    return _python_script(
        f"""
import json
import subprocess
import time

container_name = {name!r}
cadence_seconds = {cadence_seconds!r}
deadline = time.time() + {max_seconds!r}
metrics = (
    "gpu_utilization_percent",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_power_watts",
    "gpu_temperature_c",
    "host_cpu_percent",
    "host_rss_mib",
)

def parse_float(value):
    text = str(value).strip().replace("%", "")
    if not text or text in {{"[Not Supported]", "N/A", "None"}}:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def parse_mib(value):
    text = str(value).strip()
    number = parse_float(text.split()[0] if text else "")
    if number is None:
        return None
    unit = text.lower()
    if "gib" in unit or "gb" in unit:
        return number * 1024
    if "kib" in unit or "kb" in unit:
        return number / 1024
    return number

def set_metric(row, key, value, reason):
    row[key] = value
    row[key + "_missing_reason"] = None if value is not None else reason

sample_index = 0
while time.time() < deadline:
    row = {{
        "sample_index": sample_index,
        "collected_at_unix_ns": time.time_ns(),
        "source": "nvidia-smi+docker-stats",
    }}
    gpu_reason = "nvidia_smi_unavailable"
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if gpu.returncode == 0 and gpu.stdout.strip():
            parts = [part.strip() for part in gpu.stdout.splitlines()[0].split(",")]
            values = [parse_float(part) for part in (parts + [""] * 5)[:5]]
            gpu_reason = "nvidia_smi_field_unavailable"
            for key, value in zip(metrics[:5], values):
                set_metric(row, key, value, gpu_reason)
        else:
            for key in metrics[:5]:
                set_metric(row, key, None, gpu_reason)
    except Exception:
        for key in metrics[:5]:
            set_metric(row, key, None, gpu_reason)

    host_reason = "docker_stats_unavailable"
    try:
        stats = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{{{json .}}}}", container_name],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if stats.returncode == 0 and stats.stdout.strip():
            payload = json.loads(stats.stdout.splitlines()[0])
            set_metric(row, "host_cpu_percent", parse_float(payload.get("CPUPerc")), host_reason)
            memory_used = str(payload.get("MemUsage", "")).split("/")[0]
            set_metric(row, "host_rss_mib", parse_mib(memory_used), host_reason)
        else:
            set_metric(row, "host_cpu_percent", None, host_reason)
            set_metric(row, "host_rss_mib", None, host_reason)
    except Exception:
        set_metric(row, "host_cpu_percent", None, host_reason)
        set_metric(row, "host_rss_mib", None, host_reason)

    print(json.dumps(row, sort_keys=True), flush=True)
    sample_index += 1
    time.sleep(cadence_seconds)
""".strip()
    )


def _ssh_args(script: str, env: Mapping[str, str]) -> list[str]:
    target = env.get(GPU_ENV_VAR, "").strip()
    if not target:
        raise RuntimeError(f"{GPU_ENV_VAR} is required")
    return ["ssh", *SSH_OPTIONS, *shlex.split(target), script]


def _log_command(
    name: str,
    args: Sequence[str],
    result: CommandResult,
    run_dir: Path,
    env: Mapping[str, str],
) -> None:
    payload = {
        "command": " ".join(shlex.quote(item) for item in args),
        "exit_code": result.returncode,
        "stdout": _split_output(str(result.stdout)),
        "stderr": _split_output(str(result.stderr)),
    }
    _write_json(run_dir / "command_logs" / f"{name}.json", redact(payload, secrets=secret_values(env)))


def _capture_logs(*, config: Mapping[str, Any], run_dir: Path, env: Mapping[str, str]) -> Path:
    result = _remote("engine-logs", f"docker logs {shlex.quote(config['container_name'])} 2>&1 || true", timeout=120, run_dir=run_dir, env=env)
    path = run_dir / "engine_logs" / f"{config['engine']}.log"
    path.write_text(redact(result.stdout, secrets=secret_values(env)), encoding="utf-8")
    return path


def _health_payloads(stdout: str) -> list[dict[str, Any]]:
    payloads = []
    for line in stdout.splitlines():
        if line.strip():
            payloads.append(json.loads(line))
    return payloads


def _served_model_ids(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for payload in payloads:
        if payload.get("path") != "/v1/models":
            continue
        body = json.loads(str(payload["body"]))
        ids.extend(item["id"] for item in body.get("data", []) if "id" in item)
    return ids


def _hardware_from_launch(stdout: str) -> contract.HardwareInfo:
    gpu_model = "unknown"
    driver = "unknown"
    memory_mib: int | None = None
    for line in stdout.splitlines():
        match = re.match(r"^(?P<gpu>.+), (?P<driver>[0-9.]+), (?P<memory>\d+) MiB$", line.strip())
        if match:
            gpu_model = match.group("gpu")
            driver = match.group("driver")
            memory_mib = int(match.group("memory"))
            break
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", stdout)
    return contract.HardwareInfo(
        gpu_model=gpu_model,
        gpu_count=1,
        driver_version=driver,
        cuda_version=cuda_match.group(1) if cuda_match else "unknown",
        memory_mib=memory_mib,
    )


def _generated_text(raw_sse: str) -> str:
    if "data: " not in raw_sse:
        return _generated_text_from_json(raw_sse)
    chunks = []
    for line in raw_sse.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        payload = json.loads(data)
        choice = (payload.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        chunks.append(delta.get("content") or delta.get("reasoning_content") or choice.get("text") or "")
    return "".join(chunks)


def _generated_text_from_json(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content") or choice.get("text") or ""


def _source(project_root: Path) -> dict[str, Any]:
    return {
        "branch": _git(project_root, "branch", "--show-current") or "unknown",
        "commit": _git(project_root, "rev-parse", "--verify", "HEAD") or "unborn",
        "dirty": bool(_git(project_root, "status", "--short")),
    }


def _git(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _mkdirs(run_dir: Path) -> None:
    for item in (
        run_dir,
        run_dir / "raw_streams",
        run_dir / "engine_logs",
        run_dir / "command_logs",
        run_dir.parent.parent / "current",
    ):
        item.mkdir(parents=True, exist_ok=True)


def _current_state(engine: str) -> Path:
    return Path("artifacts/current") / f"{engine}.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_output(output: str) -> list[str]:
    return output.strip().splitlines()


def _as_bytes(value: bytes | str) -> bytes:
    return value if isinstance(value, bytes) else value.encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
    return str(path).replace("\\", "/")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()
