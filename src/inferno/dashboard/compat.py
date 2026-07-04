"""GPU-aware engine recommendations for the local dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

from inferno.preflight import GPU_ENV_VAR, SSH_OPTIONS, CommandResult, Runner
from inferno.preflight import collect_preflight, secret_values
from inferno.redaction import redact


STATUS_VALIDATED = "Validated"
STATUS_NEEDS_SSH = "Needs SSH"
STATUS_UNPROVEN = "Unproven on this GPU"
STATUS_LEGACY = "Legacy profile"
STATUS_FAILED = "Failed"


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver_version: str
    memory_mib: int | None
    compute_capability: str | None
    cuda_version: str | None
    docker_gpu_ok: bool


ENGINE_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "vllm",
        "label": "vLLM",
        "comparison_scope": "strict_engine_comparison",
        "default_model": "Qwen/Qwen3.5-2B",
        "config_path": "configs/engines/vllm.json",
    },
    {
        "id": "sglang",
        "label": "SGLang",
        "comparison_scope": "strict_engine_comparison",
        "default_model": "Qwen/Qwen3.5-2B",
        "config_path": "configs/engines/sglang.json",
    },
    {
        "id": "ollama",
        "label": "Ollama",
        "comparison_scope": "deployment_profile_comparison",
        "default_model": "qwen3.5:2b",
        "config_path": "configs/engines/ollama.json",
    },
    {
        "id": "tensorrtllm",
        "label": "TensorRT-LLM",
        "comparison_scope": "engine_configuration_comparison",
        "default_model": "Qwen/Qwen3.5-2B",
        "config_path": "configs/engines/tensorrtllm.json",
        "profiles": ["tensorrtllm", "tensorrtllm_t4"],
    },
)


def engine_catalog() -> dict[str, Any]:
    return {"engines": list(ENGINE_CATALOG), "statuses": dashboard_statuses()}


def dashboard_statuses() -> list[str]:
    return [STATUS_VALIDATED, STATUS_NEEDS_SSH, STATUS_UNPROVEN, STATUS_LEGACY, STATUS_FAILED]


def gpu_preflight(
    *,
    env: Mapping[str, str] | None = None,
    project_root: Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    current_env = dict(os.environ if env is None else env)
    preflight = collect_preflight(
        include_gpu=True,
        env=current_env,
        project_root=project_root,
        runner=runner,
    )
    target = current_env.get(GPU_ENV_VAR, "").strip()
    if not target:
        return {
            "status": STATUS_NEEDS_SSH,
            "gpu": None,
            "preflight": preflight,
            "recommendations": recommend_engines(None),
        }

    result = _run_probe(target, runner)
    if result.returncode != 0:
        return redact(
            {
                "status": STATUS_FAILED,
                "gpu": None,
                "preflight": preflight,
                "probe": {
                    "returncode": result.returncode,
                    "stdout": result.stdout.splitlines()[:30],
                    "stderr": result.stderr.splitlines()[:30],
                },
                "recommendations": recommend_engines(None, failed=True),
            },
            secrets=secret_values(current_env),
        )

    gpu = parse_gpu_probe(result.stdout)
    return redact(
        {
            "status": "ok" if gpu else STATUS_FAILED,
            "gpu": asdict(gpu) if gpu else None,
            "preflight": preflight,
            "recommendations": recommend_engines(gpu),
        },
        secrets=secret_values(current_env),
    )


def recommend_engines(gpu: GpuInfo | None, *, failed: bool = False) -> list[dict[str, Any]]:
    if failed:
        return [
            _recommendation(item["id"], STATUS_FAILED, [], None)
            for item in ENGINE_CATALOG
        ]
    if gpu is None:
        return [
            _recommendation(item["id"], STATUS_NEEDS_SSH, ["Set INFERNO_GPU_SSH or enter SSH for this session."], None)
            for item in ENGINE_CATALOG
        ]

    is_t4 = _is_t4(gpu)
    docker_gpu_warning = (
        []
        if gpu.docker_gpu_ok
        else ["Docker GPU smoke failed; GPU engines may fail and Ollama may fall back to CPU/slow execution."]
    )
    strict_status = STATUS_VALIDATED if is_t4 and gpu.docker_gpu_ok else STATUS_UNPROVEN
    strict_warning = (
        ["Existing validated evidence is for Tesla T4 with Qwen/Qwen3.5-2B."]
        if not is_t4
        else []
    )
    recommendations = [
        _recommendation("vllm", strict_status, [*strict_warning, *docker_gpu_warning], "vllm"),
        _recommendation("sglang", strict_status, [*strict_warning, *docker_gpu_warning], "sglang"),
        _recommendation(
            "ollama",
            STATUS_UNPROVEN if gpu.docker_gpu_ok else STATUS_FAILED,
            docker_gpu_warning
            or ["Ollama will run as deployment-profile evidence, not a strict comparison."],
            "ollama",
        ),
    ]
    if is_t4:
        recommendations.append(
            _recommendation(
                "tensorrtllm",
                STATUS_FAILED,
                [
                    "Tesla T4 / SM75 uses the legacy TensorRT-LLM 0.11.0 profile.",
                    "Qwen/Qwen3.5-2B is forced by dashboard default and is a known failed T4 path: TensorRT-LLM 0.11.0 does not support Qwen3_5ForConditionalGeneration.",
                    *docker_gpu_warning,
                ],
                "tensorrtllm_t4",
            )
        )
    else:
        recommendations.append(
            _recommendation(
                "tensorrtllm",
                STATUS_UNPROVEN if gpu.docker_gpu_ok else STATUS_FAILED,
                [
                    "Current TensorRT-LLM profile needs real validation on this GPU before readiness is claimed.",
                    *docker_gpu_warning,
                ],
                "tensorrtllm",
            )
        )
    return recommendations


def parse_gpu_probe(stdout: str) -> GpuInfo | None:
    sections = _sections(stdout)
    gpu_lines = [line for line in sections.get("INFERNO_GPU_QUERY", []) if line.strip()]
    if not gpu_lines:
        return None
    parts = [part.strip() for part in gpu_lines[0].split(",")]
    name = parts[0] if parts else "unknown"
    driver = parts[1] if len(parts) > 1 else "unknown"
    memory_mib = _int_or_none(parts[2] if len(parts) > 2 else None)
    compute = parts[3] if len(parts) > 3 and parts[3] else None
    cuda = next((line.strip() for line in sections.get("INFERNO_CUDA_QUERY", []) if line.strip()), None)
    docker_lines = [line.strip() for line in sections.get("INFERNO_DOCKER_GPU", []) if line.strip()]
    docker_gpu_ok = bool(docker_lines and docker_lines[-1] == "0")
    return GpuInfo(
        name=name,
        driver_version=driver,
        memory_mib=memory_mib,
        compute_capability=compute,
        cuda_version=cuda,
        docker_gpu_ok=docker_gpu_ok,
    )


def _recommendation(
    engine: str,
    status: str,
    warnings: list[str],
    selected_profile: str | None,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "status": status,
        "selected_profile": selected_profile,
        "warnings": warnings,
    }


def _is_t4(gpu: GpuInfo) -> bool:
    return "t4" in gpu.name.lower() or gpu.compute_capability == "7.5"


def _sections(stdout: str) -> dict[str, list[str]]:
    markers = {"INFERNO_GPU_QUERY", "INFERNO_CUDA_QUERY", "INFERNO_DOCKER_GPU"}
    current: str | None = None
    sections: dict[str, list[str]] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line in markers:
            current = line
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(raw_line)
    return sections


def _run_probe(target: str, runner: Runner | None) -> CommandResult:
    script = """set +e
echo INFERNO_GPU_QUERY
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits 2>&1
echo INFERNO_CUDA_QUERY
nvidia-smi | sed -n 's/.*CUDA Version: *\\([0-9.]*\\).*/\\1/p' | head -n 1
echo INFERNO_DOCKER_GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1
echo $?
"""
    args = ["ssh", *SSH_OPTIONS, *shlex.split(target), script]
    run = _run_command if runner is None else runner
    try:
        return run(args, 240)
    except subprocess.TimeoutExpired:
        return CommandResult(124, "", "gpu probe timed out")
    except OSError as exc:
        return CommandResult(1, "", f"failed to start ssh: {exc}")


def _run_command(args: Sequence[str], timeout: int) -> CommandResult:
    completed = subprocess.run(
        list(args),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(str(value).strip()) if value not in (None, "") else None
    except ValueError:
        return None
