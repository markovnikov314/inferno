"""Local and remote preflight collection for Project Inferno."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any, Literal

from inferno.redaction import env_secret_values, redact, ssh_secret_values

Status = Literal["ok", "warn", "error"]
Runner = Callable[[Sequence[str], int], "CommandResult"]

SCHEMA_VERSION = 1
GPU_ENV_VAR = "INFERNO_GPU_SSH"
GPU_QUERY = "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ConnectTimeout=15",
)
REMOTE_COMMANDS = {
    "remote_os": "uname -srm; cat /etc/os-release 2>/dev/null | head -n 3",
    "remote_python": "python3 --version || python --version",
    "remote_git": "git --version",
    "remote_docker": (
        "docker --version; "
        "docker info --format 'Runtimes: {{range $k, $_ := .Runtimes}}{{$k}} {{end}}"
        "Default: {{.DefaultRuntime}}'"
    ),
    "remote_gpu": GPU_QUERY,
    "remote_gpu_container": (
        "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
    ),
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def collect_preflight(
    *,
    include_gpu: bool = False,
    env: Mapping[str, str] | None = None,
    project_root: Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Collect local checks and, optionally, the SSH GPU preflight."""

    current_env = os.environ if env is None else env
    checks = collect_local_checks(project_root=project_root)

    if include_gpu:
        checks.extend(collect_gpu_checks(env=current_env, runner=runner))

    redacted_checks = redact(checks, secrets=secret_values(current_env))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall_status(redacted_checks),
        "checks": redacted_checks,
    }


def collect_local_checks(*, project_root: Path | None = None) -> list[dict[str, Any]]:
    """Collect checks that do not need network or accelerator access."""

    root = Path.cwd() if project_root is None else project_root
    checks = [
        _check(
            "python",
            "ok" if sys.version_info >= (3, 11) else "error",
            f"Python {platform.python_version()}",
            {"executable": sys.executable},
        ),
    ]

    uv_path = shutil.which("uv")
    checks.append(
        _check(
            "uv",
            "ok" if uv_path else "error",
            "uv is available" if uv_path else "uv was not found on PATH",
            {"path": uv_path},
        )
    )

    missing = [name for name in ("pyproject.toml", "uv.lock") if not (root / name).exists()]
    checks.append(
        _check(
            "project_lock",
            "ok" if not missing else "warn",
            "pyproject.toml and uv.lock are present"
            if not missing
            else f"Missing project file(s): {', '.join(missing)}",
            {"root": str(root), "missing": missing},
        )
    )

    return checks


def collect_gpu_checks(
    *,
    env: Mapping[str, str] | None = None,
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
    """Collect GPU preflight checks over SSH using ``INFERNO_GPU_SSH``."""

    current_env = os.environ if env is None else env
    target = current_env.get(GPU_ENV_VAR, "").strip()
    if not target:
        return [
            _check(
                "gpu_ssh_config",
                "warn",
                f"{GPU_ENV_VAR} is not set; skipping remote GPU preflight",
            )
        ]

    if not shutil.which("ssh"):
        return [_check("gpu_ssh", "error", "ssh was not found on PATH")]

    run = _run_command if runner is None else runner
    checks = []
    for name, command in REMOTE_COMMANDS.items():
        checks.append(_remote_check(name, target, command, run))

    return checks


def secret_values(env: Mapping[str, str]) -> tuple[str, ...]:
    """Collect values that must not appear in committed diagnostics."""

    target = env.get(GPU_ENV_VAR, "").strip()
    local_identity = (
        env.get("USERNAME", ""),
        env.get("USER", ""),
        env.get("USERPROFILE", ""),
        env.get("HOME", ""),
    )
    return (*env_secret_values(env), *local_identity, *ssh_secret_values(target))


def _remote_check(name: str, target: str, command: str, run: Runner) -> dict[str, Any]:
    args = ["ssh", *SSH_OPTIONS, *shlex.split(target), command]
    timeout = 240 if name == "remote_gpu_container" else 30

    try:
        result = run(args, timeout)
    except subprocess.TimeoutExpired:
        return _check(name, "error", f"{name} timed out", {"target": target})
    except OSError as exc:
        return _check(name, "error", f"failed to start ssh: {exc}", {"target": target})

    output = (result.stdout or result.stderr).strip()
    details = {"target": target, "returncode": result.returncode, "output": output.splitlines()[:20]}
    if result.returncode == 0:
        status: Status = "ok" if output else "warn"
        message = f"{name} succeeded" if output else f"{name} succeeded with no output"
        return _check(name, status, message, details)

    diagnostic = (result.stderr or result.stdout).strip() or f"ssh exited {result.returncode}"
    return _check(name, "error", f"{name} failed: {diagnostic}", details)


def overall_status(checks: Sequence[Mapping[str, Any]]) -> Status:
    """Collapse check statuses into one status."""

    statuses = {check.get("status") for check in checks}
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _check(
    name: str,
    status: Status,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "details": dict(details or {}),
    }


def _run_command(args: Sequence[str], timeout: int) -> CommandResult:
    completed = subprocess.run(
        list(args),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)
