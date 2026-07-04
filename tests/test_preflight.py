from pathlib import Path

from inferno import preflight
from inferno.preflight import CommandResult, collect_gpu_checks, collect_preflight


def test_local_preflight_shape_and_ok_status(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'inferno'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "C:/bin/uv.exe" if name == "uv" else None)

    result = collect_preflight(project_root=tmp_path, env={})

    assert result["schema_version"] == 1
    assert result["status"] == "ok"
    assert isinstance(result["checks"], list)
    assert {check["name"] for check in result["checks"]} == {"python", "uv", "project_lock"}
    for check in result["checks"]:
        assert set(check) == {"name", "status", "message", "details"}


def test_gpu_preflight_without_env_is_warn() -> None:
    checks = collect_gpu_checks(env={})

    assert checks == [
        {
            "name": "gpu_ssh_config",
            "status": "warn",
            "message": "INFERNO_GPU_SSH is not set; skipping remote GPU preflight",
            "details": {},
        }
    ]


def test_gpu_preflight_uses_inferno_gpu_ssh(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "C:/Windows/System32/OpenSSH/ssh.exe")

    def runner(args, timeout):
        calls.append((args, timeout))
        return CommandResult(0, "NVIDIA A100, 550.54, 40960 MiB\n", "")

    result = collect_preflight(
        include_gpu=True,
        env={
            "INFERNO_GPU_SSH": "user@gpu-host",
            "INFERNO_API_KEY": "secret-value",
            "USERNAME": "alice",
            "USERPROFILE": "C:/Users/alice",
        },
        runner=runner,
    )

    assert result["status"] == "ok"
    assert len(calls) == len(preflight.REMOTE_COMMANDS)
    gpu_check = result["checks"][-1]
    assert gpu_check["name"] == "remote_gpu_container"
    assert gpu_check["status"] == "ok"
    assert gpu_check["details"]["target"] == "[REDACTED]"
    assert gpu_check["details"]["output"] == ["NVIDIA A100, 550.54, 40960 MiB"]


def test_local_identity_is_redacted_from_paths(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "alice" / "repo"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'inferno'\n", encoding="utf-8")
    (project / "uv.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "C:/Users/alice/bin/uv.exe")

    result = collect_preflight(project_root=project, env={"USERNAME": "alice"})

    assert "alice" not in str(result)
