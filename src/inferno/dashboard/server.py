"""FastAPI server for the local dashboard."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
from pathlib import Path
import threading
import webbrowser
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inferno.dashboard.compat import GPU_ENV_VAR, engine_catalog, gpu_preflight
from inferno.dashboard.jobs import DashboardJobManager


class PreflightRequest(BaseModel):
    ssh_target: str | None = Field(default=None, alias="sshTarget")


class TestParameters(BaseModel):
    model_profile: str = Field(default="qwen35-2b", alias="modelProfile")
    prompt: str = "Explain local model serving in one concise sentence."
    batch_prompts: str = Field(default="", alias="batchPrompts")
    total_requests: int = Field(default=1, ge=1, le=128, alias="totalRequests")
    concurrent_users: int = Field(default=1, ge=1, le=16, alias="concurrentUsers")
    request_rate_rps: float = Field(default=0.1, gt=0, le=100, alias="requestRateRps")
    warmup_requests: int = Field(default=1, ge=0, le=32, alias="warmupRequests")
    kv_cache_mode: Literal["engine_default", "compact"] = Field(default="engine_default", alias="kvCacheMode")
    max_tokens: int = Field(default=32, ge=1, le=1024, alias="maxTokens")
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1, alias="topP")
    timeout_seconds: int = Field(default=180, ge=30, le=3600, alias="timeoutSeconds")


class JobRequest(BaseModel):
    engines: list[str] = Field(min_length=1)
    parameters: TestParameters = Field(default_factory=TestParameters)
    ssh_target: str | None = Field(default=None, alias="sshTarget")


def create_app(*, project_root: Path | None = None) -> FastAPI:
    root = Path.cwd() if project_root is None else project_root
    manager = DashboardJobManager(project_root=root)
    app = FastAPI(title="Inferno Dashboard", version="0.1.0")

    @app.get("/api/engines")
    def get_engines() -> dict[str, Any]:
        return engine_catalog()

    @app.post("/api/preflight")
    def post_preflight(request: PreflightRequest) -> dict[str, Any]:
        env = dict(os.environ)
        if request.ssh_target:
            env[GPU_ENV_VAR] = request.ssh_target
        return gpu_preflight(env=env, project_root=root)

    @app.post("/api/jobs")
    def create_job(request: JobRequest) -> dict[str, Any]:
        allowed = {"vllm", "sglang", "ollama", "tensorrtllm", "tensorrtllm_t4"}
        unknown = sorted(set(request.engines) - allowed)
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown engines: {unknown}")
        if not (request.ssh_target or os.environ.get(GPU_ENV_VAR)):
            raise HTTPException(status_code=400, detail="SSH access is required to run tests")
        try:
            job = manager.create_job(
                engines=request.engines,
                parameters=request.parameters.model_dump(by_alias=False),
                ssh_target=request.ssh_target,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job.snapshot()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job.snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = manager.cancel_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        async def stream():
            sent = 0
            while True:
                job = manager.get_job(job_id)
                if not job:
                    yield "event: error\ndata: {\"detail\":\"job not found\"}\n\n"
                    return
                events = job.events[sent:]
                for event in events:
                    sent += 1
                    yield f"event: message\ndata: {_json_sse(event)}\n\n"
                if job.status in {"succeeded", "failed", "cancelled"} and sent >= len(job.events):
                    yield f"event: done\ndata: {_json_sse(job.snapshot())}\n\n"
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/artifacts")
    def get_artifact(path: str) -> FileResponse:
        target = Path(path)
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        artifacts_root = (root / "artifacts").resolve()
        try:
            target.relative_to(artifacts_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="only artifact files are served") from exc
        if not target.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(target)

    dist = root / "web" / "dashboard" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")

    return app


def run_dashboard(*, host: str, port: int, open_browser: bool, project_root: Path | None = None) -> int:
    import uvicorn

    if not _is_loopback_host(host):
        raise ValueError("P11 dashboard is local-only; use 127.0.0.1, localhost, or ::1")
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(project_root=project_root), host=host, port=port)
    return 0


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _json_sse(payload: Any) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
