"""Command-line entrypoints for Project Inferno preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from inferno import artifact_contract
from inferno import hardening
from inferno import planner
from inferno.preflight import collect_preflight
from inferno import release
from inferno import router
from inferno import study
from inferno import vllm_remote


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inferno")
    subcommands = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("doctor", "Run local preflight checks"),
        ("doctor-gpu", "Run local checks plus remote GPU preflight"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--output", type=Path, help="Write redacted JSON evidence")
    up_command = subcommands.add_parser("up", help="Launch a real engine on the GPU host")
    up_command.add_argument("--engine", required=True)
    up_command.add_argument("--model")
    health_command = subcommands.add_parser("engine-health", help="Check engine health and identity")
    health_command.add_argument("--engine", required=True)
    stream_command = subcommands.add_parser("real-stream", help="Capture one real streamed response")
    stream_command.add_argument("--engine", required=True)
    stream_command.add_argument("--prompt", required=True)
    run_real_command = subcommands.add_parser("run-real", help="Run one real workload and validate artifacts")
    run_real_command.add_argument("--engine", required=True)
    run_real_command.add_argument("--config", required=True, type=Path)
    validate_command = subcommands.add_parser("validate", help="Validate a run artifact directory")
    validate_command.add_argument("--run", required=True, type=Path)
    study_command = subcommands.add_parser("study", help="Run a reproducible P3 study")
    study_command.add_argument("--engine", required=True)
    study_command.add_argument("--config", required=True, type=Path)
    report_command = subcommands.add_parser("report", help="Render a static study report")
    report_command.add_argument("--study", required=True, type=Path)
    compare_command = subcommands.add_parser("compare", help="Render a strict comparison report")
    compare_command.add_argument("--mode", required=True)
    compare_command.add_argument("--runs", required=True)
    release_command = subcommands.add_parser("release-package", help="Build a redacted release package")
    release_command.add_argument("--mode", required=True)
    release_command.add_argument("--study", required=True, type=Path)
    plan_command = subcommands.add_parser("plan-capacity", help="Run the P7 capacity planner")
    plan_command.add_argument("--config", required=True, type=Path)
    router_command = subcommands.add_parser("replay-router", help="Run the P8 offline router replay")
    router_command.add_argument("--config", required=True, type=Path)
    hardening_command = subcommands.add_parser("hardening-check", help="Run P9 hardening checks")
    hardening_command.add_argument("--config", required=True, type=Path)
    hardening_command.add_argument("--include-gpu-smoke", action="store_true")
    dashboard_command = subcommands.add_parser("dashboard", help="Run the local P11 browser dashboard")
    dashboard_command.add_argument("--host", default="127.0.0.1")
    dashboard_command.add_argument("--port", type=int, default=8765)
    dashboard_command.add_argument("--no-open", action="store_true")
    dashboard_command.add_argument("--smoke", action="store_true", help="Validate dashboard app wiring and exit")
    subcommands.add_parser("dump-contract-schemas", help="Print artifact JSON Schema snapshots")
    down_command = subcommands.add_parser("down", help="Stop a real engine on the GPU host")
    down_command.add_argument("--engine", required=True)

    args = parser.parse_args(argv)
    if args.command in {"doctor", "doctor-gpu"}:
        include_gpu = args.command == "doctor-gpu"
        result = collect_preflight(include_gpu=include_gpu, project_root=Path.cwd())
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if result["status"] == "error" else 0

    try:
        if args.command == "up":
            return vllm_remote.up(
                engine=args.engine,
                model_arg=args.model,
                project_root=Path.cwd(),
                env=os.environ,
            )
        if args.command == "engine-health":
            return vllm_remote.engine_health(
                engine=args.engine,
                project_root=Path.cwd(),
                env=os.environ,
            )
        if args.command == "real-stream":
            return vllm_remote.real_stream(
                engine=args.engine,
                prompt=args.prompt,
                project_root=Path.cwd(),
                env=os.environ,
            )
        if args.command == "run-real":
            return vllm_remote.run_real(
                engine=args.engine,
                config_path=args.config,
                project_root=Path.cwd(),
                env=os.environ,
            )
        if args.command == "validate":
            result = artifact_contract.validate_run(args.run, write=True)
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
            return 0 if result.ok else 1
        if args.command == "study":
            return study.run_study(
                engine=args.engine,
                config_path=args.config,
                project_root=Path.cwd(),
                env=os.environ,
            )
        if args.command == "report":
            return study.report_study(
                study_path=args.study,
                project_root=Path.cwd(),
            )
        if args.command == "compare":
            return study.compare_runs(
                mode=args.mode,
                runs_arg=args.runs,
                project_root=Path.cwd(),
            )
        if args.command == "release-package":
            return release.package_release(
                mode=args.mode,
                study_path=args.study,
                project_root=Path.cwd(),
            )
        if args.command == "plan-capacity":
            return planner.run_capacity_plan(
                config_path=args.config,
                project_root=Path.cwd(),
            )
        if args.command == "replay-router":
            return router.run_router_replay(
                config_path=args.config,
                project_root=Path.cwd(),
            )
        if args.command == "hardening-check":
            return hardening.run_hardening_check(
                config_path=args.config,
                project_root=Path.cwd(),
                include_gpu_smoke=args.include_gpu_smoke,
                env=os.environ,
            )
        if args.command == "dashboard":
            from inferno.dashboard.server import create_app, run_dashboard

            if args.smoke:
                app = create_app(project_root=Path.cwd())
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "title": app.title,
                            "routes": sorted(route.path for route in app.routes),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0

            return run_dashboard(
                host=args.host,
                port=args.port,
                open_browser=not args.no_open,
                project_root=Path.cwd(),
            )
        if args.command == "dump-contract-schemas":
            print(json.dumps(artifact_contract.schema_snapshots(), indent=2, sort_keys=True))
            return 0
        if args.command == "down":
            return vllm_remote.down(engine=args.engine, project_root=Path.cwd(), env=os.environ)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
