.PHONY: doctor doctor-gpu up engine-health real-stream run-real validate study report compare release-package plan-capacity replay-router hardening-check dashboard down

PYTHONPATH := src
export PYTHONPATH

ENGINE ?= vllm
MODEL ?= Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc
PROMPT ?= Explain continuous batching in one sentence.
CONFIG ?= configs/workloads/smoke_real.yaml
STUDY ?= configs/study/vllm_smoke.yaml
RUN ?=
MODE ?=
RUNS ?=
PLAN ?= configs/planner/capacity.yaml
ROUTER ?= configs/router/offline_router.yaml
HARDENING ?= configs/hardening/repo_audit.yaml

doctor:
	uv run python -m inferno.cli doctor --output artifacts/preflight/doctor.json

doctor-gpu:
	uv run python -m inferno.cli doctor-gpu --output artifacts/preflight/doctor-gpu.json

up:
	uv run python -m inferno.cli up --engine "$(ENGINE)" --model "$(MODEL)"

engine-health:
	uv run python -m inferno.cli engine-health --engine "$(ENGINE)"

real-stream:
	uv run python -m inferno.cli real-stream --engine "$(ENGINE)" --prompt "$(PROMPT)"

run-real:
	uv run python -m inferno.cli run-real --engine "$(ENGINE)" --config "$(CONFIG)"

validate:
	uv run python -m inferno.cli validate --run "$(RUN)"

study:
	uv run python -m inferno.cli study --engine "$(ENGINE)" --config "$(CONFIG)"

report:
	uv run python -m inferno.cli report --study "$(STUDY)"

compare:
	uv run python -m inferno.cli compare --mode "$(MODE)" --runs "$(RUNS)"

release-package:
	uv run python -m inferno.cli release-package --mode "$(MODE)" --study "$(STUDY)"

plan-capacity:
	uv run python -m inferno.cli plan-capacity --config "$(PLAN)"

replay-router:
	uv run python -m inferno.cli replay-router --config "$(ROUTER)"

hardening-check:
	uv run python -m inferno.cli hardening-check --config "$(HARDENING)"

dashboard:
	uv run python -m inferno.cli dashboard

down:
	uv run python -m inferno.cli down --engine "$(ENGINE)"
