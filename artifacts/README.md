# Local Artifacts

This directory is for generated evidence, not source code.

Phase outputs are grouped by workflow when commands run:

- `runs/` for real engine runs
- `studies/` for repeated study records
- `compare/` for strict, deployment-profile, and engine-configuration reports
- `planner/` for capacity-planner outputs
- `router/` for offline replay outputs
- `hardening/` for hardening gate outputs
- `dashboard/` for local dashboard job outputs and logs

The files are ignored by Git so large traces, logs, raw streams, and host-local
details stay out of published history.
