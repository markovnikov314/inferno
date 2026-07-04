# Local Artifacts

This directory stores generated evidence and local dashboard output.

Common subdirectories:

- `runs/` for engine runs
- `studies/` for repeated study records
- `compare/` for comparison reports
- `planner/` for capacity-planner outputs
- `router/` for offline replay outputs
- `hardening/` for repository audit outputs
- `dashboard/` for dashboard jobs and logs

Generated files are ignored by Git so traces, logs, raw streams, and host-local
details stay out of published history.
