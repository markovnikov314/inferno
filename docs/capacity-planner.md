# Trace-Grounded Capacity Planner

The capacity planner reads validated study artifacts and labels every estimate
by evidence range. It does not launch engines or route live traffic.

The planner accepts a traffic/SLO request, comparison type, environment
fingerprint, optional dated pricing metadata, and study `latest.json` paths.
Evidence is usable only when the comparison type and environment fingerprint
match the request and every referenced run validates.

The planner uses nearest-neighbor selection with bounded range checks. It
reports nearest evidence, distance components, observed ranges, holdout error,
baseline error, confidence, and one of `IN_RANGE`, `INTERPOLATED`,
`EXTRAPOLATED`, or `UNAVAILABLE`.

Run:

```bash
make plan-capacity PLAN=configs/planner/capacity.yaml
```

Outputs are written under `artifacts/planner/`.
