# P7 Trace-Grounded Capacity Planner

P7 adds an offline capacity planner that reads validated study artifacts and
labels every result by evidence range. It does not launch engines, route live
traffic, or create P8 serving-policy logic.

The planner accepts a traffic/SLO request, a comparison type, an environment
fingerprint, dated user-provided pricing metadata, and study `latest.json`
paths. Evidence is usable only when the study comparison type and environment
fingerprint match the request and every referenced run validates through the
existing artifact contract.

The first planner is intentionally simple: nearest-neighbor selection with
bounded range checks. It reports nearest evidence, distance components,
observed ranges, a leave-one-run-out holdout evaluation, a naive same-candidate
median baseline, confidence, and one of:

- `IN_RANGE`
- `INTERPOLATED`
- `EXTRAPOLATED`
- `UNAVAILABLE`

`EXTRAPOLATED` is not measured capacity and is never a hard recommendation.
Cost remains `UNAVAILABLE` unless the input includes dated user-provided
candidate prices.

Run:

```bash
make plan-capacity PLAN=configs/planner/p7_capacity.yaml
```

Outputs:

- `artifacts/planner/p7_capacity_strict_t4_qwen35_2b/latest.json`
- `artifacts/planner/p7_capacity_strict_t4_qwen35_2b/holdout_eval.json`
- `artifacts/planner/p7_capacity_strict_t4_qwen35_2b/report.md`
