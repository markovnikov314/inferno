# P8 Offline Serving-Policy Router Replay

P8 evaluates serving-policy choices from held-out trace replay. It does not add
a live router, live endpoint, autoscaler, dashboard, or serving control plane.

The replay reads existing validated study artifacts, groups requests by
workload block, and evaluates each block with all runs from that block held out
of policy training. The decision log records one row per replayed request with
the request features, candidate profiles, held-out outcomes, train block IDs,
and policy decisions.

Configured features include:

- canonical prompt length from `prompt_chars`
- requested output length from `sampling.max_tokens`
- prefix/cache policy from `cache_state`
- candidate engine/config/profile
- replay queue depth
- observed training headroom
- failure risk
- quality compatibility
- uncertainty/confidence

The strict P8 config evaluates these policies:

- static vLLM
- static SGLang
- transparent heuristic
- oracle lower bound, explicitly non-deployable
- proposed policy

Static llama.cpp is omitted from the strict config because llama.cpp is only a
deployment-profile comparison in this repository. Strict mode must continue to
reject llama.cpp.

Run:

```bash
make replay-router ROUTER=configs/router/p8_offline_router.yaml
```

Outputs:

- `artifacts/router/p8_offline_policy_replay/latest.json`
- `artifacts/router/p8_offline_policy_replay/decision_log.jsonl`
- `artifacts/router/p8_offline_policy_replay/leakage_check.json`
- `artifacts/router/p8_offline_policy_replay/ablation_eval.json`
- `artifacts/router/p8_offline_policy_replay/report.md`
