# Offline Serving-Policy Replay

Offline replay evaluates serving-policy choices from held-out trace data. It
reads existing validated study artifacts, holds out each workload block, and
evaluates policy decisions against the hidden outcomes for that block.

The decision log records one row per replayed request with request features,
candidate profiles, held-out outcomes, training block IDs, and policy
decisions. Configured signals include prompt length, output length, cache state,
candidate engine/profile, queue depth, observed headroom, failure risk, quality
compatibility, and uncertainty.

Run:

```bash
make replay-router ROUTER=configs/router/offline_router.yaml
```

Outputs are written under `artifacts/router/`.
