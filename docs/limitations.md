# Project Inferno v1.0 Research Core Limitations

Release claims are limited to vLLM, SGLang, Qwen/Qwen3.5-2B, the pinned engine images, the observed GPU host class, and the configured P5 workload envelope.

The P5 lite workloads are intentionally small. They exercise smoke, interactive, decode-heavy, long-context-lite, and structured-output-lite behavior, but they are not a universal serving benchmark.

LOW_SAMPLE labels are expected when completed request counts are below the configured threshold. They are not warnings to ignore; they bound how strongly the results can be interpreted.

No report may declare a universal winner. Differences outside the controlled comparison key require a new study type and cannot be attributed to engine identity alone.

The public package is redacted. Raw streams, private prompts, engine logs, host paths, SSH targets, usernames, hostnames, private IPs, tokens, and model weights are excluded.
