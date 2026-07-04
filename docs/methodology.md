# Project Inferno v1.0 Research Core Methodology

Inferno v1.0 is a strict vLLM vs SGLang research package. The engine is the experimental factor; model revision, tokenizer revision, weight format, precision, quantization, hardware, context limit, workload, prompt template, seed, sampling, cache state, warmup, retry policy, and failure policy must match within each strict comparison.

Each P5 study uses real OpenAI-compatible engine processes on a supported GPU host. A manifest is written before the first request, then raw stream evidence, request traces, telemetry, summaries, checksums, validation output, and report inputs are retained locally. Reports are rendered only from validator-backed artifacts.

Primary metrics are recomputed from client-side traces: TTFT, end-to-end latency, inter-token latency, request throughput, output-token throughput when usage data is available, and error rate. Repeat-level uncertainty is reported from independent repeats, not from request-level samples alone.

Response evidence is part of the comparison. Strict reports include E1 configuration equivalence and E2 output fingerprints; structured-output-lite also records an E3 JSON-validity status without publishing private raw output.
