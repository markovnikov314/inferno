# ADR-0003: P10 TensorRT-LLM and Expanded Workloads

## Status

Accepted for engine-configuration support. The current TensorRT-LLM release
profile is blocked on the Tesla T4 host; the legacy T4 profile has validated
real evidence.

## Context

P10 is authorized to add TensorRT-LLM support and broaden workload parameters
without weakening P5 strict vLLM/SGLang evidence or P6 deployment-profile
semantics. TensorRT-LLM uses its own official NVIDIA release container,
serving command, runtime backend, model-loading path, and engine knobs.

The current official TensorRT-LLM documentation exposes `trtllm-serve serve`
with OpenAI-compatible endpoints, and NVIDIA publishes the release container on
NGC. The current support matrix lists Blackwell, Grace Hopper, Hopper, Ada, and
Ampere GPU families; the configured GPU host is a Tesla T4. The TensorRT-LLM
v0.11 support matrix lists Turing/SM75/T4, but that generation does not expose
the current `trtllm-serve` path.

## Decision

Add TensorRT-LLM as `engine-configuration` profiles, not strict engine-only
profiles. The current profile keeps the NVIDIA TensorRT-LLM release container
and P5 Qwen model pin for newer supported GPUs. The T4 profile pins the
official NVIDIA Triton/TensorRT-LLM backend image and uses the legacy
TensorRT-LLM `LLM` API with a small TinyLlama model that validated on T4.

Both profiles record launch arguments, model/tokenizer provenance, hardware,
telemetry, checksums, validation, shutdown evidence, and engine-native metadata.

Extend workload and manifest parsing with optional, backward-compatible fields:
request rate, arrival model, concurrency/max-in-flight, prompt/output target
metadata, context length, cache/prefix policy, streaming, timeout, retry and
failure policy, quality requirement, cost metadata, and engine-specific runtime
knobs such as tensor/pipeline parallelism and KV-cache settings.

Strict comparison remains limited to the P5 vLLM/SGLang envelope. TensorRT-LLM
may only be compared with `MODE=engine-configuration` unless a future strict
comparison type proves all controlled variables match and validators pass.

## Consequences

CPU tests cover parsing, compatibility keys, strict refusal, report validation,
and a fake TensorRT-LLM artifact path.

The current-release P10 GPU run on the current host did not pass: the official
TensorRT-LLM container starts, then exits before health with a runtime `sm_75`
architecture failure. The failed artifact is retained as negative evidence and
must not be converted into readiness, strict, or winner claims.

The legacy T4 profile did pass on the same host with TensorRT-LLM 0.11.0 and
TinyLlama. This proves only the legacy T4 TensorRT-LLM profile, not current
TensorRT-LLM serving, Qwen support, or strict engine equivalence.

P10 integration is allowed only as an engine-configuration result with the
above limitation. Strict mode remains vLLM/SGLang only.
