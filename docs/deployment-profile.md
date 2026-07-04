# Deployment Profile Comparisons

Project Inferno P6 adds llama.cpp only as a deployment-profile comparison path.
It is not a strict engine-only comparison.

Required banner:

> DEPLOYMENT PROFILE COMPARISON — engine and one or more controlled deployment variables differ. Do not attribute outcomes to engine alone.

The P6 local profile intentionally differs from P5 strict vLLM/SGLang runs:

| Control | P5 strict vLLM/SGLang | P6 llama.cpp profile |
|---|---|---|
| Engine | vLLM or SGLang | llama.cpp `llama-server` |
| Weight format | Hugging Face weights | GGUF |
| Precision | half | quantized |
| Quantization | none | Q4_K_M |
| Tokenizer source | Hugging Face tokenizer revision | tokenizer metadata embedded in the GGUF artifact |
| Container/build | vLLM/SGLang pinned images | `ghcr.io/ggml-org/llama.cpp:server-cuda` pinned by digest |

Strict mode must continue to reject llama.cpp and any mismatched GGUF,
precision, quantization, tokenizer, or runtime controls. Use
`MODE=deployment-profile` for P6 reports, and interpret results as practical
profile outcomes rather than engine-only attribution.
