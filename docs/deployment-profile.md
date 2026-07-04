# Deployment Profile Comparisons

Deployment-profile reports compare practical serving setups when more than the
engine differs. They are useful for operational tradeoffs, but they should not
be read as strict engine-only attribution.

The llama.cpp profile differs from strict vLLM/SGLang runs in weight format,
precision, tokenizer source, container image, and runtime controls.

| Control | Strict vLLM/SGLang | llama.cpp profile |
|---|---|---|
| Engine | vLLM or SGLang | llama.cpp `llama-server` |
| Weight format | Hugging Face weights | GGUF |
| Precision | half | quantized |
| Quantization | none | Q4_K_M |
| Tokenizer source | Hugging Face tokenizer revision | GGUF metadata |
| Container/build | pinned vLLM/SGLang images | pinned llama.cpp CUDA image |

Use `MODE=deployment-profile` for these reports.
