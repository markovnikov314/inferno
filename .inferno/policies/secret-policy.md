# Secret Policy

- Put local secrets in ignored `.env` files or shell environment variables.
- `INFERNO_GPU_SSH` may contain a real SSH target but must never be committed.
- Redact usernames, hostnames, SSH targets, IP addresses, and token-like values
  before writing evidence.
- Public artifacts must not contain private prompts, raw private outputs, model
  weights, hostnames, internal IPs, or credentials.
