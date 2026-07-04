# Privacy And Redaction

Local private artifacts stay under `artifacts/` and remain gitignored. The public release package contains redacted derivatives only.

Included package materials:

- static reports and redacted report inputs;
- run summaries, validation records, telemetry when present, and redacted timing traces;
- run metadata with model, engine, hardware, workload, source commit, and source artifact hashes;
- release docs and a package manifest with SHA256 hashes.

Excluded package materials:

- raw streams and generated text;
- private prompts and request bodies;
- engine logs and unredacted command output;
- SSH targets, usernames, hostnames, private IPs, tokens, credentials, model weights, and cache contents.

Package validation fails if report inputs point to missing files, strict keys are uncontrolled, runs are invalid, LOW_SAMPLE labels are missing, package hashes do not match, or obvious private path/SSH patterns are present.
