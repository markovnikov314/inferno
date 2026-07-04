"""Small helpers for removing secrets from diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import re

REDACTION = "[REDACTED]"
PRIVATE_IPV4 = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3})\b"
)

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
)

TOKENIZER_METADATA_KEYS = {"tokenizer_id", "tokenizer_revision"}


def is_secret_key(key: object) -> bool:
    """Return whether a mapping key commonly carries sensitive material."""

    normalized = str(key).lower().replace("-", "_")
    if normalized in TOKENIZER_METADATA_KEYS:
        return False
    return any(part in normalized for part in SECRET_KEY_PARTS)


def redact(value: Any, *, secrets: Sequence[str] = ()) -> Any:
    """Return ``value`` with secret-looking data replaced.

    Mapping values are redacted when their key looks sensitive. String values
    also have any explicit non-empty secret values replaced in-place.
    """

    secret_values = tuple(secret for secret in secrets if secret)

    if isinstance(value, str):
        redacted = value
        for secret in secret_values:
            redacted = redacted.replace(secret, REDACTION)
        return PRIVATE_IPV4.sub(REDACTION, redacted)

    if isinstance(value, Mapping):
        return {
            key: REDACTION if is_secret_key(key) else redact(item, secrets=secret_values)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(redact(item, secrets=secret_values) for item in value)

    if isinstance(value, list):
        return [redact(item, secrets=secret_values) for item in value]

    return value


def env_secret_values(env: Mapping[str, str]) -> tuple[str, ...]:
    """Collect secret-looking environment values for output redaction."""

    return tuple(value for key, value in env.items() if value and is_secret_key(key))


def ssh_secret_values(target: str) -> tuple[str, ...]:
    """Collect sensitive pieces from an SSH target string."""

    parts = [target]
    for item in target.replace("@", " ").split():
        if item and not item.startswith("-"):
            parts.append(item)
    return tuple(dict.fromkeys(parts))
