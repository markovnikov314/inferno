from inferno.redaction import REDACTION, env_secret_values, redact, ssh_secret_values


def test_redacts_sensitive_mapping_keys_and_explicit_secret_values() -> None:
    payload = {
        "api_key": "live_secret",
        "message": "connecting with live_secret",
        "nested": {"token": "nested_secret", "safe": "visible"},
    }

    result = redact(payload, secrets=("live_secret",))

    assert result["api_key"] == REDACTION
    assert result["message"] == f"connecting with {REDACTION}"
    assert result["nested"]["token"] == REDACTION
    assert result["nested"]["safe"] == "visible"


def test_env_secret_values_only_collects_secret_looking_keys() -> None:
    env = {
        "INFERNO_API_KEY": "abc123",
        "INFERNO_GPU_SSH": "user@gpu-host",
        "PASSWORD": "pass123",
    }

    assert env_secret_values(env) == ("abc123", "pass123")


def test_tokenizer_metadata_is_not_secret() -> None:
    payload = {
        "tokenizer_id": "Qwen/Qwen3.5-4B",
        "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizer_auth_token": "secret-token",
    }

    result = redact(payload)

    assert result["tokenizer_id"] == payload["tokenizer_id"]
    assert result["tokenizer_revision"] == payload["tokenizer_revision"]
    assert result["tokenizer_auth_token"] == REDACTION


def test_redacts_ssh_target_pieces_and_private_ips() -> None:
    target = "user@gpu-host"
    private_ip = ".".join(("192", "168", "1", "20"))
    result = redact(
        f"ssh user@gpu-host from {private_ip} as user on gpu-host",
        secrets=ssh_secret_values(target),
    )

    assert "user" not in result
    assert "gpu-host" not in result
    assert private_ip not in result
