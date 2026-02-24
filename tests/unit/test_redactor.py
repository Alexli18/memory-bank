"""Tests for secret redaction — src/mb/redactor.py."""

from __future__ import annotations

import pytest

from mb.redactor import Redactor, RedactorConfig

# Build Stripe-like keys dynamically so GitHub push protection
# does not flag them as real secrets in source code.
_STRIPE_SUFFIX = "A1B2C3D4E5F6G7H8I9J0K1L2M3"
_SK_LIVE = "sk_" + "live_" + _STRIPE_SUFFIX
_PK_LIVE = "pk_" + "live_" + _STRIPE_SUFFIX
_SK_TEST = "sk_" + "test_" + _STRIPE_SUFFIX


class TestRedactorConfig:
    """RedactorConfig initialization and custom patterns."""

    def test_default_config_has_patterns(self) -> None:
        cfg = RedactorConfig()
        assert len(cfg.patterns) > 0

    def test_custom_patterns(self) -> None:
        cfg = RedactorConfig(extra_patterns=[("MY_SECRET_[0-9]+", "CUSTOM")])
        redactor = Redactor(cfg)
        result = redactor.redact("token is MY_SECRET_12345 here")
        assert "[REDACTED:CUSTOM]" in result
        assert "MY_SECRET_12345" not in result

    def test_disabled_redactor(self) -> None:
        cfg = RedactorConfig(enabled=False)
        redactor = Redactor(cfg)
        text = "AKIAIOSFODNN7EXAMPLE"
        assert redactor.redact(text) == text


class TestPassthrough:
    """Clean text must pass through unchanged."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Hello, world!",
            "No secrets here, just normal text",
            "import os\nprint(os.getcwd())",
            "The key concept is modularity",
            "API documentation is located at /docs",
            "a" * 500,
        ],
    )
    def test_clean_text_unchanged(self, text: str) -> None:
        redactor = Redactor()
        assert redactor.redact(text) == text


class TestAWSKeys:
    """AWS Access Key ID and Secret Key patterns."""

    def test_aws_access_key_id(self) -> None:
        redactor = Redactor()
        text = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = redactor.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[REDACTED:AWS_KEY]" in result

    def test_aws_access_key_inline(self) -> None:
        redactor = Redactor()
        text = "key: AKIAI44QH8DHBEXAMPLE rest of text"
        result = redactor.redact(text)
        assert "AKIAI44QH8DHBEXAMPLE" not in result

    def test_aws_secret_key(self) -> None:
        redactor = Redactor()
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = redactor.redact(text)
        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in result
        assert "[REDACTED:AWS_SECRET]" in result

    def test_aws_secret_key_double_quotes(self) -> None:
        redactor = Redactor()
        text = 'AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        result = redactor.redact(text)
        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" not in result


class TestJWT:
    """JWT token pattern."""

    def test_jwt_token(self) -> None:
        redactor = Redactor()
        token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        text = f"Authorization: Bearer {token}"
        result = redactor.redact(text)
        assert token not in result
        assert "[REDACTED:JWT]" in result

    def test_jwt_inline(self) -> None:
        redactor = Redactor()
        token = "eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJhY2NvdW50cyJ9.abc123def456"
        result = redactor.redact(f"token={token}")
        assert token not in result


class TestStripeKeys:
    """Stripe API key patterns."""

    def test_stripe_secret_key(self) -> None:
        redactor = Redactor()
        text = f"STRIPE_KEY={_SK_LIVE}"
        result = redactor.redact(text)
        assert _SK_LIVE not in result
        assert "[REDACTED:STRIPE]" in result

    def test_stripe_publishable_key(self) -> None:
        redactor = Redactor()
        result = redactor.redact(_PK_LIVE)
        assert _PK_LIVE not in result
        assert "[REDACTED:STRIPE]" in result

    def test_stripe_test_key(self) -> None:
        redactor = Redactor()
        result = redactor.redact(_SK_TEST)
        assert _SK_TEST not in result
        assert "[REDACTED:STRIPE]" in result


class TestGenericAPIKeys:
    """Generic API key patterns in context."""

    def test_api_key_assignment(self) -> None:
        redactor = Redactor()
        text = "api_key = 'abcdef1234567890abcdef1234567890ab'"
        result = redactor.redact(text)
        assert "abcdef1234567890abcdef1234567890ab" not in result
        assert "[REDACTED:API_KEY]" in result

    def test_api_key_env_var(self) -> None:
        redactor = Redactor()
        text = "API_KEY=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
        result = redactor.redact(text)
        assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in result

    def test_token_assignment(self) -> None:
        redactor = Redactor()
        text = "token=abcdef1234567890abcdef1234567890ab"
        result = redactor.redact(text)
        assert "abcdef1234567890abcdef1234567890ab" not in result

    def test_secret_assignment(self) -> None:
        redactor = Redactor()
        text = "client_secret = 'a1b2c3d4e5f6g7h8i9j0a1b2c3d4e5f6'"
        result = redactor.redact(text)
        assert "a1b2c3d4e5f6g7h8i9j0a1b2c3d4e5f6" not in result


class TestPasswords:
    """Password-like value patterns."""

    def test_password_equals(self) -> None:
        redactor = Redactor()
        text = "password=MyS3cretP@ss"
        result = redactor.redact(text)
        assert "MyS3cretP@ss" not in result
        assert "[REDACTED:PASSWORD]" in result

    def test_passwd_colon(self) -> None:
        redactor = Redactor()
        text = "passwd: hunter2"
        result = redactor.redact(text)
        assert "hunter2" not in result
        assert "[REDACTED:PASSWORD]" in result

    def test_pwd_assignment(self) -> None:
        redactor = Redactor()
        text = "db_pwd = 'sup3r_s3cret'"
        result = redactor.redact(text)
        assert "sup3r_s3cret" not in result

    def test_password_in_url(self) -> None:
        redactor = Redactor()
        text = "DATABASE_URL=postgres://user:secret123@localhost/db"
        result = redactor.redact(text)
        assert "secret123" not in result


class TestCorpus:
    """Test against a corpus of known secrets — no false negatives."""

    SECRETS = [
        ("AKIAIOSFODNN7EXAMPLE", "AWS_KEY"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.rTCH8cLoGxAm_xw68z-zXVKi9ie6xJn9tn", "JWT"),
        (_SK_LIVE, "STRIPE"),
        (_PK_LIVE, "STRIPE"),
    ]

    @pytest.mark.parametrize("secret,label", SECRETS)
    def test_known_secret_detected(self, secret: str, label: str) -> None:
        redactor = Redactor()
        text = f"value is {secret} in context"
        result = redactor.redact(text)
        assert secret not in result, f"Secret {label} was not redacted"


class TestMultipleSecrets:
    """Multiple secrets in the same text."""

    def test_multiple_secrets_redacted(self) -> None:
        redactor = Redactor()
        text = (
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
            "password=hunter2\n"
            f"STRIPE_KEY={_SK_LIVE}"
        )
        result = redactor.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "hunter2" not in result
        assert _SK_LIVE not in result
