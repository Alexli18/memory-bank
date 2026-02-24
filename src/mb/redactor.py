"""Secret redaction for event content before persistence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Default secret patterns: (regex, replacement_label)
# ---------------------------------------------------------------------------

_DEFAULT_PATTERNS: list[tuple[str, str]] = [
    # AWS Access Key ID: AKIA followed by 16 alphanumeric chars
    (r"AKIA[0-9A-Z]{16}", "AWS_KEY"),
    # AWS Secret Key: 40-char base64-ish value after aws_secret context
    (
        r"(?i)(?:aws_secret_access_key|aws_secret)\s*[=:]\s*[\"']?"
        r"([A-Za-z0-9/+=]{40})",
        "AWS_SECRET",
    ),
    # JWT tokens: three base64url-encoded segments separated by dots
    (
        r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        "JWT",
    ),
    # Stripe keys: sk_live_, pk_live_, sk_test_, pk_test_ followed by 24+ alphanumeric
    (r"[sp]k_(?:live|test)_[a-zA-Z0-9]{24,}", "STRIPE"),
    # Generic API key/token/secret in assignment context
    (
        r"(?i)(?:api[_-]?key|token|client_secret)\s*[=:]\s*[\"']?"
        r"([a-zA-Z0-9]{32,})",
        "API_KEY",
    ),
    # Password in URL: ://user:password@host
    (r"://[^:]+:([^@\s]+)@", "PASSWORD"),
    # Password-like assignments: password/passwd/pwd = value
    (
        r"(?i)(?:password|passwd|pwd)\s*[=:]\s*[\"']?"
        r"(\S+)",
        "PASSWORD",
    ),
]


@dataclass(frozen=True, slots=True)
class RedactorConfig:
    """Configuration for the Redactor."""

    enabled: bool = True
    extra_patterns: Sequence[tuple[str, str]] = field(default_factory=list)

    @property
    def patterns(self) -> list[tuple[str, str]]:
        return list(_DEFAULT_PATTERNS) + list(self.extra_patterns)


class Redactor:
    """Redacts known secret patterns from text."""

    def __init__(self, config: RedactorConfig | None = None) -> None:
        self._config = config or RedactorConfig()
        # Pre-compile all patterns for performance.
        self._compiled: list[tuple[re.Pattern[str], str, bool]] = []
        if self._config.enabled:
            for regex, label in self._config.patterns:
                has_group = "(" in regex and ")" in regex
                self._compiled.append(
                    (re.compile(regex), label, has_group)
                )

    def redact(self, text: str) -> str:
        """Return *text* with detected secrets replaced by [REDACTED:TYPE] markers."""
        if not self._config.enabled or not text:
            return text

        result = text
        for pattern, label, has_group in self._compiled:
            marker = f"[REDACTED:{label}]"
            if has_group:
                # Replace only the captured group (group 1), keep surrounding context.
                result = _replace_group(pattern, result, marker)
            else:
                result = pattern.sub(marker, result)
        return result


def _replace_group(pattern: re.Pattern[str], text: str, marker: str) -> str:
    """Replace group(1) matches within the full pattern match."""
    parts: list[str] = []
    last_end = 0
    for m in pattern.finditer(text):
        if m.lastindex and m.lastindex >= 1:
            # Replace group 1 only, keep the rest of the match.
            parts.append(text[last_end : m.start(1)])
            parts.append(marker)
            last_end = m.end(1)
        else:
            # Fallback: replace entire match if no groups.
            parts.append(text[last_end : m.start()])
            parts.append(marker)
            last_end = m.end()
    parts.append(text[last_end:])
    return "".join(parts)
