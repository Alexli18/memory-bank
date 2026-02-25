"""Tests for mb.decay — exponential decay pure functions."""

from __future__ import annotations

import math

from mb.decay import (
    DEFAULT_HALF_LIFE,
    LN2,
    decay_factor,
    decayed_quality,
    get_decay_config,
)


class TestDecayFactor:
    def test_zero_age_returns_one(self) -> None:
        now = 1_000_000.0
        assert decay_factor(now, 14.0, now=now) == 1.0

    def test_negative_age_returns_one(self) -> None:
        now = 1_000_000.0
        assert decay_factor(now + 100, 14.0, now=now) == 1.0

    def test_half_life_boundary(self) -> None:
        now = 1_000_000.0
        ts_end = now - 14 * 86400  # exactly 14 days ago
        result = decay_factor(ts_end, 14.0, now=now)
        assert abs(result - 0.5) < 1e-9

    def test_double_half_life(self) -> None:
        now = 1_000_000.0
        ts_end = now - 28 * 86400  # 28 days ago = 2 half-lives
        result = decay_factor(ts_end, 14.0, now=now)
        assert abs(result - 0.25) < 1e-9

    def test_zero_half_life_returns_one(self) -> None:
        assert decay_factor(0.0, 0.0, now=1_000_000.0) == 1.0

    def test_negative_half_life_returns_one(self) -> None:
        assert decay_factor(0.0, -5.0, now=1_000_000.0) == 1.0

    def test_custom_half_life(self) -> None:
        now = 1_000_000.0
        ts_end = now - 7 * 86400  # 7 days ago
        result = decay_factor(ts_end, 7.0, now=now)
        assert abs(result - 0.5) < 1e-9

    def test_very_old_chunk_near_zero(self) -> None:
        now = 1_000_000.0
        ts_end = now - 365 * 86400  # 1 year ago
        result = decay_factor(ts_end, 14.0, now=now)
        assert result < 0.001

    def test_monotonically_decreasing(self) -> None:
        now = 1_000_000.0
        values = [
            decay_factor(now - d * 86400, 14.0, now=now)
            for d in range(0, 60)
        ]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]


class TestDecayedQuality:
    def test_multiplies_quality_by_factor(self) -> None:
        now = 1_000_000.0
        ts_end = now - 14 * 86400
        result = decayed_quality(0.8, ts_end, 14.0, now=now)
        assert abs(result - 0.4) < 1e-9

    def test_zero_quality(self) -> None:
        now = 1_000_000.0
        assert decayed_quality(0.0, now - 86400, 14.0, now=now) == 0.0

    def test_no_decay_when_disabled(self) -> None:
        assert decayed_quality(0.8, 0.0, 0.0, now=1_000_000.0) == 0.8


class TestGetDecayConfig:
    def test_defaults_when_no_decay_key(self) -> None:
        half_life, enabled = get_decay_config({"version": "1.0"})
        assert half_life == DEFAULT_HALF_LIFE
        assert enabled is True

    def test_custom_values(self) -> None:
        config = {"decay": {"half_life_days": 7, "enabled": True}}
        half_life, enabled = get_decay_config(config)
        assert half_life == 7.0
        assert enabled is True

    def test_disabled(self) -> None:
        config = {"decay": {"half_life_days": 14, "enabled": False}}
        half_life, enabled = get_decay_config(config)
        assert half_life == 0.0
        assert enabled is False

    def test_zero_half_life_means_disabled(self) -> None:
        config = {"decay": {"half_life_days": 0, "enabled": True}}
        half_life, enabled = get_decay_config(config)
        assert half_life == 0.0
        assert enabled is False

    def test_negative_half_life_means_disabled(self) -> None:
        config = {"decay": {"half_life_days": -5, "enabled": True}}
        half_life, enabled = get_decay_config(config)
        assert half_life == 0.0
        assert enabled is False

    def test_missing_enabled_defaults_true(self) -> None:
        config = {"decay": {"half_life_days": 7}}
        half_life, enabled = get_decay_config(config)
        assert half_life == 7.0
        assert enabled is True

    def test_missing_half_life_defaults_14(self) -> None:
        config = {"decay": {"enabled": True}}
        half_life, enabled = get_decay_config(config)
        assert half_life == DEFAULT_HALF_LIFE
        assert enabled is True

    def test_ln2_constant(self) -> None:
        assert abs(LN2 - math.log(2)) < 1e-15
