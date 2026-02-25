"""Exponential decay for chunk quality scores.

Pure functions — no state, no I/O.  Uses only ``math`` and ``time`` from
stdlib so the module stays dependency-free.
"""

from __future__ import annotations

import math
import time
from typing import Any

LN2: float = math.log(2)
DEFAULT_HALF_LIFE: float = 14.0


def decay_factor(ts_end: float, half_life_days: float, now: float | None = None) -> float:
    """Return exponential decay factor in [0, 1].

    ``exp(-age_days * ln(2) / half_life_days)``

    Returns 1.0 when *age* <= 0 or *half_life_days* <= 0.
    """
    if half_life_days <= 0:
        return 1.0
    if now is None:
        now = time.time()
    age_days = (now - ts_end) / 86400.0
    if age_days <= 0:
        return 1.0
    return math.exp(-age_days * LN2 / half_life_days)


def decayed_quality(
    quality: float, ts_end: float, half_life_days: float, now: float | None = None,
) -> float:
    """Return ``quality * decay_factor(...)``."""
    return quality * decay_factor(ts_end, half_life_days, now)


def get_decay_config(config: dict[str, Any]) -> tuple[float, bool]:
    """Extract ``(half_life_days, enabled)`` from *config*.

    Looks under the ``decay`` key.  Defaults: ``(14.0, True)``.
    Returns ``(0.0, False)`` when disabled or half-life <= 0.
    """
    decay_cfg = config.get("decay", {})
    enabled: bool = decay_cfg.get("enabled", True)
    half_life_days: float = float(decay_cfg.get("half_life_days", DEFAULT_HALF_LIFE))

    if not enabled or half_life_days <= 0:
        return 0.0, False
    return half_life_days, True
