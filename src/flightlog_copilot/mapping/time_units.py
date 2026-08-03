"""Conservative timestamp-unit inference with explicit review states."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_UNIT_FACTORS: Dict[str, float] = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
}

UNIT_TOKENS = {
    "s": "seconds",
    "sec": "seconds",
    "secs": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "ms": "milliseconds",
    "msec": "milliseconds",
    "millisecond": "milliseconds",
    "milliseconds": "milliseconds",
    "us": "microseconds",
    "usec": "microseconds",
    "microsecond": "microseconds",
    "microseconds": "microseconds",
}

# Broad enough to keep both 1 Hz seconds and 1,000 Hz milliseconds plausible.
MIN_PLAUSIBLE_FREQUENCY_HZ = 0.2
MAX_PLAUSIBLE_FREQUENCY_HZ = 2000.0
MAX_PLAUSIBLE_DURATION_S = 48.0 * 60.0 * 60.0


@dataclass(frozen=True)
class TimeUnitInference:
    unit: Optional[str]
    confidence: float
    reasons: List[str]
    requires_confirmation: bool


def resolve_time_unit(
    values: pd.Series,
    column_name: Optional[str],
    selected_time_unit: str = "auto",
    mapping_unit: str = "auto",
) -> TimeUnitInference:
    """Resolve explicit choices before consulting the column name or values."""
    mapping_unit = mapping_unit or "auto"
    if selected_time_unit in TIME_UNIT_FACTORS:
        return TimeUnitInference(
            selected_time_unit,
            1.0,
            [f"사용자가 시간 단위를 '{selected_time_unit}'로 명시했습니다."],
            False,
        )
    if selected_time_unit != "auto":
        raise ValueError("time_unit은 auto, seconds, milliseconds, microseconds 중 하나여야 합니다.")
    if mapping_unit in TIME_UNIT_FACTORS:
        return TimeUnitInference(
            mapping_unit,
            1.0,
            [f"timestamp 매핑 단위를 '{mapping_unit}'로 명시했습니다."],
            False,
        )
    if mapping_unit != "auto":
        raise ValueError("timestamp 매핑 단위는 auto, seconds, milliseconds, microseconds 중 하나여야 합니다.")
    return infer_time_unit(values, column_name)


def infer_time_unit(values: pd.Series, column_name: Optional[str] = None) -> TimeUnitInference:
    """Infer a timestamp unit without silently accepting ambiguous value scales."""
    name_unit = time_unit_from_column_name(column_name)
    if name_unit is not None:
        unit, token = name_unit
        return TimeUnitInference(
            unit,
            1.0,
            [f"컬럼명 '{column_name}'의 단위 토큰 '{token}'은 {unit}를 명시합니다."],
            False,
        )

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid_count = int(len(numeric))
    distinct = numeric.drop_duplicates().sort_values()
    distinct_count = int(len(distinct))
    if distinct_count < 2:
        return TimeUnitInference(
            None,
            0.0,
            ["유효하고 서로 다른 timestamp가 2개 미만이어서 단위를 추론할 수 없습니다."],
            True,
        )

    positive_intervals = distinct.diff().dropna()
    positive_intervals = positive_intervals[positive_intervals > 0]
    if positive_intervals.empty:
        return TimeUnitInference(
            None,
            0.0,
            ["양수 sampling interval이 없어 timestamp 단위를 추론할 수 없습니다."],
            True,
        )

    median_interval = float(positive_intervals.median())
    timestamp_range = float(distinct.iloc[-1] - distinct.iloc[0])
    reasons = [
        "유효 샘플 "
        f"{valid_count}개, 서로 다른 timestamp {distinct_count}개, "
        f"sampling interval 중앙값 {median_interval:g}, timestamp 범위 {timestamp_range:g}입니다."
    ]

    candidates: List[Tuple[str, float, float]] = []
    for unit, factor in TIME_UNIT_FACTORS.items():
        frequency_hz = 1.0 / (median_interval * factor)
        duration_s = timestamp_range * factor
        if (
            MIN_PLAUSIBLE_FREQUENCY_HZ <= frequency_hz <= MAX_PLAUSIBLE_FREQUENCY_HZ
            and 0.0 < duration_s <= MAX_PLAUSIBLE_DURATION_S
        ):
            candidates.append((unit, frequency_hz, duration_s))

    if len(candidates) == 1:
        unit, frequency_hz, duration_s = candidates[0]
        reasons.append(
            f"{unit}로 변환하면 sampling frequency {frequency_hz:g} Hz, "
            f"기록 길이 {duration_s:g}초입니다."
        )
        if distinct_count >= 3:
            reasons.append("현실적인 비행 로그 범위에 해당하는 단위 후보가 하나입니다.")
            return TimeUnitInference(unit, 0.85, reasons, False)
        reasons.append("유효 샘플 수가 적어 값 기반 단위를 자동 확정하지 않습니다.")
        return TimeUnitInference(unit, 0.6, reasons, True)

    if len(candidates) > 1:
        descriptions = ", ".join(
            f"{unit}={frequency_hz:g} Hz/{duration_s:g} s"
            for unit, frequency_hz, duration_s in candidates
        )
        reasons.append(f"현실적인 sampling frequency 후보가 여러 개입니다: {descriptions}.")
        reasons.append("sampling interval만으로 timestamp 단위를 하나로 확정할 수 없습니다.")
        return TimeUnitInference(None, 0.4, reasons, True)

    reasons.append("현실적인 sampling frequency 범위에 해당하는 단위 후보가 없습니다.")
    return TimeUnitInference(None, 0.2, reasons, True)


def time_unit_from_column_name(column_name: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return a unit only for a distinct final unit token or a unit-only name."""
    if not column_name:
        return None
    normalized = unicodedata.normalize("NFKC", str(column_name)).lower().strip()
    normalized = normalized.replace("µ", "u").replace("μ", "u")
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    if not tokens:
        return None
    token = tokens[-1]
    unit = UNIT_TOKENS.get(token)
    if unit is None:
        return None
    return unit, token
