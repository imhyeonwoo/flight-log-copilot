"""Non-fatal validation messages for raw and canonical data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from flightlog_copilot.config import AnalysisSettings, MOTOR_PARAMETERS


@dataclass
class ValidationMessage:
    severity: str
    code: str
    message: str
    parameter: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def validate_data(
    raw: pd.DataFrame,
    canonical: pd.DataFrame,
    settings: AnalysisSettings = AnalysisSettings(),
    duplicate_columns: Optional[List[str]] = None,
) -> List[ValidationMessage]:
    messages: List[ValidationMessage] = []
    if raw.empty:
        messages.append(ValidationMessage("error", "empty_csv", "CSV에 데이터 행이 없습니다."))
        return messages
    if duplicate_columns:
        messages.append(ValidationMessage("error", "duplicate_columns", "중복 컬럼명: " + ", ".join(duplicate_columns)))
    if len(raw) < settings.minimum_rows:
        messages.append(ValidationMessage("warning", "short_log", f"데이터가 {len(raw)}행으로 짧아 일부 분석의 신뢰도가 낮습니다."))
    constant = [str(column) for column in raw.columns if raw[column].nunique(dropna=True) <= 1]
    if constant:
        messages.append(ValidationMessage("warning", "constant_columns", "상수 컬럼: " + ", ".join(constant)))
    missing_cells = int(raw.isna().sum().sum())
    if missing_cells:
        messages.append(ValidationMessage("warning", "missing_values", f"원본 데이터에 결측치 {missing_cells}개가 있습니다."))

    if "timestamp" not in canonical:
        messages.append(ValidationMessage("error", "missing_timestamp", "timestamp 컬럼을 찾지 못해 시계열 분석을 수행할 수 없습니다.", "timestamp"))
    else:
        timestamp = pd.to_numeric(canonical["timestamp"], errors="coerce")
        invalid = int(timestamp.isna().sum())
        duplicate = int(timestamp.duplicated().sum())
        reverse = int((timestamp.diff() < 0).sum())
        if invalid:
            messages.append(ValidationMessage("error", "invalid_timestamp", f"timestamp 숫자 변환 또는 결측 실패 {invalid}개", "timestamp"))
        if duplicate:
            messages.append(ValidationMessage("warning", "duplicate_timestamp", f"중복 timestamp {duplicate}개", "timestamp"))
        if reverse:
            messages.append(ValidationMessage("warning", "reverse_timestamp", f"역순 timestamp {reverse}개", "timestamp"))
        intervals = timestamp.dropna().sort_values().drop_duplicates().diff().dropna()
        intervals = intervals[intervals > 0]
        if not intervals.empty:
            median_interval = float(intervals.median())
            spike_count = int((intervals > median_interval * settings.dropout_multiplier).sum())
            if spike_count:
                messages.append(
                    ValidationMessage(
                        "warning",
                        "timestamp_gap_spikes",
                        f"중앙 sampling interval의 {settings.dropout_multiplier:g}배를 넘는 시간 간격이 {spike_count}개입니다.",
                        "timestamp",
                    )
                )

    numeric = canonical.select_dtypes(include=[np.number])
    infinite = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
    if infinite:
        messages.append(ValidationMessage("warning", "infinite_values", f"무한대 값 {infinite}개가 있습니다."))
    if "ekf_altitude" not in canonical:
        messages.append(ValidationMessage("error", "missing_ekf", "ekf_altitude가 없어 최소 고도 분석을 수행할 수 없습니다.", "ekf_altitude"))
    if "reference_altitude" not in canonical:
        messages.append(ValidationMessage(
            "warning",
            "missing_reference_altitude",
            "reference_altitude가 없어 기준 고도-EKF 비교를 생략합니다.",
            "reference_altitude",
        ))
    if not MOTOR_PARAMETERS.issubset(canonical.columns):
        messages.append(ValidationMessage("info", "missing_motors", "motor_1~motor_4가 모두 없어 모터 불균형 분석을 일부 또는 전부 생략합니다."))
    for parameter in ("armed", "althold_active"):
        if parameter in canonical and not canonical[parameter].fillna(False).astype(bool).any():
            messages.append(ValidationMessage("warning", f"no_{parameter}_segment", f"{parameter} 활성 구간이 없습니다.", parameter))
    return messages
