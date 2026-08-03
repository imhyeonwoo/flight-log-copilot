"""Apply only the user-confirmed mapping and explicit value transforms."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from flightlog_copilot.mapping.models import MappingProfile
from flightlog_copilot.mapping.time_units import TIME_UNIT_FACTORS, resolve_time_unit


TRUE_VALUES = {"1", "true", "yes", "on", "armed", "active"}
FALSE_VALUES = {"0", "false", "no", "off", "disarmed", "inactive"}


def transform_frame(frame: pd.DataFrame, profile: MappingProfile) -> Tuple[pd.DataFrame, List[str]]:
    output = pd.DataFrame(index=frame.index)
    warnings: List[str] = []
    for parameter, mapping in profile.parameters.items():
        if not mapping.column:
            continue
        if mapping.column not in frame.columns:
            warnings.append(f"{parameter}: CSV에 '{mapping.column}' 컬럼이 없습니다.")
            continue
        if parameter in {"armed", "althold_active"}:
            output[parameter] = _to_boolean(frame[mapping.column])
            continue
        numeric = pd.to_numeric(frame[mapping.column], errors="coerce")
        conversion_failures = int((frame[mapping.column].notna() & numeric.isna()).sum())
        if conversion_failures:
            warnings.append(f"{parameter}: 숫자 변환 실패 {conversion_failures}개")
        sign = -1.0 if mapping.invert_sign else 1.0
        converted = sign * numeric * float(mapping.scale) + float(mapping.offset)
        converted = _convert_unit(
            parameter,
            converted,
            mapping.unit,
            profile.time_unit,
            mapping.column,
        )
        output[parameter] = converted
    return output, warnings


def _to_boolean(series: pd.Series) -> pd.Series:
    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        normalized = str(value).strip().lower()
        if normalized in TRUE_VALUES:
            return True
        if normalized in FALSE_VALUES:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def _convert_unit(
    parameter: str,
    values: pd.Series,
    unit: str,
    profile_time_unit: str,
    column_name: Optional[str],
) -> pd.Series:
    if parameter == "timestamp":
        inference = resolve_time_unit(values, column_name, profile_time_unit, unit)
        if inference.requires_confirmation or inference.unit is None:
            raise ValueError(
                "timestamp 단위를 자동으로 확정할 수 없습니다. "
                "seconds, milliseconds, microseconds 중 하나를 직접 선택하십시오. "
                + " ".join(inference.reasons)
            )
        return values * TIME_UNIT_FACTORS[inference.unit]
    if parameter in {"altitude_setpoint", "ekf_altitude", "reference_altitude"}:
        factor = {"meters": 1.0, "centimeters": 1e-2, "millimeters": 1e-3}.get(unit, 1.0)
        return values * factor
    if parameter == "vertical_velocity":
        factor = {"m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3}.get(unit, 1.0)
        return values * factor
    return values
