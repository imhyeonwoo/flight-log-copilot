"""Explainable automatic mapping without coordinate/sign assumptions."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from flightlog_copilot.config import MINIMUM_REQUIRED, PARAMETER_GROUP, STANDARD_PARAMETERS
from flightlog_copilot.io.csv_loader import infer_semantic_type
from flightlog_copilot.mapping.aliases import ALIASES, NORMALIZED_ALIASES, normalize_column_name, tokenize_column_name
from flightlog_copilot.mapping.models import MappingCandidate


COORDINATE_AMBIGUOUS = {"posd", "veld", "positiond", "velocityd"}


def detect_mappings(frame: pd.DataFrame) -> Dict[str, MappingCandidate]:
    results: Dict[str, MappingCandidate] = {}
    for parameter in STANDARD_PARAMETERS:
        ranked = sorted(
            (_score_column(parameter, str(column), frame[column], frame) for column in frame.columns),
            key=lambda item: item[1],
            reverse=True,
        )
        best_column, best_score, best_reasons = ranked[0] if ranked else (None, 0.0, [])
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        ambiguous = best_score >= 0.55 and second_score >= best_score - 0.08
        coordinate_warning = normalize_column_name(best_column or "") in COORDINATE_AMBIGUOUS
        if best_score < 0.55:
            status = "미매핑"
            selected = None
        elif ambiguous or best_score < 0.78 or coordinate_warning:
            status = "확인 필요"
            selected = best_column
        else:
            status = "자동 추천"
            selected = best_column
        if coordinate_warning:
            best_reasons.append("좌표계와 부호 확인 필요: Down 축일 수 있음")
        results[parameter] = MappingCandidate(
            parameter=parameter,
            column=selected,
            confidence=round(float(best_score), 3),
            reasons=best_reasons,
            required=parameter in MINIMUM_REQUIRED,
            status=status,
            alternatives=[
                {"column": column, "confidence": round(float(score), 3)}
                for column, score, _ in ranked[:3]
                if score >= 0.35
            ],
        )
    return results


def _score_column(parameter: str, column: str, series: pd.Series, frame: pd.DataFrame) -> Tuple[str, float, List[str]]:
    normalized = normalize_column_name(column)
    canonical = normalize_column_name(parameter)
    alias_set = NORMALIZED_ALIASES[parameter]
    reasons: List[str] = []
    name_score = 0.0
    if normalized == canonical:
        name_score = 0.92
        reasons.append("표준 이름 완전 일치")
    elif normalized in alias_set:
        name_score = 0.86
        reasons.append("별칭 사전 완전 일치")
    else:
        similarity = max(
            SequenceMatcher(None, normalized, alias).ratio()
            for alias in alias_set
        )
        column_tokens = set(tokenize_column_name(column))
        alias_tokens = [set(tokenize_column_name(alias)) for alias in ALIASES[parameter]]
        token_score = max(
            (len(column_tokens & tokens) / max(len(column_tokens | tokens), 1) for tokens in alias_tokens),
            default=0.0,
        )
        name_score = 0.58 * similarity + 0.22 * token_score
        if token_score >= 0.5:
            reasons.append(f"이름 토큰 유사도 {token_score:.2f}")
        elif similarity >= 0.65:
            reasons.append(f"컬럼명 유사도 {similarity:.2f}")

    semantic = infer_semantic_type(series)
    expected_boolean = PARAMETER_GROUP[parameter] == "boolean"
    type_bonus = 0.0
    if expected_boolean and semantic == "boolean":
        type_bonus = 0.08
        reasons.append("불리언형 데이터")
    elif not expected_boolean and semantic == "numeric":
        type_bonus = 0.05
        reasons.append("숫자형 데이터")
    elif expected_boolean != (semantic == "boolean") and semantic != "unknown":
        type_bonus = -0.08
        reasons.append("예상 데이터 타입과 다름")

    range_bonus = _range_bonus(parameter, series)
    if range_bonus > 0:
        reasons.append("값 범위가 보조 근거와 일치")
    elif range_bonus < 0:
        reasons.append("값 범위가 일반적 범위와 다름")
    relationship_bonus = _relationship_bonus(parameter, frame)
    if relationship_bonus:
        reasons.append("관련 파라미터 컬럼군이 함께 존재")
    score = max(0.0, min(1.0, name_score + type_bonus + range_bonus + relationship_bonus))
    return column, score, reasons


def _range_bonus(parameter: str, series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return 0.0
    minimum, maximum = float(numeric.min()), float(numeric.max())
    if parameter in {"armed", "althold_active"}:
        return 0.04 if set(numeric.unique()).issubset({0, 1}) else -0.03
    if parameter.startswith("motor_") or parameter in {"throttle_base", "throttle_correction"}:
        return 0.03 if (-500 <= minimum <= 2500 and -500 <= maximum <= 2500) else -0.02
    if parameter == "timestamp" and maximum > minimum:
        return 0.03
    return 0.0


def _relationship_bonus(parameter: str, frame: pd.DataFrame) -> float:
    normalized_columns = {normalize_column_name(str(column)) for column in frame.columns}
    if parameter.startswith("motor_"):
        recognized = sum(
            1
            for motor in ("motor_1", "motor_2", "motor_3", "motor_4")
            if normalized_columns & NORMALIZED_ALIASES[motor]
        )
        return 0.02 if recognized >= 2 else 0.0
    related = {
        "altitude_setpoint": {"ekf_altitude", "barometer_altitude"},
        "ekf_altitude": {"altitude_setpoint", "barometer_altitude"},
        "barometer_altitude": {"ekf_altitude"},
        "armed": {"althold_active"},
        "althold_active": {"armed"},
    }.get(parameter, set())
    return 0.02 if any(normalized_columns & NORMALIZED_ALIASES[name] for name in related) else 0.0
