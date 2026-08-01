"""Serializable models for automatic candidates and confirmed mappings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from flightlog_copilot.config import DEFAULT_UNITS, MINIMUM_REQUIRED, PARAMETER_GROUP, STANDARD_PARAMETERS


@dataclass
class MappingCandidate:
    parameter: str
    column: Optional[str]
    confidence: float
    reasons: List[str] = field(default_factory=list)
    required: bool = False
    status: str = "unmapped"
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParameterMapping:
    column: Optional[str] = None
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    invert_sign: bool = False
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ParameterMapping":
        if not isinstance(payload, dict):
            raise ValueError("각 파라미터 매핑은 JSON 객체여야 합니다.")
        column = payload.get("column")
        if column is not None and not isinstance(column, str):
            raise ValueError("매핑 column은 문자열 또는 null이어야 합니다.")
        scale = float(payload.get("scale", 1.0))
        offset = float(payload.get("offset", 0.0))
        if not math.isfinite(scale) or not math.isfinite(offset):
            raise ValueError("매핑 scale과 offset은 유한한 숫자여야 합니다.")
        return cls(
            column=column,
            unit=str(payload.get("unit", "")),
            scale=scale,
            offset=offset,
            invert_sign=bool(payload.get("invert_sign", False)),
            confirmed=bool(payload.get("confirmed", True)),
        )


@dataclass
class MappingProfile:
    profile_name: str = "Untitled profile"
    time_unit: str = "auto"
    coordinate_frame: str = "확인 필요"
    header_hash: str = ""
    parameters: Dict[str, ParameterMapping] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "time_unit": self.time_unit,
            "coordinate_frame": self.coordinate_frame,
            "header_hash": self.header_hash,
            "parameters": {name: mapping.to_dict() for name, mapping in self.parameters.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MappingProfile":
        if not isinstance(payload, dict) or not isinstance(payload.get("parameters", {}), dict):
            raise ValueError("매핑 프로필에 parameters 객체가 필요합니다.")
        unknown = set(payload.get("parameters", {})) - set(STANDARD_PARAMETERS)
        if unknown:
            raise ValueError("알 수 없는 표준 파라미터: " + ", ".join(sorted(unknown)))
        time_unit = str(payload.get("time_unit", "auto"))
        if time_unit not in {"auto", "seconds", "milliseconds", "microseconds"}:
            raise ValueError("time_unit은 auto, seconds, milliseconds, microseconds 중 하나여야 합니다.")
        return cls(
            profile_name=str(payload.get("profile_name", "Imported profile")),
            time_unit=time_unit,
            coordinate_frame=str(payload.get("coordinate_frame", "확인 필요")),
            header_hash=str(payload.get("header_hash", "")),
            parameters={
                name: ParameterMapping.from_dict(value)
                for name, value in payload.get("parameters", {}).items()
            },
        )


def default_parameter_mapping(parameter: str, column: Optional[str] = None) -> ParameterMapping:
    group = PARAMETER_GROUP[parameter]
    return ParameterMapping(
        column=column,
        unit=DEFAULT_UNITS[group],
        confirmed=False,
    )


def profile_from_candidates(candidates: Dict[str, MappingCandidate], name: str = "Auto mapping") -> MappingProfile:
    profile = MappingProfile(profile_name=name)
    for parameter in STANDARD_PARAMETERS:
        candidate = candidates[parameter]
        # Low-confidence and ambiguous suggestions stay visible but are not silently selected.
        selected = candidate.column if candidate.status == "자동 추천" else None
        profile.parameters[parameter] = default_parameter_mapping(parameter, selected)
    return profile


def is_required(parameter: str) -> bool:
    return parameter in MINIMUM_REQUIRED
