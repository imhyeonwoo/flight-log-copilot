"""Serialization, validation, and identification for mapping profiles."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, List

from flightlog_copilot.mapping.models import MappingProfile


def header_hash(columns: Iterable[str]) -> str:
    canonical = "\x1f".join(str(column).strip() for column in columns)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dump_profile(profile: MappingProfile) -> str:
    return json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def load_profile(content: Any) -> MappingProfile:
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"매핑 프로필 JSON 형식이 올바르지 않습니다: {exc}") from exc
    elif isinstance(content, dict):
        payload = content
    else:
        raise ValueError("매핑 프로필은 JSON 문자열, bytes 또는 객체여야 합니다.")
    return MappingProfile.from_dict(payload)


def validate_profile_columns(profile: MappingProfile, columns: Iterable[str]) -> List[str]:
    available = set(columns)
    return sorted(
        mapping.column
        for mapping in profile.parameters.values()
        if mapping.column and mapping.column not in available
    )
