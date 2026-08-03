"""Backward-compatible facade for the official OpenAI provider."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from flightlog_copilot.llm.base import (
    FORBIDDEN_RAW_KEYS,
    LLMAnalysisError,
    find_forbidden_keys,
    reject_raw_data,
)
from flightlog_copilot.llm.models import DEFAULT_OPENAI_MODEL
from flightlog_copilot.llm.openai_client import OpenAIResponsesClient
from flightlog_copilot.llm.settings import ProviderSettings


def analyze_with_openai(
    payload: Dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    response_language: str = "ko",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> Dict[str, Any]:
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    selected_key = api_key or os.getenv("OPENAI_API_KEY")
    if client is None and not selected_key:
        raise LLMAnalysisError("API 키가 없습니다.", code="missing_key")
    request_payload = dict(payload)
    request_payload.setdefault("response_language", "English" if response_language == "en" else "Korean")
    provider_settings = ProviderSettings(
        model=selected_model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    llm_client = OpenAIResponsesClient(provider_settings, selected_key or "injected-client", sdk_client=client)
    return llm_client.generate_diagnosis(request_payload)


_reject_raw_data = reject_raw_data
_find_forbidden_keys = find_forbidden_keys

__all__ = [
    "FORBIDDEN_RAW_KEYS",
    "LLMAnalysisError",
    "analyze_with_openai",
]
