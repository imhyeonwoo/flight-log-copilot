"""Provider-aware LLM client construction."""

from __future__ import annotations

from typing import Any, Optional

from flightlog_copilot.llm.base import LLMAnalysisError, LlmClient
from flightlog_copilot.llm.models import AiProvider
from flightlog_copilot.llm.openai_client import OpenAIResponsesClient
from flightlog_copilot.llm.openai_compatible_client import LOCAL_NO_KEY_PLACEHOLDER, OpenAICompatibleClient
from flightlog_copilot.llm.settings import AiSettings


def create_llm_client(
    provider: AiProvider,
    settings: AiSettings,
    api_key: Optional[str],
    sdk_client: Optional[Any] = None,
) -> LlmClient:
    provider_settings = settings.for_provider(provider)
    if provider == "openai":
        if not api_key:
            raise LLMAnalysisError("API 키가 없습니다.", code="missing_key")
        return OpenAIResponsesClient(provider_settings, api_key, sdk_client=sdk_client)
    if provider == "openai_compatible":
        if provider_settings.api_key_mode == "no_key":
            selected_key = LOCAL_NO_KEY_PLACEHOLDER
        else:
            if not api_key:
                raise LLMAnalysisError("API 키가 없습니다.", code="missing_key")
            selected_key = api_key
        try:
            return OpenAICompatibleClient(provider_settings, selected_key, sdk_client=sdk_client)
        except ValueError as exc:
            raise LLMAnalysisError("Base URL이 올바르지 않습니다.", code="base_url") from exc
    raise LLMAnalysisError("지원하지 않는 AI Provider입니다.", code="invalid_request")
