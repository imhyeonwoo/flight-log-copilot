"""OpenAI Chat Completions compatible client."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import ValidationError

from flightlog_copilot.llm.base import LLMAnalysisError, classify_provider_error, reject_raw_data
from flightlog_copilot.llm.prompts import SYSTEM_PROMPT
from flightlog_copilot.llm.schemas import CopilotResponse
from flightlog_copilot.llm.settings import ProviderSettings, normalize_openai_compatible_base_url
from flightlog_copilot.reporting.json_report import sanitize_for_json


LOCAL_NO_KEY_PLACEHOLDER = "local-no-key"


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: ProviderSettings,
        api_key: str,
        sdk_client: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self.base_url = normalize_openai_compatible_base_url(settings.base_url)
        self._client = sdk_client or OpenAI(api_key=api_key or LOCAL_NO_KEY_PLACEHOLDER, base_url=self.base_url)

    def test_connection(self) -> None:
        self._require_model()
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=self.settings.temperature,
                max_tokens=min(self.settings.max_output_tokens, 16),
            )
            content = _message_content(response)
            if content.strip().upper() != "OK":
                raise LLMAnalysisError("AI 응답 형식이 올바르지 않습니다.", code="response_format")
        except Exception as exc:
            raise classify_provider_error(exc) from exc

    def generate_diagnosis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._require_model()
        reject_raw_data(payload)
        safe_payload = sanitize_for_json(payload)
        language_instruction = (
            "Respond in English."
            if safe_payload.get("response_language") == "English"
            else "한국어로 답변하십시오."
        )
        schema = json.dumps(CopilotResponse.model_json_schema(), ensure_ascii=False)
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + " Return one JSON object matching this JSON Schema: "
                        + schema,
                    },
                    {
                        "role": "user",
                        "content": language_instruction
                        + " Use only the following Python analysis result as evidence.\n"
                        + json.dumps(safe_payload, ensure_ascii=False, allow_nan=False),
                    },
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            content = _strip_json_fence(_message_content(response))
            return CopilotResponse.model_validate_json(content).model_dump(mode="json")
        except (ValidationError, json.JSONDecodeError, ValueError, IndexError, AttributeError) as exc:
            raise LLMAnalysisError("AI 응답 형식이 올바르지 않습니다.", code="response_format") from exc
        except LLMAnalysisError:
            raise
        except Exception as exc:
            raise classify_provider_error(exc) from exc

    def list_models(self) -> List[str]:
        try:
            page = self._client.models.list()
            model_ids = {
                str(getattr(model, "id", "")).strip()
                for model in getattr(page, "data", [])
            }
            return sorted(model_id for model_id in model_ids if model_id)
        except Exception as exc:
            raise classify_provider_error(exc) from exc

    def _require_model(self) -> None:
        if not self.settings.model.strip():
            raise LLMAnalysisError("AI 모델 ID를 입력하십시오.", code="missing_model")


def _message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMAnalysisError("AI 응답 형식이 올바르지 않습니다.", code="response_format")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise LLMAnalysisError("AI 응답 형식이 올바르지 않습니다.", code="response_format")
    return content


def _strip_json_fence(content: str) -> str:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return value
