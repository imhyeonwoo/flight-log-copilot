"""Official OpenAI Responses API client."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import ValidationError

from flightlog_copilot.llm.base import LLMAnalysisError, classify_provider_error, reject_raw_data
from flightlog_copilot.llm.models import is_likely_openai_text_model
from flightlog_copilot.llm.prompts import SYSTEM_PROMPT
from flightlog_copilot.llm.schemas import CopilotResponse
from flightlog_copilot.llm.settings import ProviderSettings
from flightlog_copilot.reporting.json_report import sanitize_for_json


class OpenAIResponsesClient:
    def __init__(
        self,
        settings: ProviderSettings,
        api_key: str,
        sdk_client: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self._client = sdk_client or OpenAI(api_key=api_key)

    def test_connection(self) -> None:
        self._require_model()
        try:
            response = self._client.responses.create(
                model=self.settings.model,
                input="Reply with exactly: OK",
                temperature=self.settings.temperature,
                max_output_tokens=min(self.settings.max_output_tokens, 16),
                store=False,
            )
            if str(getattr(response, "output_text", "")).strip().upper() != "OK":
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
        try:
            response = self._client.responses.parse(
                model=self.settings.model,
                input=[
                    {"role": "developer", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": language_instruction
                        + " Use only the following Python analysis result as evidence.\n"
                        + json.dumps(safe_payload, ensure_ascii=False, allow_nan=False),
                    },
                ],
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
                text_format=CopilotResponse,
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise LLMAnalysisError("AI 응답 형식이 올바르지 않습니다.", code="response_format")
            if isinstance(parsed, CopilotResponse):
                return parsed.model_dump(mode="json")
            return CopilotResponse.model_validate(parsed).model_dump(mode="json")
        except LLMAnalysisError:
            raise
        except (ValidationError, json.JSONDecodeError) as exc:
            raise LLMAnalysisError("AI 응답 형식이 올바르지 않습니다.", code="response_format") from exc
        except Exception as exc:
            raise classify_provider_error(exc) from exc

    def list_models(self) -> List[str]:
        try:
            page = self._client.models.list()
            model_ids = {
                str(getattr(model, "id", "")).strip()
                for model in getattr(page, "data", [])
            }
            return sorted(model_id for model_id in model_ids if model_id and is_likely_openai_text_model(model_id))
        except Exception as exc:
            raise classify_provider_error(exc) from exc

    def _require_model(self) -> None:
        if not self.settings.model.strip():
            raise LLMAnalysisError("AI 모델 ID를 입력하십시오.", code="missing_model")
