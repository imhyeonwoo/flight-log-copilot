"""OpenAI Responses API client with Pydantic structured output and safe failure."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI
from pydantic import ValidationError

from flightlog_copilot.llm.prompts import SYSTEM_PROMPT
from flightlog_copilot.llm.schemas import CopilotResponse
from flightlog_copilot.reporting.json_report import sanitize_for_json


class LLMAnalysisError(RuntimeError):
    """A recoverable optional-AI error; deterministic output remains valid."""


FORBIDDEN_RAW_KEYS = {"raw_csv", "csv", "rows", "raw_data", "dataframe", "records"}


def analyze_with_openai(
    payload: Dict[str, Any],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    response_language: str = "ko",
) -> Dict[str, Any]:
    _reject_raw_data(payload)
    selected_model = model or os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
    selected_key = api_key or os.getenv("OPENAI_API_KEY")
    if client is None and not selected_key:
        raise LLMAnalysisError("OPENAI_API_KEY가 없어 AI Copilot 호출을 건너뜁니다.")
    sdk_client = client or OpenAI(api_key=selected_key)
    safe_payload = sanitize_for_json(payload)
    language_instruction = "Respond in English." if response_language == "en" else "한국어로 답변하십시오."
    try:
        response = sdk_client.responses.parse(
            model=selected_model,
            input=[
                {"role": "developer", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": language_instruction + " Use only the following Python analysis result as evidence.\n" + json.dumps(safe_payload, ensure_ascii=False, allow_nan=False),
                },
            ],
            text_format=CopilotResponse,
        )
        parsed = response.output_parsed
        if parsed is None:
            refusal = getattr(response, "output_text", "")
            raise LLMAnalysisError(f"구조화된 AI 응답이 없습니다. {refusal}".strip())
        if isinstance(parsed, CopilotResponse):
            return parsed.model_dump(mode="json")
        return CopilotResponse.model_validate(parsed).model_dump(mode="json")
    except LLMAnalysisError:
        raise
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMAnalysisError(f"AI 응답 schema 검증에 실패했습니다: {exc}") from exc
    except Exception as exc:
        raise LLMAnalysisError(f"OpenAI API 호출에 실패했습니다: {exc}") from exc


def _reject_raw_data(payload: Dict[str, Any]) -> None:
    forbidden = _find_forbidden_keys(payload)
    if forbidden:
        raise LLMAnalysisError("원시 CSV 또는 행 데이터는 AI payload로 전달할 수 없습니다: " + ", ".join(sorted(forbidden)))


def _find_forbidden_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {str(key).lower() for key in value} & FORBIDDEN_RAW_KEYS
        for nested in value.values():
            found |= _find_forbidden_keys(nested)
        return found
    if isinstance(value, (list, tuple)):
        found: set[str] = set()
        for nested in value:
            found |= _find_forbidden_keys(nested)
        return found
    return set()
