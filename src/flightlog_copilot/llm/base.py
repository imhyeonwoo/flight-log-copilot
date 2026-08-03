"""Shared LLM client contract, payload guard, and sanitized error mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


class LLMAnalysisError(RuntimeError):
    """A recoverable optional-AI error; deterministic output remains valid."""

    def __init__(self, message: str, code: str = "unknown", status_code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class LlmClient(Protocol):
    def test_connection(self) -> None:
        ...

    def generate_diagnosis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def list_models(self) -> List[str]:
        ...


@dataclass(frozen=True)
class ModelListResult:
    model_ids: List[str]
    warning: Optional[str] = None


FORBIDDEN_RAW_KEYS = {"raw_csv", "csv", "rows", "raw_data", "dataframe", "records"}

ERROR_MESSAGES = {
    "missing_key": {
        "ko": "API 키가 없습니다.",
        "en": "No API key is available.",
    },
    "missing_model": {
        "ko": "AI 모델 ID를 입력하십시오.",
        "en": "Enter an AI model ID.",
    },
    "base_url": {
        "ko": "Base URL이 올바르지 않습니다.",
        "en": "The Base URL is invalid.",
    },
    "auth": {
        "ko": "API 키가 유효하지 않습니다.",
        "en": "The API key is invalid.",
    },
    "not_found": {
        "ko": "Endpoint 또는 모델 ID를 찾을 수 없습니다.",
        "en": "The endpoint or model ID could not be found.",
    },
    "rate_limit": {
        "ko": "API 요청 한도를 초과했습니다.",
        "en": "The API request limit was exceeded.",
    },
    "server": {
        "ko": "AI Provider 서버 오류입니다.",
        "en": "The AI provider returned a server error.",
    },
    "timeout": {
        "ko": "AI Provider 요청 시간이 초과되었습니다.",
        "en": "The AI provider request timed out.",
    },
    "network": {
        "ko": "연결에 실패했습니다. Base URL 또는 네트워크 상태를 확인하십시오.",
        "en": "Connection failed. Check the Base URL and network.",
    },
    "response_format": {
        "ko": "AI 응답 형식이 올바르지 않습니다.",
        "en": "The AI response format is invalid.",
    },
    "invalid_request": {
        "ko": "AI 요청 설정이 올바르지 않습니다. 모델과 Provider 설정을 확인하십시오.",
        "en": "The AI request is invalid. Check the model and provider settings.",
    },
    "unknown": {
        "ko": "AI Provider 호출에 실패했습니다.",
        "en": "The AI provider call failed.",
    },
}


def error_message(code: str, language: str = "ko") -> str:
    selected = ERROR_MESSAGES.get(code, ERROR_MESSAGES["unknown"])
    return selected["en" if language == "en" else "ko"]


def classify_provider_error(exc: Exception) -> LLMAnalysisError:
    if isinstance(exc, LLMAnalysisError):
        return exc
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code in {401, 403}:
        code = "auth"
    elif status_code == 404:
        code = "not_found"
    elif status_code == 429:
        code = "rate_limit"
    elif isinstance(status_code, int) and status_code >= 500:
        code = "server"
    elif status_code == 400:
        code = "invalid_request"
    else:
        class_name = exc.__class__.__name__.lower()
        if isinstance(exc, TimeoutError) or "timeout" in class_name:
            code = "timeout"
        elif "connection" in class_name or "connect" in class_name:
            code = "network"
        else:
            code = "unknown"
    return LLMAnalysisError(error_message(code), code=code, status_code=status_code)


def safe_list_models(client: LlmClient, language: str = "ko") -> ModelListResult:
    try:
        return ModelListResult(client.list_models())
    except Exception as exc:
        error = classify_provider_error(exc)
        prefix = "모델 목록을 자동으로 가져오지 못했습니다." if language != "en" else "The model list could not be loaded automatically."
        return ModelListResult([], f"{prefix} {error_message(error.code, language)}")


def reject_raw_data(payload: Dict[str, Any]) -> None:
    forbidden = find_forbidden_keys(payload)
    if forbidden:
        raise LLMAnalysisError(
            "원시 CSV 또는 행 데이터는 AI payload로 전달할 수 없습니다: " + ", ".join(sorted(forbidden)),
            code="raw_data",
        )


def find_forbidden_keys(value: Any) -> set:
    if isinstance(value, dict):
        found = {str(key).lower() for key in value} & FORBIDDEN_RAW_KEYS
        for nested in value.values():
            found |= find_forbidden_keys(nested)
        return found
    if isinstance(value, (list, tuple)):
        found = set()
        for nested in value:
            found |= find_forbidden_keys(nested)
        return found
    return set()
