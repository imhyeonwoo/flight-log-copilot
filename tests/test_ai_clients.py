import copy
import json
from types import SimpleNamespace

import pytest

from flightlog_copilot.engine import build_llm_payload
from flightlog_copilot.llm.base import LLMAnalysisError, classify_provider_error, safe_list_models
from flightlog_copilot.llm.factory import create_llm_client
from flightlog_copilot.llm.openai_client import OpenAIResponsesClient
from flightlog_copilot.llm.openai_compatible_client import LOCAL_NO_KEY_PLACEHOLDER, OpenAICompatibleClient
from flightlog_copilot.llm.settings import AiSettings, ProviderSettings
from flightlog_copilot.llm.schemas import CopilotResponse


VALID_RESPONSE = {
    "summary": "Structured diagnosis",
    "hypothesis_review": [],
    "additional_hypotheses": [],
    "recommended_experiments": [],
    "missing_data_requests": [],
}


class StatusError(Exception):
    def __init__(self, status_code, message="provider failed"):
        super().__init__(message)
        self.status_code = status_code


def _settings(provider="openai", **overrides):
    providers = {
        "openai": ProviderSettings(model="official-model", temperature=0.3, max_output_tokens=321),
        "openai_compatible": ProviderSettings(
            model="served-model",
            base_url="http://localhost:8000/",
            temperature=0.4,
            max_output_tokens=654,
            api_key_mode="api_key",
        ),
    }
    for key, value in overrides.items():
        setattr(providers[provider], key, value)
    return AiSettings(enabled=True, provider=provider, providers=providers)


def test_factory_creates_official_openai_client_with_explicit_key(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("flightlog_copilot.llm.openai_client.OpenAI", fake_openai)
    client = create_llm_client("openai", _settings(), "secret-value")
    assert isinstance(client, OpenAIResponsesClient)
    assert captured == {"api_key": "secret-value"}


def test_factory_creates_compatible_client_with_normalized_url_and_no_key(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("flightlog_copilot.llm.openai_compatible_client.OpenAI", fake_openai)
    settings = _settings("openai_compatible", api_key_mode="no_key")
    client = create_llm_client("openai_compatible", settings, None)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "http://localhost:8000/v1"
    assert captured == {"api_key": LOCAL_NO_KEY_PLACEHOLDER, "base_url": "http://localhost:8000/v1"}


def test_official_client_uses_user_temperature_and_output_limit():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=CopilotResponse.model_validate(VALID_RESPONSE))

    sdk = SimpleNamespace(responses=Responses())
    client = OpenAIResponsesClient(_settings().for_provider(), "secret", sdk_client=sdk)
    result = client.generate_diagnosis({"features": {"rmse": 0.2}})
    assert result["summary"] == "Structured diagnosis"
    assert captured["temperature"] == 0.3
    assert captured["max_output_tokens"] == 321
    assert captured["model"] == "official-model"
    assert "raw_csv" not in json.dumps(captured["input"])


def test_compatible_client_uses_chat_completions_and_validates_json():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content=json.dumps(VALID_RESPONSE))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    client = OpenAICompatibleClient(_settings("openai_compatible").for_provider(), "secret", sdk_client=sdk)
    result = client.generate_diagnosis({"features": {"rmse": 0.2}})
    assert result["summary"] == "Structured diagnosis"
    assert captured["model"] == "served-model"
    assert captured["max_tokens"] == 654
    assert "response_format" not in captured


def test_official_connection_test_uses_a_minimal_non_stored_request():
    captured = {}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="OK")

    client = OpenAIResponsesClient(
        _settings().for_provider(),
        "secret",
        sdk_client=SimpleNamespace(responses=Responses()),
    )
    client.test_connection()
    assert captured["input"] == "Reply with exactly: OK"
    assert captured["max_output_tokens"] == 16
    assert captured["store"] is False


def test_compatible_model_list_returns_sorted_ids():
    sdk = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="z-model"), SimpleNamespace(id="a-model")]
            )
        )
    )
    client = OpenAICompatibleClient(
        _settings("openai_compatible").for_provider(),
        "secret",
        sdk_client=sdk,
    )
    assert client.list_models() == ["a-model", "z-model"]


@pytest.mark.parametrize(
    "status,code",
    [(401, "auth"), (403, "auth"), (404, "not_found"), (429, "rate_limit"), (503, "server")],
)
def test_http_errors_are_mapped_without_leaking_provider_message(status, code):
    secret = "sk-never-print-this"
    error = classify_provider_error(StatusError(status, f"failed Authorization Bearer {secret}"))
    assert error.code == code
    assert secret not in str(error)
    assert "Bearer" not in str(error)


def test_model_list_failure_returns_manual_entry_fallback():
    class FailingClient:
        def list_models(self):
            raise StatusError(404)

    result = safe_list_models(FailingClient(), language="en")
    assert result.model_ids == []
    assert "model ID" in result.warning


def test_api_failure_does_not_mutate_local_analysis_result():
    local_result = {
        "flight_phase": "altitude_hold",
        "analysis_window": {"start_s": 0.0, "end_s": 1.0, "rows": 2},
        "metrics": {},
        "rule_based_hypotheses": [],
        "missing_parameters": [],
        "analysis_warnings": [],
    }
    before = copy.deepcopy(local_result)
    payload = build_llm_payload(local_result)

    class Responses:
        def parse(self, **kwargs):
            raise StatusError(503)

    sdk = SimpleNamespace(responses=Responses())
    client = OpenAIResponsesClient(_settings().for_provider(), "secret", sdk_client=sdk)
    with pytest.raises(LLMAnalysisError, match="서버 오류"):
        client.generate_diagnosis(payload)
    assert local_result == before
