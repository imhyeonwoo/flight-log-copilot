import json

import pytest

from flightlog_copilot.llm.settings import (
    AiSettings,
    ProviderSettings,
    load_ai_settings,
    normalize_openai_compatible_base_url,
    save_ai_settings,
)


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:8000",
        "http://localhost:8000/",
        "http://localhost:8000/v1",
        "http://localhost:8000/v1/",
    ],
)
def test_compatible_base_url_normalization_avoids_duplicate_v1(value):
    assert normalize_openai_compatible_base_url(value) == "http://localhost:8000/v1"


def test_invalid_compatible_base_url_is_rejected():
    with pytest.raises(ValueError, match="Base URL"):
        normalize_openai_compatible_base_url("localhost:8000")
    with pytest.raises(ValueError, match="비밀번호"):
        normalize_openai_compatible_base_url("http://user:secret@localhost:8000")


def test_provider_settings_remain_independent_and_direct_model_round_trips(tmp_path):
    settings = AiSettings(
        enabled=True,
        provider="openai_compatible",
        providers={
            "openai": ProviderSettings(model="custom-openai-model", temperature=0.1, max_output_tokens=512),
            "openai_compatible": ProviderSettings(
                model="served-local-alias",
                base_url="http://localhost:8000",
                temperature=0.7,
                max_output_tokens=4096,
                api_key_mode="no_key",
            ),
        },
    )
    path = tmp_path / "ai_settings.json"
    save_ai_settings(settings, path)
    restored = load_ai_settings(path).settings
    assert restored.for_provider("openai").model == "custom-openai-model"
    assert restored.for_provider("openai").temperature == 0.1
    assert restored.for_provider("openai_compatible").model == "served-local-alias"
    assert restored.for_provider("openai_compatible").temperature == 0.7
    assert restored.for_provider("openai_compatible").api_key_mode == "no_key"


def test_settings_json_never_contains_api_key_or_masked_value(tmp_path):
    secret = "sk-test-super-secret-A1B2"
    path = tmp_path / "ai_settings.json"
    save_ai_settings(AiSettings(enabled=True), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert secret not in serialized
    assert "A1B2" not in serialized
    assert "masked" not in serialized
    assert not _contains_exact_key(payload, "api_key")
    assert not (tmp_path / "ai_settings.json.tmp").exists()


def test_corrupt_settings_are_backed_up_without_overwriting(tmp_path):
    path = tmp_path / "ai_settings.json"
    path.write_text("{broken", encoding="utf-8")
    loaded = load_ai_settings(path)
    assert loaded.warning
    assert loaded.backup_path is not None
    assert loaded.backup_path.read_text(encoding="utf-8") == "{broken"
    assert not path.exists()
    assert loaded.settings.enabled is False


def _contains_exact_key(value, expected):
    if isinstance(value, dict):
        return expected in value or any(_contains_exact_key(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_exact_key(item, expected) for item in value)
    return False
