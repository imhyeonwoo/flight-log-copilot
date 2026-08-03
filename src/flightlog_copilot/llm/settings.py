"""Non-secret AI settings with platform-specific, atomic persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from platformdirs import user_config_dir

from flightlog_copilot.llm.models import AiProvider, DEFAULT_OPENAI_MODEL, SUPPORTED_PROVIDERS


@dataclass
class ProviderSettings:
    model: str = ""
    temperature: float = 0.2
    max_output_tokens: int = 2048
    base_url: str = ""
    api_key_mode: str = "api_key"

    def to_dict(self, provider: AiProvider) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if provider == "openai_compatible":
            payload["base_url"] = (
                normalize_openai_compatible_base_url(self.base_url)
                if self.base_url.strip()
                else ""
            )
            payload["api_key_mode"] = self.api_key_mode
        return payload


def default_provider_settings() -> Dict[AiProvider, ProviderSettings]:
    return {
        "openai": ProviderSettings(model=DEFAULT_OPENAI_MODEL),
        "openai_compatible": ProviderSettings(
            model="",
            base_url="http://localhost:11434",
            api_key_mode="api_key",
        ),
    }


@dataclass
class AiSettings:
    enabled: bool = False
    provider: AiProvider = "openai"
    providers: Dict[AiProvider, ProviderSettings] = field(default_factory=default_provider_settings)

    def for_provider(self, provider: Optional[AiProvider] = None) -> ProviderSettings:
        return self.providers[provider or self.provider]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "providers": {
                provider: self.providers[provider].to_dict(provider)
                for provider in SUPPORTED_PROVIDERS
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AiSettings":
        if not isinstance(payload, dict):
            raise ValueError("AI 설정은 JSON 객체여야 합니다.")
        provider = str(payload.get("provider", "openai"))
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError("지원하지 않는 AI Provider입니다.")
        defaults = default_provider_settings()
        provider_payloads = payload.get("providers", {})
        if not isinstance(provider_payloads, dict):
            raise ValueError("AI 설정의 providers는 JSON 객체여야 합니다.")
        parsed: Dict[AiProvider, ProviderSettings] = {}
        for provider_name in SUPPORTED_PROVIDERS:
            values = provider_payloads.get(provider_name, {})
            if not isinstance(values, dict):
                raise ValueError(f"{provider_name} 설정은 JSON 객체여야 합니다.")
            default = defaults[provider_name]
            temperature = float(values.get("temperature", default.temperature))
            max_output_tokens = int(values.get("max_output_tokens", default.max_output_tokens))
            if not 0.0 <= temperature <= 2.0:
                raise ValueError("temperature는 0과 2 사이여야 합니다.")
            if not 1 <= max_output_tokens <= 131072:
                raise ValueError("max_output_tokens는 1 이상 131072 이하여야 합니다.")
            key_mode = str(values.get("api_key_mode", default.api_key_mode))
            if provider_name == "openai_compatible" and key_mode not in {"api_key", "no_key", "bearer"}:
                raise ValueError("api_key_mode가 올바르지 않습니다.")
            base_url = str(values.get("base_url", default.base_url)).strip()
            if provider_name == "openai_compatible" and base_url:
                base_url = normalize_openai_compatible_base_url(base_url)
            parsed[provider_name] = ProviderSettings(
                model=str(values.get("model", default.model)).strip(),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                base_url=base_url,
                api_key_mode=key_mode,
            )
        return cls(enabled=bool(payload.get("enabled", False)), provider=provider, providers=parsed)


@dataclass(frozen=True)
class SettingsLoadResult:
    settings: AiSettings
    warning: Optional[str] = None
    backup_path: Optional[Path] = None


def normalize_openai_compatible_base_url(base_url: str) -> str:
    """Return a validated SDK base URL ending in exactly one /v1."""
    value = str(base_url).strip()
    if not value:
        raise ValueError("OpenAI Compatible Base URL을 입력하십시오.")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Base URL은 http:// 또는 https://로 시작하는 유효한 주소여야 합니다.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Base URL에 사용자 이름이나 비밀번호를 포함할 수 없습니다.")
    if parsed.query or parsed.fragment:
        raise ValueError("Base URL에는 query string 또는 fragment를 사용할 수 없습니다.")
    path = parsed.path.rstrip("/")
    if path.lower() != "/v1" and not path.lower().endswith("/v1"):
        path = path + "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path or "/v1", "", ""))


def ai_settings_path() -> Path:
    return Path(user_config_dir(appname="FlightLogCopilot", appauthor=False)) / "ai_settings.json"


def load_ai_settings(path: Optional[Path] = None) -> SettingsLoadResult:
    target = Path(path) if path is not None else ai_settings_path()
    if not target.exists():
        return SettingsLoadResult(AiSettings())
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return SettingsLoadResult(AiSettings.from_dict(payload))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        backup = _corrupt_backup_path(target)
        try:
            target.replace(backup)
        except OSError:
            backup = None
        warning = "AI 설정 파일이 손상되어 기본 설정으로 실행합니다."
        if backup is not None:
            warning += f" 손상 파일은 '{backup.name}'으로 백업했습니다."
        return SettingsLoadResult(AiSettings(), warning=warning, backup_path=backup)


def save_ai_settings(settings: AiSettings, path: Optional[Path] = None) -> Path:
    target = Path(path) if path is not None else ai_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    payload = settings.to_dict()
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _corrupt_backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.corrupt-{stamp}.bak")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.corrupt-{stamp}-{index}.bak")
        index += 1
    return candidate
