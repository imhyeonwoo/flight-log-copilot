"""API-key resolution and OS keyring persistence without plaintext fallback."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Protocol

from flightlog_copilot.llm.models import AiProvider
from flightlog_copilot.llm.settings import normalize_openai_compatible_base_url

try:
    import keyring as _system_keyring
except ImportError:  # pragma: no cover - exercised through injected unavailable backend
    _system_keyring = None


SERVICE_NAME = "FlightLogCopilot"
ENVIRONMENT_KEYS = {
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}


class KeyringBackend(Protocol):
    def get_password(self, service_name: str, username: str) -> Optional[str]:
        ...

    def set_password(self, service_name: str, username: str, password: str) -> None:
        ...

    def delete_password(self, service_name: str, username: str) -> None:
        ...


@dataclass(frozen=True)
class SecretResolution:
    api_key: Optional[str]
    source: str
    masked: Optional[str]
    warning: Optional[str] = None


@dataclass(frozen=True)
class SecretSaveResult:
    stored: bool
    persistent: bool
    masked: Optional[str]
    warning: Optional[str] = None


@dataclass(frozen=True)
class SecretDeleteResult:
    deleted: bool
    environment_key_remains: bool
    warning: Optional[str] = None


def secret_account(provider: AiProvider, base_url: str = "") -> str:
    if provider == "openai":
        return "openai_api_key"
    normalized = normalize_openai_compatible_base_url(base_url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"openai_compatible:{digest}"


def mask_api_key(api_key: str) -> str:
    value = str(api_key)
    return "····" + value[-4:] if len(value) > 4 else "····"


def resolve_api_key(
    provider: AiProvider,
    session_keys: Mapping[str, str],
    base_url: str = "",
    backend: Optional[KeyringBackend] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> SecretResolution:
    account = secret_account(provider, base_url)
    session_value = str(session_keys.get(account, "")).strip()
    if session_value:
        return SecretResolution(session_value, "session", mask_api_key(session_value))

    selected_backend = backend if backend is not None else _system_keyring
    warning = None
    if selected_backend is not None:
        try:
            stored = selected_backend.get_password(SERVICE_NAME, account)
            if stored:
                return SecretResolution(stored, "keyring", mask_api_key(stored))
        except Exception:
            warning = "운영체제 비밀 저장소를 읽을 수 없습니다."
    else:
        warning = "운영체제 비밀 저장소를 사용할 수 없습니다."

    env = environment if environment is not None else os.environ
    env_value = str(env.get(ENVIRONMENT_KEYS[provider], "")).strip()
    if env_value:
        return SecretResolution(env_value, "environment", mask_api_key(env_value), warning)
    return SecretResolution(None, "none", None, warning)


def save_api_key(
    provider: AiProvider,
    api_key: str,
    session_keys: MutableMapping[str, str],
    base_url: str = "",
    backend: Optional[KeyringBackend] = None,
) -> SecretSaveResult:
    value = str(api_key).strip()
    if not value:
        return SecretSaveResult(False, False, None, "빈 API 키는 저장하지 않았습니다. 기존 키는 유지됩니다.")
    account = secret_account(provider, base_url)
    session_keys[account] = value
    selected_backend = backend if backend is not None else _system_keyring
    if selected_backend is None:
        return SecretSaveResult(
            True,
            False,
            mask_api_key(value),
            "운영체제 비밀 저장소를 사용할 수 없어 API 키가 현재 세션에만 저장됩니다. 앱을 종료하면 다시 입력해야 합니다.",
        )
    try:
        selected_backend.set_password(SERVICE_NAME, account, value)
        return SecretSaveResult(True, True, mask_api_key(value))
    except Exception:
        return SecretSaveResult(
            True,
            False,
            mask_api_key(value),
            "운영체제 비밀 저장소 저장에 실패하여 API 키가 현재 세션에만 저장됩니다. 앱을 종료하면 다시 입력해야 합니다.",
        )


def delete_api_key(
    provider: AiProvider,
    session_keys: MutableMapping[str, str],
    base_url: str = "",
    backend: Optional[KeyringBackend] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> SecretDeleteResult:
    account = secret_account(provider, base_url)
    deleted = session_keys.pop(account, None) is not None
    selected_backend = backend if backend is not None else _system_keyring
    warning = None
    if selected_backend is not None:
        try:
            if selected_backend.get_password(SERVICE_NAME, account) is not None:
                selected_backend.delete_password(SERVICE_NAME, account)
                deleted = True
        except Exception:
            warning = "운영체제 비밀 저장소의 API 키를 삭제하지 못했습니다."
    env = environment if environment is not None else os.environ
    environment_remains = bool(str(env.get(ENVIRONMENT_KEYS[provider], "")).strip())
    return SecretDeleteResult(deleted, environment_remains, warning)
