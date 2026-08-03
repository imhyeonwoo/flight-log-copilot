from flightlog_copilot.llm.secrets import (
    SERVICE_NAME,
    delete_api_key,
    mask_api_key,
    resolve_api_key,
    save_api_key,
    secret_account,
)


class MemoryKeyring:
    def __init__(self):
        self.values = {}

    def get_password(self, service_name, username):
        return self.values.get((service_name, username))

    def set_password(self, service_name, username, password):
        self.values[(service_name, username)] = password

    def delete_password(self, service_name, username):
        self.values.pop((service_name, username), None)


class FailingKeyring:
    def get_password(self, service_name, username):
        raise RuntimeError("backend unavailable")

    def set_password(self, service_name, username, password):
        raise RuntimeError("backend unavailable")

    def delete_password(self, service_name, username):
        raise RuntimeError("backend unavailable")


def test_api_key_mask_exposes_only_last_four_characters():
    secret = "sk-test-not-recoverable-A1B2"
    masked = mask_api_key(secret)
    assert masked == "····A1B2"
    assert secret not in masked
    assert "not-recoverable" not in masked
    assert mask_api_key("abc") == "····"


def test_empty_input_keeps_existing_keyring_key():
    backend = MemoryKeyring()
    account = secret_account("openai")
    backend.set_password(SERVICE_NAME, account, "existing-secret")
    session = {}
    result = save_api_key("openai", "   ", session, backend=backend)
    assert result.stored is False
    assert backend.get_password(SERVICE_NAME, account) == "existing-secret"
    assert session == {}


def test_new_key_replaces_keyring_and_is_immediately_used_from_session():
    backend = MemoryKeyring()
    account = secret_account("openai")
    backend.set_password(SERVICE_NAME, account, "old-secret")
    session = {}
    result = save_api_key("openai", "  new-secret-Z9Y8  ", session, backend=backend)
    assert result.persistent is True
    assert backend.get_password(SERVICE_NAME, account) == "new-secret-Z9Y8"
    resolved = resolve_api_key("openai", session, backend=backend, environment={"OPENAI_API_KEY": "env-secret"})
    assert resolved.source == "session"
    assert resolved.api_key == "new-secret-Z9Y8"


def test_key_delete_removes_session_and_keyring_but_not_environment():
    backend = MemoryKeyring()
    account = secret_account("openai")
    backend.set_password(SERVICE_NAME, account, "stored-secret")
    session = {account: "session-secret"}
    result = delete_api_key(
        "openai",
        session,
        backend=backend,
        environment={"OPENAI_API_KEY": "environment-secret"},
    )
    assert result.deleted is True
    assert result.environment_key_remains is True
    assert session == {}
    assert backend.get_password(SERVICE_NAME, account) is None


def test_keyring_failure_uses_session_only_fallback():
    session = {}
    result = save_api_key("openai", "session-only-A1B2", session, backend=FailingKeyring())
    assert result.stored is True
    assert result.persistent is False
    assert "세션" in result.warning
    resolved = resolve_api_key("openai", session, backend=FailingKeyring(), environment={})
    assert resolved.source == "session"
    assert resolved.api_key == "session-only-A1B2"


def test_environment_fallback_and_priority_do_not_expose_value():
    backend = MemoryKeyring()
    environment = {"OPENAI_COMPATIBLE_API_KEY": "environment-C3D4"}
    resolved = resolve_api_key(
        "openai_compatible",
        {},
        base_url="http://localhost:8000/",
        backend=backend,
        environment=environment,
    )
    assert resolved.source == "environment"
    assert resolved.masked == "····C3D4"
    backend.set_password(
        SERVICE_NAME,
        secret_account("openai_compatible", "http://localhost:8000/v1"),
        "keyring-B2C3",
    )
    resolved = resolve_api_key(
        "openai_compatible",
        {},
        base_url="http://localhost:8000",
        backend=backend,
        environment=environment,
    )
    assert resolved.source == "keyring"
    assert resolved.api_key == "keyring-B2C3"


def test_compatible_keys_are_separated_by_normalized_base_url():
    first = secret_account("openai_compatible", "http://localhost:8000")
    same = secret_account("openai_compatible", "http://localhost:8000/v1/")
    other = secret_account("openai_compatible", "http://localhost:9000")
    assert first == same
    assert first != other
