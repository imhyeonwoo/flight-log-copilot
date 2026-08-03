"""FlightLog Copilot Streamlit application."""

from __future__ import annotations

import hashlib
import html
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

from flightlog_copilot.config import AnalysisSettings, MINIMUM_REQUIRED, PARAMETER_GROUP, STANDARD_PARAMETERS
from flightlog_copilot.engine import build_llm_payload, run_deterministic_analysis
from flightlog_copilot.io.csv_loader import CSVLoadError, load_csv, profile_columns
from flightlog_copilot.io.mapping_profile import dump_profile, header_hash, load_profile, validate_profile_columns
from flightlog_copilot.i18n import SUPPORTED_LANGUAGES, localize_values, translate_text
from flightlog_copilot.llm import LLMAnalysisError, create_llm_client, error_message, safe_list_models
from flightlog_copilot.llm.models import DEFAULT_OPENAI_MODELS
from flightlog_copilot.llm.secrets import delete_api_key, resolve_api_key, save_api_key
from flightlog_copilot.llm.settings import (
    AiSettings,
    ProviderSettings,
    load_ai_settings,
    normalize_openai_compatible_base_url,
    save_ai_settings,
)
from flightlog_copilot.mapping.detector import detect_mappings
from flightlog_copilot.mapping.models import MappingProfile, ParameterMapping
from flightlog_copilot.mapping.time_units import resolve_time_unit
from flightlog_copilot.mapping.transformer import transform_frame
from flightlog_copilot.preprocessing.segmentation import FlightSegment, detect_segments, select_time_range
from flightlog_copilot.preprocessing.validation import validate_data
from flightlog_copilot.reporting import render_json_report, render_markdown_report


def _language() -> str:
    return st.session_state.get("language", "ko")


def _t(korean: str, english: str) -> str:
    return english if _language() == "en" else korean


def _tx(value: str) -> str:
    return str(translate_text(value, _language()))


def _default_unit(parameter: str) -> str:
    return _unit_options(parameter)[0]


def _unit_options(parameter: str) -> List[str]:
    group = PARAMETER_GROUP[parameter]
    return {
        "time": ["auto", "seconds", "milliseconds", "microseconds"],
        "altitude": ["meters", "centimeters", "millimeters"],
        "velocity": ["m/s", "cm/s", "mm/s"],
        "pwm": ["us", "normalized", "raw"],
        "boolean": ["boolean"],
    }[group]


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value.strip())
    return cleaned or "flightlog_mapping"


def _initialize_ai_settings() -> None:
    if st.session_state.get("ai_settings_initialized"):
        return
    loaded = load_ai_settings()
    settings = loaded.settings
    st.session_state["ai_settings_initialized"] = True
    st.session_state["ai_settings_warning"] = loaded.warning
    st.session_state["ai_enabled"] = settings.enabled
    st.session_state["ai_provider"] = settings.provider
    st.session_state["ai_api_key_session"] = {}
    st.session_state["ai_fetched_models_openai"] = []
    st.session_state["ai_fetched_models_openai_compatible"] = []

    openai_settings = settings.for_provider("openai")
    default_ids = [option.id for option in DEFAULT_OPENAI_MODELS]
    use_custom = openai_settings.model not in default_ids
    st.session_state["ai_openai_use_custom_model"] = use_custom
    st.session_state["ai_openai_model_select"] = openai_settings.model if not use_custom else default_ids[0]
    st.session_state["ai_openai_model_custom"] = openai_settings.model if use_custom else ""
    st.session_state["ai_openai_temperature"] = openai_settings.temperature
    st.session_state["ai_openai_max_output_tokens"] = openai_settings.max_output_tokens

    compatible = settings.for_provider("openai_compatible")
    st.session_state["ai_openai_compatible_base_url"] = compatible.base_url
    st.session_state["ai_openai_compatible_model"] = compatible.model
    st.session_state["ai_openai_compatible_fetched_select"] = "__manual__"
    st.session_state["ai_openai_compatible_temperature"] = compatible.temperature
    st.session_state["ai_openai_compatible_max_output_tokens"] = compatible.max_output_tokens
    st.session_state["ai_openai_compatible_api_key_mode"] = compatible.api_key_mode


def _current_ai_settings() -> AiSettings:
    openai_model = (
        str(st.session_state.get("ai_openai_model_custom", "")).strip()
        if st.session_state.get("ai_openai_use_custom_model")
        else str(st.session_state.get("ai_openai_model_select", "")).strip()
    )
    compatible_choice = st.session_state.get("ai_openai_compatible_fetched_select", "__manual__")
    compatible_model = (
        str(compatible_choice).strip()
        if compatible_choice != "__manual__"
        else str(st.session_state.get("ai_openai_compatible_model", "")).strip()
    )
    return AiSettings(
        enabled=bool(st.session_state.get("ai_enabled", False)),
        provider=st.session_state.get("ai_provider", "openai"),
        providers={
            "openai": ProviderSettings(
                model=openai_model,
                temperature=float(st.session_state.get("ai_openai_temperature", 0.2)),
                max_output_tokens=int(st.session_state.get("ai_openai_max_output_tokens", 2048)),
            ),
            "openai_compatible": ProviderSettings(
                model=compatible_model,
                base_url=str(st.session_state.get("ai_openai_compatible_base_url", "")).strip(),
                temperature=float(st.session_state.get("ai_openai_compatible_temperature", 0.2)),
                max_output_tokens=int(st.session_state.get("ai_openai_compatible_max_output_tokens", 2048)),
                api_key_mode=str(st.session_state.get("ai_openai_compatible_api_key_mode", "api_key")),
            ),
        },
    )


def _active_secret(settings: AiSettings):
    provider = settings.provider
    provider_settings = settings.for_provider()
    if provider == "openai_compatible" and provider_settings.api_key_mode == "no_key":
        return None
    return resolve_api_key(
        provider,
        st.session_state.setdefault("ai_api_key_session", {}),
        base_url=provider_settings.base_url,
    )


def _register_ai_api_key(provider: str, input_key: str) -> None:
    value = str(st.session_state.get(input_key, ""))
    base_url = str(st.session_state.get("ai_openai_compatible_base_url", "")) if provider == "openai_compatible" else ""
    try:
        result = save_api_key(
            provider,
            value,
            st.session_state.setdefault("ai_api_key_session", {}),
            base_url=base_url,
        )
        if result.stored:
            st.session_state[f"ai_key_persistent_{provider}"] = result.persistent
            level = "success" if result.persistent else "warning"
            message = (
                _t(f"API 키를 저장했습니다: {result.masked}", f"API key saved: {result.masked}")
                if result.persistent
                else str(result.warning)
            )
        else:
            level, message = "warning", str(result.warning)
        st.session_state["ai_settings_notice"] = (level, message)
        st.session_state[input_key] = ""
    except ValueError as exc:
        st.session_state["ai_settings_notice"] = ("error", str(exc))


def _delete_ai_api_key(provider: str, confirm_key: str) -> None:
    if not st.session_state.get(confirm_key):
        st.session_state["ai_settings_notice"] = (
            "warning",
            _t("API 키 삭제 확인을 먼저 선택하십시오.", "Select the API-key deletion confirmation first."),
        )
        return
    base_url = str(st.session_state.get("ai_openai_compatible_base_url", "")) if provider == "openai_compatible" else ""
    try:
        result = delete_api_key(
            provider,
            st.session_state.setdefault("ai_api_key_session", {}),
            base_url=base_url,
        )
        st.session_state.pop(f"ai_key_persistent_{provider}", None)
        if result.environment_key_remains:
            message = _t(
                "이 API 키는 환경변수에서도 로드됩니다. 앱에서 환경변수를 삭제할 수 없으므로 시스템 환경변수를 직접 변경해야 합니다.",
                "An API key is also loaded from the environment. The app cannot delete it; change the system environment variable directly.",
            )
            level = "warning"
        elif result.warning:
            message, level = result.warning, "warning"
        elif result.deleted:
            message, level = _t("저장된 API 키를 삭제했습니다.", "The stored API key was deleted."), "success"
        else:
            message, level = _t("삭제할 저장된 API 키가 없습니다.", "There is no stored API key to delete."), "info"
        st.session_state["ai_settings_notice"] = (level, message)
        st.session_state[confirm_key] = False
    except ValueError as exc:
        st.session_state["ai_settings_notice"] = ("error", str(exc))


def _render_ai_settings() -> None:
    _initialize_ai_settings()
    with st.expander(_t("AI 설정", "AI settings"), expanded=False):
        warning = st.session_state.pop("ai_settings_warning", None)
        if warning:
            st.warning(_tx(warning))
        notice = st.session_state.pop("ai_settings_notice", None)
        if notice:
            getattr(st, notice[0])(_tx(str(notice[1])))

        enabled = st.toggle(_t("AI Copilot 활성화", "Enable AI Copilot"), key="ai_enabled")
        provider = st.selectbox(
            "AI Provider",
            ["openai", "openai_compatible"],
            key="ai_provider",
            disabled=not enabled,
            format_func=lambda value: "OpenAI" if value == "openai" else "OpenAI Compatible",
        )

        if provider == "openai_compatible":
            base_url = st.text_input(
                "Base URL",
                key="ai_openai_compatible_base_url",
                placeholder="http://localhost:11434",
                disabled=not enabled,
            )
            try:
                normalized = normalize_openai_compatible_base_url(base_url)
                st.caption(_t(f"요청 기준 URL: {normalized}", f"Normalized API URL: {normalized}"))
            except ValueError as exc:
                if enabled:
                    st.warning(_tx(str(exc)))
            key_mode = st.radio(
                _t("인증 방식", "Authentication mode"),
                ["api_key", "no_key", "bearer"],
                key="ai_openai_compatible_api_key_mode",
                disabled=not enabled,
                format_func=lambda value: {
                    "api_key": _t("실제 API 키 사용", "Use an API key"),
                    "no_key": _t("API 키 없이 연결", "Connect without an API key"),
                    "bearer": _t("임의 Bearer 문자열 사용", "Use a custom Bearer value"),
                }[value],
            )
        else:
            key_mode = "api_key"

        settings = _current_ai_settings()
        provider_settings = settings.for_provider()
        if not enabled:
            secret = None
            st.caption(_t(
                "AI Copilot을 활성화하면 Provider 설정과 API 키 관리 기능을 사용할 수 있습니다.",
                "Enable AI Copilot to configure the provider and manage API keys.",
            ))
        elif key_mode == "no_key":
            st.info(_t(
                "로컬 서버 연결용 내부 placeholder만 사용하며 저장하거나 화면에 표시하지 않습니다.",
                "An internal local-server placeholder is used and is never stored or displayed.",
            ))
            secret = None
        else:
            try:
                secret = _active_secret(settings)
            except ValueError as exc:
                secret = None
                if enabled:
                    st.warning(_tx(str(exc)))
            if secret is not None:
                if secret.warning:
                    st.warning(_tx(secret.warning))
                if secret.source == "environment":
                    st.info(_t(
                        "API 키 사용 가능: 환경변수에서 로드됨",
                        "API key available: loaded from an environment variable",
                    ))
                elif secret.api_key:
                    persistent = secret.source == "keyring" or bool(st.session_state.get(f"ai_key_persistent_{provider}"))
                    label = _t("API 키 저장됨", "API key stored") if persistent else _t("세션 API 키 사용 중", "Session API key in use")
                    st.success(f"{label}: {secret.masked}")
                    if not persistent:
                        st.caption(_t(
                            "이 키는 앱 종료 시 사라집니다.",
                            "This key is discarded when the app exits.",
                        ))
                else:
                    st.info(_t("API 키가 등록되지 않았습니다.", "No API key is registered."))
            input_key = f"ai_new_api_key_{provider}"
            st.text_input(
                _t("API 키", "API key"),
                type="password",
                key=input_key,
                disabled=not enabled,
                placeholder=_t("새 키 또는 Bearer 문자열", "New key or Bearer value"),
            )
            st.button(
                _t("새 API 키 등록", "Register new API key"),
                key=f"ai_register_key_{provider}",
                disabled=not enabled,
                on_click=_register_ai_api_key,
                args=(provider, input_key),
            )
            confirm_key = f"ai_delete_key_confirm_{provider}"
            st.checkbox(
                _t("저장된 API 키 삭제 확인", "Confirm stored API-key deletion"),
                key=confirm_key,
                disabled=not enabled,
            )
            st.button(
                _t("저장된 API 키 삭제", "Delete stored API key"),
                key=f"ai_delete_key_{provider}",
                disabled=not enabled,
                on_click=_delete_ai_api_key,
                args=(provider, confirm_key),
            )

        if provider == "openai":
            fetched = list(st.session_state.get("ai_fetched_models_openai", []))
            default_ids = [option.id for option in DEFAULT_OPENAI_MODELS]
            model_options = list(dict.fromkeys([*default_ids, *fetched]))
            if st.session_state.get("ai_openai_model_select") not in model_options:
                st.session_state["ai_openai_model_select"] = model_options[0]
            st.checkbox(
                _t("모델 ID 직접 입력", "Enter model ID manually"),
                key="ai_openai_use_custom_model",
                disabled=not enabled,
            )
            if st.session_state.get("ai_openai_use_custom_model"):
                st.text_input(
                    _t("OpenAI 모델 ID", "OpenAI model ID"),
                    key="ai_openai_model_custom",
                    placeholder=_t("실제 OpenAI API 모델 ID", "Actual OpenAI API model ID"),
                    disabled=not enabled,
                )
            else:
                selected_model = st.selectbox(
                    _t("모델 선택", "Select model"),
                    model_options,
                    key="ai_openai_model_select",
                    disabled=not enabled,
                )
                option = next((item for item in DEFAULT_OPENAI_MODELS if item.id == selected_model), None)
                if option:
                    st.caption(option.description)
        else:
            fetched = list(st.session_state.get("ai_fetched_models_openai_compatible", []))
            if fetched:
                options = ["__manual__", *fetched]
                if st.session_state.get("ai_openai_compatible_fetched_select") not in options:
                    st.session_state["ai_openai_compatible_fetched_select"] = "__manual__"
                st.selectbox(
                    _t("서버 모델 목록", "Server model list"),
                    options,
                    key="ai_openai_compatible_fetched_select",
                    disabled=not enabled,
                    format_func=lambda value: _t("직접 입력", "Manual entry") if value == "__manual__" else value,
                )
            if st.session_state.get("ai_openai_compatible_fetched_select", "__manual__") == "__manual__":
                st.text_input(
                    _t("모델 ID", "Model ID"),
                    key="ai_openai_compatible_model",
                    placeholder=_t("서버에서 실제 제공하는 모델 ID", "Model ID actually served by the server"),
                    disabled=not enabled,
                )

        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key=f"ai_{provider}_temperature",
            disabled=not enabled,
        )
        st.number_input(
            _t("최대 출력 토큰", "Maximum output tokens"),
            min_value=1,
            max_value=131072,
            step=128,
            key=f"ai_{provider}_max_output_tokens",
            disabled=not enabled,
        )

        action_columns = st.columns(2)
        if action_columns[0].button(_t("모델 목록 불러오기", "Load model list"), disabled=not enabled):
            current = _current_ai_settings()
            try:
                current_secret = _active_secret(current)
                client = create_llm_client(
                    current.provider,
                    current,
                    current_secret.api_key if current_secret else None,
                )
                model_result = safe_list_models(client, _language())
                if model_result.warning:
                    st.warning(model_result.warning)
                else:
                    st.session_state[f"ai_fetched_models_{provider}"] = model_result.model_ids
                    st.success(_t(f"모델 {len(model_result.model_ids)}개를 불러왔습니다.", f"Loaded {len(model_result.model_ids)} models."))
                    st.rerun()
            except (LLMAnalysisError, ValueError) as exc:
                code = getattr(exc, "code", "base_url" if isinstance(exc, ValueError) else "unknown")
                st.error(error_message(code, _language()))

        if action_columns[1].button(_t("AI 연결 테스트", "Test AI connection"), disabled=not enabled):
            current = _current_ai_settings()
            try:
                current_secret = _active_secret(current)
                client = create_llm_client(
                    current.provider,
                    current,
                    current_secret.api_key if current_secret else None,
                )
                with st.spinner(_t("AI Provider 연결을 확인하는 중입니다...", "Testing the AI provider connection...")):
                    client.test_connection()
                provider_label = "OpenAI" if provider == "openai" else "OpenAI Compatible"
                st.success(_t(
                    f"연결 성공 · Provider: {provider_label} · Model: {current.for_provider().model}",
                    f"Connection successful · Provider: {provider_label} · Model: {current.for_provider().model}",
                ))
            except (LLMAnalysisError, ValueError) as exc:
                code = getattr(exc, "code", "base_url" if isinstance(exc, ValueError) else "unknown")
                st.error(error_message(code, _language()))

        if st.button(_t("AI 설정 저장", "Save AI settings")):
            try:
                current = _current_ai_settings()
                if current.provider == "openai_compatible":
                    normalized = normalize_openai_compatible_base_url(current.for_provider().base_url)
                    current.for_provider().base_url = normalized
                save_ai_settings(current)
                st.success(_t("API 키를 제외한 AI 설정을 저장했습니다.", "AI settings were saved without the API key."))
            except (OSError, ValueError) as exc:
                st.error(_t(f"AI 설정 저장에 실패했습니다: {exc}", f"Failed to save AI settings: {exc}"))


def _profile_signature(profile: MappingProfile) -> tuple:
    return (
        profile.profile_name,
        profile.time_unit,
        profile.coordinate_frame,
        profile.reference_altitude_source,
        tuple(
            (
                parameter,
                mapping.column,
                mapping.unit,
                float(mapping.scale),
                float(mapping.offset),
                bool(mapping.invert_sign),
            )
            for parameter, mapping in sorted(profile.parameters.items())
        ),
    )


@st.cache_data(show_spinner=False, max_entries=4)
def _prepare_upload(content: bytes, filename: str) -> tuple:
    loaded_result = load_csv(content, filename)
    return loaded_result, profile_columns(loaded_result.data), detect_mappings(loaded_result.data)


def _render_validation(messages: list) -> None:
    if not messages:
        st.success(_t("데이터 유효성 검사에서 보고할 문제가 없습니다.", "No data-validation issues were found."))
        return
    with st.expander(_t("데이터 유효성 검사", "Data validation"), expanded=True):
        for message in messages:
            if message.severity == "error":
                st.error(_validation_message(message))
            elif message.severity == "warning":
                st.warning(_validation_message(message))
            else:
                st.info(_validation_message(message))


def _validation_message(message: object) -> str:
    if _language() != "en":
        return str(message.message)
    parameter = getattr(message, "parameter", None)
    translations = {
        "empty_csv": "The CSV has no data rows.",
        "duplicate_columns": "The CSV contains duplicate column names.",
        "short_log": "The log is short, so some analysis results may be less reliable.",
        "constant_columns": "One or more columns contain only a constant value.",
        "missing_values": "The source data contains missing values.",
        "missing_timestamp": "The timestamp column is missing, so time-series analysis is unavailable.",
        "invalid_timestamp": "Some timestamp values are missing or failed numeric conversion.",
        "duplicate_timestamp": "Duplicate timestamps were found.",
        "reverse_timestamp": "Reverse-order timestamps were found.",
        "timestamp_gap_spikes": "Sampling-interval spikes above the configured threshold were found.",
        "infinite_values": "Infinite numeric values were found.",
        "missing_ekf": "ekf_altitude is missing, so minimum altitude analysis is unavailable.",
        "missing_reference_altitude": "reference_altitude is missing; the reference-altitude/EKF comparison was skipped.",
        "missing_motors": "motor_1 through motor_4 are incomplete; motor-imbalance analysis was partly or fully skipped.",
        "no_armed_segment": "No active armed segment was found.",
        "no_althold_active_segment": "No active AltHold segment was found.",
    }
    translated = translations.get(getattr(message, "code", ""))
    if translated and parameter and getattr(message, "code", "").startswith("no_"):
        return f"No active {parameter} segment was found."
    return translated or _tx(str(message.message))


def _select_segment(
    frame: pd.DataFrame,
    segments: Dict[str, List[FlightSegment]],
    mode: str,
) -> Tuple[pd.DataFrame, str]:
    if mode == "full":
        return frame.copy(), "full_log"
    if mode in {"auto_althold", "althold"}:
        options = segments["althold"]
        labels = [f"{segment.start_s:.3f}–{segment.end_s:.3f} s ({segment.rows} {_t('행', 'rows')})" for segment in options]
        selected_label = st.selectbox(_t("AltHold 구간", "AltHold segment"), labels, key="segment_althold")
        segment = options[labels.index(selected_label)]
        return select_time_range(frame, segment.start_s, segment.end_s), "altitude_hold"
    if mode == "armed":
        options = segments["armed"]
        labels = [f"{segment.start_s:.3f}–{segment.end_s:.3f} s ({segment.rows} {_t('행', 'rows')})" for segment in options]
        selected_label = st.selectbox(_t("Armed 구간", "Armed segment"), labels, key="segment_armed")
        segment = options[labels.index(selected_label)]
        return select_time_range(frame, segment.start_s, segment.end_s), "armed"
    timestamp = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
    minimum, maximum = float(timestamp.min()), float(timestamp.max())
    selected = st.slider(_t("시작·종료 시간 (s)", "Start and end time (s)"), minimum, maximum, (minimum, maximum), key="segment_range")
    return select_time_range(frame, selected[0], selected[1]), "custom"


def _render_metrics(result: dict, frame: pd.DataFrame) -> None:
    metrics = result["metrics"]
    timing = metrics["timing"]
    altitude = metrics["altitude"]
    actuator = metrics["actuator"]
    sensor = metrics["sensor_comparison"]
    frequency = metrics["frequency"]
    cards = st.columns(5)
    cards[0].metric(_t("분석 행 수", "Analysis rows"), result["analysis_window"]["rows"])
    cards[1].metric(_t("지속 시간", "Duration"), _metric(timing.get("duration_s"), "s"))
    cards[2].metric("Altitude RMSE", _metric(altitude.get("rmse_m"), "m"))
    cards[3].metric(_t("Correction 포화", "Correction saturation"), _metric_percent(actuator.get("correction_saturation_ratio")))
    cards[4].metric(_t("주요 진동", "Dominant vibration"), _metric(frequency.get("dominant_frequency_hz"), "Hz"))
    tab_altitude, tab_output, tab_psd, tab_sampling, tab_missing = st.tabs([
        _t("고도", "Altitude"),
        _t("출력", "Output"),
        "PSD",
        "Sampling",
        _t("계산 불가 항목", "Unavailable analyses"),
    ])
    with tab_altitude:
        columns = [name for name in ("altitude_setpoint", "ekf_altitude", "reference_altitude") if name in frame]
        if "timestamp" in frame and columns:
            st.line_chart(frame.set_index("timestamp")[columns])
        if {"altitude_setpoint", "ekf_altitude", "timestamp"}.issubset(frame.columns):
            error = frame[["timestamp"]].copy()
            error["altitude_error"] = frame["altitude_setpoint"] - frame["ekf_altitude"]
            st.line_chart(error.set_index("timestamp"))
        st.json(localize_values(altitude, _language()))
        st.markdown(f"**{_t('기준 고도-EKF 비교', 'Reference-altitude/EKF comparison')}**")
        st.json(localize_values(sensor, _language()))
    with tab_output:
        columns = [name for name in ("throttle_base", "throttle_correction", "motor_1", "motor_2", "motor_3", "motor_4") if name in frame]
        if "timestamp" in frame and columns:
            st.line_chart(frame.set_index("timestamp")[columns])
        st.json(localize_values(actuator, _language()))
    with tab_psd:
        if frequency.get("available"):
            psd = pd.DataFrame({"frequency_hz": frequency["frequencies_hz"], "PSD": frequency["psd"]}).set_index("frequency_hz")
            st.line_chart(psd)
        else:
            st.warning(_tx(frequency.get("reason", "계산 불가")))
    with tab_sampling:
        intervals = timing.get("sampling_intervals_s", [])
        if intervals:
            st.line_chart(pd.DataFrame({"sampling_interval_s": intervals}))
        st.json(localize_values({key: value for key, value in timing.items() if key != "sampling_intervals_s"}, _language()))
    with tab_missing:
        unavailable = [(name, values.get("reason", "필요한 파라미터가 없습니다.")) for name, values in metrics.items() if not values.get("available")]
        if unavailable:
            for name, reason in unavailable:
                st.warning(f"{name}: {_tx(reason)}")
        else:
            st.success(_t("모든 핵심 분석 모듈이 계산되었습니다.", "All core analysis modules were calculated."))


def _metric(value: Optional[float], unit: str) -> str:
    return _t("계산 불가", "Unavailable") if value is None else f"{value:.3f} {unit}"


def _metric_percent(value: Optional[float]) -> str:
    return _t("계산 불가", "Unavailable") if value is None else f"{value * 100:.1f}%"


def _bullet_list(values: List[str]) -> None:
    if not values:
        st.write(_t("없음", "None"))
    else:
        for value in values:
            st.markdown(f"- {_tx(value)}")


def _bullet_section(title: str, values: List[str]) -> None:
    st.markdown(f"**{title}**")
    _bullet_list(values)


st.set_page_config(page_title="FlightLog Copilot", page_icon="✈️", layout="wide")
with st.sidebar:
    if "language_choice" not in st.session_state:
        st.session_state["language_choice"] = "KO"
    language_choice = st.segmented_control(
        "Language / 언어",
        options=["KO", "EN"],
        key="language_choice",
    )
selected_language = SUPPORTED_LANGUAGES.get(language_choice or "KO", "ko")
previous_language = st.session_state.get("language")
if previous_language and previous_language != selected_language and "ai_result" in st.session_state:
    st.session_state.pop("ai_result", None)
    st.session_state["ai_language_notice"] = True
st.session_state["language"] = selected_language
st.title("FlightLog Copilot")
st.caption(_t(
    "AI-Powered UAV Flight Log Diagnosis · Python이 계산하고 AI는 구조화된 결과만 설명합니다.",
    "AI-Powered UAV Flight Log Diagnosis · Python calculates the evidence; AI explains only structured results.",
))

with st.sidebar:
    st.header(_t("분석 설정", "Analysis settings"))
    correction_limit = st.number_input("Correction limit (us)", min_value=1.0, value=160.0, step=10.0)
    dropout_multiplier = st.number_input(_t("Dropout 판정 배수", "Dropout threshold multiplier"), min_value=1.5, value=3.0, step=0.5)
    motor_pwm_min = st.number_input(_t("Motor PWM 최솟값 (us)", "Minimum motor PWM (us)"), value=1000.0, step=10.0)
    motor_pwm_max = st.number_input(_t("Motor PWM 최댓값 (us)", "Maximum motor PWM (us)"), value=2000.0, step=10.0)
    st.info(_t("좌표계·축 방향·부호는 자동 확정하지 않습니다.", "Coordinate frame, axis direction, and sign are never auto-confirmed."))
    _render_ai_settings()

st.header(_t("Step 1. CSV 업로드", "Step 1. Upload CSV"))
uploaded = st.file_uploader(_t("STM32 UAV 비행 로그 CSV", "STM32 UAV flight-log CSV"), type=["csv", "txt"])
if uploaded is None:
    st.info(_t("CSV를 업로드하면 컬럼 프로파일과 자동 매핑 후보를 확인할 수 있습니다.", "Upload a CSV to inspect its column profile and automatic mapping candidates."))
    st.stop()

raw_bytes = uploaded.getvalue()
file_id = hashlib.sha256(raw_bytes).hexdigest()[:16]
if st.session_state.get("file_id") != file_id:
    for key in list(st.session_state):
        if key.startswith(("map_", "profile_", "analysis_", "segment_")):
            del st.session_state[key]
    st.session_state["file_id"] = file_id
    st.session_state["mapping_confirmed"] = False
    for stale_key in (
        "analysis_result",
        "ai_result",
        "confirmed_profile",
        "canonical_frame",
        "transform_warnings",
        "imported_profile_id",
        "profile_import_notice",
    ):
        st.session_state.pop(stale_key, None)

try:
    loaded, column_profile_base, candidates = _prepare_upload(raw_bytes, uploaded.name)
except CSVLoadError as exc:
    st.error(_tx(str(exc)))
    st.stop()

raw_frame = loaded.data
summary_columns = st.columns(4)
summary_columns[0].metric(_t("행", "Rows"), f"{len(raw_frame):,}")
summary_columns[1].metric(_t("열", "Columns"), f"{len(raw_frame.columns):,}")
summary_columns[2].metric(_t("인코딩", "Encoding"), loaded.encoding)
summary_columns[3].metric(_t("구분자", "Delimiter"), repr(loaded.delimiter))
for warning in loaded.warnings:
    st.warning(_tx(warning))
with st.expander(_t("원본 데이터 미리보기", "Source-data preview"), expanded=True):
    st.dataframe(raw_frame.head(100), use_container_width=True)
with st.expander(_t("컬럼 프로파일", "Column profile")):
    column_profile = column_profile_base
    if _language() == "en":
        column_profile = column_profile.rename(columns={
            "순서": "Order",
            "컬럼명": "Column",
            "추정 타입": "Inferred type",
            "유효 개수": "Valid count",
            "결측 비율": "Missing ratio",
            "최솟값": "Minimum",
            "최댓값": "Maximum",
            "평균": "Mean",
            "표준편차": "Std. dev.",
            "고유값 개수": "Unique count",
            "앞부분 샘플": "Leading samples",
        })
    st.dataframe(column_profile, use_container_width=True, hide_index=True)

st.header(_t("Step 2. 파라미터 매핑", "Step 2. Parameter mapping"))
candidate_rows = [
    {
        _t("표준 파라미터", "Canonical parameter"): name,
        _t("자동 추천", "Auto suggestion"): candidates[name].column,
        _t("신뢰도", "Confidence"): candidates[name].confidence,
        _t("상태", "Status"): _tx(candidates[name].status),
        _t("필수", "Requirement"): _t("필수", "Required") if name in MINIMUM_REQUIRED else _t("선택", "Optional"),
        _t("근거", "Evidence"): "; ".join(_tx(reason) for reason in candidates[name].reasons),
    }
    for name in STANDARD_PARAMETERS
]
st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)

imported_profile_file = st.file_uploader(_t("기존 매핑 JSON 불러오기", "Import an existing mapping JSON"), type=["json"], key="profile_upload")
if imported_profile_file is not None:
    imported_bytes = imported_profile_file.getvalue()
    imported_id = hashlib.sha256(imported_bytes).hexdigest()
    if st.session_state.get("imported_profile_id") != imported_id:
        try:
            imported = load_profile(imported_bytes)
            st.session_state["imported_profile_id"] = imported_id
            st.session_state["profile_name"] = imported.profile_name
            if imported.time_unit in {"auto", "seconds", "milliseconds", "microseconds"}:
                st.session_state["profile_time_unit"] = imported.time_unit
            if imported.coordinate_frame in {"확인 필요", "NED", "ENU", "FRD", "FLU", "기타"}:
                st.session_state["profile_coordinate_frame"] = imported.coordinate_frame
            st.session_state["profile_reference_altitude_source"] = imported.reference_altitude_source
            for parameter, mapping in imported.parameters.items():
                selection = mapping.column if mapping.column in raw_frame.columns else ("__direct__" if mapping.column else "__none__")
                st.session_state[f"map_select_{parameter}"] = selection
                st.session_state[f"map_direct_{parameter}"] = mapping.column or ""
                st.session_state[f"map_unit_{parameter}"] = mapping.unit if mapping.unit in _unit_options(parameter) else _default_unit(parameter)
                st.session_state[f"map_scale_{parameter}"] = mapping.scale
                st.session_state[f"map_offset_{parameter}"] = mapping.offset
                st.session_state[f"map_invert_{parameter}"] = mapping.invert_sign
            missing_imported = validate_profile_columns(imported, raw_frame.columns)
            if missing_imported:
                st.session_state["profile_import_notice"] = ("warning", "현재 CSV에 없는 프로필 컬럼: " + ", ".join(missing_imported))
            elif imported.header_hash and imported.header_hash == header_hash(raw_frame.columns):
                st.session_state["profile_import_notice"] = ("success", "현재 CSV와 헤더 해시가 일치하는 매핑 프로필입니다.")
            else:
                st.session_state["profile_import_notice"] = ("success", "매핑 프로필을 불러왔습니다. 컬럼과 좌표계 설정을 확인하십시오.")
            st.rerun()
        except ValueError as exc:
            st.error(_tx(str(exc)))

notice = st.session_state.get("profile_import_notice")
if notice:
    getattr(st, notice[0])(_tx(notice[1]))

profile_controls = st.columns([2, 1, 1, 1])
if "profile_name" not in st.session_state:
    st.session_state["profile_name"] = "STM32_AltHold_profile"
profile_name = profile_controls[0].text_input(_t("프로필 이름", "Profile name"), key="profile_name")
time_unit = profile_controls[1].selectbox(_t("시간 단위", "Time unit"), ["auto", "seconds", "milliseconds", "microseconds"], key="profile_time_unit")
coordinate_frame = profile_controls[2].selectbox(
    _t("좌표계", "Coordinate frame"),
    ["확인 필요", "NED", "ENU", "FRD", "FLU", "기타"],
    key="profile_coordinate_frame",
    format_func=lambda value: {"확인 필요": _t("확인 필요", "Review required"), "기타": _t("기타", "Other")}.get(value, value),
)
reference_altitude_source = profile_controls[3].selectbox(
    _t("기준 고도 출처", "Reference altitude source"),
    ["auto", "barometer", "gnss", "other"],
    key="profile_reference_altitude_source",
    format_func=lambda value: {
        "auto": _t("자동 판별", "Auto-detect"),
        "barometer": "Barometer",
        "gnss": "GNSS",
        "other": _t("기타", "Other"),
    }[value],
)

st.warning(_t(
    "`pos_d`, `vel_d` 등은 NED 좌표계의 Down 축일 수 있습니다. 실제 고도/상승속도로 사용하기 전에 좌표계와 부호를 확인하고 필요하면 직접 부호 반전을 선택하십시오.",
    "`pos_d`, `vel_d`, and similar columns may use the NED Down axis. Verify the coordinate frame and sign before using them as altitude or climb rate, and explicitly invert the sign when required.",
))
st.info(_t(
    "reference_altitude에는 EKF와 비교할 Barometer 또는 GNSS 고도를 매핑하십시오. GNSS를 사용할 때는 타원체/MSL 고도 기준을 확인해야 합니다.",
    "Map the Barometer or GNSS altitude used for EKF comparison to reference_altitude. For GNSS, verify whether altitude uses the ellipsoid or MSL datum.",
))

mappings: Dict[str, ParameterMapping] = {}
mapping_errors: List[str] = []
for parameter in STANDARD_PARAMETERS:
    candidate = candidates[parameter]
    with st.container(border=True):
        row = st.columns([1.2, 1.2, 1.5, 0.9, 0.8, 0.8, 0.7, 0.7, 0.9])
        row[0].markdown(f"**{parameter}**\n\n{_t('필수', 'Required') if parameter in MINIMUM_REQUIRED else _t('선택', 'Optional')}")
        row[1].write(candidate.column or _t("추천 없음", "No suggestion"))
        options = ["__none__", *[str(column) for column in raw_frame.columns], "__direct__"]
        key = f"map_select_{parameter}"
        default_selection = candidate.column if candidate.status == "자동 추천" and candidate.column in raw_frame.columns else "__none__"
        if st.session_state.get(key) == "사용하지 않음":
            st.session_state[key] = "__none__"
        elif st.session_state.get(key) == "직접 입력":
            st.session_state[key] = "__direct__"
        if key not in st.session_state:
            st.session_state[key] = default_selection
        selected = row[2].selectbox(
            _t("사용자 선택", "User selection"),
            options,
            key=key,
            label_visibility="collapsed",
            format_func=lambda value: {"__none__": _t("사용하지 않음", "Do not use"), "__direct__": _t("직접 입력", "Enter manually")}.get(value, value),
        )
        unit = row[3].selectbox(_t("단위", "Unit"), _unit_options(parameter), key=f"map_unit_{parameter}", label_visibility="collapsed")
        scale = row[4].number_input("Scale", value=1.0, format="%.6f", key=f"map_scale_{parameter}", label_visibility="collapsed")
        offset = row[5].number_input("Offset", value=0.0, format="%.6f", key=f"map_offset_{parameter}", label_visibility="collapsed")
        invert = row[6].checkbox(_t("부호 반전", "Invert sign"), key=f"map_invert_{parameter}")
        row[7].write(f"{candidate.confidence:.2f}")
        row[8].write(_tx("사용자 확인" if st.session_state.get("mapping_confirmed") else candidate.status))
        direct_value = ""
        if selected == "__direct__":
            direct_value = st.text_input(
                _t("직접 컬럼명", "Manual column name"),
                key=f"map_direct_{parameter}",
                placeholder=_t("CSV의 정확한 컬럼명", "Exact CSV column name"),
            )
            if direct_value and direct_value not in raw_frame.columns:
                mapping_errors.append(_t(
                    f"{parameter}: 직접 입력한 '{direct_value}' 컬럼이 CSV에 없습니다.",
                    f"{parameter}: manually entered column '{direct_value}' is not present in the CSV.",
                ))
        column = direct_value if selected == "__direct__" else (None if selected == "__none__" else selected)
        mappings[parameter] = ParameterMapping(
            column=column,
            unit=unit,
            scale=float(scale),
            offset=float(offset),
            invert_sign=bool(invert),
            confirmed=False,
        )

if time_unit == "auto":
    timestamp_mapping = mappings.get("timestamp")
    if timestamp_mapping and timestamp_mapping.column in raw_frame.columns:
        timestamp_values = pd.to_numeric(raw_frame[timestamp_mapping.column], errors="coerce")
        timestamp_values = (
            (-1.0 if timestamp_mapping.invert_sign else 1.0)
            * timestamp_values
            * float(timestamp_mapping.scale)
            + float(timestamp_mapping.offset)
        )
        time_inference = resolve_time_unit(
            timestamp_values,
            timestamp_mapping.column,
            selected_time_unit=time_unit,
            mapping_unit=timestamp_mapping.unit,
        )
        confidence_label = (
            _t("높음", "High")
            if time_inference.confidence >= 0.8
            else _t("중간", "Medium")
            if time_inference.confidence >= 0.6
            else _t("낮음", "Low")
        )
        reason_text = "; ".join(_tx(reason) for reason in time_inference.reasons)
        if time_inference.requires_confirmation or time_inference.unit is None:
            mapping_errors.append(_t(
                "시간 단위를 자동으로 확정할 수 없습니다. seconds, milliseconds, microseconds 중 하나를 직접 선택하십시오.",
                "The timestamp unit cannot be confirmed automatically. Select seconds, milliseconds, or microseconds manually.",
            ) + f"\n\n{_t('신뢰도', 'Confidence')}: {confidence_label} ({time_inference.confidence:.2f})"
            + f"\n\n{_t('근거', 'Reason')}: {reason_text}")
        else:
            st.info(
                f"{_t('추정 시간 단위', 'Inferred timestamp unit')}: {time_inference.unit}"
                f"\n\n{_t('신뢰도', 'Confidence')}: {confidence_label} ({time_inference.confidence:.2f})"
                f"\n\n{_t('근거', 'Reason')}: {reason_text}"
            )

if coordinate_frame == "확인 필요":
    st.info(_t(
        "좌표계가 아직 확정되지 않았습니다. 고도 부호 해석은 사용자가 설정한 부호 반전만 적용합니다.",
        "The coordinate frame is not confirmed. Altitude-sign interpretation uses only the sign inversion explicitly selected by the user.",
    ))
for error in mapping_errors:
    st.error(error)

profile = MappingProfile(
    profile_name=profile_name,
    time_unit=time_unit,
    coordinate_frame=coordinate_frame,
    reference_altitude_source=reference_altitude_source,
    header_hash=header_hash(raw_frame.columns),
    parameters=mappings,
)
saved_profile_payload = st.session_state.get("confirmed_profile")
if st.session_state.get("mapping_confirmed") and saved_profile_payload:
    saved_profile = MappingProfile.from_dict(saved_profile_payload)
    if _profile_signature(profile) != _profile_signature(saved_profile):
        st.session_state["mapping_confirmed"] = False
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("ai_result", None)
        st.warning(_t(
            "매핑 또는 변환 설정이 변경되었습니다. 변경 사항을 적용하려면 다시 `매핑 확정`을 누르십시오.",
            "The mapping or transform settings changed. Select `Confirm mapping` again to apply them.",
        ))
st.download_button(
    _t("현재 매핑 JSON 다운로드", "Download current mapping JSON"),
    dump_profile(profile),
    file_name=f"{_safe_filename(profile_name)}.json",
    mime="application/json",
    disabled=bool(mapping_errors),
)

if st.button(_t("매핑 확정", "Confirm mapping"), type="primary", disabled=bool(mapping_errors)):
    confirmed_profile = MappingProfile.from_dict(profile.to_dict())
    for mapping in confirmed_profile.parameters.values():
        mapping.confirmed = True
    canonical, transform_warnings = transform_frame(raw_frame, confirmed_profile)
    st.session_state["confirmed_profile"] = confirmed_profile.to_dict()
    st.session_state["canonical_frame"] = canonical
    st.session_state["transform_warnings"] = transform_warnings
    st.session_state["mapping_confirmed"] = True
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("ai_result", None)
    st.rerun()

if not st.session_state.get("mapping_confirmed"):
    st.info(_t(
        "`매핑 확정`을 눌러야 분석 구간 선택과 정량 분석을 실행할 수 있습니다.",
        "Select `Confirm mapping` before choosing an analysis segment or running quantitative analysis.",
    ))
    st.stop()

confirmed_profile = MappingProfile.from_dict(st.session_state["confirmed_profile"])
canonical = st.session_state["canonical_frame"]
st.success(_t(
    "매핑이 확정되었습니다. 아래 분석은 이 사용자 확정 매핑만 사용합니다.",
    "The mapping is confirmed. The analysis below uses only this user-confirmed mapping.",
))
for warning in st.session_state.get("transform_warnings", []):
    st.warning(_tx(warning))

settings = AnalysisSettings(
    correction_limit=float(correction_limit),
    motor_pwm_min=float(motor_pwm_min),
    motor_pwm_max=float(motor_pwm_max),
    dropout_multiplier=float(dropout_multiplier),
)
validation = validate_data(raw_frame, canonical, settings, loaded.duplicate_columns)
_render_validation(validation)

st.header(_t("Step 3. 분석 구간 선택", "Step 3. Select analysis segment"))
segments = detect_segments(canonical)
segment_options = ["full"]
if segments["althold"]:
    segment_options.insert(0, "auto_althold")
    segment_options.append("althold")
if segments["armed"]:
    segment_options.append("armed")
if "timestamp" in canonical:
    segment_options.append("custom")
legacy_segment_modes = {
    "전체 로그": "full",
    "자동 감지된 AltHold 구간": "auto_althold",
    "AltHold 구간": "althold",
    "Armed 구간": "armed",
    "직접 시간 범위": "custom",
}
if st.session_state.get("segment_mode") in legacy_segment_modes:
    st.session_state["segment_mode"] = legacy_segment_modes[st.session_state["segment_mode"]]
segment_labels = {
    "full": _t("전체 로그", "Full log"),
    "auto_althold": _t("자동 감지된 AltHold 구간", "Auto-detected AltHold segment"),
    "althold": _t("AltHold 구간", "AltHold segment"),
    "armed": _t("Armed 구간", "Armed segment"),
    "custom": _t("직접 시간 범위", "Custom time range"),
}
selection_mode = st.radio(
    _t("구간 방식", "Segment mode"),
    segment_options,
    horizontal=True,
    key="segment_mode",
    format_func=lambda value: segment_labels[value],
)
analysis_frame, flight_phase = _select_segment(canonical, segments, selection_mode)

if "timestamp" in analysis_frame and not analysis_frame.empty:
    time_valid = pd.to_numeric(analysis_frame["timestamp"], errors="coerce").dropna()
    if not time_valid.empty:
        segment_cards = st.columns(3)
        segment_cards[0].metric(_t("시작 시간", "Start time"), f"{time_valid.min():.3f} s")
        segment_cards[1].metric(_t("종료 시간", "End time"), f"{time_valid.max():.3f} s")
        segment_cards[2].metric(_t("분석 행 수", "Analysis rows"), f"{len(analysis_frame):,}")
        chart_columns = [name for name in ("altitude_setpoint", "ekf_altitude", "reference_altitude") if name in analysis_frame]
        if chart_columns:
            st.line_chart(analysis_frame.set_index("timestamp")[chart_columns])
else:
    st.warning(_t("선택한 분석 구간에 유효한 행이 없습니다.", "The selected analysis segment has no valid rows."))

if st.button(_t("정량 분석 실행", "Run quantitative analysis"), type="primary", disabled=analysis_frame.empty):
    mapping_columns = {name: mapping.column for name, mapping in confirmed_profile.parameters.items()}
    result = run_deterministic_analysis(
        canonical,
        analysis_frame,
        mapping_columns,
        [message.to_dict() for message in validation],
        flight_phase,
        settings,
        reference_altitude_source=confirmed_profile.reference_altitude_source,
    )
    st.session_state["analysis_result"] = result
    st.session_state.pop("ai_result", None)
    st.rerun()

result = st.session_state.get("analysis_result")
if not result:
    st.stop()

st.header(_t("Step 4. 정량 분석", "Step 4. Quantitative analysis"))
_render_metrics(result, analysis_frame)

st.header(_t("Step 5. 가설 진단", "Step 5. Hypothesis diagnosis"))
st.caption(_t(
    "진단 우선순위 점수는 확률이 아닙니다. 어떤 규칙이 점수를 올리거나 낮췄는지 확인할 수 있습니다.",
    "Diagnostic priority scores are not probabilities. Expand each hypothesis to inspect which rules increased or decreased its score.",
))
for hypothesis in result["rule_based_hypotheses"]:
    with st.expander(f"{hypothesis['score']:>3}/100 · {_tx(hypothesis['title'])}"):
        columns = st.columns(2)
        with columns[0]:
            st.markdown(f"**{_t('근거', 'Evidence')}**")
            _bullet_list(hypothesis["evidence"])
            st.markdown(f"**{_t('반대 근거', 'Counter-evidence')}**")
            _bullet_list(hypothesis["counter_evidence"])
            st.markdown(f"**{_t('부족한 파라미터', 'Missing parameters')}**")
            _bullet_list(hypothesis["missing_parameters"])
        with columns[1]:
            st.markdown(f"**{_t('분석 한계', 'Limitations')}**")
            _bullet_list(hypothesis["limitations"])
            st.markdown(f"**{_t('권장 검증 실험', 'Recommended validation tests')}**")
            _bullet_list(hypothesis["recommended_tests"])
        st.markdown(f"**{_t('점수 계산 상세', 'Score calculation details')}**")
        st.dataframe(pd.DataFrame(localize_values(hypothesis["score_details"], _language())), use_container_width=True, hide_index=True)

st.header(_t("Step 6. AI Copilot", "Step 6. AI Copilot"))
if st.session_state.pop("ai_language_notice", False):
    st.info(_t(
        "언어가 변경되어 기존 AI 응답만 초기화했습니다. Python 분석 결과와 매핑은 유지됩니다.",
        "The existing AI response was cleared after the language change. Python analysis and mapping remain unchanged.",
    ))
llm_payload = build_llm_payload(result, response_language=_language())
with st.expander(_t("외부 AI Provider에 전달되는 구조화 데이터 미리보기", "Preview structured data sent to the external AI provider")):
    st.json(localize_values(llm_payload, _language()))
st.caption(_t(
    "원시 CSV, 개별 행, API 키, 로컬 경로는 외부 AI Provider에 전달되지 않습니다.",
    "Raw CSV data, individual rows, API keys, and local paths are never sent to the external AI provider.",
))
ai_settings = _current_ai_settings()
if not ai_settings.enabled:
    st.info(_t(
        "AI Copilot이 비활성화되어 있습니다. 사이드바의 AI 설정에서 활성화할 수 있습니다.",
        "AI Copilot is disabled. Enable it in AI settings in the sidebar.",
    ))
if st.button(_t("AI 진단 실행", "Run AI diagnosis"), disabled=not ai_settings.enabled):
    try:
        secret = _active_secret(ai_settings)
        llm_client = create_llm_client(
            ai_settings.provider,
            ai_settings,
            secret.api_key if secret else None,
        )
        with st.spinner(_t("구조화된 분석 결과를 설명하는 중입니다...", "Explaining the structured analysis results...")):
            st.session_state["ai_result"] = llm_client.generate_diagnosis(llm_payload)
        st.success(_t("AI Copilot 분석이 완료되었습니다.", "AI Copilot analysis is complete."))
        st.rerun()
    except (LLMAnalysisError, ValueError) as exc:
        code = getattr(exc, "code", "base_url" if isinstance(exc, ValueError) else "unknown")
        st.error(error_message(code, _language()))
        st.info(_t(
            "Python 정량 분석과 규칙 기반 가설 결과는 계속 사용할 수 있습니다.",
            "The Python quantitative analysis and rule-based hypotheses remain available.",
        ))

ai_result = st.session_state.get("ai_result")
if ai_result:
    result = dict(result)
    result["ai_copilot"] = ai_result
    st.subheader(_t("AI 요약", "AI summary"))
    st.write(ai_result["summary"])
    if ai_result["hypothesis_review"]:
        st.subheader(_t("규칙 기반 가설 검토", "Rule-based hypothesis review"))
        for review in sorted(ai_result["hypothesis_review"], key=lambda item: item["priority"]):
            with st.expander(f"{_t('우선순위', 'Priority')} {review['priority']} · {review['hypothesis_id']}"):
                st.write(review["explanation"])
                _bullet_section(_t("근거", "Evidence"), review["supporting_evidence"])
                _bullet_section(_t("반대 근거", "Counter-evidence"), review["counter_evidence"])
                _bullet_section(_t("불확실성", "Uncertainties"), review["uncertainties"])
    if ai_result["additional_hypotheses"]:
        st.warning(_t(
            "아래는 AI Provider가 추가로 제안한 미검증 가설이며 확정된 결과가 아닙니다.",
            "The hypotheses below are unverified suggestions from the AI provider, not confirmed findings.",
        ))
        _bullet_list(ai_result["additional_hypotheses"])
    if ai_result["recommended_experiments"]:
        st.subheader(_t("추가 검증 실험", "Additional validation experiments"))
        for experiment in ai_result["recommended_experiments"]:
            with st.expander(experiment["title"]):
                st.write(experiment["purpose"])
                _bullet_section(_t("변경 변수", "Variables to change"), experiment["variables_to_change"])
                _bullet_section(_t("기록 변수", "Variables to log"), experiment["variables_to_log"])
                st.markdown(f"**{_t('예상 관찰', 'Expected observation')}:** {experiment['expected_observation']}")
                _bullet_section(_t("안전 주의", "Safety notes"), experiment["safety_notes"])

st.header(_t("Step 7. 보고서", "Step 7. Reports"))
markdown_report = render_markdown_report(result, language=_language())
localized_result = localize_values(result, _language())
json_report = render_json_report(localized_result)
profile_json = dump_profile(confirmed_profile)
html_report = f"<!doctype html><html lang='{_language()}'><meta charset='utf-8'><title>FlightLog Copilot Report</title><style>body{{max-width:1000px;margin:40px auto;font-family:system-ui;line-height:1.55}}pre{{white-space:pre-wrap}}</style><body><pre>" + html.escape(markdown_report) + "</pre></body></html>"
downloads = st.columns(4)
downloads[0].download_button(_t("Markdown 보고서", "Markdown report"), markdown_report, "flightlog_report.md", "text/markdown")
downloads[1].download_button(_t("JSON 분석 결과", "JSON analysis"), json_report, "flightlog_analysis.json", "application/json")
downloads[2].download_button(_t("매핑 JSON", "Mapping JSON"), profile_json, "flightlog_mapping.json", "application/json")
downloads[3].download_button(_t("HTML 보고서", "HTML report"), html_report, "flightlog_report.html", "text/html")
with st.expander(_t("Markdown 보고서 미리보기", "Markdown report preview")):
    st.markdown(markdown_report)
