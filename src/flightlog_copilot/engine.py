"""High-level deterministic analysis orchestration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from flightlog_copilot.analysis import (
    analyze_actuators,
    analyze_altitude_tracking,
    analyze_frequency,
    analyze_sensor_comparison,
    analyze_timing,
)
from flightlog_copilot.config import AnalysisSettings
from flightlog_copilot.diagnosis import evaluate_hypotheses
from flightlog_copilot.reporting.json_report import sanitize_for_json


def run_deterministic_analysis(
    full_frame: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    parameter_mapping: Dict[str, Optional[str]],
    validation_messages: Optional[List[Dict[str, Any]]] = None,
    flight_phase: str = "custom",
    settings: AnalysisSettings = AnalysisSettings(),
) -> Dict[str, Any]:
    metrics = {
        "timing": analyze_timing(full_frame, analysis_frame, settings.dropout_multiplier),
        "altitude": analyze_altitude_tracking(analysis_frame, settings.steady_state_fraction, settings.settling_tolerance_fraction),
        "actuator": analyze_actuators(analysis_frame, settings.correction_limit, settings.motor_pwm_min, settings.motor_pwm_max),
        "sensor_comparison": analyze_sensor_comparison(analysis_frame),
        "frequency": analyze_frequency(analysis_frame),
    }
    hypotheses = evaluate_hypotheses(metrics, analysis_frame.columns)
    if "timestamp" in analysis_frame and not analysis_frame.empty:
        time = pd.to_numeric(analysis_frame["timestamp"], errors="coerce").dropna()
        start_s = float(time.min()) if not time.empty else None
        end_s = float(time.max()) if not time.empty else None
    else:
        start_s = end_s = None
    result = {
        "schema_version": "1.0",
        "flight_phase": flight_phase,
        "analysis_window": {"start_s": start_s, "end_s": end_s, "rows": int(len(analysis_frame))},
        "parameter_mapping": {key: value for key, value in parameter_mapping.items() if value},
        "metrics": metrics,
        "rule_based_hypotheses": hypotheses,
        "missing_parameters": sorted(set(parameter_mapping) - set(analysis_frame.columns)),
        "validation_messages": validation_messages or [],
        "analysis_warnings": [
            message.get("message", "")
            for message in (validation_messages or [])
            if message.get("severity") in {"error", "warning"}
        ],
    }
    return sanitize_for_json(result)


def build_llm_payload(result: Dict[str, Any], response_language: Optional[str] = None) -> Dict[str, Any]:
    metrics = result.get("metrics", {})
    altitude = metrics.get("altitude", {})
    actuator = metrics.get("actuator", {})
    sensor = metrics.get("sensor_comparison", {})
    frequency = metrics.get("frequency", {})
    features = {
        "altitude_rmse_m": altitude.get("rmse_m"),
        "mean_throttle_correction_pwm": actuator.get("mean_throttle_correction_pwm"),
        "correction_saturation_ratio": actuator.get("correction_saturation_ratio"),
        "dominant_frequency_hz": frequency.get("dominant_frequency_hz"),
        "barometer_ekf_bias_m": sensor.get("barometer_ekf_bias_m"),
    }
    payload = {
            "flight_phase": result.get("flight_phase"),
            "analysis_window": result.get("analysis_window"),
            "parameter_mapping": result.get("parameter_mapping", {}),
            "features": features,
            "rule_based_hypotheses": result.get("rule_based_hypotheses", []),
            "missing_parameters": result.get("missing_parameters", []),
            "analysis_warnings": result.get("analysis_warnings", []),
        }
    if response_language:
        payload["response_language"] = "English" if response_language == "en" else "Korean"
    return sanitize_for_json(payload)
