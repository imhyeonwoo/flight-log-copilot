"""Altitude tracking and conservative step-response analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def analyze_altitude_tracking(
    frame: pd.DataFrame,
    steady_state_fraction: float = 0.2,
    settling_tolerance_fraction: float = 0.05,
) -> Dict[str, Any]:
    required = ["altitude_setpoint", "ekf_altitude"]
    missing = [name for name in required if name not in frame]
    if missing:
        return {"available": False, "missing_parameters": missing, "reason": "고도 추종 파라미터가 부족합니다."}
    columns = required + (["timestamp"] if "timestamp" in frame else [])
    data = frame[columns].copy()
    for column in required + (["timestamp"] if "timestamp" in data else []):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required)
    if len(data) < 2:
        return {"available": False, "missing_parameters": [], "reason": "유효 고도 샘플이 2개 미만입니다."}
    error = data["altitude_setpoint"] - data["ekf_altitude"]
    absolute = error.abs()
    steady_count = max(1, int(round(len(error) * steady_state_fraction)))
    result: Dict[str, Any] = {
        "available": True,
        "samples": int(len(data)),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "mae_m": float(absolute.mean()),
        "mean_error_m": float(error.mean()),
        "error_std_m": float(error.std(ddof=0)),
        "max_absolute_error_m": float(absolute.max()),
        "absolute_error_p95_m": float(np.percentile(absolute, 95)),
        "steady_state_error_m": float(error.iloc[-steady_count:].mean()),
        "overshoot_m": None,
        "undershoot_m": None,
        "rise_time_s": None,
        "settling_time_s": None,
        "dynamic_metrics_reason": None,
    }
    result.update(_step_metrics(data, settling_tolerance_fraction))
    return result


def _step_metrics(data: pd.DataFrame, tolerance_fraction: float) -> Dict[str, Optional[float]]:
    setpoint = data["altitude_setpoint"]
    changes = setpoint.diff().abs().fillna(0)
    full_range = float(setpoint.max() - setpoint.min())
    threshold = max(1e-6, 0.05 * max(full_range, float(setpoint.abs().max()), 1.0))
    candidates = changes[changes > threshold]
    if candidates.empty or "timestamp" not in data:
        return {"dynamic_metrics_reason": "명확한 setpoint 변화 또는 timestamp가 없어 동적 지표를 계산하지 않았습니다."}
    change_index = candidates.idxmax()
    location = data.index.get_loc(change_index)
    if location < 1 or len(data) - location < 5:
        return {"dynamic_metrics_reason": "setpoint 변화 전후 데이터가 부족합니다."}
    initial = float(setpoint.iloc[max(0, location - 3):location].median())
    final = float(setpoint.iloc[-max(3, len(data) // 10):].median())
    step = final - initial
    if abs(step) <= threshold:
        return {"dynamic_metrics_reason": "지속되는 setpoint 단계를 확인하지 못했습니다."}
    response = data["ekf_altitude"].iloc[location:]
    time = data["timestamp"].iloc[location:]
    direction = 1.0 if step > 0 else -1.0
    directed = direction * (response - initial)
    amplitude = abs(step)
    rise_10 = _first_time(directed, time, 0.1 * amplitude)
    rise_90 = _first_time(directed, time, 0.9 * amplitude)
    rise_time = rise_90 - rise_10 if rise_10 is not None and rise_90 is not None and rise_90 >= rise_10 else None
    directed_error = direction * (response - final)
    overshoot = max(0.0, float(directed_error.max()))
    undershoot = max(0.0, float((-directed_error).max()))
    tolerance = max(abs(step) * tolerance_fraction, 1e-6)
    outside = np.flatnonzero(np.abs((response - final).to_numpy()) > tolerance)
    settling_time = None
    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] + 1 < len(response):
        settling_time = float(time.iloc[outside[-1] + 1] - time.iloc[0])
    return {
        "overshoot_m": overshoot,
        "undershoot_m": undershoot,
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "dynamic_metrics_reason": None,
    }


def _first_time(values: pd.Series, time: pd.Series, threshold: float) -> Optional[float]:
    reached = np.flatnonzero(values.to_numpy() >= threshold)
    if len(reached) == 0:
        return None
    return float(time.iloc[reached[0]])
