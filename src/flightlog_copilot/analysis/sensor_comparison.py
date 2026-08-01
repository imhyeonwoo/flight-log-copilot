"""Barometer/EKF comparison with correlation caveats."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import signal


def analyze_sensor_comparison(frame: pd.DataFrame) -> Dict[str, Any]:
    required = ["barometer_altitude", "ekf_altitude"]
    missing = [name for name in required if name not in frame]
    if missing:
        return {"available": False, "missing_parameters": missing, "reason": "Barometer-EKF 비교 파라미터가 부족합니다."}
    data = frame.copy()
    for column in data.columns:
        if column not in {"armed", "althold_active"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    paired = data[required].dropna()
    if len(paired) < 3:
        return {"available": False, "missing_parameters": [], "reason": "동시에 유효한 Barometer/EKF 샘플이 3개 미만입니다."}
    difference = paired["barometer_altitude"] - paired["ekf_altitude"]
    lag_s, max_corr = _cross_correlation(data)
    result: Dict[str, Any] = {
        "available": True,
        "barometer_mean_m": float(paired["barometer_altitude"].mean()),
        "ekf_altitude_mean_m": float(paired["ekf_altitude"].mean()),
        "barometer_std_m": float(paired["barometer_altitude"].std(ddof=0)),
        "ekf_altitude_std_m": float(paired["ekf_altitude"].std(ddof=0)),
        "barometer_ekf_bias_m": float(difference.mean()),
        "difference_std_m": float(difference.std(ddof=0)),
        "pearson_correlation": _safe_correlation(paired.iloc[:, 0], paired.iloc[:, 1]),
        "estimated_lag_s": lag_s,
        "max_cross_correlation": max_corr,
        "mean_motor_pwm_vs_abs_error_correlation": None,
        "throttle_correction_vs_barometer_change_correlation": None,
        "caveat": "상관관계와 cross-correlation lag는 인과관계를 증명하지 않습니다.",
    }
    motors = [name for name in ("motor_1", "motor_2", "motor_3", "motor_4") if name in data]
    if motors:
        motor_mean = data[motors].mean(axis=1)
        abs_error = (data["barometer_altitude"] - data["ekf_altitude"]).abs()
        result["mean_motor_pwm_vs_abs_error_correlation"] = _safe_correlation(motor_mean, abs_error)
    if "throttle_correction" in data:
        barometer_change = data["barometer_altitude"].diff()
        result["throttle_correction_vs_barometer_change_correlation"] = _safe_correlation(
            data["throttle_correction"], barometer_change
        )
    return result


def _safe_correlation(left: pd.Series, right: pd.Series) -> Optional[float]:
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].std() == 0 or paired.iloc[:, 1].std() == 0:
        return None
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return float(value) if pd.notna(value) else None


def _cross_correlation(data: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    columns = ["barometer_altitude", "ekf_altitude"] + (["timestamp"] if "timestamp" in data else [])
    paired = data[columns].dropna()
    if len(paired) < 4:
        return None, None
    left = paired["barometer_altitude"].to_numpy(dtype=float)
    right = paired["ekf_altitude"].to_numpy(dtype=float)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return None, None
    correlations = signal.correlate(left, right, mode="full") / denominator
    lags = signal.correlation_lags(len(left), len(right), mode="full")
    best = int(np.argmax(correlations))
    sample_lag = int(lags[best])
    dt = 1.0
    if "timestamp" in paired:
        differences = paired["timestamp"].diff().dropna()
        differences = differences[differences > 0]
        if not differences.empty:
            dt = float(differences.median())
    return float(sample_lag * dt), float(correlations[best])
