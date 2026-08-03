"""Reference-altitude/EKF comparison with correlation caveats."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import signal


def analyze_sensor_comparison(frame: pd.DataFrame, reference_source: str = "other") -> Dict[str, Any]:
    reference_source = reference_source if reference_source in {"barometer", "gnss", "other"} else "other"
    required = ["reference_altitude", "ekf_altitude"]
    missing = [name for name in required if name not in frame]
    if missing:
        return {
            "available": False,
            "reference_altitude_source": reference_source,
            "missing_parameters": missing,
            "reason": "기준 고도-EKF 비교 파라미터가 부족합니다.",
        }
    data = frame.copy()
    for column in data.columns:
        if column not in {"armed", "althold_active"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    paired = data[required].dropna()
    if len(paired) < 3:
        return {
            "available": False,
            "reference_altitude_source": reference_source,
            "missing_parameters": [],
            "reason": "동시에 유효한 기준 고도/EKF 샘플이 3개 미만입니다.",
        }
    difference = paired["reference_altitude"] - paired["ekf_altitude"]
    lag_s, max_corr = _cross_correlation(data)
    result: Dict[str, Any] = {
        "available": True,
        "reference_altitude_source": reference_source,
        "reference_mean_m": float(paired["reference_altitude"].mean()),
        "ekf_altitude_mean_m": float(paired["ekf_altitude"].mean()),
        "reference_std_m": float(paired["reference_altitude"].std(ddof=0)),
        "ekf_altitude_std_m": float(paired["ekf_altitude"].std(ddof=0)),
        "reference_ekf_bias_m": float(difference.mean()),
        "difference_std_m": float(difference.std(ddof=0)),
        "pearson_correlation": _safe_correlation(paired.iloc[:, 0], paired.iloc[:, 1]),
        "estimated_lag_s": lag_s,
        "max_cross_correlation": max_corr,
        "mean_motor_pwm_vs_abs_error_correlation": None,
        "throttle_correction_vs_reference_change_correlation": None,
        "caveat": "상관관계와 cross-correlation lag는 인과관계를 증명하지 않습니다.",
        "source_caveat": _source_caveat(reference_source),
    }
    motors = [name for name in ("motor_1", "motor_2", "motor_3", "motor_4") if name in data]
    if motors:
        motor_mean = data[motors].mean(axis=1)
        abs_error = (data["reference_altitude"] - data["ekf_altitude"]).abs()
        result["mean_motor_pwm_vs_abs_error_correlation"] = _safe_correlation(motor_mean, abs_error)
    if "throttle_correction" in data:
        reference_change = data["reference_altitude"].diff()
        result["throttle_correction_vs_reference_change_correlation"] = _safe_correlation(
            data["throttle_correction"], reference_change
        )
    return result


def _source_caveat(reference_source: str) -> str:
    if reference_source == "barometer":
        return "Barometer 고도는 기압·프로펠러 바람·설치 위치의 영향을 받을 수 있습니다."
    if reference_source == "gnss":
        return "GNSS 고도는 타원체/MSL 기준과 위성 가시성의 영향을 받을 수 있습니다."
    return "기준 고도의 센서 종류와 수직 datum을 확인해야 합니다."


def _safe_correlation(left: pd.Series, right: pd.Series) -> Optional[float]:
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].std() == 0 or paired.iloc[:, 1].std() == 0:
        return None
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return float(value) if pd.notna(value) else None


def _cross_correlation(data: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    columns = ["reference_altitude", "ekf_altitude"] + (["timestamp"] if "timestamp" in data else [])
    paired = data[columns].dropna()
    if len(paired) < 4:
        return None, None
    left = paired["reference_altitude"].to_numpy(dtype=float)
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
