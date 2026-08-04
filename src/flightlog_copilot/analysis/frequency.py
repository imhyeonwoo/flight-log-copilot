"""Welch PSD analysis for altitude error or EKF altitude."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid


def analyze_frequency(frame: pd.DataFrame, minimum_samples: int = 32) -> Dict[str, Any]:
    source = "altitude_error" if {"altitude_setpoint", "ekf_altitude"}.issubset(frame.columns) else "ekf_altitude"
    if "timestamp" not in frame or "ekf_altitude" not in frame:
        return {"available": False, "missing_parameters": [name for name in ("timestamp", "ekf_altitude") if name not in frame], "reason": "주파수 분석 파라미터가 부족합니다."}
    value = (
        pd.to_numeric(frame["altitude_setpoint"], errors="coerce") - pd.to_numeric(frame["ekf_altitude"], errors="coerce")
        if source == "altitude_error"
        else pd.to_numeric(frame["ekf_altitude"], errors="coerce")
    )
    data = pd.DataFrame({"timestamp": pd.to_numeric(frame["timestamp"], errors="coerce"), "value": value})
    data = data.replace([np.inf, -np.inf], np.nan).dropna().sort_values("timestamp").drop_duplicates("timestamp")
    if len(data) < minimum_samples:
        return {
            "available": False,
            "source": source,
            "samples": int(len(data)),
            "reason": f"주파수 분석에는 최소 {minimum_samples}개의 유효 샘플이 필요합니다.",
            "warning": "데이터 길이가 부족합니다.",
        }
    differences = data["timestamp"].diff().dropna()
    differences = differences[differences > 0]
    if differences.empty:
        return {"available": False, "source": source, "reason": "양수 sampling interval이 없습니다."}
    dt = float(differences.median())
    fs = 1.0 / dt
    uniform_time = np.arange(float(data["timestamp"].iloc[0]), float(data["timestamp"].iloc[-1]) + dt * 0.5, dt)
    if len(uniform_time) < minimum_samples:
        return {"available": False, "source": source, "reason": "보간 후 샘플 수가 부족합니다."}
    uniform_value = np.interp(uniform_time, data["timestamp"].to_numpy(), data["value"].to_numpy())
    uniform_value = signal.detrend(uniform_value, type="linear")
    nperseg = min(len(uniform_value), 1024)
    frequencies, psd = signal.welch(
        uniform_value,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        detrend=False,
        scaling="density",
    )
    nonzero = frequencies > 0
    frequencies, psd = frequencies[nonzero], psd[nonzero]
    if len(frequencies) == 0 or not np.isfinite(psd).any():
        return {"available": False, "source": source, "reason": "유효한 PSD를 계산하지 못했습니다."}
    peaks, _ = signal.find_peaks(psd)
    if len(peaks) == 0:
        peaks = np.array([int(np.nanargmax(psd))])
    ranked = peaks[np.argsort(psd[peaks])[::-1]][:3]
    dominant = int(ranked[0])
    total_power = (
        float(trapezoid(psd, x=frequencies))
        if len(frequencies) > 1
        else float(psd.sum())
    )
    bin_width = float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else fs / nperseg
    peak_ratio = float(psd[dominant] * bin_width / total_power) if total_power > 0 else None
    return {
        "available": True,
        "source": source,
        "samples": int(len(uniform_value)),
        "sampling_frequency_hz": fs,
        "dominant_frequency_hz": float(frequencies[dominant]),
        "dominant_peak_power": float(psd[dominant]),
        "dominant_peak_power_ratio": peak_ratio,
        "top_peaks": [
            {"frequency_hz": float(frequencies[index]), "power": float(psd[index])}
            for index in ranked
        ],
        "minimum_analyzable_frequency_hz": float(fs / len(uniform_value)),
        "nyquist_frequency_hz": float(fs / 2.0),
        "frequencies_hz": frequencies.astype(float).tolist(),
        "psd": psd.astype(float).tolist(),
        "irregular_sampling_interpolated": bool(float(differences.std(ddof=0) / dt) > 0.01),
    }
