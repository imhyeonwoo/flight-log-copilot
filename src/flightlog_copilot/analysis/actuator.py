"""Throttle correction and motor output saturation metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def analyze_actuators(
    frame: pd.DataFrame,
    correction_limit: float = 160.0,
    motor_pwm_min: float = 1000.0,
    motor_pwm_max: float = 2000.0,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "missing_parameters": []}
    if "throttle_base" in frame:
        throttle = pd.to_numeric(frame["throttle_base"], errors="coerce").dropna()
        if not throttle.empty:
            result["mean_throttle_base_pwm"] = float(throttle.mean())
            result["available"] = True
    else:
        result["missing_parameters"].append("throttle_base")

    if "throttle_correction" in frame:
        correction = pd.to_numeric(frame["throttle_correction"], errors="coerce")
        valid = correction.dropna()
        if not valid.empty:
            positive = valid >= correction_limit
            negative = valid <= -correction_limit
            saturated = valid.abs() >= correction_limit
            result.update(
                {
                    "mean_throttle_correction_pwm": float(valid.mean()),
                    "throttle_correction_std_pwm": float(valid.std(ddof=0)),
                    "correction_limit_pwm": float(correction_limit),
                    "correction_saturation_ratio": float(saturated.mean()),
                    "positive_saturation_ratio": float(positive.mean()),
                    "negative_saturation_ratio": float(negative.mean()),
                    "max_continuous_saturation_s": _max_continuous_duration(frame, correction.abs() >= correction_limit),
                }
            )
            result["available"] = True
    else:
        result["missing_parameters"].append("throttle_correction")

    motors = [name for name in ("motor_1", "motor_2", "motor_3", "motor_4") if name in frame]
    result["motor_metrics"] = {}
    for motor in motors:
        values = pd.to_numeric(frame[motor], errors="coerce").dropna()
        if values.empty:
            continue
        result["motor_metrics"][motor] = {
            "mean_pwm": float(values.mean()),
            "std_pwm": float(values.std(ddof=0)),
            "saturation_ratio": float(((values <= motor_pwm_min) | (values >= motor_pwm_max)).mean()),
        }
        result["available"] = True
    if len(motors) >= 2:
        motor_values = frame[motors].apply(pd.to_numeric, errors="coerce")
        result["mean_motor_spread_pwm"] = float((motor_values.max(axis=1) - motor_values.min(axis=1)).mean())
    missing_motors = sorted(set(("motor_1", "motor_2", "motor_3", "motor_4")) - set(motors))
    result["missing_parameters"].extend(missing_motors)
    if not result["available"]:
        result["reason"] = "출력 관련 파라미터가 없습니다."
    return result


def _max_continuous_duration(frame: pd.DataFrame, mask: pd.Series) -> float:
    active = mask.fillna(False).to_numpy(dtype=bool)
    if not active.any():
        return 0.0
    if "timestamp" not in frame:
        longest = max((len(run) for run in _true_runs(active)), default=0)
        return float(longest)
    time = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(dtype=float)
    durations = []
    for run in _true_runs(active):
        start, end = run[0], run[-1]
        duration = time[end] - time[start] if np.isfinite(time[start]) and np.isfinite(time[end]) else 0.0
        durations.append(float(max(duration, 0.0)))
    return max(durations, default=0.0)


def _true_runs(mask: Iterable[bool]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index, active in enumerate(mask):
        if active:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs
