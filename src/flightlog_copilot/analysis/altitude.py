"""Altitude tracking and conservative, event-based step-response analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepEvent:
    index: int
    timestamp_s: float
    initial_setpoint_m: float
    final_setpoint_m: float
    amplitude_m: float
    direction: str
    next_event_index: Optional[int]


def analyze_altitude_tracking(
    frame: pd.DataFrame,
    steady_state_fraction: float = 0.2,
    settling_tolerance_fraction: float = 0.05,
    absolute_settling_tolerance_m: float = 0.01,
    minimum_settling_dwell_s: float = 0.5,
) -> Dict[str, Any]:
    required = ["altitude_setpoint", "ekf_altitude"]
    missing = [name for name in required if name not in frame]
    if missing:
        return {"available": False, "missing_parameters": missing, "reason": "고도 추종 파라미터가 부족합니다."}
    columns = required + (["timestamp"] if "timestamp" in frame else [])
    data = frame[columns].copy()
    for column in columns:
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
        "primary_step_index": None,
        "step_responses": [],
    }
    result.update(_step_metrics(
        data.reset_index(drop=True),
        settling_tolerance_fraction,
        absolute_settling_tolerance_m,
        minimum_settling_dwell_s,
    ))
    return result


def detect_step_events(
    data: pd.DataFrame,
    absolute_minimum_step_m: float = 0.05,
    relative_step_threshold: float = 0.05,
    noise_multiplier: float = 5.0,
    plateau_window_s: float = 0.5,
    transition_merge_gap_s: float = 0.25,
    maximum_transition_duration_s: float = 0.75,
    minimum_plateau_samples: int = 2,
) -> List[StepEvent]:
    """Detect discrete setpoint events with stable plateaus on both sides."""
    if not {"timestamp", "altitude_setpoint"}.issubset(data.columns):
        return []
    working = data[["timestamp", "altitude_setpoint"]].copy().reset_index(drop=True)
    time = pd.to_numeric(working["timestamp"], errors="coerce").to_numpy(dtype=float)
    setpoint = pd.to_numeric(working["altitude_setpoint"], errors="coerce").to_numpy(dtype=float)
    if len(time) < minimum_plateau_samples * 2 + 1:
        return []
    if not np.isfinite(time).all() or not np.isfinite(setpoint).all() or np.any(np.diff(time) <= 0):
        return []

    full_range = float(np.ptp(setpoint))
    if full_range <= 0:
        return []
    differences = np.diff(setpoint)
    difference_median = float(np.median(differences))
    difference_noise = float(1.4826 * np.median(np.abs(differences - difference_median)))
    change_threshold = max(
        float(absolute_minimum_step_m),
        float(relative_step_threshold) * full_range,
        float(noise_multiplier) * difference_noise,
    )
    change_positions = (np.flatnonzero(np.abs(differences) >= change_threshold) + 1).tolist()
    if not change_positions:
        return []

    groups = _group_transition_positions(change_positions, time, transition_merge_gap_s)
    events: List[StepEvent] = []
    for group_index, group in enumerate(groups):
        transition_start = int(group[0])
        transition_end = int(group[-1])
        if float(time[transition_end] - time[transition_start]) > maximum_transition_duration_s:
            continue

        previous_boundary = int(groups[group_index - 1][-1]) if group_index > 0 else 0
        next_boundary = int(groups[group_index + 1][0]) if group_index + 1 < len(groups) else len(setpoint)
        initial_indices = _plateau_indices_before(
            time,
            transition_start,
            previous_boundary,
            plateau_window_s,
            minimum_plateau_samples,
        )
        final_indices = _plateau_indices_after(
            time,
            transition_end,
            next_boundary,
            plateau_window_s,
            minimum_plateau_samples,
        )
        if initial_indices is None or final_indices is None:
            continue

        initial_values = setpoint[initial_indices]
        final_values = setpoint[final_indices]
        initial = float(np.median(initial_values))
        final = float(np.median(final_values))
        amplitude = final - initial
        if abs(amplitude) < change_threshold:
            continue
        plateau_tolerance = max(
            absolute_minimum_step_m * 0.5,
            abs(amplitude) * 0.02,
            difference_noise * noise_multiplier,
        )
        if float(np.ptp(initial_values)) > plateau_tolerance or float(np.ptp(final_values)) > plateau_tolerance:
            continue

        events.append(StepEvent(
            index=transition_start,
            timestamp_s=float(time[transition_start]),
            initial_setpoint_m=initial,
            final_setpoint_m=final,
            amplitude_m=float(amplitude),
            direction="up" if amplitude > 0 else "down",
            next_event_index=next_boundary if next_boundary < len(setpoint) else None,
        ))
    return events


def _step_metrics(
    data: pd.DataFrame,
    tolerance_fraction: float,
    absolute_tolerance_m: float,
    minimum_dwell_s: float,
) -> Dict[str, Any]:
    input_reason = _dynamic_input_reason(data)
    if input_reason is not None:
        return {
            "dynamic_metrics_reason": input_reason,
            "primary_step_index": None,
            "step_responses": [],
        }

    events = detect_step_events(data)
    if not events:
        return {
            "dynamic_metrics_reason": "명확한 plateau를 가진 setpoint step을 찾지 못했습니다.",
            "primary_step_index": None,
            "step_responses": [],
        }

    responses = [
        _analyze_step_event(data, event, tolerance_fraction, absolute_tolerance_m, minimum_dwell_s)
        for event in events
    ]
    primary_index = max(range(len(events)), key=lambda index: abs(events[index].amplitude_m))
    primary = responses[primary_index]
    return {
        "overshoot_m": primary["overshoot_m"],
        "undershoot_m": primary["undershoot_m"],
        "rise_time_s": primary["rise_time_s"],
        "settling_time_s": primary["settling_time_s"],
        "dynamic_metrics_reason": primary["reason"],
        "primary_step_index": int(primary_index),
        "step_responses": responses,
    }


def _analyze_step_event(
    data: pd.DataFrame,
    event: StepEvent,
    tolerance_fraction: float,
    absolute_tolerance_m: float,
    minimum_dwell_s: float,
) -> Dict[str, Any]:
    response_end = event.next_event_index if event.next_event_index is not None else len(data)
    response = data.iloc[event.index:response_end]
    time = response["timestamp"].to_numpy(dtype=float)
    altitude = response["ekf_altitude"].to_numpy(dtype=float)
    direction = 1.0 if event.amplitude_m > 0 else -1.0
    amplitude = abs(event.amplitude_m)
    progress = direction * (altitude - event.initial_setpoint_m)

    time_10 = _interpolated_crossing_time(time, progress, 0.1 * amplitude)
    time_90 = _interpolated_crossing_time(time, progress, 0.9 * amplitude)
    reached_10 = time_10 is not None
    reached_90 = time_90 is not None
    rise_time = (
        float(time_90 - time_10)
        if time_10 is not None and time_90 is not None and time_90 >= time_10
        else None
    )

    directed_final_error = direction * (altitude - event.final_setpoint_m)
    overshoot = max(0.0, float(np.max(directed_final_error)))
    undershoot_limit = float(time_10) if time_10 is not None else float(time[-1])
    undershoot_values = -direction * (altitude[time <= undershoot_limit] - event.initial_setpoint_m)
    undershoot = max(0.0, float(np.max(undershoot_values))) if len(undershoot_values) else 0.0

    tolerance = max(amplitude * tolerance_fraction, absolute_tolerance_m)
    settling_entry = _settling_entry_time(
        time,
        np.abs(altitude - event.final_setpoint_m) <= tolerance,
        minimum_dwell_s,
    )
    settled = settling_entry is not None
    settling_time = float(settling_entry - event.timestamp_s) if settling_entry is not None else None

    if not reached_10:
        reason = "응답이 step amplitude의 10% threshold에 도달하지 못했습니다."
    elif not reached_90:
        reason = "응답이 step amplitude의 90% threshold에 도달하지 못했습니다."
    elif not settled:
        reason = "다음 step 또는 분석 구간 종료 전 settling dwell 조건을 충족하지 못했습니다."
    else:
        reason = None

    return {
        "timestamp_s": float(event.timestamp_s),
        "direction": event.direction,
        "initial_setpoint_m": float(event.initial_setpoint_m),
        "final_setpoint_m": float(event.final_setpoint_m),
        "amplitude_m": float(event.amplitude_m),
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "overshoot_m": float(overshoot),
        "undershoot_m": float(undershoot),
        "reached_10_percent": bool(reached_10),
        "reached_90_percent": bool(reached_90),
        "settled": bool(settled),
        "reason": reason,
    }


def _dynamic_input_reason(data: pd.DataFrame) -> Optional[str]:
    if "timestamp" not in data:
        return "timestamp가 없어 동적 지표를 계산하지 않았습니다."
    time = pd.to_numeric(data["timestamp"], errors="coerce").to_numpy(dtype=float)
    if len(time) < 5:
        return "step-response 분석 구간이 너무 짧습니다."
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        return "timestamp가 유효하지 않거나 엄격하게 증가하지 않아 동적 지표를 계산하지 않았습니다."
    return None


def _group_transition_positions(
    positions: Sequence[int],
    time: np.ndarray,
    maximum_gap_s: float,
) -> List[List[int]]:
    groups: List[List[int]] = [[int(positions[0])]]
    for position in positions[1:]:
        if float(time[position] - time[groups[-1][-1]]) <= maximum_gap_s:
            groups[-1].append(int(position))
        else:
            groups.append([int(position)])
    return groups


def _plateau_indices_before(
    time: np.ndarray,
    transition_start: int,
    lower_boundary: int,
    window_s: float,
    minimum_samples: int,
) -> Optional[np.ndarray]:
    # Prefer a time-based plateau; sparse logs fall back to the nearest samples.
    available = np.arange(max(0, lower_boundary), transition_start, dtype=int)
    if len(available) < minimum_samples:
        return None
    timed = available[time[available] >= time[transition_start] - window_s]
    return timed if len(timed) >= minimum_samples else available[-minimum_samples:]


def _plateau_indices_after(
    time: np.ndarray,
    transition_end: int,
    upper_boundary: int,
    window_s: float,
    minimum_samples: int,
) -> Optional[np.ndarray]:
    # Prefer a time-based plateau; sparse logs fall back to the nearest samples.
    available = np.arange(transition_end, min(len(time), upper_boundary), dtype=int)
    if len(available) < minimum_samples:
        return None
    timed = available[time[available] <= time[transition_end] + window_s]
    return timed if len(timed) >= minimum_samples else available[:minimum_samples]


def _interpolated_crossing_time(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> Optional[float]:
    reached = np.flatnonzero(values >= threshold)
    if len(reached) == 0:
        return None
    index = int(reached[0])
    if index == 0:
        return float(time[0])
    previous_value = float(values[index - 1])
    current_value = float(values[index])
    if current_value <= previous_value:
        return float(time[index])
    fraction = (threshold - previous_value) / (current_value - previous_value)
    fraction = min(1.0, max(0.0, float(fraction)))
    return float(time[index - 1] + fraction * (time[index] - time[index - 1]))


def _settling_entry_time(
    time: np.ndarray,
    inside_tolerance: np.ndarray,
    minimum_dwell_s: float,
) -> Optional[float]:
    run_start: Optional[int] = None
    for index, inside in enumerate(inside_tolerance):
        if inside and run_start is None:
            run_start = index
        if not inside and run_start is not None:
            if float(time[index - 1] - time[run_start]) >= minimum_dwell_s:
                return float(time[run_start])
            run_start = None
    if run_start is not None and float(time[-1] - time[run_start]) >= minimum_dwell_s:
        return float(time[run_start])
    return None
