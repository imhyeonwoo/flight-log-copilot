"""Reference-altitude/EKF comparison with conservative lag qualification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from flightlog_copilot.config import SensorComparisonSettings


LAG_SIGN_CONVENTION = "positive means reference_altitude lags ekf_altitude"


@dataclass(frozen=True)
class LagAnalysisResult:
    estimated_lag_s: Optional[float] = None
    max_cross_correlation: Optional[float] = None
    lag_reliable: bool = False
    lag_confidence: str = "unavailable"
    lag_reason: Optional[str] = None
    lag_sign_convention: str = LAG_SIGN_CONVENTION
    lag_search_limit_s: Optional[float] = None
    lag_at_search_boundary: bool = False
    second_best_correlation: Optional[float] = None
    correlation_peak_separation: Optional[float] = None
    effective_overlap_samples: int = 0
    original_sample_count: int = 0
    resampled_sample_count: int = 0
    median_sampling_interval_s: Optional[float] = None
    resampled_sampling_frequency_hz: Optional[float] = None
    irregular_sampling_interpolated: bool = False
    large_gap_exclusion_applied: bool = False
    duplicate_timestamp_count: int = 0
    timestamp_reordered: bool = False
    possible_sign_mismatch: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_sensor_comparison(
    frame: pd.DataFrame,
    reference_source: str = "other",
    lag_settings: SensorComparisonSettings = SensorComparisonSettings(),
) -> Dict[str, Any]:
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
    data = data.replace([np.inf, -np.inf], np.nan)
    paired = data[required].dropna()
    if len(paired) < 3:
        return {
            "available": False,
            "reference_altitude_source": reference_source,
            "missing_parameters": [],
            "reason": "동시에 유효한 기준 고도/EKF 샘플이 3개 미만입니다.",
        }

    difference = paired["reference_altitude"] - paired["ekf_altitude"]
    lag_result = analyze_cross_correlation(data, lag_settings)
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
        **lag_result.to_dict(),
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


def analyze_cross_correlation(
    data: pd.DataFrame,
    settings: SensorComparisonSettings = SensorComparisonSettings(),
) -> LagAnalysisResult:
    """Estimate reference-vs-EKF lag after safe timebase preparation.

    Duplicate timestamps are averaged after a stable timestamp sort. Gaps larger
    than the configured median-interval multiple split the record, and only the
    longest contiguous common-valid segment is resampled, so interpolation never
    bridges a detected large dropout.
    """
    if "timestamp" not in data:
        return LagAnalysisResult(lag_reason="timestamp가 없어 시간 지연을 계산할 수 없습니다.")

    columns = ["timestamp", "reference_altitude", "ekf_altitude"]
    working = data[columns].apply(pd.to_numeric, errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).dropna()
    original_count = int(len(working))
    if original_count < max(4, settings.minimum_overlap_samples):
        return LagAnalysisResult(
            lag_reason="공통 유효 샘플이 부족하여 시간 지연을 계산할 수 없습니다.",
            original_sample_count=original_count,
        )

    raw_time = working["timestamp"].to_numpy(dtype=float)
    reordered = bool(np.any(np.diff(raw_time) < 0))
    working = working.sort_values("timestamp", kind="mergesort")
    duplicate_count = int(len(working) - working["timestamp"].nunique())
    # Averaging is deterministic and preserves both sensors at their shared time.
    working = working.groupby("timestamp", as_index=False, sort=True)[
        ["reference_altitude", "ekf_altitude"]
    ].mean()
    if len(working) < max(4, settings.minimum_overlap_samples):
        return LagAnalysisResult(
            lag_reason="공통 유효 샘플이 부족하여 시간 지연을 계산할 수 없습니다.",
            original_sample_count=original_count,
            duplicate_timestamp_count=duplicate_count,
            timestamp_reordered=reordered,
        )

    time = working["timestamp"].to_numpy(dtype=float)
    intervals = np.diff(time)
    valid_intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(valid_intervals) == 0:
        return LagAnalysisResult(
            lag_reason="유효한 timestamp 간격을 계산할 수 없습니다.",
            original_sample_count=original_count,
            duplicate_timestamp_count=duplicate_count,
            timestamp_reordered=reordered,
        )
    median_interval = float(np.median(valid_intervals))
    if not np.isfinite(median_interval) or median_interval <= 0:
        return LagAnalysisResult(
            lag_reason="유효한 timestamp 간격을 계산할 수 없습니다.",
            original_sample_count=original_count,
            duplicate_timestamp_count=duplicate_count,
            timestamp_reordered=reordered,
        )

    gap_limit = settings.maximum_interpolation_gap_multiplier * median_interval
    split_positions = (np.flatnonzero(intervals > gap_limit) + 1).tolist()
    segment_indices = np.split(np.arange(len(working), dtype=int), split_positions)
    eligible_segments = [indices for indices in segment_indices if len(indices) >= 2]
    if not eligible_segments:
        return LagAnalysisResult(
            lag_reason="공통 유효 시간 구간이 너무 짧습니다.",
            original_sample_count=original_count,
            median_sampling_interval_s=median_interval,
            duplicate_timestamp_count=duplicate_count,
            timestamp_reordered=reordered,
        )
    # Prefer duration, then sample count; max() keeps the earliest exact tie.
    selected_indices = max(
        eligible_segments,
        key=lambda indices: (float(time[indices[-1]] - time[indices[0]]), len(indices)),
    )
    selected = working.iloc[selected_indices]
    selected_time = selected["timestamp"].to_numpy(dtype=float)
    duration = float(selected_time[-1] - selected_time[0])
    large_gap_excluded = bool(split_positions)
    if duration <= 0:
        return LagAnalysisResult(
            lag_reason="공통 유효 시간 구간이 너무 짧습니다.",
            original_sample_count=original_count,
            median_sampling_interval_s=median_interval,
            large_gap_exclusion_applied=large_gap_excluded,
            duplicate_timestamp_count=duplicate_count,
            timestamp_reordered=reordered,
        )

    grid_count = int(np.floor(duration / median_interval + 1e-9)) + 1
    uniform_time = selected_time[0] + np.arange(grid_count, dtype=float) * median_interval
    if len(uniform_time) < max(4, settings.minimum_overlap_samples):
        return LagAnalysisResult(
            lag_reason="공통 유효 시간 구간이 너무 짧습니다.",
            original_sample_count=original_count,
            resampled_sample_count=int(len(uniform_time)),
            median_sampling_interval_s=median_interval,
            resampled_sampling_frequency_hz=float(1.0 / median_interval),
            large_gap_exclusion_applied=large_gap_excluded,
            duplicate_timestamp_count=duplicate_count,
            timestamp_reordered=reordered,
        )

    reference = np.interp(
        uniform_time,
        selected_time,
        selected["reference_altitude"].to_numpy(dtype=float),
    )
    ekf = np.interp(
        uniform_time,
        selected_time,
        selected["ekf_altitude"].to_numpy(dtype=float),
    )
    same_grid = len(selected_time) == len(uniform_time) and np.allclose(
        selected_time,
        uniform_time,
        rtol=1e-7,
        atol=max(1e-12, median_interval * 1e-7),
    )
    interpolated = not same_grid
    common_fields = {
        "original_sample_count": original_count,
        "resampled_sample_count": int(len(uniform_time)),
        "median_sampling_interval_s": median_interval,
        "resampled_sampling_frequency_hz": float(1.0 / median_interval),
        "irregular_sampling_interpolated": interpolated,
        "large_gap_exclusion_applied": large_gap_excluded,
        "duplicate_timestamp_count": duplicate_count,
        "timestamp_reordered": reordered,
    }

    reference_detrended = signal.detrend(reference, type="linear")
    ekf_detrended = signal.detrend(ekf, type="linear")
    reference_std = float(np.std(reference_detrended, ddof=0))
    ekf_std = float(np.std(ekf_detrended, ddof=0))
    if (
        not np.isfinite(reference_std)
        or not np.isfinite(ekf_std)
        or reference_std < settings.minimum_signal_std
        or ekf_std < settings.minimum_signal_std
    ):
        return LagAnalysisResult(
            lag_reason="detrending 후 신호 변화가 너무 작아 시간 지연을 계산할 수 없습니다.",
            **common_fields,
        )
    reference_normalized = (reference_detrended - reference_detrended.mean()) / reference_std
    ekf_normalized = (ekf_detrended - ekf_detrended.mean()) / ekf_std

    effective_limit = min(max(0.0, settings.max_lag_s), duration * 0.25)
    maximum_lag_samples = int(np.floor(effective_limit / median_interval + 1e-9))
    if maximum_lag_samples < 1:
        return LagAnalysisResult(
            lag_reason="공통 유효 시간 구간이 너무 짧습니다.",
            lag_search_limit_s=float(maximum_lag_samples * median_interval),
            **common_fields,
        )
    actual_search_limit = float(min(effective_limit, maximum_lag_samples * median_interval))
    minimum_overlap = max(
        int(settings.minimum_overlap_samples),
        int(np.ceil(settings.minimum_overlap_ratio * len(uniform_time))),
    )

    candidate_lags: List[int] = []
    candidate_correlations: List[float] = []
    candidate_overlaps: List[int] = []
    for lag_samples in range(-maximum_lag_samples, maximum_lag_samples + 1):
        reference_overlap, ekf_overlap = _overlap_for_lag(
            reference_normalized,
            ekf_normalized,
            lag_samples,
        )
        overlap = int(len(reference_overlap))
        if overlap < minimum_overlap:
            continue
        correlation = _array_correlation(reference_overlap, ekf_overlap, settings.minimum_signal_std)
        if correlation is None:
            continue
        candidate_lags.append(lag_samples)
        candidate_correlations.append(correlation)
        candidate_overlaps.append(overlap)

    if not candidate_correlations:
        return LagAnalysisResult(
            lag_reason="유효 overlap을 가진 lag 후보가 없습니다.",
            lag_search_limit_s=actual_search_limit,
            **common_fields,
        )

    correlations = np.asarray(candidate_correlations, dtype=float)
    lags = np.asarray(candidate_lags, dtype=int)
    maximum_correlation = float(np.max(correlations))
    tied = np.flatnonzero(np.isclose(correlations, maximum_correlation, rtol=0.0, atol=1e-12))
    best_position = int(min(tied, key=lambda index: abs(int(lags[index]))))
    best_lag_samples = int(lags[best_position])
    best_overlap = int(candidate_overlaps[best_position])
    estimated_lag = float(best_lag_samples * median_interval)
    estimated_lag = float(np.clip(estimated_lag, -actual_search_limit, actual_search_limit))

    second_best = _second_best_peak(
        correlations,
        best_position,
        max(0, int(settings.peak_guard_band_samples)),
    )
    peak_separation = (
        float(maximum_correlation - second_best)
        if second_best is not None
        else None
    )
    boundary_tolerance = max(0, int(settings.boundary_tolerance_samples))
    at_boundary = bool(
        best_position <= boundary_tolerance
        or best_position >= len(correlations) - 1 - boundary_tolerance
    )
    strongest_negative = float(np.min(correlations))
    possible_sign_mismatch = bool(
        strongest_negative <= -settings.minimum_correlation
        and abs(strongest_negative) > maximum_correlation + settings.minimum_peak_separation
    )

    reason: Optional[str] = None
    if at_boundary:
        reason = "최대 correlation peak가 검색 범위 경계에 있습니다."
    elif possible_sign_mismatch:
        reason = "강한 음의 correlation이 감지되어 센서 축 또는 부호 매핑 확인이 필요합니다."
    elif maximum_correlation < settings.minimum_correlation:
        reason = "최대 cross-correlation이 신뢰 기준보다 낮습니다."
    elif peak_separation is not None and peak_separation < settings.minimum_peak_separation:
        reason = "비슷한 correlation peak가 반복되어 지연 후보가 모호합니다."
    elif best_overlap < minimum_overlap:
        reason = "최적 lag의 유효 overlap 샘플이 부족합니다."

    reliable = reason is None
    if reliable and (
        maximum_correlation >= settings.high_confidence_correlation
        and (
            peak_separation is None
            or peak_separation >= settings.high_confidence_peak_separation
        )
    ):
        confidence = "high"
    elif reliable:
        confidence = "medium"
    else:
        confidence = "low"

    return LagAnalysisResult(
        estimated_lag_s=estimated_lag,
        max_cross_correlation=maximum_correlation,
        lag_reliable=reliable,
        lag_confidence=confidence,
        lag_reason=reason,
        lag_search_limit_s=actual_search_limit,
        lag_at_search_boundary=at_boundary,
        second_best_correlation=second_best,
        correlation_peak_separation=peak_separation,
        effective_overlap_samples=best_overlap,
        possible_sign_mismatch=possible_sign_mismatch,
        **common_fields,
    )


def _overlap_for_lag(
    reference: np.ndarray,
    ekf: np.ndarray,
    lag_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # corr(reference(t), ekf(t - lag)): positive means reference is later.
    if lag_samples > 0:
        return reference[lag_samples:], ekf[:-lag_samples]
    if lag_samples < 0:
        offset = abs(lag_samples)
        return reference[:-offset], ekf[offset:]
    return reference, ekf


def _array_correlation(left: np.ndarray, right: np.ndarray, minimum_std: float) -> Optional[float]:
    if len(left) < 3 or len(right) < 3:
        return None
    left_std = float(np.std(left, ddof=0))
    right_std = float(np.std(right, ddof=0))
    if left_std < minimum_std or right_std < minimum_std:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _second_best_peak(
    correlations: np.ndarray,
    best_position: int,
    guard_band_samples: int,
) -> Optional[float]:
    peaks = set(int(index) for index in signal.find_peaks(correlations)[0])
    if len(correlations) == 1:
        peaks.add(0)
    else:
        if correlations[0] >= correlations[1]:
            peaks.add(0)
        if correlations[-1] >= correlations[-2]:
            peaks.add(len(correlations) - 1)
    peaks.add(best_position)
    alternatives = [
        position
        for position in peaks
        if abs(position - best_position) > guard_band_samples
    ]
    if not alternatives:
        return None
    return float(max(correlations[position] for position in alternatives))


def _source_caveat(reference_source: str) -> str:
    if reference_source == "barometer":
        return "Barometer 고도는 기압·프로펠러 바람·설치 위치의 영향을 받을 수 있습니다."
    if reference_source == "gnss":
        return "GNSS 고도는 타원체/MSL 기준과 위성 가시성의 영향을 받을 수 있습니다."
    return "기준 고도의 센서 종류와 수직 datum을 확인해야 합니다."


def _safe_correlation(left: pd.Series, right: pd.Series) -> Optional[float]:
    paired = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].std() == 0 or paired.iloc[:, 1].std() == 0:
        return None
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return float(value) if pd.notna(value) else None
