import numpy as np
import pandas as pd
import pytest

from flightlog_copilot.analysis.sensor_comparison import analyze_sensor_comparison
from flightlog_copilot.config import SensorComparisonSettings
from flightlog_copilot.preprocessing.validation import validate_data


def _rich_signal(time):
    return (
        np.sin(2 * np.pi * 0.37 * time)
        + 0.55 * np.sin(2 * np.pi * 0.83 * time + 0.4)
        + 0.25 * np.sin(2 * np.pi * 1.61 * time + 1.1)
        + 0.5 * np.exp(-((time - 6.2) / 0.5) ** 2)
        - 0.35 * np.exp(-((time - 13.1) / 0.8) ** 2)
    )


def _lag_frame(time, delay_s=0.0):
    return pd.DataFrame({
        "timestamp": time,
        "reference_altitude": _rich_signal(time - delay_s),
        "ekf_altitude": _rich_signal(time),
    })


def test_sensor_comparison_settings_defaults_are_explicit():
    settings = SensorComparisonSettings()
    assert settings.max_lag_s == 2.0
    assert settings.minimum_correlation == 0.5
    assert settings.minimum_peak_separation == 0.05
    assert settings.minimum_overlap_samples == 20
    assert settings.minimum_overlap_ratio == 0.3
    assert settings.maximum_interpolation_gap_multiplier == 3.0
    assert settings.minimum_signal_std == 1e-6
    assert settings.boundary_tolerance_samples == 1


def test_reference_altitude_comparison_supports_gnss():
    frame = pd.DataFrame({
        "timestamp": [0.0, 1.0, 2.0, 3.0],
        "reference_altitude": [10.2, 11.2, 12.2, 13.2],
        "ekf_altitude": [10.0, 11.0, 12.0, 13.0],
        "throttle_correction": [0.0, 1.0, 2.0, 3.0],
    })
    result = analyze_sensor_comparison(frame, "gnss")
    assert result["available"] is True
    assert result["reference_altitude_source"] == "gnss"
    assert result["reference_ekf_bias_m"] == pytest.approx(0.2)
    assert result["reference_mean_m"] == pytest.approx(11.7)
    assert "GNSS" in result["source_caveat"]
    assert "throttle_correction_vs_reference_change_correlation" in result
    assert not any("barometer" in key for key in result)


def test_reference_altitude_comparison_reports_missing_reference():
    result = analyze_sensor_comparison(pd.DataFrame({"ekf_altitude": [1.0, 2.0]}), "barometer")
    assert result["available"] is False
    assert result["missing_parameters"] == ["reference_altitude"]
    assert result["reference_altitude_source"] == "barometer"


def test_validation_uses_generic_reference_altitude_warning():
    raw = pd.DataFrame({"time": range(20), "ekf": range(20)})
    canonical = pd.DataFrame({"timestamp": range(20), "ekf_altitude": range(20)})
    messages = validate_data(raw, canonical)
    missing = next(message for message in messages if message.code == "missing_reference_altitude")
    assert missing.parameter == "reference_altitude"
    assert "Barometer" not in missing.message


def test_zero_lag_is_reliable_for_identical_rich_signals():
    time = np.arange(0.0, 20.0, 0.04)
    result = analyze_sensor_comparison(_lag_frame(time))
    assert result["estimated_lag_s"] == pytest.approx(0.0, abs=0.04)
    assert result["max_cross_correlation"] > 0.99
    assert result["lag_reliable"] is True
    assert result["lag_confidence"] == "high"
    assert result["lag_sign_convention"] == "positive means reference_altitude lags ekf_altitude"


@pytest.mark.parametrize("delay_s", [0.32, -0.20])
def test_known_positive_and_negative_reference_delay(delay_s):
    time = np.arange(0.0, 20.0, 0.04)
    result = analyze_sensor_comparison(_lag_frame(time, delay_s))
    assert result["estimated_lag_s"] == pytest.approx(delay_s, abs=0.04)
    assert result["lag_reliable"] is True


def test_irregular_sampling_is_resampled_before_known_lag_analysis():
    regular_time = np.arange(0.0, 20.0, 0.04)
    rng = np.random.default_rng(9)
    irregular_time = regular_time + rng.normal(0.0, 0.004, len(regular_time))
    irregular_time[0] = 0.0
    irregular_time = np.maximum.accumulate(irregular_time)
    result = analyze_sensor_comparison(_lag_frame(irregular_time, 0.32))
    assert result["estimated_lag_s"] == pytest.approx(0.32, abs=0.05)
    assert result["irregular_sampling_interpolated"] is True
    assert result["resampled_sampling_frequency_hz"] == pytest.approx(25.0, rel=0.05)


def test_detrending_removes_offset_and_linear_trend_before_lag_analysis():
    time = np.arange(0.0, 20.0, 0.04)
    frame = _lag_frame(time, 0.32)
    frame["reference_altitude"] += 12.0 + 0.07 * time
    frame["ekf_altitude"] += -4.0 - 0.03 * time
    result = analyze_sensor_comparison(frame)
    assert result["estimated_lag_s"] == pytest.approx(0.32, abs=0.04)
    assert result["lag_reliable"] is True


def test_independent_noise_does_not_produce_reliable_lag():
    time = np.arange(0.0, 20.0, 0.04)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "timestamp": time,
        "reference_altitude": rng.normal(size=len(time)),
        "ekf_altitude": rng.normal(size=len(time)),
    })
    result = analyze_sensor_comparison(frame)
    assert result["lag_reliable"] is False
    assert result["lag_confidence"] == "low"
    assert result["max_cross_correlation"] < 0.5


def test_constant_signal_reports_unavailable_lag():
    time = np.arange(0.0, 20.0, 0.04)
    frame = _lag_frame(time)
    frame["reference_altitude"] = 1.0
    result = analyze_sensor_comparison(frame)
    assert result["estimated_lag_s"] is None
    assert result["lag_reliable"] is False
    assert result["lag_confidence"] == "unavailable"
    assert "신호 변화" in result["lag_reason"]


def test_periodic_signal_with_repeated_peaks_is_ambiguous():
    time = np.arange(0.0, 20.0, 0.04)
    frame = pd.DataFrame({
        "timestamp": time,
        "reference_altitude": np.sin(2 * np.pi * (time - 0.2)),
        "ekf_altitude": np.sin(2 * np.pi * time),
    })
    result = analyze_sensor_comparison(frame)
    assert result["lag_reliable"] is False
    assert result["correlation_peak_separation"] < 0.05
    assert "모호" in result["lag_reason"]


def test_delay_outside_configured_range_is_a_boundary_peak():
    time = np.arange(0.0, 20.0, 0.04)
    settings = SensorComparisonSettings(max_lag_s=0.4)
    result = analyze_sensor_comparison(_lag_frame(time, 0.8), lag_settings=settings)
    assert result["lag_at_search_boundary"] is True
    assert result["lag_reliable"] is False
    assert result["lag_search_limit_s"] <= 0.4
    assert "경계" in result["lag_reason"]


def test_large_dropout_selects_one_segment_without_bridging_gap():
    time = np.arange(0.0, 20.0, 0.04)
    keep = ~((time > 7.0) & (time < 11.0))
    result = analyze_sensor_comparison(_lag_frame(time[keep], 0.32))
    assert result["large_gap_exclusion_applied"] is True
    assert result["resampled_sample_count"] < result["original_sample_count"]
    assert result["estimated_lag_s"] == pytest.approx(0.32, abs=0.04)


def test_sign_inversion_is_not_treated_as_valid_positive_correlation_lag():
    time = np.arange(0.0, 20.0, 0.04)
    frame = _lag_frame(time)
    frame["reference_altitude"] *= -1.0
    result = analyze_sensor_comparison(frame)
    assert result["possible_sign_mismatch"] is True
    assert result["lag_reliable"] is False
    assert "부호" in result["lag_reason"]


def test_missing_timestamp_keeps_static_sensor_statistics():
    time = np.arange(0.0, 20.0, 0.04)
    frame = _lag_frame(time).drop(columns="timestamp")
    result = analyze_sensor_comparison(frame)
    assert result["available"] is True
    assert result["pearson_correlation"] > 0.99
    assert result["estimated_lag_s"] is None
    assert result["lag_reliable"] is False
    assert "timestamp" in result["lag_reason"]


def test_duplicate_and_reverse_timestamps_are_aggregated_and_sorted():
    time = np.arange(0.0, 20.0, 0.04)
    frame = _lag_frame(time, 0.32)
    frame = pd.concat([frame, frame.iloc[[100]]], ignore_index=True)
    frame = frame.iloc[::-1].reset_index(drop=True)
    result = analyze_sensor_comparison(frame)
    assert result["duplicate_timestamp_count"] == 1
    assert result["timestamp_reordered"] is True
    assert result["estimated_lag_s"] == pytest.approx(0.32, abs=0.04)
