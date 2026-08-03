import pandas as pd
import pytest

from flightlog_copilot.analysis.sensor_comparison import analyze_sensor_comparison
from flightlog_copilot.preprocessing.validation import validate_data


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
