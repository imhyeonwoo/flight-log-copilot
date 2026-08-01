import numpy as np
import pandas as pd

from flightlog_copilot.analysis.frequency import analyze_frequency


def test_welch_detects_synthetic_frequency():
    sample_rate = 25.0
    time = np.arange(0, 60, 1 / sample_rate)
    vibration_hz = 0.3
    frame = pd.DataFrame({
        "timestamp": time,
        "ekf_altitude": np.sin(2 * np.pi * vibration_hz * time),
    })
    result = analyze_frequency(frame)
    assert result["available"]
    assert abs(result["dominant_frequency_hz"] - vibration_hz) < 0.04


def test_frequency_handles_short_data():
    result = analyze_frequency(pd.DataFrame({"timestamp": [0, 1], "ekf_altitude": [0, 1]}))
    assert not result["available"]
