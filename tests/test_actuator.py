import pandas as pd

from flightlog_copilot.analysis.actuator import analyze_actuators


def test_correction_saturation_ratio_and_motor_spread():
    frame = pd.DataFrame({
        "timestamp": [0.0, 0.1, 0.2, 0.3],
        "throttle_correction": [-160, 0, 160, 200],
        "motor_1": [1500, 1510, 1520, 1530],
        "motor_2": [1550, 1560, 1570, 1580],
    })
    result = analyze_actuators(frame, correction_limit=160)
    assert result["correction_saturation_ratio"] == 0.75
    assert result["mean_motor_spread_pwm"] == 50.0
