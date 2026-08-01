import numpy as np
import pandas as pd

from flightlog_copilot.analysis.altitude import analyze_altitude_tracking


def test_altitude_rmse():
    frame = pd.DataFrame({"timestamp": [0, 1, 2], "altitude_setpoint": [1, 1, 1], "ekf_altitude": [0, 1, 2]})
    result = analyze_altitude_tracking(frame)
    assert np.isclose(result["rmse_m"], np.sqrt(2 / 3))
    assert result["rise_time_s"] is None


def test_missing_altitude_parameter_is_non_fatal():
    result = analyze_altitude_tracking(pd.DataFrame({"ekf_altitude": [1, 2]}))
    assert not result["available"]
    assert "altitude_setpoint" in result["missing_parameters"]
