import numpy as np
import pandas as pd

from flightlog_copilot.analysis.altitude import analyze_altitude_tracking, detect_step_events


def _step_frame(timestamp, setpoint, response):
    return pd.DataFrame({
        "timestamp": np.asarray(timestamp, dtype=float),
        "altitude_setpoint": np.asarray(setpoint, dtype=float),
        "ekf_altitude": np.asarray(response, dtype=float),
    })


def _single_step(response_builder, direction="up"):
    time = np.arange(0.0, 3.1, 0.1)
    if direction == "up":
        setpoint = np.where(time < 0.5, 0.0, 1.0)
        response = response_builder(time)
    else:
        setpoint = np.where(time < 0.5, 1.0, 0.0)
        response = response_builder(time)
    return _step_frame(time, setpoint, response)


def test_altitude_rmse():
    frame = pd.DataFrame({"timestamp": [0, 1, 2], "altitude_setpoint": [1, 1, 1], "ekf_altitude": [0, 1, 2]})
    result = analyze_altitude_tracking(frame)
    assert np.isclose(result["rmse_m"], np.sqrt(2 / 3))
    assert result["rise_time_s"] is None


def test_missing_altitude_parameter_is_non_fatal():
    result = analyze_altitude_tracking(pd.DataFrame({"ekf_altitude": [1, 2]}))
    assert not result["available"]
    assert "altitude_setpoint" in result["missing_parameters"]


def test_rising_step_has_interpolated_rise_time_and_settles():
    frame = _single_step(lambda time: np.where(time < 0.5, 0.0, np.minimum(time - 0.5, 1.0)))
    result = analyze_altitude_tracking(frame)
    step = result["step_responses"][0]
    assert 0.79 <= step["rise_time_s"] <= 0.81
    assert step["overshoot_m"] == 0.0
    assert step["undershoot_m"] == 0.0
    assert step["reached_10_percent"] is True
    assert step["reached_90_percent"] is True
    assert step["settled"] is True
    assert step["settling_time_s"] is not None
    assert result["rise_time_s"] == step["rise_time_s"]


def test_rising_step_overshoot_is_measured_beyond_final_setpoint():
    def response(time):
        values = np.zeros_like(time)
        active = time >= 0.5
        values[active] = np.interp(time[active], [0.5, 1.3, 1.5, 1.7], [0.0, 0.9, 1.1, 1.0])
        return values

    step = analyze_altitude_tracking(_single_step(response))["step_responses"][0]
    assert np.isclose(step["overshoot_m"], 0.1)
    assert step["undershoot_m"] == 0.0


def test_rising_step_undershoot_counts_only_wrong_way_motion_from_initial_target():
    def response(time):
        values = np.zeros_like(time)
        active = time >= 0.5
        values[active] = np.interp(
            time[active],
            [0.5, 0.6, 0.7, 1.7],
            [0.0, -0.1, 0.0, 1.0],
        )
        return values

    step = analyze_altitude_tracking(_single_step(response))["step_responses"][0]
    assert np.isclose(step["undershoot_m"], 0.1)
    assert step["undershoot_m"] != 1.0


def test_falling_step_uses_direction_normalization():
    def response(time):
        values = np.ones_like(time)
        active = time >= 0.5
        values[active] = np.interp(
            time[active],
            [0.5, 0.6, 0.7, 1.5, 1.7],
            [1.0, 1.1, 0.9, -0.1, 0.0],
        )
        return values

    step = analyze_altitude_tracking(_single_step(response, direction="down"))["step_responses"][0]
    assert step["direction"] == "down"
    assert step["amplitude_m"] == -1.0
    assert step["rise_time_s"] > 0.0
    assert np.isclose(step["overshoot_m"], 0.1)
    assert np.isclose(step["undershoot_m"], 0.1)
    assert step["settled"] is True


def test_unreached_target_does_not_create_rise_or_settling_time():
    frame = _single_step(lambda time: np.where(time < 0.5, 0.0, np.minimum(time - 0.5, 0.7)))
    step = analyze_altitude_tracking(frame)["step_responses"][0]
    assert step["reached_10_percent"] is True
    assert step["reached_90_percent"] is False
    assert step["rise_time_s"] is None
    assert step["settled"] is False
    assert step["settling_time_s"] is None
    assert step["overshoot_m"] == 0.0
    assert "90%" in step["reason"]


def test_multiple_steps_are_independent_and_largest_is_primary():
    time = np.arange(0.0, 12.1, 0.1)
    setpoint = np.select(
        [time < 2.0, time < 5.0, time < 8.0],
        [0.0, 1.0, 0.5],
        default=2.0,
    )
    result = analyze_altitude_tracking(_step_frame(time, setpoint, setpoint.copy()))
    assert len(result["step_responses"]) == 3
    assert [step["amplitude_m"] for step in result["step_responses"]] == [1.0, -0.5, 1.5]
    assert result["primary_step_index"] == 2
    assert result["rise_time_s"] == result["step_responses"][2]["rise_time_s"]
    assert all(step["overshoot_m"] == 0.0 for step in result["step_responses"])


def test_each_step_response_stops_before_the_next_event():
    time = np.arange(0.0, 4.1, 0.1)
    setpoint = np.select([time < 1.0, time < 2.0], [0.0, 1.0], default=2.0)
    response = np.select([time < 1.0, time < 2.0], [0.0, 1.0], default=3.0)
    result = analyze_altitude_tracking(_step_frame(time, setpoint, response))
    assert len(result["step_responses"]) == 2
    assert result["step_responses"][0]["overshoot_m"] == 0.0
    assert result["step_responses"][1]["overshoot_m"] == 1.0


def test_equal_amplitude_steps_choose_the_earliest_as_primary():
    time = np.arange(0.0, 7.1, 0.1)
    setpoint = np.select([time < 2.0, time < 5.0], [0.0, 1.0], default=0.0)
    result = analyze_altitude_tracking(_step_frame(time, setpoint, setpoint.copy()))
    assert len(result["step_responses"]) == 2
    assert result["primary_step_index"] == 0


def test_adjacent_transition_samples_merge_into_one_step():
    time = np.arange(0.0, 2.1, 0.1)
    setpoint = np.zeros_like(time)
    setpoint[time >= 0.5] = 0.3
    setpoint[time >= 0.6] = 0.7
    setpoint[time >= 0.7] = 1.0
    events = detect_step_events(_step_frame(time, setpoint, setpoint.copy()))
    assert len(events) == 1
    assert events[0].timestamp_s == 0.5
    assert events[0].amplitude_m == 1.0


def test_continuous_ramp_is_not_treated_as_step():
    time = np.arange(0.0, 5.1, 0.1)
    setpoint = time / time[-1]
    result = analyze_altitude_tracking(_step_frame(time, setpoint, setpoint.copy()))
    assert result["step_responses"] == []
    assert result["rise_time_s"] is None
    assert result["settling_time_s"] is None
    assert "plateau" in result["dynamic_metrics_reason"]


def test_too_short_window_reports_unavailable_dynamic_metrics():
    result = analyze_altitude_tracking(_step_frame(
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.5, 1.0],
    ))
    assert result["rise_time_s"] is None
    assert result["settling_time_s"] is None
    assert result["step_responses"] == []
    assert "짧" in result["dynamic_metrics_reason"]


def test_settling_ignores_brief_tolerance_entry_and_uses_later_dwell():
    time = np.arange(0.0, 4.1, 0.1)
    setpoint = np.where(time < 1.0, 0.0, 1.0)
    response = np.zeros_like(time)
    active = time >= 1.0
    response[active] = np.interp(
        time[active],
        [1.0, 1.4, 1.5, 1.7, 1.8, 2.2, 2.3],
        [0.0, 0.9, 0.98, 1.0, 0.8, 0.8, 1.0],
    )
    step = analyze_altitude_tracking(_step_frame(time, setpoint, response))["step_responses"][0]
    assert step["settled"] is True
    assert step["settling_time_s"] >= 1.2


def test_final_samples_inside_tolerance_do_not_satisfy_dwell():
    time = np.arange(0.0, 3.1, 0.1)
    setpoint = np.where(time < 1.0, 0.0, 1.0)
    response = np.where(time < 1.0, 0.0, 0.7)
    response[-2:] = 1.0
    step = analyze_altitude_tracking(_step_frame(time, setpoint, response))["step_responses"][0]
    assert step["settled"] is False
    assert step["settling_time_s"] is None


def test_irregular_time_uses_elapsed_time_and_reverse_time_is_rejected():
    time = np.array([0.0, 0.12, 0.27, 0.45, 0.62, 0.8, 1.05, 1.33, 1.62, 1.95, 2.35])
    setpoint = np.where(time < 0.62, 0.0, 1.0)
    response = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.9, 1.0, 1.0, 1.0])
    result = analyze_altitude_tracking(_step_frame(time, setpoint, response))
    step = result["step_responses"][0]
    assert np.isclose(step["rise_time_s"], 0.53)
    assert np.isclose(step["settling_time_s"], 1.0)

    reverse_time = time.copy()
    reverse_time[6] = 0.7
    invalid = analyze_altitude_tracking(_step_frame(reverse_time, setpoint, response))
    assert invalid["rise_time_s"] is None
    assert invalid["step_responses"] == []
    assert "timestamp" in invalid["dynamic_metrics_reason"]
