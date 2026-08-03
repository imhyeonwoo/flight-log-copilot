from flightlog_copilot.diagnosis.rules import evaluate_hypotheses


def _find(results, hypothesis_id):
    return next(item for item in results if item["id"] == hypothesis_id)


def test_rule_scores_are_explainable_not_probabilities():
    metrics = {
        "timing": {"available": True, "sampling_jitter_ratio": 0.2, "dropout_intervals": [{}], "duplicate_timestamp_count": 0, "reverse_timestamp_count": 0},
        "altitude": {"available": True, "steady_state_error_m": 0.25, "error_std_m": 0.3},
        "actuator": {"available": True, "mean_throttle_correction_pwm": 40, "correction_saturation_ratio": 0.25, "max_continuous_saturation_s": 1.0, "motor_metrics": {}},
        "sensor_comparison": {"available": False},
        "frequency": {"available": False},
    }
    results = evaluate_hypotheses(metrics, {"timestamp", "ekf_altitude", "altitude_setpoint", "throttle_correction"})
    saturation = _find(results, "throttle_correction_saturation")
    assert 0 <= saturation["score"] <= 100
    assert saturation["evidence"]
    assert saturation["score_details"][0]["delta"] > 0
    assert "probability" not in saturation


def test_missing_parameters_do_not_crash_rules():
    results = evaluate_hypotheses({}, {"timestamp"})
    motor = _find(results, "motor_propeller_imbalance")
    assert motor["score"] == 0
    assert set(motor["missing_parameters"]) == {"motor_1", "motor_2", "motor_3", "motor_4"}
    reference = _find(results, "reference_ekf_bias")
    assert set(reference["missing_parameters"]) == {"reference_altitude", "ekf_altitude"}
