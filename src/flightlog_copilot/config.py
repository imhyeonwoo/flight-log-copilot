"""Application-wide defaults and canonical parameter metadata."""

from dataclasses import dataclass


STANDARD_PARAMETERS = (
    "timestamp",
    "altitude_setpoint",
    "ekf_altitude",
    "reference_altitude",
    "vertical_velocity",
    "throttle_base",
    "throttle_correction",
    "motor_1",
    "motor_2",
    "motor_3",
    "motor_4",
    "armed",
    "althold_active",
)

MINIMUM_REQUIRED = {"timestamp", "ekf_altitude"}
ALTHOLD_PARAMETERS = {"altitude_setpoint", "throttle_correction", "althold_active"}
REFERENCE_PARAMETERS = {"reference_altitude"}
MOTOR_PARAMETERS = {"motor_1", "motor_2", "motor_3", "motor_4"}

PARAMETER_GROUP = {
    "timestamp": "time",
    "altitude_setpoint": "altitude",
    "ekf_altitude": "altitude",
    "reference_altitude": "altitude",
    "vertical_velocity": "velocity",
    "throttle_base": "pwm",
    "throttle_correction": "pwm",
    "motor_1": "pwm",
    "motor_2": "pwm",
    "motor_3": "pwm",
    "motor_4": "pwm",
    "armed": "boolean",
    "althold_active": "boolean",
}

DEFAULT_UNITS = {
    "time": "auto",
    "altitude": "meters",
    "velocity": "m/s",
    "pwm": "us",
    "boolean": "boolean",
}


@dataclass(frozen=True)
class AnalysisSettings:
    correction_limit: float = 160.0
    motor_pwm_min: float = 1000.0
    motor_pwm_max: float = 2000.0
    dropout_multiplier: float = 3.0
    minimum_rows: int = 20
    steady_state_fraction: float = 0.2
    settling_tolerance_fraction: float = 0.05
