"""Aliases and normalization for common embedded-flight-controller logs."""

from __future__ import annotations

import re
import unicodedata


ALIASES = {
    "timestamp": ["timestamp", "time", "t", "time_s", "elapsed_time", "timestamp_ms", "timestamp_us", "tick"],
    "altitude_setpoint": ["altitude_setpoint", "alt_sp", "altitude_sp", "height_setpoint", "target_altitude", "z_setpoint"],
    "ekf_altitude": ["ekf_altitude", "ekf_alt", "estimated_altitude", "height_estimate", "altitude_estimate", "pos_d", "position_d"],
    "barometer_altitude": ["barometer_altitude", "baro_alt", "baro_altitude", "bmp_altitude", "pressure_altitude"],
    "vertical_velocity": ["vertical_velocity", "vertical_speed", "vz", "vel_z", "vel_d", "velocity_d"],
    "throttle_base": ["throttle_base", "base_throttle", "base_pwm", "hover_pwm"],
    "throttle_correction": ["throttle_correction", "throttle_corr", "alt_corr", "altitude_correction", "z_correction"],
    "armed": ["armed", "is_armed", "arm_state", "vehicle_armed"],
    "althold_active": ["althold_active", "alt_hold_active", "althold", "alt_hold", "altitude_hold", "mode_althold"],
}

for motor_number in range(1, 5):
    ALIASES[f"motor_{motor_number}"] = [
        f"motor_{motor_number}",
        f"motor{motor_number}",
        f"m{motor_number}",
        f"pwm{motor_number}",
        f"pwm_{motor_number}",
        f"motor_pwm_{motor_number}",
        f"motorpwm{motor_number}",
    ]

UNIT_SUFFIXES = {
    "s", "sec", "secs", "second", "seconds", "ms", "msec", "milliseconds",
    "us", "usec", "microseconds", "m", "meter", "meters", "cm", "mm", "mps", "hz", "pwm",
}


def tokenize_column_name(name: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(name)).lower().strip()
    text = re.sub(r"([a-z])([0-9])", r"\1_\2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1_\2", text)
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    while tokens and tokens[-1] in UNIT_SUFFIXES:
        tokens.pop()
    return tokens


def normalize_column_name(name: str) -> str:
    return "".join(tokenize_column_name(name))


NORMALIZED_ALIASES = {
    parameter: {normalize_column_name(alias) for alias in aliases}
    for parameter, aliases in ALIASES.items()
}
