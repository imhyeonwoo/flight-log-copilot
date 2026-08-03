import json

import pandas as pd
import pytest

from flightlog_copilot.io.mapping_profile import dump_profile, load_profile, validate_profile_columns
from flightlog_copilot.mapping.aliases import infer_reference_altitude_source, normalize_column_name
from flightlog_copilot.mapping.detector import detect_mappings
from flightlog_copilot.mapping.models import MappingProfile, ParameterMapping
from flightlog_copilot.mapping.transformer import transform_frame


def test_column_normalization_removes_units_and_separators():
    assert normalize_column_name(" EKF-Alt (m) ") == "ekfalt"
    assert normalize_column_name("Motor.PWM_1 [us]") == "motorpwm1"


def test_alias_and_motor_mapping():
    frame = pd.DataFrame({"time_s": [0.0, 0.1], "estimated_altitude": [1.0, 1.1], "PWM-1 (us)": [1500, 1510]})
    candidates = detect_mappings(frame)
    assert candidates["timestamp"].column == "time_s"
    assert candidates["ekf_altitude"].column == "estimated_altitude"
    assert candidates["motor_1"].column == "PWM-1 (us)"


def test_barometer_and_gnss_map_to_reference_altitude_with_source():
    barometer = detect_mappings(pd.DataFrame({"baro_alt": [1.0, 1.1]}))
    gnss = detect_mappings(pd.DataFrame({"gps_altitude": [20.0, 20.1]}))
    assert barometer["reference_altitude"].column == "baro_alt"
    assert gnss["reference_altitude"].column == "gps_altitude"
    assert infer_reference_altitude_source("baro_alt (m)") == "barometer"
    assert infer_reference_altitude_source("GPS Altitude [m]") == "gnss"


def test_low_confidence_is_not_auto_selected():
    frame = pd.DataFrame({"mystery": [3.14, 2.72]})
    result = detect_mappings(frame)["ekf_altitude"]
    assert result.status == "미매핑"
    assert result.column is None


def test_user_mapping_transform_sign_scale_offset_and_time_units():
    frame = pd.DataFrame({"clock_us": [0, 1_000_000], "pos_d": [10.0, 20.0], "manual_alt": [2.0, 3.0]})
    profile = MappingProfile(
        time_unit="microseconds",
        parameters={
            "timestamp": ParameterMapping(column="clock_us"),
            "ekf_altitude": ParameterMapping(column="manual_alt", scale=2.0, offset=1.0, invert_sign=True, unit="meters"),
        },
    )
    transformed, warnings = transform_frame(frame, profile)
    assert not warnings
    assert transformed["timestamp"].tolist() == [0.0, 1.0]
    assert transformed["ekf_altitude"].tolist() == [-3.0, -5.0]


def test_mapping_profile_round_trip_and_missing_column_warning():
    profile = MappingProfile(
        profile_name="test",
        reference_altitude_source="gnss",
        parameters={"timestamp": ParameterMapping(column="t")},
    )
    restored = load_profile(dump_profile(profile))
    assert restored.profile_name == "test"
    assert restored.reference_altitude_source == "gnss"
    assert validate_profile_columns(restored, ["other"]) == ["t"]


def test_legacy_barometer_profile_is_migrated():
    restored = load_profile(json.dumps({
        "parameters": {"barometer_altitude": {"column": "baro_alt", "unit": "meters"}},
    }))
    assert "barometer_altitude" not in restored.parameters
    assert restored.parameters["reference_altitude"].column == "baro_alt"
    assert restored.reference_altitude_source == "barometer"


def test_pos_d_requires_confirmation_and_never_inverts_automatically():
    frame = pd.DataFrame({"pos_d": [0.0, -1.0], "time": [0.0, 1.0]})
    result = detect_mappings(frame)["ekf_altitude"]
    assert result.status != "자동 추천"
    assert result.column == "pos_d"
    assert any("부호" in reason for reason in result.reasons)


def test_velocity_unit_conversion_and_profile_validation():
    profile = MappingProfile(parameters={
        "vertical_velocity": ParameterMapping(column="vz", unit="cm/s"),
        "reference_altitude": ParameterMapping(column="gps_alt", unit="centimeters"),
    })
    transformed, _ = transform_frame(pd.DataFrame({"vz": [100.0], "gps_alt": [250.0]}), profile)
    assert transformed["vertical_velocity"].item() == 1.0
    assert transformed["reference_altitude"].item() == 2.5
    with pytest.raises(ValueError, match="time_unit"):
        load_profile('{"time_unit":"minutes","parameters":{}}')
    with pytest.raises(ValueError, match="reference_altitude_source"):
        load_profile('{"reference_altitude_source":"lidar","parameters":{}}')
