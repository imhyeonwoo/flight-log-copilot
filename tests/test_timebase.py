import pandas as pd
import pytest

from flightlog_copilot.analysis.timing import analyze_timing
from flightlog_copilot.mapping.models import MappingProfile, ParameterMapping
from flightlog_copilot.mapping.time_units import (
    infer_time_unit,
    time_unit_from_column_name,
)
from flightlog_copilot.mapping.transformer import transform_frame
from flightlog_copilot.preprocessing.timebase import sorted_unique_time


def test_seconds_milliseconds_and_microseconds_conversion():
    for unit, values in [("seconds", [0, 1]), ("milliseconds", [0, 1000]), ("microseconds", [0, 1_000_000])]:
        profile = MappingProfile(time_unit=unit, parameters={"timestamp": ParameterMapping(column="t")})
        converted, _ = transform_frame(pd.DataFrame({"t": values}), profile)
        assert converted["timestamp"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "column,values,expected",
    [
        ("timestamp_ms", [0, 40, 80], [0.0, 0.04, 0.08]),
        ("timestamp_us", [0, 40_000, 80_000], [0.0, 0.04, 0.08]),
        ("time_s", [0, 1, 2], [0.0, 1.0, 2.0]),
        ("time_s", [0, 2, 4, 6], [0.0, 2.0, 4.0, 6.0]),
    ],
)
def test_auto_time_unit_uses_explicit_column_tokens(column, values, expected):
    profile = MappingProfile(
        time_unit="auto",
        parameters={"timestamp": ParameterMapping(column=column, unit="auto")},
    )
    converted, _ = transform_frame(pd.DataFrame({column: values}), profile)
    assert converted["timestamp"].tolist() == expected


def test_legacy_empty_mapping_unit_uses_safe_auto_inference():
    profile = MappingProfile(
        time_unit="auto",
        parameters={"timestamp": ParameterMapping(column="time_s")},
    )
    converted, _ = transform_frame(pd.DataFrame({"time_s": [0, 1, 2]}), profile)
    assert converted["timestamp"].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "column,expected_unit",
    [
        ("Seconds", "seconds"),
        ("time_s", "seconds"),
        ("timestamp-s", "seconds"),
        ("elapsed.s", "seconds"),
        ("time(sec)", "seconds"),
        ("time [seconds]", "seconds"),
        ("Milliseconds", "milliseconds"),
        ("time_ms", "milliseconds"),
        ("timestamp-ms", "milliseconds"),
        ("elapsed.ms", "milliseconds"),
        ("time(msec)", "milliseconds"),
        ("time [milliseconds]", "milliseconds"),
        ("Microseconds", "microseconds"),
        ("time_us", "microseconds"),
        ("timestamp-us", "microseconds"),
        ("timestamp_usec", "microseconds"),
        ("time(microseconds)", "microseconds"),
        ("time [µs]", "microseconds"),
    ],
)
def test_supported_timestamp_column_unit_tokens(column, expected_unit):
    result = time_unit_from_column_name(column)
    assert result is not None
    assert result[0] == expected_unit


@pytest.mark.parametrize("column", ["timestamp", "timestamps", "ticks"])
def test_timestamp_like_names_are_not_mistaken_for_seconds(column):
    assert time_unit_from_column_name(column) is None
    inference = infer_time_unit(pd.Series([0, 1, 2, 3]), column)
    assert inference.unit is None
    assert inference.requires_confirmation is True


def test_ambiguous_auto_timestamp_requires_user_confirmation():
    inference = infer_time_unit(pd.Series([0, 1, 2, 3]), "timestamp")
    assert inference.unit is None
    assert inference.confidence < 0.5
    assert inference.requires_confirmation is True
    assert any("여러 개" in reason for reason in inference.reasons)

    profile = MappingProfile(
        time_unit="auto",
        parameters={"timestamp": ParameterMapping(column="timestamp", unit="auto")},
    )
    with pytest.raises(ValueError, match="직접 선택"):
        transform_frame(pd.DataFrame({"timestamp": [0, 1, 2, 3]}), profile)


def test_explicit_profile_time_unit_overrides_column_suffix():
    profile = MappingProfile(
        time_unit="seconds",
        parameters={"timestamp": ParameterMapping(column="timestamp_ms", unit="auto")},
    )
    converted, _ = transform_frame(pd.DataFrame({"timestamp_ms": [0, 1, 2]}), profile)
    assert converted["timestamp"].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "values,expected_unit",
    [
        ([0.0, 0.04, 0.08], "seconds"),
        ([0, 40, 80], "milliseconds"),
        ([0, 40_000, 80_000], "microseconds"),
    ],
)
def test_value_based_inference_requires_one_realistic_frequency_candidate(values, expected_unit):
    inference = infer_time_unit(pd.Series(values), "clock")
    assert inference.unit == expected_unit
    assert inference.confidence >= 0.8
    assert inference.requires_confirmation is False


def test_duplicate_reverse_and_irregular_sampling():
    frame = pd.DataFrame({"timestamp": [0.0, 0.1, 0.1, 0.05, 1.0]})
    cleaned, duplicate, reverse = sorted_unique_time(frame)
    assert duplicate == 1
    assert reverse == 1
    metrics = analyze_timing(frame, frame, dropout_multiplier=3)
    assert metrics["duplicate_timestamp_count"] == 1
    assert metrics["reverse_timestamp_count"] == 1
    assert metrics["dropout_intervals"]
