import pandas as pd

from flightlog_copilot.analysis.timing import analyze_timing
from flightlog_copilot.mapping.models import MappingProfile, ParameterMapping
from flightlog_copilot.mapping.transformer import transform_frame
from flightlog_copilot.preprocessing.timebase import sorted_unique_time


def test_seconds_milliseconds_and_microseconds_conversion():
    for unit, values in [("seconds", [0, 1]), ("milliseconds", [0, 1000]), ("microseconds", [0, 1_000_000])]:
        profile = MappingProfile(time_unit=unit, parameters={"timestamp": ParameterMapping(column="t")})
        converted, _ = transform_frame(pd.DataFrame({"t": values}), profile)
        assert converted["timestamp"].tolist() == [0.0, 1.0]


def test_duplicate_reverse_and_irregular_sampling():
    frame = pd.DataFrame({"timestamp": [0.0, 0.1, 0.1, 0.05, 1.0]})
    cleaned, duplicate, reverse = sorted_unique_time(frame)
    assert duplicate == 1
    assert reverse == 1
    metrics = analyze_timing(frame, frame, dropout_multiplier=3)
    assert metrics["duplicate_timestamp_count"] == 1
    assert metrics["reverse_timestamp_count"] == 1
    assert metrics["dropout_intervals"]
