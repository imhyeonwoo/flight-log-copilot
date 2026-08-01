import pytest

from flightlog_copilot.io.csv_loader import CSVLoadError, load_csv, profile_columns


def test_loader_detects_delimiter_and_profiles_columns():
    result = load_csv(b"time_s;ekf_alt;armed\n0;1.0;0\n1;1.2;1\n")
    assert result.delimiter == ";"
    profile = profile_columns(result.data)
    assert profile.loc[profile["컬럼명"] == "armed", "추정 타입"].item() == "boolean"


def test_loader_rejects_headerless_numeric_csv():
    with pytest.raises(CSVLoadError, match="헤더"):
        load_csv(b"1,2,3\n4,5,6\n")


def test_loader_reports_duplicate_headers():
    result = load_csv(b"time,time,value\n0,0,1\n")
    assert result.duplicate_columns == ["time"]
