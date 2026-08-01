from pathlib import Path

from flightlog_copilot.engine import build_llm_payload, run_deterministic_analysis
from flightlog_copilot.io.csv_loader import load_csv
from flightlog_copilot.mapping.detector import detect_mappings
from flightlog_copilot.mapping.models import MappingProfile, ParameterMapping
from flightlog_copilot.mapping.transformer import transform_frame
from flightlog_copilot.preprocessing.segmentation import detect_segments, select_time_range
from flightlog_copilot.preprocessing.validation import validate_data
from flightlog_copilot.reporting import render_json_report, render_markdown_report


def test_sample_log_end_to_end_without_openai():
    sample = Path(__file__).parents[1] / "sample_data" / "sample_alt_hold.csv"
    loaded = load_csv(sample.read_bytes(), sample.name)
    candidates = detect_mappings(loaded.data)
    profile = MappingProfile(
        time_unit="seconds",
        parameters={
            name: ParameterMapping(
                column=candidate.column,
                unit="meters" if "altitude" in name else "us",
                confirmed=True,
            )
            for name, candidate in candidates.items()
            if candidate.column
        },
    )
    canonical, warnings = transform_frame(loaded.data, profile)
    assert not warnings
    segment = detect_segments(canonical)["althold"][0]
    selected = select_time_range(canonical, segment.start_s, segment.end_s)
    validation = validate_data(loaded.data, canonical)
    result = run_deterministic_analysis(
        canonical,
        selected,
        {name: mapping.column for name, mapping in profile.parameters.items()},
        [message.to_dict() for message in validation],
        "altitude_hold",
    )
    assert result["analysis_window"]["rows"] == 401
    assert abs(result["metrics"]["frequency"]["dominant_frequency_hz"] - 0.3) < 0.04
    assert len(result["rule_based_hypotheses"]) == 12
    payload = build_llm_payload(result)
    assert set(payload) == {
        "flight_phase",
        "analysis_window",
        "parameter_mapping",
        "features",
        "rule_based_hypotheses",
        "missing_parameters",
        "analysis_warnings",
    }
    assert render_markdown_report(result).startswith("# FlightLog Copilot")
    assert '"flight_phase": "altitude_hold"' in render_json_report(result)
