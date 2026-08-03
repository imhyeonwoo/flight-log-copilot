import re

from flightlog_copilot.engine import build_llm_payload
from flightlog_copilot.i18n import localize_values, translate_text
from flightlog_copilot.reporting.markdown_report import render_markdown_report


def test_mapping_and_dynamic_evidence_translate_to_english():
    assert translate_text("확인 필요", "en") == "Review required"
    assert translate_text("correction 포화 비율이 25.0%입니다.", "en") == "Correction saturation ratio is 25.0%."
    assert translate_text("기준 고도-EKF 평균 bias가 0.250 m입니다.", "en") == "Mean reference-altitude/EKF bias is 0.250 m."
    assert translate_text("확인 필요", "ko") == "확인 필요"


def test_english_report_and_llm_language_are_consistent():
    result = {
        "flight_phase": "altitude_hold",
        "analysis_window": {"start_s": 1.0, "end_s": 2.0, "rows": 2},
        "parameter_mapping": {"timestamp": "time_s"},
        "validation_messages": [],
        "metrics": {},
        "rule_based_hypotheses": [{
            "title": "Throttle correction 포화",
            "score": 70,
            "evidence": ["correction 포화 비율이 25.0%입니다."],
            "counter_evidence": [],
            "missing_parameters": [],
            "limitations": [],
            "recommended_tests": [],
        }],
        "missing_parameters": [],
        "analysis_warnings": [],
    }
    localized = localize_values(result, "en")
    assert localized["rule_based_hypotheses"][0]["title"] == "Throttle correction saturation"
    report = render_markdown_report(result, language="en")
    assert report.startswith("# FlightLog Copilot Diagnostic Report")
    assert "Diagnostic priority score: 70/100" in report
    assert not re.search("[가-힣]", report)
    payload = build_llm_payload(result, response_language="en")
    assert payload["response_language"] == "English"


def test_corrupt_ai_settings_warning_is_translated_with_backup_name():
    warning = (
        "AI 설정 파일이 손상되어 기본 설정으로 실행합니다. "
        "손상 파일은 'ai_settings.json.corrupt-20260803-120000.bak'으로 백업했습니다."
    )
    translated = translate_text(warning, "en")
    assert "using default settings" in translated
    assert "ai_settings.json.corrupt-20260803-120000.bak" in translated
