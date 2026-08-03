import pytest
from pydantic import ValidationError

from flightlog_copilot.engine import build_llm_payload
from flightlog_copilot.llm.client import LLMAnalysisError, analyze_with_openai
from flightlog_copilot.llm.schemas import CopilotResponse


VALID_RESPONSE = {
    "summary": "정량 지표를 바탕으로 추가 검증이 필요합니다.",
    "hypothesis_review": [{
        "hypothesis_id": "incorrect_hover_pwm",
        "priority": 1,
        "explanation": "지속 correction 편향이 있습니다.",
        "supporting_evidence": ["mean correction"],
        "counter_evidence": [],
        "uncertainties": ["hover PWM 미기록"],
    }],
    "additional_hypotheses": ["배터리 전압 강하(미검증)"],
    "recommended_experiments": [{
        "title": "Hover 기준 출력 A/B 테스트",
        "purpose": "편향 원인 확인",
        "variables_to_change": ["hover PWM"],
        "variables_to_log": ["battery voltage"],
        "expected_observation": "correction 평균 변화",
        "safety_notes": ["작은 단계로 변경"],
    }],
    "missing_data_requests": ["battery voltage"],
}


def test_gpt_response_schema_validation():
    parsed = CopilotResponse.model_validate(VALID_RESPONSE)
    assert parsed.hypothesis_review[0].priority == 1
    with pytest.raises(ValidationError):
        CopilotResponse.model_validate({"summary": "incomplete"})


def test_client_rejects_raw_csv_payload_before_api_call():
    with pytest.raises(LLMAnalysisError, match="원시 CSV"):
        analyze_with_openai({"raw_csv": "a,b"}, client=object())
    with pytest.raises(LLMAnalysisError, match="원시 CSV"):
        analyze_with_openai({"nested": {"records": [[1, 2]]}}, client=object())


def test_client_parses_mocked_structured_response():
    class Response:
        output_parsed = CopilotResponse.model_validate(VALID_RESPONSE)

    class Responses:
        def parse(self, **kwargs):
            assert "raw_csv" not in kwargs["input"][1]["content"]
            return Response()

    class Client:
        responses = Responses()

    payload = build_llm_payload({
        "analysis_window": {"start_s": 0.0, "end_s": 10.0, "rows": 401},
        "metrics": {"altitude": {"rmse_m": 0.2}},
    })
    result = analyze_with_openai(payload, client=Client(), model="test-model")
    assert result["summary"].startswith("정량")
