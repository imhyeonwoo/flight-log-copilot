"""Human-readable deterministic analysis report."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from flightlog_copilot.i18n import localize_values


def render_markdown_report(result: Dict[str, Any], language: str = "ko") -> str:
    result = localize_values(result, language)
    english = language == "en"
    window = result.get("analysis_window", {})
    lines = [
        "# FlightLog Copilot Diagnostic Report" if english else "# FlightLog Copilot 진단 보고서",
        "",
        "> Quantitative calculations and explainable rule-based findings from the Python analysis engine."
        if english else "> Python 분석 엔진의 정량 계산과 설명 가능한 규칙 기반 진단 결과입니다.",
        "",
        "## Analysis Window" if english else "## 분석 구간",
        "",
        f"- {'Flight phase' if english else '비행 구간'}: {result.get('flight_phase', 'unknown')}",
        f"- {'Reference altitude source' if english else '기준 고도 출처'}: {result.get('reference_altitude_source', 'other')}",
        f"- {'Start' if english else '시작'}: {_format(window.get('start_s'), language)} s",
        f"- {'End' if english else '종료'}: {_format(window.get('end_s'), language)} s",
        f"- {'Analysis rows' if english else '분석 행 수'}: {window.get('rows', 0)}",
        "",
        "## Data Quality" if english else "## 데이터 품질",
        "",
    ]
    messages = result.get("validation_messages", [])
    if messages:
        for message in messages:
            lines.append(f"- **{str(message.get('severity', 'info')).upper()}** — {message.get('message', '')}")
    else:
        lines.append("- No data-quality issues were reported." if english else "- 보고할 데이터 품질 문제가 없습니다.")
    lines.extend(["", "## Quantitative Analysis" if english else "## 정량 분석", ""])
    for section, metrics in result.get("metrics", {}).items():
        lines.append(f"### {section}")
        lines.append("")
        if not metrics.get("available"):
            lines.append(
                f"{'Unavailable' if english else '계산 불가'}: "
                f"{metrics.get('reason', 'Required data is unavailable.' if english else '필요한 데이터가 없습니다.')}"
            )
        else:
            for key, value in metrics.items():
                if key in {"available", "frequencies_hz", "psd", "sampling_intervals_s", "dropout_intervals", "top_peaks", "motor_metrics"}:
                    continue
                lines.append(f"- {key}: {_format(value, language)}")
        lines.append("")
    lines.extend(
        [
            "## Rule-based Hypotheses" if english else "## 규칙 기반 가설",
            "",
            "> Diagnostic priority scores are not probabilities; they rank the relative rule-based evidence found in this log."
            if english else "> 진단 우선순위 점수는 확률이 아니며, 현재 로그에서 확인된 규칙 기반 근거의 상대적 우선순위입니다.",
            "",
        ]
    )
    for hypothesis in result.get("rule_based_hypotheses", []):
        lines.extend(
            [
                f"### {hypothesis['title']}",
                "",
                f"**{'Diagnostic priority score' if english else '진단 우선순위 점수'}: {hypothesis['score']}/100**",
                "",
                _list("Evidence" if english else "근거", hypothesis.get("evidence", []), language),
                _list("Counter-evidence" if english else "반대 근거", hypothesis.get("counter_evidence", []), language),
                _list("Missing parameters" if english else "부족한 파라미터", hypothesis.get("missing_parameters", []), language),
                _list("Limitations" if english else "분석 한계", hypothesis.get("limitations", []), language),
                _list("Recommended validation tests" if english else "권장 검증 실험", hypothesis.get("recommended_tests", []), language),
                "",
            ]
        )
    ai = result.get("ai_copilot")
    if ai:
        lines.extend(["## AI Copilot", "", str(ai.get("summary", "")), ""])
        if ai.get("additional_hypotheses"):
            lines.extend([
                "### Unverified hypotheses additionally suggested by the AI provider" if english else "### AI Provider가 추가로 제안한 미검증 가설",
                "",
                "> The items below were not verified by the rule-based engine." if english else "> 아래 항목은 규칙 기반으로 검증되지 않았습니다.",
                "",
            ])
            for item in ai["additional_hypotheses"]:
                lines.append(f"- {item}")
        lines.append("")
    lines.extend([
        "## Interpretation Notes" if english else "## 해석 주의사항",
        "",
        "- Correlation and cross-correlation do not establish causality." if english else "- 상관관계와 cross-correlation은 인과관계를 증명하지 않습니다.",
        "- Interpret coordinate frame, axis direction, units, and sign using the confirmed mapping profile."
        if english else "- 좌표계, 축 방향, 단위 및 부호는 확정된 매핑 프로필을 기준으로 해석해야 합니다.",
        "",
    ])
    return "\n".join(lines)


def _list(title: str, values: Iterable[Any], language: str = "ko") -> str:
    rendered = list(values)
    empty = "None" if language == "en" else "없음"
    return f"- **{title}:** " + ("; ".join(str(value) for value in rendered) if rendered else empty)


def _format(value: Any, language: str = "ko") -> str:
    if value is None:
        return "Unavailable" if language == "en" else "계산 불가"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
