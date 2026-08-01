"""Korean/English presentation helpers for deterministic analysis output."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Pattern, Tuple


SUPPORTED_LANGUAGES = {"KO": "ko", "EN": "en"}


ENGLISH_TEXT: Dict[str, str] = {
    # Mapping states and evidence
    "미매핑": "Unmapped",
    "확인 필요": "Review required",
    "자동 추천": "Auto-suggested",
    "사용자 확인": "User confirmed",
    "표준 이름 완전 일치": "Exact canonical-name match",
    "별칭 사전 완전 일치": "Exact alias-dictionary match",
    "불리언형 데이터": "Boolean data",
    "숫자형 데이터": "Numeric data",
    "예상 데이터 타입과 다름": "Data type differs from expectation",
    "값 범위가 보조 근거와 일치": "Value range supports the mapping",
    "값 범위가 일반적 범위와 다름": "Value range differs from the usual range",
    "관련 파라미터 컬럼군이 함께 존재": "Related parameter columns are present",
    "좌표계와 부호 확인 필요: Down 축일 수 있음": "Coordinate frame and sign need review: this may be a Down axis",
    # Analysis availability and caveats
    "timestamp 파라미터가 없습니다.": "The timestamp parameter is missing.",
    "유효하고 서로 다른 timestamp가 2개 미만입니다.": "Fewer than two valid, distinct timestamps are available.",
    "양수 sampling interval이 없습니다.": "No positive sampling interval is available.",
    "고도 추종 파라미터가 부족합니다.": "Altitude-tracking parameters are missing.",
    "유효 고도 샘플이 2개 미만입니다.": "Fewer than two valid altitude samples are available.",
    "명확한 setpoint 변화 또는 timestamp가 없어 동적 지표를 계산하지 않았습니다.": "Dynamic metrics were skipped because there is no clear setpoint change or timestamp.",
    "setpoint 변화 전후 데이터가 부족합니다.": "There is not enough data before and after the setpoint change.",
    "지속되는 setpoint 단계를 확인하지 못했습니다.": "No sustained setpoint step was found.",
    "출력 관련 파라미터가 없습니다.": "Actuator-output parameters are missing.",
    "Barometer-EKF 비교 파라미터가 부족합니다.": "Barometer-EKF comparison parameters are missing.",
    "동시에 유효한 Barometer/EKF 샘플이 3개 미만입니다.": "Fewer than three paired Barometer/EKF samples are valid.",
    "상관관계와 cross-correlation lag는 인과관계를 증명하지 않습니다.": "Correlation and cross-correlation lag do not establish causality.",
    "주파수 분석 파라미터가 부족합니다.": "Frequency-analysis parameters are missing.",
    "데이터 길이가 부족합니다.": "The data record is too short.",
    "보간 후 샘플 수가 부족합니다.": "Too few samples remain after interpolation.",
    "유효한 PSD를 계산하지 못했습니다.": "A valid PSD could not be calculated.",
    "필요한 데이터가 없습니다.": "Required data is unavailable.",
    "필요한 파라미터가 없습니다.": "Required parameters are unavailable.",
    "계산 불가": "Unavailable",
    "없음": "None",
    # CSV, profile, and optional-AI messages
    "빈 CSV 파일입니다.": "The CSV file is empty.",
    "CSV 헤더가 없습니다.": "The CSV has no header.",
    "첫 행이 모두 숫자여서 헤더가 없는 CSV로 판단했습니다.": "The first row is entirely numeric, so the CSV appears to have no header.",
    "CSV에 데이터 행이 없습니다.": "The CSV has no data rows.",
    "컬럼이 하나뿐입니다. 구분자 자동 감지 결과를 확인하십시오.": "Only one column was found. Verify the detected delimiter.",
    "매핑 프로필은 JSON 문자열, bytes 또는 객체여야 합니다.": "The mapping profile must be a JSON string, bytes, or an object.",
    "각 파라미터 매핑은 JSON 객체여야 합니다.": "Each parameter mapping must be a JSON object.",
    "매핑 column은 문자열 또는 null이어야 합니다.": "A mapping column must be a string or null.",
    "매핑 scale과 offset은 유한한 숫자여야 합니다.": "Mapping scale and offset must be finite numbers.",
    "매핑 프로필에 parameters 객체가 필요합니다.": "The mapping profile requires a parameters object.",
    "time_unit은 auto, seconds, milliseconds, microseconds 중 하나여야 합니다.": "time_unit must be auto, seconds, milliseconds, or microseconds.",
    "현재 CSV와 헤더 해시가 일치하는 매핑 프로필입니다.": "The mapping profile header hash matches the current CSV.",
    "매핑 프로필을 불러왔습니다. 컬럼과 좌표계 설정을 확인하십시오.": "The mapping profile was imported. Verify its columns and coordinate-frame settings.",
    "OPENAI_API_KEY가 없어 AI Copilot 호출을 건너뜁니다.": "OPENAI_API_KEY is missing, so the AI Copilot call was skipped.",
    # Score result labels
    "충족": "Matched",
    "미충족": "Not matched",
    # Hypothesis titles
    "Hover PWM 또는 기준 추력 부정확": "Incorrect hover PWM or baseline thrust",
    "Altitude P gain 과대": "Altitude P gain too high",
    "Altitude I gain 과대": "Altitude I gain too high",
    "적분기 windup": "Integrator windup",
    "Throttle correction 포화": "Throttle correction saturation",
    "Barometer propwash 또는 기체 진동": "Barometer propwash or airframe vibration",
    "Barometer 기준점 또는 EKF altitude bias": "Barometer reference or EKF altitude bias",
    "센서 또는 제어 응답 지연": "Sensor or control-response delay",
    "Task 주기 불안정": "Task-period instability",
    "Timestamp 또는 로깅 문제": "Timestamp or logging problem",
    "CG 비대칭": "Center-of-gravity asymmetry",
    "모터 또는 프로펠러 추력 불균형": "Motor or propeller thrust imbalance",
    # Recommended tests
    "안전하게 고정된 조건에서 hover 기준 출력을 작은 단계로 조정하고 정상상태 correction 평균을 비교합니다.": "Under safely constrained conditions, adjust the hover baseline in small steps and compare the steady-state correction mean.",
    "P gain을 소폭 낮춘 A/B 로그에서 고도 오차 진동과 settling을 비교합니다.": "Slightly reduce P gain and compare altitude-error oscillation and settling in A/B logs.",
    "I gain을 소폭 낮추고 장주기 오차 및 correction 편향 변화를 비교합니다.": "Slightly reduce I gain and compare long-period error and correction bias.",
    "적분기 상태와 anti-windup clamp를 로그에 추가하고 포화 해제 후 복귀 시간을 확인합니다.": "Log the integrator state and anti-windup clamp, then inspect recovery time after saturation clears.",
    "correction limit 및 실제 출력 한계를 확인하고 포화 구간의 setpoint·전압·추력을 함께 기록합니다.": "Verify the correction and physical output limits, and log setpoint, voltage, and thrust during saturation.",
    "프로펠러 정지/회전 조건과 barometer 폼·위치 변경 전후의 오차 PSD를 비교합니다.": "Compare error PSD with propellers stopped/running and before/after changing barometer foam or placement.",
    "시동 전 영점 구간과 착륙 후 영점 구간의 Barometer-EKF bias를 비교합니다.": "Compare Barometer-EKF bias in the pre-arm and post-landing zero-reference periods.",
    "센서 원시값·필터 출력·제어기 출력을 동일 timestamp로 기록해 지연 위치를 분리합니다.": "Log raw sensor, filtered, and controller outputs on a shared timestamp to isolate the source of delay.",
    "제어 task 실행 시간과 deadline miss 카운터를 추가로 기록합니다.": "Log control-task execution time and deadline-miss counters.",
    "단조 clock과 sequence counter를 함께 기록해 누락·중복·재정렬을 확인합니다.": "Log a monotonic clock and sequence counter to detect loss, duplication, or reordering.",
    "기체 무게중심을 측정하고 동일 hover 조건의 모터 평균 출력을 비교합니다.": "Measure the airframe center of gravity and compare mean motor outputs under the same hover condition.",
    "모터/프로펠러 위치 교환 전후의 출력 편차와 진동 peak 이동을 비교합니다.": "Swap motor/propeller positions and compare output imbalance and vibration-peak movement.",
    # Limitations and common evidence
    "필요한 로그 파라미터가 부족하여 점수가 과소평가될 수 있습니다.": "Missing log parameters may cause this score to be underestimated.",
    "I gain과 적분기 상태가 직접 기록되지 않아 간접 지표만 사용했습니다.": "Only indirect indicators were used because I gain and integrator state were not logged.",
    "적분기 내부 상태가 없으므로 windup을 확정할 수 없습니다.": "Windup cannot be confirmed without the internal integrator state.",
    "cross-correlation lag는 공통 입력과 필터 특성의 영향도 받습니다.": "Cross-correlation lag is also affected by common inputs and filter characteristics.",
    "로그 timestamp 변동과 실제 제어 task 주기 변동은 서로 다를 수 있습니다.": "Log-timestamp variation may differ from actual control-task timing variation.",
    "CG 비대칭과 모터 추력 차이를 자세·전류 로그 없이 분리할 수 없습니다.": "CG asymmetry cannot be separated from motor-thrust differences without attitude and current logs.",
    "PWM 명령만으로 실제 모터/프로펠러 추력을 직접 측정할 수 없습니다.": "PWM commands alone do not directly measure actual motor or propeller thrust.",
    "평균 throttle correction 편향이 작습니다.": "Mean throttle-correction bias is small.",
    "정상상태 고도 오차가 작습니다.": "Steady-state altitude error is small.",
    "고도 오차 변동이 작습니다.": "Altitude-error variation is small.",
    "뚜렷한 제어 대역 진동 peak가 없습니다.": "No clear control-band vibration peak is present.",
    "correction 포화 비율이 낮습니다.": "The correction saturation ratio is low.",
    "주요 진동이 저주파 적분 거동과 일치하지 않습니다.": "The dominant vibration does not match low-frequency integral behavior.",
    "긴 연속 포화가 없습니다.": "No long continuous saturation is present.",
    "포화 점유율이 낮습니다.": "The saturation duty ratio is low.",
    "settling이 빠릅니다.": "Settling is fast.",
    "correction 포화가 거의 없습니다.": "Correction saturation is negligible.",
    "센서 차이 변동이 작습니다.": "Sensor-difference variation is small.",
    "모터 출력과 센서 오차의 상관이 약합니다.": "Motor output and sensor error are weakly correlated.",
    "고주파 진동 peak 근거가 약합니다.": "Evidence for a high-frequency vibration peak is weak.",
    "평균 센서 bias가 작습니다.": "Mean sensor bias is small.",
    "bias가 차이의 변동보다 커서 지속 오프셋 형태입니다.": "The bias exceeds difference variation, indicating a persistent offset.",
    "bias가 변동에 비해 지배적이지 않습니다.": "Bias is not dominant relative to variation.",
    "추정 lag가 작습니다.": "Estimated lag is small.",
    "cross-correlation이 약해 lag 해석이 제한됩니다.": "Weak cross-correlation limits lag interpretation.",
    "sampling jitter가 낮습니다.": "Sampling jitter is low.",
    "큰 sampling gap 후보가 없습니다.": "No large sampling-gap candidates were found.",
    "중복 timestamp가 없습니다.": "No duplicate timestamps were found.",
    "역순 timestamp가 없습니다.": "No reverse-order timestamps were found.",
    "큰 시간 간격이 없습니다.": "No large time gaps were found.",
    "모터 출력 spread가 작습니다.": "Motor-output spread is small.",
    "모터 출력 변동이 작습니다.": "Motor-output variation is small.",
    # Rule names
    "지속 correction 편향": "Persistent correction bias",
    "정상상태 고도 오차": "Steady-state altitude error",
    "고도 오차 분산": "Altitude-error variance",
    "오차 진동 peak": "Error vibration peak",
    "correction 포화": "Correction saturation",
    "저주파 진동": "Low-frequency vibration",
    "연속 포화": "Continuous saturation",
    "포화 점유율": "Saturation duty ratio",
    "느린 복귀": "Slow recovery",
    "correction limit 도달": "Correction limit reached",
    "Barometer-EKF 차이 변동": "Barometer-EKF difference variation",
    "모터 출력과 센서 오차 상관": "Motor-output and sensor-error correlation",
    "진동 대역 peak": "Vibration-band peak",
    "평균 센서 bias": "Mean sensor bias",
    "변동 대비 지속 bias": "Persistent bias relative to variation",
    "lag 추정 일관성": "Lag-estimate consistency",
    "dropout 후보": "Dropout candidates",
    "중복 timestamp": "Duplicate timestamps",
    "역순 timestamp": "Reverse-order timestamps",
    "logging dropout 후보": "Logging-dropout candidates",
    "모터 평균 spread": "Mean motor spread",
    "모터 출력 불균형": "Motor-output imbalance",
    "모터 출력 변동": "Motor-output variation",
}


DynamicRule = Tuple[Pattern[str], Callable[[re.Match[str]], str]]

DYNAMIC_ENGLISH: List[DynamicRule] = [
    (re.compile(r"CSV 텍스트 인코딩을 판별하지 못했습니다: (.+)"), lambda m: f"The CSV text encoding could not be determined: {m.group(1)}"),
    (re.compile(r"CSV를 읽지 못했습니다: (.+)"), lambda m: f"The CSV could not be read: {m.group(1)}"),
    (re.compile(r"중복 컬럼명: (.+)"), lambda m: f"Duplicate column names: {m.group(1)}"),
    (re.compile(r"매핑 프로필 JSON 형식이 올바르지 않습니다: (.+)"), lambda m: f"The mapping-profile JSON is invalid: {m.group(1)}"),
    (re.compile(r"알 수 없는 표준 파라미터: (.+)"), lambda m: f"Unknown canonical parameters: {m.group(1)}"),
    (re.compile(r"현재 CSV에 없는 프로필 컬럼: (.+)"), lambda m: f"Profile columns missing from the current CSV: {m.group(1)}"),
    (re.compile(r"데이터가 (\d+)행으로 짧아 일부 분석의 신뢰도가 낮습니다\."), lambda m: f"The log contains only {m.group(1)} rows, so some analysis results may be less reliable."),
    (re.compile(r"상수 컬럼: (.+)"), lambda m: f"Constant-value columns: {m.group(1)}"),
    (re.compile(r"원본 데이터에 결측치 (\d+)개가 있습니다\."), lambda m: f"The source data contains {m.group(1)} missing values."),
    (re.compile(r"timestamp 숫자 변환 또는 결측 실패 (\d+)개"), lambda m: f"{m.group(1)} timestamp values are missing or failed numeric conversion."),
    (re.compile(r"중복 timestamp (\d+)개"), lambda m: f"Duplicate timestamps: {m.group(1)}"),
    (re.compile(r"역순 timestamp (\d+)개"), lambda m: f"Reverse-order timestamps: {m.group(1)}"),
    (re.compile(r"중앙 sampling interval의 ([0-9.]+)배를 넘는 시간 간격이 (\d+)개입니다\."), lambda m: f"{m.group(2)} time gaps exceed {m.group(1)} times the median sampling interval."),
    (re.compile(r"무한대 값 (\d+)개가 있습니다\."), lambda m: f"The data contains {m.group(1)} infinite values."),
    (re.compile(r"(.+) 활성 구간이 없습니다\."), lambda m: f"No active {m.group(1)} segment was found."),
    (re.compile(r"구조화된 AI 응답이 없습니다\.?(.*)"), lambda m: f"No structured AI response was returned. {m.group(1).strip()}".strip()),
    (re.compile(r"AI 응답 schema 검증에 실패했습니다: (.+)"), lambda m: f"AI response schema validation failed: {m.group(1)}"),
    (re.compile(r"OpenAI API 호출에 실패했습니다: (.+)"), lambda m: f"OpenAI API call failed: {m.group(1)}"),
    (re.compile(r"원시 CSV 또는 행 데이터는 AI payload로 전달할 수 없습니다: (.+)"), lambda m: f"Raw CSV or row data cannot be sent in the AI payload: {m.group(1)}"),
    (re.compile(r"이름 토큰 유사도 ([0-9.]+)"), lambda m: f"Name-token similarity {m.group(1)}"),
    (re.compile(r"컬럼명 유사도 ([0-9.]+)"), lambda m: f"Column-name similarity {m.group(1)}"),
    (re.compile(r"주파수 분석에는 최소 (\d+)개의 유효 샘플이 필요합니다\."), lambda m: f"Frequency analysis requires at least {m.group(1)} valid samples."),
    (re.compile(r"평균 throttle correction 절댓값이 ([0-9.+-]+) us입니다\."), lambda m: f"Mean absolute throttle correction is {m.group(1)} us."),
    (re.compile(r"정상상태 고도 오차 절댓값이 ([0-9.+-]+) m입니다\."), lambda m: f"Absolute steady-state altitude error is {m.group(1)} m."),
    (re.compile(r"고도 오차 표준편차가 ([0-9.+-]+) m입니다\."), lambda m: f"Altitude-error standard deviation is {m.group(1)} m."),
    (re.compile(r"([0-9.+-]+) Hz peak가 PSD power의 유의한 부분을 차지합니다\."), lambda m: f"The {m.group(1)} Hz peak accounts for a significant share of PSD power."),
    (re.compile(r"correction 포화 비율이 ([0-9.+-]+)%입니다\."), lambda m: f"Correction saturation ratio is {m.group(1)}%."),
    (re.compile(r"주요 진동이 ([0-9.+-]+) Hz로 저주파입니다\."), lambda m: f"The dominant vibration is low-frequency at {m.group(1)} Hz."),
    (re.compile(r"연속 포화가 최대 ([0-9.+-]+)초 지속됩니다\."), lambda m: f"Continuous saturation lasts up to {m.group(1)} s."),
    (re.compile(r"포화가 분석 구간의 ([0-9.+-]+)%입니다\."), lambda m: f"Saturation occupies {m.group(1)}% of the analysis window."),
    (re.compile(r"settling time이 ([0-9.+-]+)초입니다\."), lambda m: f"Settling time is {m.group(1)} s."),
    (re.compile(r"센서 차이 표준편차가 ([0-9.+-]+) m입니다\."), lambda m: f"Sensor-difference standard deviation is {m.group(1)} m."),
    (re.compile(r"평균 motor PWM과 절대 센서 오차의 상관계수가 ([0-9.+-]+)입니다\."), lambda m: f"Correlation between mean motor PWM and absolute sensor error is {m.group(1)}."),
    (re.compile(r"주요 진동 주파수는 ([0-9.+-]+) Hz입니다\."), lambda m: f"Dominant vibration frequency is {m.group(1)} Hz."),
    (re.compile(r"Barometer-EKF 평균 bias가 ([0-9.+-]+) m입니다\."), lambda m: f"Mean Barometer-EKF bias is {m.group(1)} m."),
    (re.compile(r"추정 lag 절댓값이 ([0-9.+-]+)초입니다\."), lambda m: f"Absolute estimated lag is {m.group(1)} s."),
    (re.compile(r"최대 cross-correlation이 ([0-9.+-]+)입니다\."), lambda m: f"Maximum cross-correlation is {m.group(1)}."),
    (re.compile(r"sampling jitter ratio가 ([0-9.+-]+)입니다\."), lambda m: f"Sampling jitter ratio is {m.group(1)}."),
    (re.compile(r"큰 sampling gap 후보가 (\d+)개입니다\."), lambda m: f"There are {m.group(1)} large sampling-gap candidates."),
    (re.compile(r"중복 timestamp가 (\d+)개입니다\."), lambda m: f"There are {m.group(1)} duplicate timestamps."),
    (re.compile(r"역순 timestamp가 (\d+)개입니다\."), lambda m: f"There are {m.group(1)} reverse-order timestamps."),
    (re.compile(r"큰 시간 간격이 (\d+)개입니다\."), lambda m: f"There are {m.group(1)} large time gaps."),
    (re.compile(r"행별 모터 최대-최소 출력 차이 평균이 ([0-9.+-]+) us입니다\."), lambda m: f"Mean row-wise max-min motor-output difference is {m.group(1)} us."),
    (re.compile(r"모터 출력 spread가 ([0-9.+-]+) us입니다\."), lambda m: f"Motor-output spread is {m.group(1)} us."),
    (re.compile(r"가장 큰 모터 출력 표준편차가 ([0-9.+-]+) us입니다\."), lambda m: f"Largest motor-output standard deviation is {m.group(1)} us."),
    (re.compile(r"(.+): CSV에 '(.+)' 컬럼이 없습니다\."), lambda m: f"{m.group(1)}: column '{m.group(2)}' is missing from the CSV."),
    (re.compile(r"(.+): 숫자 변환 실패 (\d+)개"), lambda m: f"{m.group(1)}: {m.group(2)} values failed numeric conversion."),
]


def translate_text(value: Any, language: str) -> Any:
    """Translate a user-facing Korean result string; preserve technical values."""
    if language != "en" or not isinstance(value, str):
        return value
    if value in ENGLISH_TEXT:
        return ENGLISH_TEXT[value]
    for pattern, formatter in DYNAMIC_ENGLISH:
        match = pattern.fullmatch(value)
        if match:
            return formatter(match)
    return value


def localize_values(value: Any, language: str) -> Any:
    """Recursively translate string values while preserving stable JSON keys."""
    if isinstance(value, dict):
        return {key: localize_values(item, language) for key, item in value.items()}
    if isinstance(value, list):
        return [localize_values(item, language) for item in value]
    if isinstance(value, tuple):
        return tuple(localize_values(item, language) for item in value)
    return translate_text(value, language)
