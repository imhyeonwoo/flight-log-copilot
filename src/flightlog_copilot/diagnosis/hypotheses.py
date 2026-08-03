"""Supported diagnosis hypotheses and safe validation experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


HYPOTHESIS_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "incorrect_hover_pwm",
        "title": "Hover PWM 또는 기준 추력 부정확",
        "requires": ["throttle_correction"],
        "recommended_tests": ["안전하게 고정된 조건에서 hover 기준 출력을 작은 단계로 조정하고 정상상태 correction 평균을 비교합니다."],
    },
    {
        "id": "altitude_p_gain_high",
        "title": "Altitude P gain 과대",
        "requires": ["altitude_setpoint", "ekf_altitude"],
        "recommended_tests": ["P gain을 소폭 낮춘 A/B 로그에서 고도 오차 진동과 settling을 비교합니다."],
    },
    {
        "id": "altitude_i_gain_high",
        "title": "Altitude I gain 과대",
        "requires": ["altitude_setpoint", "ekf_altitude", "throttle_correction"],
        "recommended_tests": ["I gain을 소폭 낮추고 장주기 오차 및 correction 편향 변화를 비교합니다."],
    },
    {
        "id": "integrator_windup",
        "title": "적분기 windup",
        "requires": ["altitude_setpoint", "ekf_altitude", "throttle_correction"],
        "recommended_tests": ["적분기 상태와 anti-windup clamp를 로그에 추가하고 포화 해제 후 복귀 시간을 확인합니다."],
    },
    {
        "id": "throttle_correction_saturation",
        "title": "Throttle correction 포화",
        "requires": ["throttle_correction"],
        "recommended_tests": ["correction limit 및 실제 출력 한계를 확인하고 포화 구간의 setpoint·전압·추력을 함께 기록합니다."],
    },
    {
        "id": "reference_sensor_noise_or_vibration",
        "title": "기준 고도 센서 노이즈 또는 진동",
        "requires": ["reference_altitude", "ekf_altitude"],
        "recommended_tests": ["기준 고도 센서의 설치·수신 환경을 바꾸기 전후로 기준 고도-EKF 오차 PSD를 비교합니다."],
    },
    {
        "id": "reference_ekf_bias",
        "title": "기준 고도 또는 EKF altitude bias",
        "requires": ["reference_altitude", "ekf_altitude"],
        "recommended_tests": ["기준 고도의 원점·수직 datum을 확인하고 비행 전후의 기준 고도-EKF bias를 비교합니다."],
    },
    {
        "id": "sensor_control_delay",
        "title": "센서 또는 제어 응답 지연",
        "requires": ["reference_altitude", "ekf_altitude", "timestamp"],
        "recommended_tests": ["센서 원시값·필터 출력·제어기 출력을 동일 timestamp로 기록해 지연 위치를 분리합니다."],
    },
    {
        "id": "task_period_instability",
        "title": "Task 주기 불안정",
        "requires": ["timestamp"],
        "recommended_tests": ["제어 task 실행 시간과 deadline miss 카운터를 추가로 기록합니다."],
    },
    {
        "id": "timestamp_logging_problem",
        "title": "Timestamp 또는 로깅 문제",
        "requires": ["timestamp"],
        "recommended_tests": ["단조 clock과 sequence counter를 함께 기록해 누락·중복·재정렬을 확인합니다."],
    },
    {
        "id": "cg_asymmetry",
        "title": "CG 비대칭",
        "requires": ["motor_1", "motor_2", "motor_3", "motor_4"],
        "recommended_tests": ["기체 무게중심을 측정하고 동일 hover 조건의 모터 평균 출력을 비교합니다."],
    },
    {
        "id": "motor_propeller_imbalance",
        "title": "모터 또는 프로펠러 추력 불균형",
        "requires": ["motor_1", "motor_2", "motor_3", "motor_4"],
        "recommended_tests": ["모터/프로펠러 위치 교환 전후의 출력 편차와 진동 peak 이동을 비교합니다."],
    },
]


def new_hypotheses() -> Dict[str, Dict[str, Any]]:
    hypotheses: Dict[str, Dict[str, Any]] = {}
    for definition in deepcopy(HYPOTHESIS_DEFINITIONS):
        definition.update(
            {
                "score": 0,
                "evidence": [],
                "counter_evidence": [],
                "missing_parameters": [],
                "limitations": [],
                "score_details": [],
            }
        )
        hypotheses[definition["id"]] = definition
    return hypotheses
