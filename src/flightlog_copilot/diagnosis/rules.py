"""Rule-based evaluation of all supported UAV root-cause candidates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from flightlog_copilot.diagnosis.hypotheses import new_hypotheses
from flightlog_copilot.diagnosis.scoring import apply_rule


def evaluate_hypotheses(metrics: Dict[str, Any], available_parameters: Iterable[str]) -> List[Dict[str, Any]]:
    hypotheses = new_hypotheses()
    available = set(available_parameters)
    altitude = metrics.get("altitude", {})
    actuator = metrics.get("actuator", {})
    sensor = metrics.get("sensor_comparison", {})
    frequency = metrics.get("frequency", {})
    timing = metrics.get("timing", {})

    _score_hover(hypotheses["incorrect_hover_pwm"], altitude, actuator)
    _score_p_gain(hypotheses["altitude_p_gain_high"], altitude, frequency)
    _score_i_gain(hypotheses["altitude_i_gain_high"], altitude, actuator, frequency)
    _score_windup(hypotheses["integrator_windup"], altitude, actuator)
    _score_saturation(hypotheses["throttle_correction_saturation"], actuator)
    _score_reference_noise(hypotheses["reference_sensor_noise_or_vibration"], sensor, frequency)
    _score_bias(hypotheses["reference_ekf_bias"], sensor)
    _score_delay(hypotheses["sensor_control_delay"], sensor)
    _score_task(hypotheses["task_period_instability"], timing)
    _score_timestamp(hypotheses["timestamp_logging_problem"], timing)
    _score_cg(hypotheses["cg_asymmetry"], actuator)
    _score_motor(hypotheses["motor_propeller_imbalance"], actuator, frequency)

    for hypothesis in hypotheses.values():
        hypothesis["missing_parameters"] = sorted(set(hypothesis.pop("requires", [])) - available)
        if hypothesis["missing_parameters"]:
            hypothesis["limitations"].append("필요한 로그 파라미터가 부족하여 점수가 과소평가될 수 있습니다.")
        hypothesis["score"] = max(0, min(100, int(hypothesis["score"])))
    return sorted(hypotheses.values(), key=lambda item: (-item["score"], item["id"]))


def _value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key) if metrics.get("available") else None
    return float(value) if isinstance(value, (int, float)) else None


def _score_hover(h: Dict[str, Any], altitude: Dict[str, Any], actuator: Dict[str, Any]) -> None:
    correction = _value(actuator, "mean_throttle_correction_pwm")
    steady = _value(altitude, "steady_state_error_m")
    apply_rule(h, None if correction is None else abs(correction) >= 25, 45, "지속 correction 편향", f"평균 throttle correction 절댓값이 {abs(correction or 0):.1f} us입니다.", "평균 throttle correction 편향이 작습니다.", correction, ">= 25 us")
    apply_rule(h, None if steady is None else abs(steady) >= 0.15, 25, "정상상태 고도 오차", f"정상상태 고도 오차 절댓값이 {abs(steady or 0):.3f} m입니다.", "정상상태 고도 오차가 작습니다.", steady, ">= 0.15 m")


def _score_p_gain(h: Dict[str, Any], altitude: Dict[str, Any], frequency: Dict[str, Any]) -> None:
    error_std = _value(altitude, "error_std_m")
    peak_ratio = _value(frequency, "dominant_peak_power_ratio")
    dominant = _value(frequency, "dominant_frequency_hz")
    apply_rule(h, None if error_std is None else error_std >= 0.2, 30, "고도 오차 분산", f"고도 오차 표준편차가 {error_std or 0:.3f} m입니다.", "고도 오차 변동이 작습니다.", error_std, ">= 0.2 m")
    oscillatory = None if peak_ratio is None or dominant is None else peak_ratio >= 0.15 and 0.15 <= dominant <= 3.0
    apply_rule(h, oscillatory, 45, "오차 진동 peak", f"{dominant or 0:.3f} Hz peak가 PSD power의 유의한 부분을 차지합니다.", "뚜렷한 제어 대역 진동 peak가 없습니다.", peak_ratio, ">= 0.15")


def _score_i_gain(h: Dict[str, Any], altitude: Dict[str, Any], actuator: Dict[str, Any], frequency: Dict[str, Any]) -> None:
    saturation = _value(actuator, "correction_saturation_ratio")
    dominant = _value(frequency, "dominant_frequency_hz")
    apply_rule(h, None if saturation is None else saturation >= 0.1, 30, "correction 포화", f"correction 포화 비율이 {(saturation or 0) * 100:.1f}%입니다.", "correction 포화 비율이 낮습니다.", saturation, ">= 0.1")
    apply_rule(h, None if dominant is None else 0 < dominant <= 0.5, 25, "저주파 진동", f"주요 진동이 {dominant or 0:.3f} Hz로 저주파입니다.", "주요 진동이 저주파 적분 거동과 일치하지 않습니다.", dominant, "<= 0.5 Hz")
    h["limitations"].append("I gain과 적분기 상태가 직접 기록되지 않아 간접 지표만 사용했습니다.")


def _score_windup(h: Dict[str, Any], altitude: Dict[str, Any], actuator: Dict[str, Any]) -> None:
    duration = _value(actuator, "max_continuous_saturation_s")
    saturation = _value(actuator, "correction_saturation_ratio")
    settling = _value(altitude, "settling_time_s")
    apply_rule(h, None if duration is None else duration >= 0.5, 40, "연속 포화", f"연속 포화가 최대 {duration or 0:.2f}초 지속됩니다.", "긴 연속 포화가 없습니다.", duration, ">= 0.5 s")
    apply_rule(h, None if saturation is None else saturation >= 0.2, 25, "포화 점유율", f"포화가 분석 구간의 {(saturation or 0) * 100:.1f}%입니다.", "포화 점유율이 낮습니다.", saturation, ">= 0.2")
    apply_rule(h, None if settling is None else settling >= 3.0, 15, "느린 복귀", f"settling time이 {settling or 0:.2f}초입니다.", "settling이 빠릅니다.", settling, ">= 3 s")
    h["limitations"].append("적분기 내부 상태가 없으므로 windup을 확정할 수 없습니다.")


def _score_saturation(h: Dict[str, Any], actuator: Dict[str, Any]) -> None:
    ratio = _value(actuator, "correction_saturation_ratio")
    apply_rule(h, None if ratio is None else ratio >= 0.05, 70, "correction limit 도달", f"correction 포화 비율이 {(ratio or 0) * 100:.1f}%입니다.", "correction 포화가 거의 없습니다.", ratio, ">= 0.05")


def _score_reference_noise(h: Dict[str, Any], sensor: Dict[str, Any], frequency: Dict[str, Any]) -> None:
    difference_std = _value(sensor, "difference_std_m")
    motor_corr = _value(sensor, "mean_motor_pwm_vs_abs_error_correlation")
    dominant = _value(frequency, "dominant_frequency_hz")
    apply_rule(h, None if difference_std is None else difference_std >= 0.15, 30, "기준 고도-EKF 차이 변동", f"센서 차이 표준편차가 {difference_std or 0:.3f} m입니다.", "센서 차이 변동이 작습니다.", difference_std, ">= 0.15 m")
    apply_rule(h, None if motor_corr is None else motor_corr >= 0.35, 35, "모터 출력과 센서 오차 상관", f"평균 motor PWM과 절대 센서 오차의 상관계수가 {motor_corr or 0:.2f}입니다.", "모터 출력과 센서 오차의 상관이 약합니다.", motor_corr, ">= 0.35")
    apply_rule(h, None if dominant is None else dominant >= 1.0, 15, "진동 대역 peak", f"주요 진동 주파수는 {dominant or 0:.2f} Hz입니다.", "고주파 진동 peak 근거가 약합니다.", dominant, ">= 1 Hz")


def _score_bias(h: Dict[str, Any], sensor: Dict[str, Any]) -> None:
    bias = _value(sensor, "reference_ekf_bias_m")
    spread = _value(sensor, "difference_std_m")
    apply_rule(h, None if bias is None else abs(bias) >= 0.2, 55, "평균 센서 bias", f"기준 고도-EKF 평균 bias가 {bias or 0:.3f} m입니다.", "평균 센서 bias가 작습니다.", bias, ">= 0.2 m absolute")
    stable_bias = None if bias is None or spread is None else abs(bias) >= 0.2 and spread < abs(bias)
    apply_rule(h, stable_bias, 20, "변동 대비 지속 bias", "bias가 차이의 변동보다 커서 지속 오프셋 형태입니다.", "bias가 변동에 비해 지배적이지 않습니다.", spread, "std < abs(bias)")
    h["limitations"].append("기준 고도의 원점 또는 수직 datum 차이도 지속 bias를 만들 수 있습니다.")


def _score_delay(h: Dict[str, Any], sensor: Dict[str, Any]) -> None:
    lag = _value(sensor, "estimated_lag_s")
    correlation = _value(sensor, "max_cross_correlation")
    apply_rule(h, None if lag is None else abs(lag) >= 0.1, 50, "cross-correlation lag", f"추정 lag 절댓값이 {abs(lag or 0):.3f}초입니다.", "추정 lag가 작습니다.", lag, ">= 0.1 s absolute")
    apply_rule(h, None if correlation is None else correlation >= 0.5, 15, "lag 추정 일관성", f"최대 cross-correlation이 {correlation or 0:.2f}입니다.", "cross-correlation이 약해 lag 해석이 제한됩니다.", correlation, ">= 0.5")
    h["limitations"].append("cross-correlation lag는 공통 입력과 필터 특성의 영향도 받습니다.")


def _score_task(h: Dict[str, Any], timing: Dict[str, Any]) -> None:
    jitter = _value(timing, "sampling_jitter_ratio")
    dropouts = timing.get("dropout_intervals", []) if timing.get("available") else []
    apply_rule(h, None if jitter is None else jitter >= 0.1, 50, "sampling jitter", f"sampling jitter ratio가 {jitter or 0:.3f}입니다.", "sampling jitter가 낮습니다.", jitter, ">= 0.1")
    apply_rule(h, None if not timing.get("available") else len(dropouts) > 0, 20, "dropout 후보", f"큰 sampling gap 후보가 {len(dropouts)}개입니다.", "큰 sampling gap 후보가 없습니다.", len(dropouts), "> 0")
    h["limitations"].append("로그 timestamp 변동과 실제 제어 task 주기 변동은 서로 다를 수 있습니다.")


def _score_timestamp(h: Dict[str, Any], timing: Dict[str, Any]) -> None:
    duplicate = timing.get("duplicate_timestamp_count") if timing else None
    reverse = timing.get("reverse_timestamp_count") if timing else None
    dropouts = timing.get("dropout_intervals") if timing.get("available") else None
    apply_rule(h, None if duplicate is None else duplicate > 0, 30, "중복 timestamp", f"중복 timestamp가 {duplicate or 0}개입니다.", "중복 timestamp가 없습니다.", duplicate, "> 0")
    apply_rule(h, None if reverse is None else reverse > 0, 35, "역순 timestamp", f"역순 timestamp가 {reverse or 0}개입니다.", "역순 timestamp가 없습니다.", reverse, "> 0")
    apply_rule(h, None if dropouts is None else len(dropouts) > 0, 25, "logging dropout 후보", f"큰 시간 간격이 {len(dropouts or [])}개입니다.", "큰 시간 간격이 없습니다.", len(dropouts or []), "> 0")


def _score_cg(h: Dict[str, Any], actuator: Dict[str, Any]) -> None:
    spread = _value(actuator, "mean_motor_spread_pwm")
    apply_rule(h, None if spread is None else spread >= 80, 50, "모터 평균 spread", f"행별 모터 최대-최소 출력 차이 평균이 {spread or 0:.1f} us입니다.", "모터 출력 spread가 작습니다.", spread, ">= 80 us")
    h["limitations"].append("CG 비대칭과 모터 추력 차이를 자세·전류 로그 없이 분리할 수 없습니다.")


def _score_motor(h: Dict[str, Any], actuator: Dict[str, Any], frequency: Dict[str, Any]) -> None:
    spread = _value(actuator, "mean_motor_spread_pwm")
    motor_metrics = actuator.get("motor_metrics", {}) if actuator.get("available") else {}
    max_std = max((float(values.get("std_pwm", 0)) for values in motor_metrics.values()), default=None)
    apply_rule(h, None if spread is None else spread >= 100, 45, "모터 출력 불균형", f"모터 출력 spread가 {spread or 0:.1f} us입니다.", "모터 출력 spread가 작습니다.", spread, ">= 100 us")
    apply_rule(h, None if max_std is None else max_std >= 60, 25, "모터 출력 변동", f"가장 큰 모터 출력 표준편차가 {max_std or 0:.1f} us입니다.", "모터 출력 변동이 작습니다.", max_std, ">= 60 us")
    h["limitations"].append("PWM 명령만으로 실제 모터/프로펠러 추력을 직접 측정할 수 없습니다.")
