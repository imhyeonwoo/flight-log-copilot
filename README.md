# FlightLog Copilot

> AI-Powered UAV Flight Log Diagnosis

FlightLog Copilot combines a deterministic Python analysis engine with
the OpenAI API to detect anomalies, evaluate possible root causes, and
recommend evidence-based validation experiments from UAV flight logs.

FlightLog Copilot은 STM32 기반 UAV 비행제어기의 CSV 로그를 표준 파라미터로 매핑하고, Python으로 수치 지표와 이상 징후를 계산한 뒤, 설명 가능한 규칙으로 원인 가설의 진단 우선순위를 정하는 Streamlit 애플리케이션입니다. OpenAI 연동은 선택 기능입니다.

사이드바의 `KO / EN` 선택기로 화면, 진단 표현, AI 응답 언어와 다운로드 보고서를 한국어 또는 영어로 전환할 수 있습니다. 언어를 변경해도 업로드·확정 매핑·분석 결과는 유지됩니다.

## 문제 정의와 역할 분리

비행 로그의 컬럼 이름·단위·좌표계는 펌웨어마다 다릅니다. 이름만 보고 `pos_d`를 고도로 단정하거나 원시 CSV 전체를 모델에 맡기면 부호와 단위 오해가 진단 전체를 왜곡할 수 있습니다.

- Python 엔진: CSV 로딩, 매핑 변환, 구간 선택, 모든 정량 계산, 데이터 품질 검사, 규칙 기반 점수 계산
- OpenAI API: Python이 만든 구조화 JSON만 받아 가설 관계와 우선순위를 설명하고 추가 검증 실험을 제안
- 사용자: 좌표계, 축 방향, 단위, 부호와 최종 매핑을 확인

원시 CSV와 개별 로그 행은 OpenAI 요청에 포함되지 않습니다. API 키가 없거나 호출이 실패해도 Python 분석과 보고서는 정상 동작합니다.

## 전체 흐름

```text
CSV upload
  -> column profile
  -> explainable auto-mapping
  -> user-confirmed mapping and transforms
  -> flight-segment selection
  -> deterministic metrics
  -> rule-based hypotheses
  -> optional structured OpenAI explanation
  -> Markdown / JSON / HTML reports
```

코드는 UI(`app.py`), 입출력·매핑·전처리, 정량 분석, 규칙 진단, OpenAI 연동, 보고서 렌더러로 분리되어 있습니다.

## 설치와 실행

Python 3.9 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 샘플을 시험하려면 `sample_data/sample_alt_hold.csv`를 업로드합니다.

## OpenAI API 설정

`.env.example`을 `.env`로 복사하고 키를 입력합니다. `.env`는 Git에서 제외됩니다.

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.6-terra
```

모델은 Streamlit의 AI Copilot 단계에서도 변경할 수 있습니다. SDK는 Responses API의 Pydantic Structured Output을 사용하며, 응답이 schema를 통과하지 못하면 AI 결과만 실패 처리합니다.

## CSV 파라미터

최소 분석 필수 파라미터:

- `timestamp`
- `ekf_altitude`

선택 파라미터:

- 고도 추종: `altitude_setpoint`, `throttle_correction`, `althold_active`
- Barometer 비교: `barometer_altitude`
- 출력: `throttle_base`, `motor_1`~`motor_4`
- 구간 감지: `armed`, `althold_active`
- 기타: `vertical_velocity`

파라미터가 없으면 전체 프로그램을 중단하지 않고 해당 분석을 `계산 불가`로 표시합니다.

## 자동 매핑과 수동 매핑

자동 매핑은 정규화 이름, 별칭, 토큰 유사도, dtype, 값 범위를 조합해 후보와 신뢰도, 근거를 제공합니다. 값 범위만으로 물리 의미를 확정하지 않으며 낮은 점수나 비슷한 후보가 있으면 `확인 필요`로 남깁니다.

각 표준 파라미터에서 다음을 최종 지정할 수 있습니다.

- CSV 컬럼 또는 `사용하지 않음` 또는 `직접 입력`
- 원본 단위
- scale과 offset
- 부호 반전

적용식은 다음과 같습니다.

```text
converted_value = sign × raw_value × scale + offset
```

그 뒤 시간은 초, 고도는 미터로 변환됩니다. 실제 분석에는 자동 후보가 아니라 사용자가 `매핑 확정`한 설정만 사용됩니다.

## 좌표계와 부호 주의사항

애플리케이션은 NED/ENU, FRD/FLU, 위/아래 양의 방향을 자동 확정하지 않습니다. `pos_d`와 `vel_d`는 Down 축일 수 있으므로 후보로는 보여도 자동 부호 반전하지 않습니다. 기체 정의를 확인한 뒤 사용자가 직접 `부호 반전`을 선택해야 합니다.

## 매핑 프로필

Step 2에서 현재 프로필을 JSON으로 다운로드하거나 과거 프로필을 업로드할 수 있습니다. 프로필에는 이름, 시간 단위, 좌표계, 헤더 해시, 컬럼·단위·scale·offset·부호 설정이 포함됩니다. 불러온 프로필의 컬럼이 현재 CSV에 없으면 경고하며, 같은 헤더 해시라면 일치 사실을 표시합니다.

## 분석 지표

- 시간: sampling interval 평균·중앙값·표준편차, 주파수, jitter, 중복/역순 timestamp, dropout 후보
- 고도: RMSE, MAE, 평균/표준편차/95 percentile/정상상태 오차와 유효한 경우에만 step-response 지표
- 출력: throttle correction 포화, 양/음 포화, 연속 포화, 모터 평균·표준편차·spread·포화
- 센서: Barometer-EKF bias·분산·Pearson correlation·cross-correlation lag 및 보조 상관
- 주파수: 등간격 보간, detrend, Hann window, Welch PSD, 상위 peak와 Nyquist 정보

상관관계와 cross-correlation lag는 인과관계를 증명하지 않는다는 경고가 보고서에 포함됩니다.

## 규칙 기반 진단

12개 가설에 대해 근거, 반대 근거, 누락 파라미터, 한계, 권장 실험과 각 점수 증감 규칙을 출력합니다. 예:

```text
진단 우선순위 점수: 82/100
```

이 점수는 발생 확률이 아닙니다. 현재 로그에서 규칙 근거를 검토할 순서를 정하기 위한 설명 가능한 우선순위 점수입니다. GPT가 추가한 가설은 별도의 `미검증 가설` 영역에만 표시됩니다.

## 샘플 실행

`sample_data/sample_alt_hold.csv`는 25 Hz, 0.3 Hz 고도 진동, AltHold 활성 구간과 약한 모터 출력 비대칭을 포함한 합성 로그입니다.

1. 샘플 CSV 업로드
2. 자동 추천을 검토하고 모든 필요한 매핑 선택
3. 시간 단위 `seconds`, 고도 단위 `meters` 확인
4. `매핑 확정`
5. 자동 AltHold 구간 선택 후 `정량 분석 실행`

Welch 분석의 주요 peak는 bin 해상도 범위에서 약 0.3 Hz로 검출되어야 합니다. 실제 수치는 선택 구간과 설정에 따라 달라집니다.

## 테스트

```powershell
python -m pytest
```

테스트는 컬럼 정규화와 별칭/모터 매핑, 모호한 매핑, 사용자 변환, 시간 단위, timestamp 품질, RMSE, 출력 포화, 25 Hz 합성 로그의 0.3 Hz Welch peak, 누락 파라미터, 규칙 점수, GPT schema 및 mock 응답을 다룹니다. 실제 OpenAI API는 호출하지 않습니다.

## 현재 한계

- 펌웨어별 별칭 사전은 알려진 일반 이름을 중심으로 하며 새 형식은 수동 매핑이 필요합니다.
- 자세, 배터리 전압, 적분기 내부 상태가 없으면 CG·추력·windup 원인을 완전히 분리할 수 없습니다.
- step-response 지표는 명확하고 지속되는 단일 setpoint 변화가 있을 때만 보수적으로 계산합니다.
- 그래프 구간 선택은 Streamlit 슬라이더 기반이며 자유형 brush selection은 제공하지 않습니다.
- 규칙 임계값은 초기 엔지니어링 기본값으로, 기체별 검증과 보정이 필요합니다.

## 향후 계획

- 펌웨어별 검증된 alias/profile 라이브러리
- 기체 설정별 규칙 threshold preset과 회귀 평가 데이터셋
- attitude·battery·integrator 로그를 이용한 가설 분리 강화
- interactive brush selection과 보고서 차트 내장
- 반복 비행 간 비교 및 추세 분석
