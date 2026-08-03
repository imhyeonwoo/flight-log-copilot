# FlightLog Copilot

> AI-Powered UAV Flight Log Diagnosis

FlightLog Copilot combines a deterministic Python analysis engine with
optional OpenAI or OpenAI-compatible AI providers to detect anomalies, evaluate possible root causes, and
recommend evidence-based validation experiments from UAV flight logs.

FlightLog Copilot은 STM32 기반 UAV 비행제어기의 CSV 로그를 표준 파라미터로 매핑하고, Python으로 수치 지표와 이상 징후를 계산한 뒤, 설명 가능한 규칙으로 원인 가설의 진단 우선순위를 정하는 Streamlit 애플리케이션입니다. 외부 AI Provider 연동은 선택 기능입니다.

사이드바의 `KO / EN` 선택기로 화면, 진단 표현, AI 응답 언어와 다운로드 보고서를 한국어 또는 영어로 전환할 수 있습니다. 언어를 변경해도 업로드·확정 매핑·분석 결과는 유지됩니다.

## 문제 정의와 역할 분리

비행 로그의 컬럼 이름·단위·좌표계는 펌웨어마다 다릅니다. 이름만 보고 `pos_d`를 고도로 단정하거나 원시 CSV 전체를 모델에 맡기면 부호와 단위 오해가 진단 전체를 왜곡할 수 있습니다.

- Python 엔진: CSV 로딩, 매핑 변환, 구간 선택, 모든 정량 계산, 데이터 품질 검사, 규칙 기반 점수 계산
- AI Provider: Python이 만든 구조화 JSON만 받아 가설 관계와 우선순위를 설명하고 추가 검증 실험을 제안
- 사용자: 좌표계, 축 방향, 단위, 부호와 최종 매핑을 확인

원시 CSV와 개별 로그 행은 외부 AI 요청에 포함되지 않습니다. API 키가 없거나 호출이 실패해도 Python 분석과 보고서는 정상 동작합니다.

## 전체 흐름

```text
CSV upload
  -> column profile
  -> explainable auto-mapping
  -> user-confirmed mapping and transforms
  -> flight-segment selection
  -> deterministic metrics
  -> rule-based hypotheses
  -> optional structured AI-provider explanation
  -> Markdown / JSON / HTML reports
```

코드는 UI(`app.py`), 입출력·매핑·전처리, 정량 분석, 규칙 진단, Provider별 AI 클라이언트, 보고서 렌더러로 분리되어 있습니다. `src/flightlog_copilot/llm/` 아래에서 비밀 저장, 일반 설정, OpenAI Responses API, OpenAI Compatible Chat Completions, Factory와 응답 schema를 분리합니다.

## 설치와 실행

Python 3.9 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 샘플을 시험하려면 `sample_data/sample_alt_hold.csv`를 업로드합니다.

## AI Provider와 API 키 설정

사이드바의 `AI 설정`에서 다음 순서로 설정합니다.

1. `AI Copilot 활성화` 선택
2. `OpenAI` 또는 `OpenAI Compatible` Provider 선택
3. API 키 등록 또는 Compatible의 no-key 인증 방식 선택
4. 모델 선택 또는 모델 ID 직접 입력
5. 필요하면 `모델 목록 불러오기` 실행
6. `AI 연결 테스트` 실행
7. `AI 설정 저장` 실행
8. 정량 분석 뒤 `AI 진단 실행`

Provider별 설정은 독립적으로 유지됩니다.

| Provider | API 방식 | Provider별 설정 |
| --- | --- | --- |
| OpenAI | 공식 SDK Responses API와 Pydantic Structured Output | API 키, 모델 선택/직접 입력, temperature, 최대 출력 토큰 |
| OpenAI Compatible | `{base_url}/v1/chat/completions` | Base URL, API 키/no-key/Bearer, 직접 모델 ID, temperature, 최대 출력 토큰 |

Compatible Base URL은 `http://localhost:8000`, trailing slash, `/v1`, `/v1/` 입력을 모두 `http://localhost:8000/v1` 형태로 정규화합니다. `GET {base_url}/v1/models` 조회가 실패해도 직접 모델 ID 입력은 계속 사용할 수 있습니다. OpenAI 모델 드롭다운의 유지보수용 기본 목록은 `llm/models.py`에 있고, 목록에 없는 실제 모델 ID도 직접 입력할 수 있습니다.

### API 키 저장과 로딩 우선순위

UI에서 등록한 키는 Python `keyring`을 통해 운영체제 비밀 저장소에 저장합니다. Windows에서는 사용 가능한 경우 Windows Credential Manager가 사용됩니다. OpenAI Compatible 키는 정규화된 Base URL의 해시별로 분리됩니다.

실제 요청 키의 우선순위는 다음과 같습니다.

1. 현재 Streamlit 세션에서 새로 등록한 키
2. OS keyring 키
3. `OPENAI_API_KEY` 또는 `OPENAI_COMPATIBLE_API_KEY` 환경변수
4. 키 없음

keyring을 사용할 수 없거나 저장에 실패하면 평문 파일로 대체 저장하지 않고 현재 Streamlit 세션에만 유지합니다. 이 경우 앱 종료 후 다시 입력해야 합니다. 저장된 키는 입력창에 다시 채우지 않으며 화면에는 마지막 네 자리만 표시합니다. 환경변수 키는 앱에서 삭제하지 않으며 시스템 환경변수를 직접 변경해야 합니다.

API 키 삭제는 `저장된 API 키 삭제 확인`을 선택한 뒤 삭제 버튼을 누릅니다. API 키는 `ai_settings.json`, 매핑 프로필, 로그 분석 결과, Markdown/JSON/HTML 보고서에 포함되지 않습니다. 일반 AI 설정만 `platformdirs` 사용자 설정 디렉터리의 `FlightLogCopilot/ai_settings.json`에 atomic write로 저장됩니다. 손상된 설정 파일은 백업한 뒤 기본 설정으로 실행합니다.

환경변수 방식은 선택 fallback입니다. 필요하면 `.env.example`을 `.env`로 복사하되 실제 키를 Git에 커밋하지 마십시오.

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=
OPENAI_COMPATIBLE_API_KEY=
```

연결 테스트는 짧은 `OK` 요청만 보냅니다. 인증 오류, endpoint/모델 404, 요청 한도 429, 서버 오류, timeout과 네트워크 오류를 실제 키나 Authorization 헤더 없이 사용자 메시지로 변환합니다.

## CSV 파라미터

최소 분석 필수 파라미터:

- `timestamp`
- `ekf_altitude`

선택 파라미터:

- 고도 추종: `altitude_setpoint`, `throttle_correction`, `althold_active`
- 기준 고도 비교: `reference_altitude` (Barometer 또는 GNSS)
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

`reference_altitude`에는 EKF와 비교할 Barometer 또는 GNSS 고도를 지정합니다. 기준 고도 출처는 `auto`, `barometer`, `gnss`, `other` 중 하나로 프로필에 저장되며, `auto`는 선택한 원본 컬럼 이름으로 출처를 판별합니다. Barometer와 GNSS 후보가 모두 있거나 자동 판별이 모호하면 사용자가 직접 컬럼과 출처를 선택해야 합니다.

적용식은 다음과 같습니다.

```text
converted_value = sign × raw_value × scale + offset
```

그 뒤 시간은 초, 고도는 미터로 변환됩니다. 실제 분석에는 자동 후보가 아니라 사용자가 `매핑 확정`한 설정만 사용됩니다.

## 좌표계와 부호 주의사항

애플리케이션은 NED/ENU, FRD/FLU, 위/아래 양의 방향을 자동 확정하지 않습니다. `pos_d`와 `vel_d`는 Down 축일 수 있으므로 후보로는 보여도 자동 부호 반전하지 않습니다. 기체 정의를 확인한 뒤 사용자가 직접 `부호 반전`을 선택해야 합니다.

## 매핑 프로필

Step 2에서 현재 프로필을 JSON으로 다운로드하거나 과거 프로필을 업로드할 수 있습니다. 프로필에는 이름, 시간 단위, 좌표계, 기준 고도 출처, 헤더 해시, 컬럼·단위·scale·offset·부호 설정이 포함됩니다. 불러온 프로필의 컬럼이 현재 CSV에 없으면 경고하며, 같은 헤더 해시라면 일치 사실을 표시합니다. 기존 `barometer_altitude` 프로필은 불러올 때 `reference_altitude`와 `barometer` 출처로 자동 변환됩니다.

## 분석 지표

- 시간: sampling interval 평균·중앙값·표준편차, 주파수, jitter, 중복/역순 timestamp, dropout 후보
- 고도: RMSE, MAE, 평균/표준편차/95 percentile/정상상태 오차와 유효한 경우에만 step-response 지표
- 출력: throttle correction 포화, 양/음 포화, 연속 포화, 모터 평균·표준편차·spread·포화
- 센서: 기준 고도-EKF bias·분산·Pearson correlation·cross-correlation lag 및 보조 상관
- 주파수: 등간격 보간, detrend, Hann window, Welch PSD, 상위 peak와 Nyquist 정보

상관관계와 cross-correlation lag는 인과관계를 증명하지 않는다는 경고가 보고서에 포함됩니다.

## 규칙 기반 진단

12개 가설에 대해 근거, 반대 근거, 누락 파라미터, 한계, 권장 실험과 각 점수 증감 규칙을 출력합니다. 예:

```text
진단 우선순위 점수: 82/100
```

이 점수는 발생 확률이 아닙니다. 현재 로그에서 규칙 근거를 검토할 순서를 정하기 위한 설명 가능한 우선순위 점수입니다. AI Provider가 추가한 가설은 별도의 `미검증 가설` 영역에만 표시됩니다.

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

테스트는 컬럼 정규화와 별칭/모터 매핑, 기준 고도, 사용자 변환, 시간 단위, timestamp 품질, 정량 지표, 규칙 점수, Provider별 설정 독립성, Base URL 정규화, atomic 설정 저장과 손상 백업, keyring·세션·환경변수 우선순위, 키 마스킹·교체·삭제, Provider Factory, 모델 목록 fallback, HTTP 오류 분류, 구조화 AI 응답과 원시 데이터 차단을 다룹니다. 실제 외부 AI API는 호출하지 않습니다.

## 현재 한계

- 펌웨어별 별칭 사전은 알려진 일반 이름을 중심으로 하며 새 형식은 수동 매핑이 필요합니다.
- 자세, 배터리 전압, 적분기 내부 상태가 없으면 CG·추력·windup 원인을 완전히 분리할 수 없습니다.
- step-response 지표는 명확하고 지속되는 단일 setpoint 변화가 있을 때만 보수적으로 계산합니다.
- GNSS 기준 고도는 타원체/MSL 수직 datum과 위성 가시성에 따라 EKF와 일정한 오프셋 또는 노이즈 차이가 생길 수 있습니다.
- 그래프 구간 선택은 Streamlit 슬라이더 기반이며 자유형 brush selection은 제공하지 않습니다.
- 규칙 임계값은 초기 엔지니어링 기본값으로, 기체별 검증과 보정이 필요합니다.
- OpenAI Compatible 서버는 `/v1/chat/completions`와 JSON 객체 응답을 제공해야 하며, `/v1/models` 지원은 선택 사항입니다.
- 일부 모델이나 호환 서버는 temperature 또는 출력 토큰 옵션을 지원하지 않을 수 있으므로 연결 테스트에서 실제 설정을 확인해야 합니다.
- OS keyring backend가 없는 환경에서는 API 키가 세션에만 유지됩니다.

## 향후 계획

- 펌웨어별 검증된 alias/profile 라이브러리
- 기체 설정별 규칙 threshold preset과 회귀 평가 데이터셋
- attitude·battery·integrator 로그를 이용한 가설 분리 강화
- interactive brush selection과 보고서 차트 내장
- 반복 비행 간 비교 및 추세 분석
