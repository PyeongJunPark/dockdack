# Mark1.5–1.7 동결 추론 묶음

`models/mark1_series/`는 2026-09-27 완료된 CUDA 연구 실행(`outputs/mark1/mark1-series-100x2-contractfix-20260927/`)의 시장별 선택 시드 6개와 2021년 보정 기준값을 그대로 복사한 읽기 전용 추론 산출물이다. `manifest.json`·`manifest.sha256`·개별 가중치 SHA-256·학습 코드 SHA-256을 검사하며, 원본 SQLite와 주문 API는 추론 시 열지 않는다. 이 묶음을 만든 명령은 `python -m scripts.export_mark1_series_bundle`이다. 기존 출력 폴더를 임의로 덮어쓰지 않는다.

| 모델 | 국내 선택 | 미국 선택 | 출력 |
|---|---|---|---|
| Mark1.5 | LSTM 시드 42 | LSTM 시드 41 | 비용 후 다음 날 시가→종가 순수익률 예측, % |
| Mark1.6 | CNN 시드 42 | CNN 시드 43 | 비용 후 양의 수익 가능성 점수, 0–1; 보정된 승률 아님 |
| Mark1.7 | 같은 날짜 전종목 attention 시드 41 | 시드 41 | 맥락 의존 상대 순위 점수; 확률 아님 |

각 시장별 실제 숫자 기준값과 가중치 SHA-256은 manifest에 기록했다. 전략 식별자는 각각 `mark1-5-prototype`, `mark1-6-prototype`, `mark1-7-prototype`이다. `signal_phase=preopen`, `exit_after_sessions=0`, `exit_timing=preclose`는 *매매 의도 메타데이터*일 뿐, 이 모듈이 주문하거나 청산하는 기능은 아니다. 2022년 개발 구간에서 6개 조합 모두 왕복 20bp 차감 후 손실이었으므로 `research_qualified=false`, `deployment_allowed=false`로 고정했다.

```python
from dockdack.mark1_series_inference import MarkSeriesPredictor

predictor = MarkSeriesPredictor("models/mark1_series", "domestic", "mark1.5")
results = predictor.score_many(windows, [("005930", "KRX"), ("000660", "KRX")])
```

`windows`는 위 순서와 같은 30개 **완료된** 일봉 OHLCV 배열 `[종목, 30, 5]`다. 호출자가 목표 거래일·봉 완료 여부·현재 감시 후보 전체를 확인해야 한다. 이 **추론 API 자체**에는 호가, 현재가, 계좌 또는 주문 기능이 없다. 후속 일반 GUI 연결은 별도 장전 신호 작업자와 주문 엔진을 통해 이 API를 사용하며 모의 전용이다. 특히 Mark1.7은 *한 목표 거래일의 전체 적격 종목*을 한 번의 `score_many` 호출로 보내야 한다. 종목을 나눠 호출하면 attention 맥락과 점수가 달라지므로 `score_one`은 거부한다. 점수와 동결 기준값은 모델별·시장별로 비교하고 서로 평균내지 않는다.

실행에는 **Python 3.11 이상, NumPy 2.x, PyTorch 2.6 이상**이 필요하다. 이 컴퓨터의 `DockDack.vbs`는 우선 `.venv-ml-cuda/Scripts/pythonw.exe`를 사용하며, 그 환경에서 Python 3.13.5·PyTorch 2.11.0+cu128을 확인했다. CPU 추론은 GPU를 요구하지 않지만 PyTorch 패키지는 필요하다. VBS의 최종 대체 환경 `.venv`에 PyTorch가 없다면 이 모델의 추론 작업자는 시작하지 못하므로 GUI 연결 시 환경 점검과 실패 표시가 필요하다. `pyproject.toml`의 기본 의존성에는 PyTorch가 없고 `ml`/`prototype` 추가 의존성에만 있다.

`tests/test_mark1_series_inference.py`는 국내 92종목·미국 100종목의 실제 같은 날짜 교차단면에서 저장된 GPU 점수와 CPU 추론을 대조한다. CNN·attention은 절대/상대 `3e-5` 이내, LSTM은 CPU/CuDNN 수치차를 감안해 `1e-3` 이내이며 그 기준일의 원시 통과 판정은 모두 일치했다. 전체 역사 후보를 같은 512 배치로 재검산한 수치는 아래와 같다.

| 시장·모델 | 후보 수 | 최대 CPU↔GPU 점수차 | 원시 기준 통과 반전 | 추론 수치 경계폭 |
|---|---:|---:|---:|---:|
| 국내 1.5 | 154,216 | 0.0048404 | 4 | ±0.005 |
| 미국 1.5 | 172,680 | 0.0026769 | 3 | ±0.003 |
| 국내 1.6 | 154,216 | 0.00001252 | 1 | ±0.00002 |
| 미국 1.6 | 172,680 | 0.00001404 | 3 | ±0.00002 |
| 국내 1.7 | 154,216 | 0.000000477 | 0 | ±0.000001 |
| 미국 1.7 | 172,680 | 0.000000596 | 0 | ±0.000001 |

각 경계폭은 **관측된 최대 차이보다 위로 올림**한 값이며 미래 점수에 대한 수학적 상한은 아니다. 추론은 원래 `score`와 `raw_above_frozen_threshold`를 그대로 반환하지만, 점수가 기준값±경계폭 안에 들면 `uncertain_numeric_boundary=true`, `above_frozen_threshold=false`, `signal_decision=HOLD_NUMERIC_BOUNDARY`로 처리한다. 그 밖에서만 `BUY_CANDIDATE` 또는 `HOLD`를 반환한다. 따라서 이 CPU 신호는 GPU 백테스트와 **완전히 동일하지 않으며**, 안전 경계 때문에 일부 매수 후보를 추가로 보류한다. 현재 카탈로그 생존편향, 가상 시가·종가 체결, 비블라인드 2022–24 평가라는 연구 제한도 유지된다.
