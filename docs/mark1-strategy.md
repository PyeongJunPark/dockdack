# mark_1: 30개 완료 일봉 + 현재가 질문

이 브랜치는 정제 DB에 기반한 **연구용 1차 모델 비교**다. 모의 자동매매를 켜거나 주문한 결과가 아니다. [실측 비교 보고서](../reports/mark1-20260916/REPORT.md), [GUI 실행 안내](MARK1_GUI.md)를 함께 읽는다.

## 학습 목표와 매매 조건

- 31번째 토큰은 진행 중인 날의 **후보 진입가**다. 앞의 30개는 완료된 거래일의 OHLCV다.
- 지도학습 정답은 `당일 고가 >= 후보가 × 1.01 AND 당일 저가 > 후보가 × 0.991`이다.
- 두 경계를 모두 건드린 일봉은 실패다. 이는 **익절이 손절보다 먼저**라는 정답과 다르다. 일봉만으로 순서를 복원할 수 없어서 보수적으로 정한 하루 전체 사건이다.
- 사용자가 **과거 30일 + 당일 가상 매수가**, **양쪽 경계 도달 시 손절 우선**을 확인했다. 기존 라벨이 이 보수적 규칙과 일치하므로 가중치·문턱 재학습 없이 [전체 평가 기간 포트폴리오 백테스트](../reports/mark1-backtest-20260916/REPORT.md)를 진행했다. 당일 미도달 포지션의 보유/종가 청산은 구분해서 제시한다.
- 미보유일 때 보정된 모델 확률 **> 0.5**이면 BUY, 정확히 0.5이면 HOLD다.
- 보유 시 실제 평균 매입가 대비 현재가 **+1% 이상 / −0.9% 이하**이면 SELL이다. 매도는 학습할 확률이 아니라 명확한 가격 규칙이다.
- 현재가로 생성한 BUY는 주문 직전 새 현재가에서도 모델을 다시 확인한다. 조회와 체결 사이 가격 변화, 틱 단위 반올림, 갭, 슬리피지로 실제 손익이 경계를 넘을 수 있다.
- +1%/−0.9%는 비용 차감 전 가격 변화다. 자동주문과 감시는 시작 시 OFF다.

**중요:** 현재가가 후보가로 입력될 수 있지만, 평가 기준은 실제 시가다. 장중 진입 이후의 남은 시간에 대한 50% 확률로 검증된 것은 아니다. 분봉/틱과 진입 시각이 있어야 해당 목표와 실제 체결을 추가 검증할 수 있다.

## 가격 증강

학습 표본마다 실제 시가의 `[0.99, 0.995, 1, 1.005, 1.01]`을 후보가로 넣고 후보가마다 정답을 다시 만든다. 후보가를 미래 고가/저가에 맞춰 선택하지 않는다. 표본의 모든 복제본은 같은 학습 구간에 남긴다.

당일 최종 고가·저가·종가·거래량은 **라벨 계산에만 사용**하며 입력하지 않는다. 관측되지 않은 후보가격이 존재하므로 증강은 실제 거래 5개가 아니라 반사실적 질의 5개다. 가격만 바꾸고 과거 30봉을 같은 값으로 보존한다.

MLP, LSTM, GRU, TCN, Transformer를 동일 조건으로 비교한다. 가격 증강의 효과를 분리하려고 같은 원본 시가를 5번 반복한 MLP 대조군도 학습한다. 결과가 나쁘더라도 숨기지 않고, 증강 대조군이 이기면 그 모델을 선택한다.

## 정제 데이터와 누수 방지

실험 입력 파일:

```text
C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1/domestic_daily_clean.sqlite3
C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1/us_daily_clean.sqlite3
```

DB를 읽기 전용으로 열고, `clean-daily-v1`의 승인 `training_samples`에 등록된 30+1 거래일을 확인한다. 달력 연속성, 세그먼트, 가격·거래량·통화를 검증한다. 과거 다음 종가 라벨 `target_up`은 재사용하지 않는다. 학습기간 승인 이력이 있는 종목만 선택한다.

원본/정제 DB는 덮어쓰지 않는다. 캐시는 해시·정제본 경로·기간·표본 상한·시드를 키로 사용한다. 비어 있지 않은 WAL을 거부하고 읽기 전후 DB SHA-256을 비교한다. DB와 거대한 캐시는 Git에 포함하지 않는다.

| 용도 | 대상 기간 | 시장별 상한 | 사용 방식 |
|---|---|---:|---|
| 학습 | 2010–2021 | 원본 200,000개 | 가격 5개, epoch당 1,000,000개 제시 |
| epoch 선택 | 2022 | 60,000개 | 실제 시가만, log loss 조기 종료 |
| 확률 보정 | 2023 | 60,000개 | 양의 기울기 Platt 보정 |
| 모델 선택 | 2024 | 60,000개 | Brier, 동률 log loss |
| 최종 평가 | 2025 이후 | 60,000개 | 모델 선택 완료 후 한 번 평가 |

각 경계에서 과거 입력 30봉이 이전 기간에 걸친 표본은 제거한다. 표본은 고정 시드로 무작위 추출하며 라벨로 비율을 맞추지 않는다. 학습 사건 수, 증강 제시 수, 모델 파라미터 수는 서로 다른 수치다. 이번 결과는 단일 시드·작은 모델·제한된 epoch의 비교로, 모든 딥러닝 중 최적이라는 뜻이 아니다.

## 실행

현재 컴퓨터에서는 RTX 5080을 지원하는 기존 CUDA 환경을 사용했다. 기본 `.venv`는 CPU PyTorch여서 학습 명령에 CUDA 환경을 명시한다. 저장소 루트는 `C:/Users/user/Desktop/dockdack-mark_1`이다.

```powershell
& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.train_mark1 `
  --db-dir 'C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1' `
  --output-dir outputs/mark1/my-new-run --device cuda

& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.report_mark1 `
  --run outputs/mark1/experiment-20260916 --destination reports/mark1-20260916
```

학습은 이미 존재하는 결과 폴더에 덮어쓰지 않으므로 새 이름을 사용한다. 기본 6 epoch 상한/patience 2이며 CUDA가 없으면 CPU로 몰래 전환하지 않고 종료한다. CPU가 필요하면 `--device cpu`를 명시한다. GPU 최대 메모리 할당 비율은 65%이며 모든 모델을 순차 실행한다. Windows CUDA RNN 종료 문제를 피하려고 LSTM/GRU의 cuDNN만 해당 호출 안에서 비활성화한다.

추론은 GUI 없이도 가능하고, 모델을 불러오기만 해서는 주문하지 않는다.

```python
from decimal import Decimal
from dockdack.mark1_inference import Predictor

predictor = Predictor('models/mark1/domestic.pt', device='cpu')
# history_30: 날짜 오름차순의 완료 30일 [시가, 고가, 저가, 종가, 거래량]
result = predictor.predict(history_30, current_price=Decimal('70000'))
print(result['probability_success'], result['predicts_success'])
```

직접 추론 API에는 날짜가 없으므로 호출자가 30개 완료 거래일·종목·시장 일치를 보장해야 한다. GUI 연결 어댑터는 거래일 달력과 날짜를 별도로 검사한다. 체크포인트의 시장·특징·타깃·문턱값·익절/손절·보정·가중치를 검증하여 mark0 체크포인트와 혼용하지 않는다.

## 해석 제한

확률 보정이란 예측 60%인 사례가 실제로도 비슷한 비율로 성공하는지 확인하는 절차다. 보정용 데이터와 모델 학습 데이터를 분리했어도 미래의 시장 변화에는 어긋날 수 있다. Brier와 log loss는 보정뿐 아니라 분별력에도 영향을 받는다. [scikit-learn 공식 설명](https://scikit-learn.org/stable/modules/calibration.html)

일봉 기반 매매 모의실험은 봉 안의 가격 순서와 체결을 가정해야 한다. 이 보고서는 이를 실제 체결로 부르지 않는다. [TradingView 공식 전략 설명](https://www.tradingview.com/pine-script-docs/concepts/strategies/)

저유동성 정제는 유통주식수의 완전한 역사적 검증이 아니다. 현재 종목목록 생존 편향·조정주가·상장 이후 시점·타깃 관측 가능성 문제도 남는다. 분봉 검증, 여러 시드/기간의 반복 검증과 비용·체결 모델을 추가하기 전 실거래에 적합하다고 판단할 수 없다.
