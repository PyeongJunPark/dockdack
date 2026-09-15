# PyTorch LSTM 일봉 분류 예제

`examples/train_lstm_daily.py`는 수집한 SQLite DB에서 한 종목의 일봉을 읽어 다음 유효 거래일의 종가가 마지막 입력일 종가보다 높은지 학습하는 예제다. 기본 종목은 삼성전자 `005930/KRX`, 시작일은 `2010-01-01`이다.

## 실행

프로젝트 루트에서 학습 라이브러리를 한 번 설치한다. 예제는 독립 스크립트라 `--no-install-project`로 브로커 패키지 자체의 재설치를 생략하고, `--inexact`로 기존 선택 의존성을 보존한다.

```powershell
uv sync --extra ml --inexact --no-install-project
```

이후에는 설치한 환경으로 실행한다.

```powershell
uv run --no-sync python examples/train_lstm_daily.py --epochs 20
```

빠르게 실행 흐름만 확인하려면 다음처럼 기간과 학습 횟수를 줄인다.

```powershell
uv run --no-sync python examples/train_lstm_daily.py --start 2020-01-01 --epochs 2 --device cpu
```

미국 DB의 애플로 바꾸는 예:

```powershell
uv run --no-sync python examples/train_lstm_daily.py --db data/kiwoom_daily/us_daily.sqlite3 --symbol AAPL --exchange ND --epochs 20
```

미국 거래소 코드는 NASDAQ `ND`, NYSE `NY`, AMEX `NA`다. 한 번 실행하면 지정한 한 종목으로 학습하며 전 종목을 합쳐서 학습하지 않는다. `--device auto`는 설치된 PyTorch에서 CUDA를 사용할 수 있으면 GPU를, 그렇지 않으면 CPU를 선택한다. CUDA 지원 PyTorch 설치는 별도 환경 설정이다.

## 모델과 학습 데이터

기본 구조는 **입력 `[배치, 60일, 5개 특징]` → 2층 LSTM(은닉 크기 64) → Dropout(0.2) → Linear(출력 1)** 이다. 학습 가능한 파라미터는 기본 설정에서 51,521개다.

- 가격 특징 4개: 당일 시가·고가·저가·종가를 전일 종가로 나눈 로그 비율
- 거래량 특징 1개: `log(1 + 당일 거래량) - log(1 + 전일 거래량)`
- 입력: 마지막 완료 거래일까지의 최근 60일 특징
- 정답: 그 다음 유효 거래일 종가가 입력 마지막 날 종가보다 높으면 `1`, 하락·보합이면 `0`
- 학습: `BCEWithLogitsLoss`, Adam(학습률 0.001), 기울기 크기 제한 1.0
- 예측: 모델 출력에 Sigmoid를 적용해 상승 확률값을 구하고 0.5를 기준으로 `UP` 또는 `NOT_UP` 출력

결측값·0 이하 가격·음수 거래량·OHLC 관계가 잘못된 행은 제외하고 제외 건수를 출력한다. 제외된 날짜가 있으면 다음 *남아 있는 유효 일봉*이 예측 대상이다. 거래량 0은 허용하며, 첫 일봉은 전일 비교에 사용하므로 특징 생성 과정에서 입력 행 하나가 줄어든다.

원본 유효 일봉 수를 `N`, 입력 길이를 `L`이라고 하면 학습 가능한 윈도 수는 `N - L - 1`개다. 원본 행 수, 겹치는 윈도 수, 모델 파라미터 수는 서로 다른 개념이며 실행 시 각각 출력한다.

## 시간순 검증

정답 날짜 순으로 앞 70%를 학습, 다음 15%를 검증, 마지막 15%를 테스트로 나눈다. 검증·테스트의 입력 윈도에는 그 날짜에 이미 관측 가능한 앞 구간의 일봉도 포함된다. 윈도끼리 일부 과거 입력은 겹치지만 정답 날짜는 겹치지 않는다. 학습용 배치만 섞고 시간 구간 자체는 섞지 않는다.

표준화의 평균·표준편차는 학습 입력에 등장한 행에서만 계산한다. 검증 손실이 가장 낮은 가중치를 저장하며 기본 5회 연속 개선이 없으면 학습을 중단한다. 모델 선택을 끝낸 뒤 테스트를 한 번 평가한다.

테스트에서는 정확도·상승 precision/recall/F1을 출력 파일에 기록한다. 학습 구간의 다수 클래스를 항상 예측하는 단순 기준의 테스트 정확도도 함께 비교한다. 이 기준보다 LSTM이 좋지 않을 수도 있다.

## 저장 결과와 재예측

기본 출력은 실행별 `outputs/lstm/종목_실행시각/` 폴더다. `--output-dir`로 새 출력 폴더를 지정할 수 있으며 기존 폴더는 덮어쓰지 않는다. 학습 결과물은 Git에서 제외한다.

- `best_model.pt`: 검증 기준 최적 가중치, 모델 구조, 표준화 기준, 데이터 설정
- `metrics.json`: 데이터 구간, 샘플 수, 파라미터 수, 최적 epoch, 테스트 지표
- `history.json`, `training_curve.png`: epoch별 학습·검증 손실과 정확도
- `test_predictions.json`: 각 테스트 정답 날짜, 상승 예측값, 실제 상승 여부
- `latest_prediction.json`: DB에 저장된 가장 최근 유효 일봉 기준의 다음 일봉 예측

저장된 모델로 학습 없이 다시 예측하려면 실제 출력 경로를 넣는다.

```powershell
uv run --no-sync python examples/train_lstm_daily.py --checkpoint outputs/lstm/실제_실행폴더/best_model.pt
```

이때 종목·입력 길이·표준화는 체크포인트 설정을 사용한다. DB 위치가 바뀌었다면 `--db 새경로.sqlite3`를 함께 지정한다. 출력의 `as_of_date`는 현재 날짜가 아니라 DB의 마지막 유효 일봉 날짜다. 기존 DB가 갱신되지 않았다면 과거 시점의 예측이다.

이 예제는 분류 모델의 학습 흐름을 익히기 위한 것이며 주문 API를 호출하지 않는다. 종가 방향 정확도는 거래 수익률이 아니다. 완성된 종가를 입력으로 사용한 뒤 같은 종가로 체결됐다고 가정할 수 없으므로, 실제 매매 연결에는 신호 발생 시점과 다음 거래일 체결 시점을 정한 별도 검증이 필요하다. 현재 수집 DB는 종목별 갱신일이 다르고 과거 상장폐지 종목 전체를 담고 있지 않다.

API 참고: [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

저장된 예측과 현재가·전일 종가·평균 매입가를 결합하는 방법은 [LSTM 매매 신호 예제](trading-signals.md)를 참고한다. 키움 모의투자 조회 또는 직접 입력으로 `BUY`/`SELL`/`HOLD`를 출력하며 주문은 전송하지 않는다.
