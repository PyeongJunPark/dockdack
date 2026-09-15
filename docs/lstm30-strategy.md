# 30개 일봉 모델과 +1% 익절 / −0.8% 손절 모듈

이 모듈은 기존 60일 방향 분류 예제와 별개다. `dockdack.ml30`은 학습된 모델 추론을, `dockdack.lstm30_adapter`는 기존 자동매매 엔진에 전달할 JSON 신호 생성을 담당한다. 코드가 자동주문을 켜거나 주문 API를 호출하지 않는다. 현재 어댑터는 모의투자 전용이다.

## 학습하는 것과 규칙으로 처리하는 것

입력은 시간순으로 정렬한 **완료된 일봉 정확히 30개**, 각 행의 열 순서는 `open, high, low, close, volume`이다. 예측 정답은 마지막 입력 종가를 `C_t`라고 할 때 `다음 유효 관측봉 종가 >= C_t × 1.01`이다. 몇 거래일 동안의 최대 상승폭이나 장중 고가 터치 여부를 예측하는 모델이 아니다. 잘못된 봉을 제거했으므로 거래정지·누락이 있으면 바로 다음 거래소 영업일과 다를 수 있다.

| 상태 | 판단 | 신호 |
| --- | --- | --- |
| 미보유 | 모델의 다음 종가 +1% 이상 상승 확률이 0.5 이상 | 매수 |
| 보유 | 현재가 ≥ 실제 평균 매입가 × 1.01 | 익절 매도 |
| 보유 | 현재가 ≤ 실제 평균 매입가 × 0.992 | 손절 매도 |
| 그 외 | 확률 미달, 보유 중 범위 내, 입력 오류 | 관망 |

보유 중에는 모델이 고장 나거나 일봉이 부족해도 유효한 현재가와 잔고로 익절·손절을 먼저 판단한다. 추가 매수와 `NOT_UP`만을 이유로 한 매도는 하지 않는다. 이전 예제의 ‘전일보다 낮으면 매수 / 높으면 매도’ 조건도 이 새 모듈에는 없다.

0.5는 **예측 확률 기준**이며 0.5% 수익률이 아니다. 예측 사건의 기준 가격은 마지막 완료 일봉의 종가다. 이후 장중 실제 매수가가 달라지므로 그 매수가에서 +1% 수익이 발생한다는 뜻이 아니다. 익절·손절은 평균 체결 매입가 기준의 수수료·세금·환율을 제외한 가격 수익률이다. 시세 지연·가격 급변·미체결로 실제 수익/손실이 +1% / −0.8%와 다를 수 있다.

## 모델 및 학습 프로토콜

- 구조: 2층 LSTM, 은닉 크기 128, 층 사이 Dropout 0.2, 출력 로짓 1개. 학습 파라미터 202,369개.
- 특징 7개: 첫 입력 종가 대비 OHLC 로그 비율 4개, 윈도 평균을 뺀 `log1p(volume)`, 종가 로그 변화량(첫 행 0), 고저 로그 비율.
- 특징은 해당 30봉 안에서만 계산한다. 추가 과거 봉이나 전체 기간에서 구한 정규화 통계가 필요 없다.
- 손실은 가중치 없는 `BCEWithLogitsLoss`, Adam, 기울기 노름 제한 1.0이다.
- 국내와 미국은 별도 모델이다. 각 시장에서 여러 종목의 윈도를 함께 학습하며, 종목별 모델 512개를 만드는 방식이 아니다.
- 첫 실행은 시장별 최대 512종목, 2015년부터의 데이터다. 학습 기간 유효 일봉이 300개 이상인 후보를 고정 seed의 해시 순서로 선택한다. 검증/테스트 성과로 종목을 고르지 않는다.
- 국내는 DB의 코스피·코스닥 분류(`0`, `10`), 미국은 ETF가 아닌 것으로 표시된 종목(`is_etf=0`)을 사용한다. 우선주·리츠·스팩 등까지 완전히 제거한 증권유형 원장은 아니며, 실제 주문 시에는 main의 보통주 정책을 그대로 적용한다.
- 정답 날짜 기준 공통 분할: 2022년 말까지 학습, 2023–2024년 검증, 2025년 이후 테스트. 학습 윈도만 epoch마다 섞는다.
- 검증 BCE가 가장 낮은 모델을 선택한 뒤 테스트를 평가한다. 최대 8 epoch, 검증 3회 연속 미개선이면 조기 종료한다. 테스트 결과를 보고 임계값을 낮추거나 모델을 다시 고르지 않는다.

30봉 입력을 위한 **학습 샘플 하나에는 31번째 봉의 정답**이 필요하다. 종목별 유효 원본 봉이 N개면 윈도는 N−30개다. 원본 봉 수, 서로 겹치는 학습 윈도 수, 파라미터 수는 서로 다른 값이다. 입력 구간은 겹쳐도 서로 다른 종목을 이어 붙여 하나의 윈도를 만들지는 않는다.

현재 DB는 과거에 수집한 종목 목록과 종목별로 다른 갱신일을 갖는다. 상장폐지 종목 누락·생존편향·시점별 종목 구성 편향이 남아 있다. 테스트의 정확도·정밀도·ROC AUC는 분류 성능이며 거래 수익률이 아니다. 일봉만으로 익절과 손절이 같은 날 모두 닿았을 때의 선후 관계를 알 수 없어, 실현 수익률 백테스트를 했다고 표시하지 않는다.

## GPU 학습 환경과 실행

이 컴퓨터의 `.venv-ml-cuda`는 기존 Python 3.13의 CUDA PyTorch를 읽도록 만든 별도 환경이다. 기존 프로젝트 `.venv`나 전역 패키지를 교체하지 않았다. 실제 사용 버전은 PyTorch `2.11.0+cu128`, RTX 5080이다. CUDA 설치 조합은 [PyTorch 공식 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/)에서 확인할 수 있다.

다른 컴퓨터에서 독립 환경을 새로 만들 때:

```powershell
uv venv .venv-ml-cuda --python 3.13
uv pip install --python .venv-ml-cuda/Scripts/python.exe -r requirements-ml-cuda.txt
```

프로젝트 루트에서 실행한다. 출력 폴더는 새 경로여야 하며 기존 학습 결과를 덮어쓰지 않는다.

```powershell
.venv-ml-cuda/Scripts/python.exe -m examples.train_lstm30 --market domestic --device cuda --output-dir outputs/ml30/domestic-new-run
.venv-ml-cuda/Scripts/python.exe -m examples.train_lstm30 --market us --device cuda --output-dir outputs/ml30/us-new-run
```

기본 배치는 1,024개, GPU 메모리 제한은 해당 프로세스 기준 총 VRAM의 25%다. GPU 연산 점유율 제한은 아니며 다른 앱의 속도에 영향을 줄 수 있다. `--batch-size`, `--gpu-memory-fraction`, `--max-symbols`, `--epochs`로 변경한다. `--device cuda`에서 CUDA를 못 쓰면 CPU로 조용히 전환하지 않고 실패한다. DB는 읽기 전용으로 연다.

이 컴퓨터의 초기 본학습은 cuDNN을 사용해 학습·테스트·파일 저장을 마쳤지만, 프로세스 종료 시 Windows 네이티브 오류(`0xC0000409`, `ucrtbase.dll`)가 발생했다. 별도 작은 테스트에서 cuDNN LSTM + Dropout 조합일 때 재현됐고, cuDNN을 끄고 Dropout 0.2를 유지하면 정상 종료됐다. 따라서 현재 기본 `--cudnn auto`는 Windows CUDA에서 cuDNN을 끈다. CUDA GPU 연산과 모델 구조는 유지되며 실행 속도는 달라질 수 있다. `--cudnn off`로 명시할 수도 있다.

안전 경로는 실제 국내 DB 8종목으로 학습·검증·테스트·저장까지 다시 실행해 종료 코드 0을 확인했다. 이 소규모 점검의 모델을 본학습 모델로 대체하지 않았다. 본학습의 두 체크포인트는 새 CPU 프로세스에서 재로드·추론하고 main의 실제 신호 파서까지 연결해 별도로 확인했다.

중단된 학습은 같은 데이터와 구조 설정으로 다음처럼 이어간다. 이미 테스트 결과를 보고 추가 학습한 실험은 새 독립 테스트를 거친 것처럼 해석하면 안 된다.

```powershell
.venv-ml-cuda/Scripts/python.exe -m examples.train_lstm30 --market domestic --device cuda --resume outputs/ml30/domestic-new-run/last_model.pt
```

출력 폴더에는 `best_model.pt`, 재개용 `last_model.pt`, 종목·제외 이유·분할별 샘플 목록 `manifest.json`, `run_config.json`, `history.json`, `metrics.json`, 테스트 예측 `test_predictions.npz`가 저장된다. 데이터와 모델 산출물 및 CUDA 환경은 Git에서 제외한다.

## 2026-09-15 본학습 결과

| 시장 | 종목 | 유효 원본 일봉 | 학습 윈도 | 검증 / 테스트 윈도 | 최적 epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 국내 | 512 | 1,336,960 | 867,456 | 250,368 / 203,776 | 8 (총 8회) |
| 미국 | 512 | 1,234,749 | 750,082 | 256,932 / 212,375 | 4 (총 7회, 조기 종료) |

본학습 체크포인트는 각각 `outputs/ml30/domestic-20260915-5080/best_model.pt`, `outputs/ml30/us-20260915-5080/best_model.pt`다. 테스트 구간의 마지막 정답 날짜는 국내 2026-08-21, 미국 2026-09-11이며 종목별 마지막 날짜는 다르다.

확률 0.5 이상 매수 후보의 실제 다음 유효 종가 +1% 달성 비율은 국내 **45.81% (3,434개 후보)**, 미국 **39.38% (325개 후보)**였다. ROC AUC는 각각 0.6328 / 0.6555, 테스트 BCE는 0.5509 / 0.5492로 학습 양성비율 상수 기준 0.5788 / 0.5868보다 낮았다. 그러나 매수 후보의 성공 비율이 예측 기준 50%에 미치지 못했고, 전체 정확도도 항상 음성으로 판단하는 기준보다 약간 낮았다. 현재 모델을 검증된 수익 전략으로 취급하지 않는다. 테스트 결과를 본 뒤 임계값을 변경하지 않았다.

상세 요약과 학습 곡선은 로컬 `outputs/ml30/report-20260915/report.md`와 `learning-curves.png`에 있다. 안전 종료 점검은 `outputs/ml30/cuda-safe-smoke-20260915`에 별도로 보관한다.

## 파이썬 모듈로 추론

```python
from dockdack.ml30 import Predictor

predictor = Predictor("outputs/ml30/domestic-20260915-5080/best_model.pt")
# completed_ohlcv: 과거→최근 순서의 숫자 배열 [30, 5]
prediction = predictor.predict(completed_ohlcv)
print(prediction)  # probability_ge_1pct, buy_threshold, predicts_gain
```

국내 모델은 국내, 미국 모델은 미국 입력에 연결한다. 한 번 로드한 `Predictor`를 재사용한다. CPU 추론도 가능하며 추론할 때마다 GPU 학습을 실행하지 않는다.

`Predictor(path, device="cuda:0")`로 GPU 추론을 선택할 수 있다. Windows CUDA에서는 추론 호출 안에서만 cuDNN을 우회하고 기존 backend 설정을 복원한다. 본학습의 국내·미국 체크포인트 모두 CPU/GPU 예측 차이 0.00001 미만 및 정상 종료를 확인했다.

## 기존 main 자동매매 코드와 연결

최신 main의 `signal_bridge` 차트/신호 JSON 버전 1 계약을 사용한다. 이 브랜치에 main을 자동 병합하거나 GUI에 새 모드를 등록한 것은 아니다. 모델 모듈과 어댑터를 main에 반영한 뒤 외부 신호 모드에 연결할 수 있다.

1. main이 내보낸 최신 차트(`charts.json` 또는 종목별 `charts_updates/*.json`)를 어댑터에 전달한다. 개별 업데이트를 사용하면 전체 관심종목 조회 완료까지 기다리지 않는다.
2. `position_provider(stock)` 콜백으로 같은 종목의 실제 모의계좌 보유·매도가능 수량·평균 매입가와 수신 시각을 제공한다. 잔고가 없다는 것과 잔고 조회 실패는 구분한다.
3. `LSTM30SignalProducer`가 JSON payload와 별도 진단 정보를 반환한다. payload만 main의 신호 입력 파일에 원자적으로 저장한다.
4. main은 출처, 수량·금액 상한, 만료·중복, 현재가와 잔고 등을 다시 검사한다. 익절 신호에는 `cost_profit_pct="1"`, 손절에는 `cost_loss_pct="0.8"`이 포함되어 주문 직전에도 실제 원가 조건을 확인한다.

모델 입력은 **완료봉 30개**다. main이 내보내는 30개 중 당일 미완료 봉이 하나 있으면 사용할 수 있는 것은 29개라 관망한다. 장중에는 main의 조회 N을 최소 31로 두어 완료봉 30개를 제공하거나, 이미 보관한 완료봉으로 보충해야 한다. 어댑터는 부족한 봉을 만들어 채우지 않는다.

신규 매수에서는 거래소 캘린더로 마지막 완료봉이 해당 시장의 직전 거래일인지 검사한다. 오래된 봉, 거래정지 등으로 직전 거래일 봉이 없는 종목, 캘린더를 확인할 수 없는 경우는 관망한다. 이 검사는 실제 보유 포지션의 익절·손절 판단을 막지 않는다.

`position_provider` 반환값은 다음 필드를 가진다. `market`, `symbol`, `exchange`, `currency`는 차트와 같아야 한다.

```python
{
    "market": "domestic", "symbol": "005930", "exchange": "KRX", "currency": "KRW",
    "quantity": "0", "sellable_quantity": "0", "average_price": None,
    "fetched_at": "시간대가 있는 실제 조회 시각"
}
```

호출 형태:

```python
from dockdack.lstm30_adapter import LSTM30SignalProducer, atomic_json

producer = LSTM30SignalProducer(
    {"domestic": domestic_model, "us": us_model},
    position_provider=read_position,
    quantity=order_quantity,  # 사용자가 정한 양의 정수
    max_krw=krw_cap, max_usd=usd_cap,  # 사용자가 정한 시장별 주문 총액 상한
    state_path="outputs/ml30/signals.state.json",
)
payload, diagnostics = producer(charts)
atomic_json("outputs/ml30/signals.preview.json", payload)
```

출처 ID는 `lstm30-mark0`다. `quantity`는 매수 수량 및 한 번의 매도 최대 수량이다. 매도 수량은 보유량·매도가능 수량·금액 상한에도 제한되어 전량 청산을 보장하지 않는다. 거래 상한 0인 시장에는 매수/매도 신호를 내지 않는다.

현재가·잔고는 조회 후 15초 이내여야 하고 신호 유효기간은 내보내기 기준 2분이다. 현재가가 거래소 실시간 체결값인지는 별도 문제다. 잘못된 모델/입력은 관망하며, 거래 일정·미체결·중복 주문 차단은 기존 실행기가 맡는다. 같은 차트 export는 같은 신호 ID·내용·시간을 재사용한다. 단일 프로세스가 파일을 쓰고 재시작 시 상태 파일을 유지한다. HOLD는 이미 접수된 주문 취소가 아니다.

CLI도 제공한다. 아래 변수의 주문 수량과 금액 상한은 실행 전에 사용자가 정해야 한다. 기본 출력은 **실제 연결 파일이 아닌 preview**이며 명령을 실행해도 자동주문이 켜지지 않는다.

```powershell
.venv-ml-cuda/Scripts/python.exe -m examples.publish_lstm30 --charts .dockdack/exchange/charts.json --domestic-checkpoint outputs/ml30/domestic-20260915-5080/best_model.pt --us-checkpoint outputs/ml30/us-20260915-5080/best_model.pt --kiwoom-demo --quantity $orderQuantity --max-krw $krwCap --max-usd $usdCap
```

한 번 실행하고 종료한다. 상시 신호 생성을 원하면 main의 새 차트 업데이트마다 callable을 호출하거나 별도 생산자 루프로 연결해야 한다. 이 작업에서는 상시 감시나 자동주문을 활성화하지 않는다.
