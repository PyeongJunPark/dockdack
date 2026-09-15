# LSTM + 현재가 매매 신호

저장된 LSTM 체크포인트의 `UP` / `NOT_UP`과 현재가·보유 정보를 결합해 `BUY`, `SELL`, `HOLD`를 반환한다. **주문은 전송하지 않는다.** 한 번 실행할 때 한 종목의 신호를 JSON으로 출력하며, 백그라운드 감시·알림 전송·중복 신호 방지는 포함하지 않는다.

## 판단 규칙

위에서 먼저 일치한 조건을 적용한다. 전일은 달력상 어제가 아니라 직전 완료 거래일이다.

| 우선순위 | 조건 | 신호 |
| --- | --- | --- |
| 1 | 보유 수량 > 0, 현재가 ≥ 평균 매입가 × 1.01 | `SELL` — 1% 이익 실현 |
| 2 | `UP`, 현재가 < 전일 종가 | `BUY` |
| 3 | `NOT_UP`, 현재가 > 전일 종가, 보유 수량 > 0 | `SELL` |
| 그 외 | 동가, 조건 불일치, 매도할 보유 수량 없음 | `HOLD` |

수익률은 `(현재가 / 평균 매입가 - 1) × 100`으로, 수수료·세금·환율 변동을 반영하지 않은 가격 수익률이다. 정확히 +1%도 매도 조건에 포함한다. 이익 실현은 예측이나 일봉 입력을 구할 수 없어도 먼저 판단한다. 평균 매입가는 체결 후 계좌의 실제 보유 원가를 사용하며, 전일 종가를 매입가로 간주하지 않는다.

보유 중에도 두 번째 조건이면 `BUY`가 나올 수 있다. 추가 매수 여부·수량·예산·주문 가능 수량은 이 신호 코드가 결정하지 않는다. `NOT_UP`은 모델상 하락 또는 보합 분류이며, 특정 수익을 보장하는 예측이 아니다.

## 키움 모의투자 조회로 실행

프로젝트 루트의 `.env`에 모의투자 키가 설정되어 있어야 한다. 설정은 [브로커 문서](kiwoom-broker.md)를 참고한다. 프로젝트 루트에서:

```powershell
uv sync --extra ml --inexact --no-install-project
uv run --no-sync python -m examples.emit_lstm_signal --checkpoint outputs/lstm/samsung-example-smoke-20260914/best_model.pt --kiwoom-demo
```

위 경로는 이 컴퓨터에서 앞서 학습한 삼성전자 예제 체크포인트다. 다른 환경에서는 [학습 예제](lstm-example.md)로 생성한 `best_model.pt` 경로로 바꾼다. 체크포인트는 Git에 포함하지 않는다. 신뢰할 수 있는 본인 체크포인트만 사용한다.

체크포인트의 종목·거래소·입력 길이·표준화를 그대로 사용한다. 국내 `KRX`, 미국 `NA`/`ND`/`NY`를 지원하며, 미국 종목은 그 종목으로 학습한 체크포인트가 필요하다.

실행 시 모의계좌 보유 수량·평균 매입가와 현재가를 조회한다. 이익 실현 조건이 아니면 키움에서 최근 일봉을 새로 조회해 재학습 없이 추론한다. 기본 60일 특징에 필요한 유효 일봉은 61개다. 현지 날짜의 당일 일봉은 미완료일 수 있으므로 제외한다. 미국 날짜는 뉴욕 시간, 국내 날짜는 서울 시간으로 계산한다.

LSTM은 직전 완료 일봉까지 보고 **다음 유효 거래일 종가가 마지막 입력 종가보다 높을지** 분류한다. 현재가를 LSTM의 미완료 일봉으로 넣지 않고, 이미 구한 분류 결과에 매매 조건을 적용한다.

일봉 부족·잘못된 정규화·추론 오류는 `HOLD`와 오류 사유를 반환한다. 현재가/계좌 조회 실패는 오류로 종료하며, 잔고를 임의로 0으로 간주하지 않는다. `observed_at`은 조회 시각이지 시세의 거래소 체결 시각이 아니다. 거래소 휴장일 판별, 장 운영 시간 제한, API 시세 지연 및 일봉 최신성의 별도 검증은 없으므로, 장외 시간이나 오래된 서버 응답을 실제 주문 근거로 쓰면 안 된다.

## 직접 가격을 넣어 확인

다음은 **2026-08-21 일봉에 가상 현재가와 가상 보유 정보를 붙인 과거 데이터 예제**다. 현재 시세를 뜻하지 않는다.

```powershell
uv run --no-sync python -m examples.emit_lstm_signal --checkpoint outputs/lstm/samsung-example-smoke-20260914/best_model.pt --current-price 282000 --previous-close 281500 --previous-date 2026-08-21 --quantity 1 --average-price 285000
```

로컬 DB에서 `--previous-date`까지의 일봉만 읽는다. 마지막 유효 일봉의 날짜와 종가가 전달한 전일 값과 다르면 `HOLD` / `STALE_OR_MISMATCHED_DAILY_DATA`로 반환한다. 사용자가 전달한 날짜 자체가 실제 직전 거래일인지는 자동 판별하지 않는다. DB 위치가 다르면 `--db 경로.sqlite3`를 지정한다. 보유 중이면 `--quantity`와 `--average-price`가 필요하며, 미보유 기본값은 수량 0이다.

1% 이익 실현만 확인하는 가상 예:

```powershell
uv run --no-sync python -m examples.emit_lstm_signal --checkpoint outputs/lstm/samsung-example-smoke-20260914/best_model.pt --current-price 282800 --previous-close 281500 --previous-date 2026-08-21 --quantity 1 --average-price 280000
```

이 경우 정확히 +1%이므로 LSTM/일봉 조회를 생략하고 `SELL` / `TAKE_PROFIT_1PCT`를 출력한다.

## 파이썬에서 규칙만 호출

```python
from dockdack.signals import evaluate_signal

signal = evaluate_signal(
    current_price="99",
    previous_close="100",
    predicted_direction="UP",
    position_quantity="0",
)
print(signal.to_dict())  # action: BUY
```

이 함수는 PyTorch나 API 연결 없이도 사용할 수 있다. 호출자가 같은 종목의 해당 거래일 예측과 최신 현재가·보유 정보를 제공해야 한다. 현재가 업데이트마다 호출하도록 연결할 수 있지만, 실제 주문을 붙이려면 중복 주문 방지·체결 상태 관리·포지션 한도 등을 별도로 구현해야 한다.

현재 LSTM은 학습 흐름을 확인하는 예제이며, 이 매매 규칙의 수익률 백테스트나 실전 검증을 마친 상태가 아니다. 신호 발생과 주문 체결은 별개다.
