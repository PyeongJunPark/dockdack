# 키움증권 브로커 모듈

`dockdack.KiwoomBroker`는 키움증권 REST API를 이용해 다음 기능을 제공한다.

- 국내주식과 미국주식 현재가 및 종목정보 조회
- 종목명·종목코드 검색과 사용자 정의 조건 필터
- 영웅문4/영웅문Global에 저장한 국내·미국 조건검색식 실행
- 국내·미국 계좌의 예수금, 주문가능금액, 보유종목, 평가손익 조회
- 국내·미국 매수·매도, 미체결 주문 조회, 주문 취소
- 모의투자/실전투자 도메인 전환, API 호출 간격 제한, 연속조회 처리
- 실전 주문 이중 안전장치

키움 API의 주문 접수 성공은 체결 완료를 뜻하지 않는다. 주문 후 `list_open_orders()`와 체결내역 API로 상태를 별도 확인해야 한다.

## 설치

Python 3.11 이상에서 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

키움 저장 조건검색도 사용할 때만 WebSocket 선택 의존성을 설치한다.

```powershell
python -m pip install -e ".[conditions]"
```

## App Key 입력 위치

실제 키는 프로젝트 최상위의 `.env` 파일에 입력한다. `.env.example`은 공개 가능한 양식이고, `.env`는 Git에서 제외된 로컬 비밀정보 파일이다.

```dotenv
DOCKDACK_TRADING_MODE=demo

DOCKDACK_KIWOOM_DEMO_DOMESTIC_APP_KEY=
DOCKDACK_KIWOOM_DEMO_DOMESTIC_SECRET_KEY=
DOCKDACK_KIWOOM_DEMO_US_APP_KEY=
DOCKDACK_KIWOOM_DEMO_US_SECRET_KEY=

DOCKDACK_KIWOOM_REAL_DOMESTIC_APP_KEY=
DOCKDACK_KIWOOM_REAL_DOMESTIC_SECRET_KEY=
DOCKDACK_KIWOOM_REAL_US_APP_KEY=
DOCKDACK_KIWOOM_REAL_US_SECRET_KEY=

DOCKDACK_ALLOW_LIVE_ORDERS=false
DOCKDACK_HTTP_TIMEOUT_SECONDS=15
```

예를 들어 국내 모의투자 App Key는 다음과 같이 `=` 오른쪽에 공백 없이 넣는다.

```dotenv
DOCKDACK_KIWOOM_DEMO_DOMESTIC_APP_KEY=실제_App_Key
```

국내와 미국 계좌가 같은 App Key/Secret을 사용한다면 두 시장 항목에 같은 값을 넣어도 된다. 호환을 위해 기존 `DOCKDACK_KIWOOM_DEMO_APP_KEY`, `DOCKDACK_KIWOOM_REAL_APP_KEY` 형식도 fallback으로 지원한다.

`.env`는 `KiwoomBroker.from_env()` 호출 시 자동으로 읽힌다. PowerShell이나 시스템에 같은 환경변수가 이미 설정되어 있으면 환경변수 값을 우선한다.

`DOCKDACK_TRADING_MODE`는 이번 실행에서 사용할 키를 결정한다.

- `demo`: 국내·미국 모의투자 키 사용
- `real`: 국내·미국 실전투자 키 사용

키, Secret, 접근 토큰을 채팅, 캡처, README, `.env.example`, Python 소스에 입력하지 않는다.

## 기본 사용법

```python
from dockdack import DomesticExchange, KiwoomBroker, Market, USExchange

broker = KiwoomBroker.from_env()

# 현재가
samsung = broker.quote_domestic("005930")
apple = broker.quote_us("AAPL", exchange=USExchange.NASDAQ)
print(samsung.name, samsung.price, samsung.currency)
print(apple.name, apple.price, apple.currency)

# 종목명/코드 검색
matches = broker.search_stocks("삼성", market=Market.DOMESTIC)
print([(stock.symbol, stock.name) for stock in matches])

# 예수금, 평가금액, 보유종목
account = broker.account_domestic(exchange=DomesticExchange.KRX)
print(account.cash, account.total_evaluation, account.positions)

# 모의 지정가 매수
order = broker.build_order(
    market=Market.DOMESTIC,
    side="buy",
    symbol="005930",
    quantity=1,
    exchange=DomesticExchange.KRX,
    price="70000",
)
print("예상 주문금액:", order.estimated_notional)
result = broker.place_order(order)
print("주문번호:", result.order_number)

# 미체결 확인 및 전량 취소(quantity=0)
print(broker.list_open_orders(Market.DOMESTIC, exchange=DomesticExchange.KRX))
broker.cancel_order(
    market=Market.DOMESTIC,
    original_order_number=result.order_number,
    symbol="005930",
    exchange=DomesticExchange.KRX,
    quantity=0,
)
```

저장 조건검색식은 비동기 함수로 실행한다.

```python
import asyncio

from dockdack import KiwoomBroker, Market


async def main() -> None:
    broker = KiwoomBroker.from_env()
    conditions = await broker.list_saved_conditions(Market.DOMESTIC)
    print(conditions)
    if conditions:
        stocks = await broker.run_saved_condition(
            Market.DOMESTIC,
            conditions[0].sequence,
        )
        print([(stock.symbol, stock.name, stock.price) for stock in stocks])


asyncio.run(main())
```

## 실전 주문 안전장치

실전 조회는 `real` 환경과 실전 키로 사용할 수 있지만, 실전 주문은 다음 두 조건을 모두 만족해야 한다.

1. `DOCKDACK_ALLOW_LIVE_ORDERS=true` 또는 `KiwoomConfig(..., allow_live_orders=True)`
2. 주문 호출에 `confirm_live_order="LIVE_ORDER"` 전달

처음에는 반드시 모의투자에서 조회, 주문, 미체결 확인, 취소까지 검증한 뒤 실전 환경으로 전환한다.

공식 참고 자료: [키움 REST API 포털](https://openapi.kiwoom.com/), [키움 공식 Python 예제](https://github.com/Kiwoom-Securities/Kiwoom-REST-API)
