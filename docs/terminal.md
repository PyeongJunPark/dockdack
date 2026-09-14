# 키움주식 터미널

Python 3.11 이상. 프로젝트 루트에서 실행한다. 기존 `KiwoomBroker` 위에 대화형 메뉴와
한 줄 명령을 제공한다. 한국 종목코드와 미국 티커를 자동 구분하며 미국 거래소는
키움의 `usa10098`로 정확히 일치하는 티커를 조회한다. 확인되지 않거나 거래소가 여러 개면
명령을 중단한다. 거래소 조회 결과는 현재 터미널 세션에서만 재사용한다.

## 준비

1. 키움 REST API 사용 신청 후 사용할 모의/실전 환경의 App Key, Secret Key를 준비한다.
2. `.env.example` 양식대로 프로젝트 루트에 `.env`를 만들고 사용할 시장의 키를 입력한다.
   국내만 사용하면 국내 키만, 미국만 사용하면 미국 키만 있으면 된다.
3. 실행한다. `uv run`이 의존성과 실행 명령을 설치한다.

```powershell
uv run dockdack
```

uv 없이 설치하려면:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m dockdack
```

모의투자 설정 예시 (`.env`, 비밀정보는 Git에 올리지 않는다):

```dotenv
DOCKDACK_KIWOOM_DEMO_DOMESTIC_APP_KEY=발급받은_국내_모의_App_Key
DOCKDACK_KIWOOM_DEMO_DOMESTIC_SECRET_KEY=발급받은_국내_모의_Secret_Key
DOCKDACK_KIWOOM_DEMO_US_APP_KEY=발급받은_미국_모의_App_Key
DOCKDACK_KIWOOM_DEMO_US_SECRET_KEY=발급받은_미국_모의_Secret_Key
DOCKDACK_ALLOW_LIVE_ORDERS=false
```

`python -m dockdack`, `uv run python -m dockdack`도 같은 기능이다.
CLI는 `.env`의 `DOCKDACK_TRADING_MODE`와 관계없이 기본값이 `demo`다.
실전은 명시적인 `--mode real`로만 선택한다. Python 라이브러리의 기존 환경변수 동작은 그대로다.

## 대화형 메뉴

종목코드/티커 입력 후 다음 메뉴가 나온다.

| 입력 | 기능 |
| --- | --- |
| `1` | 현재가, 전일대비, 등락률 조회 |
| `2` / `3` | 지정가 매수 / 매도 (수량과 단가 입력) |
| `4` / `5` | 시장가 매수 / 매도 (수량 입력) |
| `6` | 선택 종목의 미체결 주문 조회 |
| `7` | 선택한 시장의 잔고·보유 종목 조회 |
| `8` | 종목 변경 |
| `9` / `10` | 현재가를 자동 조회해 그 가격으로 지정가 매수 / 매도 |
| `q` | 종료 |

한국 종목코드는 앞자리 0을 포함해 `005930`처럼 입력한다. 미국은 `AAPL`, `NVDA`처럼
키움에 등록된 티커를 입력하며 소문자도 허용한다. 숫자로 시작하는 6자리 코드는 한국으로
구분한다. 자동 구분이 애매한 종목은 한 줄 명령에서 `--market domestic` 또는 `--market us`를 지정한다.

주문에는 모의/실전, 종목, 거래소, 매수/매도, 수량, 주문 유형, 단가가 표시된다.
모의 주문은 `y`, 실전 주문은 `LIVE_ORDER`를 정확히 입력해야 전송된다. Enter 또는 다른 입력은
해당 주문을 취소한다. 실제 확인 전에 API 주문 요청을 보내지 않는다.

## 한 줄 명령

아래 가격은 사용법 설명을 위한 예시이며 현재가가 아니다.

```powershell
# 현재가 (미국 거래소 자동 조회)
uv run dockdack quote 005930
uv run dockdack quote AAPL

# 지정가 매수 / 매도
uv run dockdack buy 005930 1 --type limit --price 70000
uv run dockdack sell 005930 1 --type limit --price 71000
uv run dockdack buy AAPL 1 --type limit --price 200.25
uv run dockdack sell AAPL 1 --type limit --price 210.50

# 시장가 매수 / 매도 (--price를 넣지 않는다)
uv run dockdack buy 005930 1 --type market
uv run dockdack sell 005930 1 --type market
uv run dockdack buy AAPL 1 --type market
uv run dockdack sell AAPL 1 --type market

# 주문 미리보기: 확인 입력 및 주문 전송 없음
uv run dockdack buy AAPL 1 --type limit --price 200.25 --dry-run

# 미국 거래소를 이미 아는 경우 자동 조회 생략
uv run dockdack quote AAPL --exchange NASDAQ

# 현재가를 조회해 해당 가격으로 지정가 주문 (--price 불필요)
uv run dockdack buy AAPL 1 --type current
uv run dockdack sell AAPL 1 --type current

# 선택 종목의 미체결 / 해당 시장 잔고
uv run dockdack orders AAPL
uv run dockdack balance 005930

uv run dockdack --help
uv run dockdack buy --help
```

`--type current`는 조회한 가격을 지정가로 사용하므로 즉시 체결을 보장하지 않는다.
확인 화면의 가격 그대로 한 번 전송하며 확인 후 재조회하거나 주문 가격을 자동 변경하지 않는다.
미국 모의투자는 시장가가 거절될 수 있으므로 이 방식으로 지정가 주문할 수 있다.
장 종료 등 API 오류는 그대로 반환하며 예약 주문이나 재주문으로 자동 전환하지 않는다.

`--dry-run`도 설정된 키를 로드한다. 미국 거래소를 생략하면 거래소 확인을 위한 조회 API를
호출하지만 주문은 전송하지 않는다. `--exchange NASDAQ`처럼 거래소를 지정하면 미리보기는
네트워크 요청 없이 동작한다. 단, `--type current`는 거래소 지정 여부와 무관하게
미리보기에서도 현재가 조회 API를 호출한다.

국내 거래소 기본값은 KRX다. `--exchange NXT`/`SOR`는 실전에서만 선택할 수 있다.
미국 거래소는 `NASDAQ`/`ND`, `NYSE`/`NY`, `AMEX`/`NA`를 지원한다.
잔고는 해당 시장·거래소 조회 결과이며, 국내 잔고 조회에는 KRX 또는 NXT를 사용한다.

수량은 1 이상의 정수, 지정가는 양의 유한한 숫자만 허용한다. 한국 가격은 정수 원 단위다.
호가단위, 거래 가능 시간, 계좌의 주문 가능 금액/수량 등 최종 주문 가능 여부는 키움에서
검증하므로 API 오류 메시지를 확인한다. 시장가는 체결가격과 최종 금액이 확정되지 않은 주문이다.

## 실전 환경

실전 시세 조회는 실전 키 설정 후 다음처럼 실행한다. `--mode`는 하위 명령 앞에 둔다.

```powershell
uv run dockdack --mode real quote AAPL
uv run dockdack --mode real
```

실전 주문은 `.env`에 `DOCKDACK_ALLOW_LIVE_ORDERS=true`를 설정하고 주문별로
`LIVE_ORDER`를 입력해야 한다. 이 설정은 CLI가 자동으로 바꾸지 않는다.
사용할 시장의 `DOCKDACK_KIWOOM_REAL_DOMESTIC_*` 또는 `DOCKDACK_KIWOOM_REAL_US_*` 키를 설정한다.

주문 API의 성공은 **접수**이지 체결 완료가 아니다. 주문번호와 미체결 조회를 확인하고
체결 여부는 영웅문 체결내역에서 확인한다. 미체결 목록이 비어 있어도 체결되었다고 단정할 수 없다.
연결 끊김·타임아웃으로 결과를 받지 못한 경우 자동 재주문하지 않는다. 재시도 전에 내역을 확인한다.
터미널 종료나 Ctrl+C는 이미 전송한 주문을 취소하지 않는다.

## 검증 및 공식 명세

테스트는 가짜 HTTP 응답을 사용하며 실제 키·계좌 없이 실행한다.

```powershell
uv run python -m unittest discover -s tests -v
```

2026-09-14에 [키움 공식 API 가이드](https://openapi.kiwoom.com/guide/apiguide)에서
국내 현재가 `ka10001`, 매수/매도 `kt10000`/`kt10001`, 미국 현재가 `usa20100`,
거래소 조회 `usa10098`, 매수/매도 `ust20000`/`ust20001`을 확인했다.
미국 주문 유형은 지정가 `00`, 시장가 `03`, 국내는 지정가 `0`, 시장가 `3`을 사용한다.
미국 주문 명세에는 모의투자 도메인도 제공된다. 실제 계좌의 서비스 신청·지원 상태와
시세/주문 동작은 발급받은 키로 별도 확인해야 한다.
