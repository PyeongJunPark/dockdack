# 거래대금 TOP100 · 차트 내보내기 · 외부 매매 신호

기본은 모의투자다. 실전은 [환경 선택과 안전장치](trading-modes.md)를 거친 별도 환경에서 사용한다.
신호를 만드는 모델/전략은 별도 코드에서 실행하고, DockDack은 **JSON 데이터만** 읽는다.
외부 코드를 로드하거나 실행하지 않는다. 신호 수신과 자동주문 활성화는 별도이며 기본은 OFF다.

실전 신호에는 최상위 `"trading_mode": "real"`이 반드시 필요하다. 내보내기에도 선택한 환경이 포함되며,
다른 환경의 신호와 내장 모의 테스트 신호는 실전에서 거절한다. 모의는 기존 환경 필드 없는 계약도 호환한다.

## 빠른 시작

1. `uv run --extra gui dockdack-gui` → 왼쪽 **관심종목 · 자동매매**.
2. **거래대금 TOP100 추가 · 한국 + 미국**을 누른다. 시장별 100종목을 별도로 선정한다.
   기존 관심종목과 종목별 N, 규칙, 주문 이력은 유지한다. 새 종목은 입력된 N(기본 30거래일)을 사용한다.
3. **전체 1회 조회** → **차트 JSON 내보내기**. 기본 출력은 `.dockdack/exchange/charts.json`이다.
4. 다른 코드가 차트를 읽고 아래 규격의 신호를 `.dockdack/exchange/signals.json`에 쓴다.
5. 상단 **외부 신호 연결 → 연결 설정**에서 출처 `source_id`, 입력/출력 경로, 주문당 최대 수량과 KRW/USD 상한을 입력한다.
   금액 0은 해당 시장 주문 차단이다. **파일 검사 (주문 없음)**는 JSON 형식/정책만 검사하며 규칙·주문을 생성하지 않는다.
   설정의 **신호 파일 1회 읽기**는 실제 신호를 접수해 대기 규칙을 생성할 수 있지만, 주문을 켜거나 전송하지 않는다.
6. **외부 신호 모드**를 선택한 뒤 **감시 시작 (조회만)**. 이 모드에서는 수동 가격/SMA 규칙을 실행하지 않는다.
7. 주문을 허용할 때만 **자동주문 켜기 (ON)**를 한 번 누르고 출처/상한을 확인한다.
   조회 중에도 예약할 수 있으며 전체 관심종목·잔고 등의 검증에 성공한 뒤 ON이 된다. 실패하면 OFF를 유지하고 예약을 취소한다.
   모니터만 시작하면 주문은 전송되지 않는다. **자동주문 끄기 (OFF)**는 ON 예약도 취소한다.
8. **감시·주문 중지**는 다음 전송을 막는다. 이미 접수된 주문은 취소하지 않는다.

외부 모드 감시에서는 **각 종목 조회 직후** `<출력 파일명>_updates/<시장>_<거래소>_<종목>.json`을 원자적으로 갱신한다.
예를 들어 `charts.json`이면 `charts_updates/domestic_KRX_005930.json`이다. 한 순회가 끝나면 전체 `charts.json`도 갱신한다.
각 파일은 동일한 차트 계약과 고유 `export_id`를 사용하며 개별 파일의 `stocks`에는 한 종목이 들어간다.
외부 코드는 개별 파일의 새 `export_id`를 처리하면 전체 순회 완료를 기다릴 필요가 없다. 종목 조회 사이와 주문 직전에
새 입력 파일을 확인하고, 새 신호가 있는 종목을 먼저 처리한다. 신호 입력 자체는 자동주문을 켜지 않는다.
창을 닫으면 감시가 끝나며, 재시작 시 모드/정책을 다시 설정하고 활성화를 확인해야 한다.

200종목 전체를 조회하려면 수백 번의 API 요청이 필요해 **한 순회에 수 분**이 걸릴 수 있다.
설정한 30초는 한 순회 **완료 후 대기시간**이지 200종목의 동시 갱신 주기가 아니다.
파일 확인도 실시간 이벤트가 아니며, 현재 HTTP 요청·차트 페이지 조회·대기시간만큼 늦어질 수 있다.
오래된 신호는 실행하지 않고 만료시킨다. 전체 200종목에 동시에 주문을 내는 고빈도 엔진은 아니다.

### 연결 상태를 읽는 방법

- **내장 모의 신호기**는 `random-demo` 테스트 코드이며 외부 모델 연결을 뜻하지 않는다.
- **종목별 즉시 출력** 폴더는 종목 조회가 끝날 때마다 갱신된다. **전체 차트 파일**은 전체 순회 종료 시 갱신된다. 화면의 성공 시각도 각각 분리한다.
- 파일이 존재하거나 형식 검사를 통과했다고 외부 프로그램이 정상 동작한다고 단정하지 않는다. 실제 읽기·검증 접수 시각과 오류를 별도로 확인한다.
- HOLD/매수·매도 후보/접수는 주문·체결이 아니다. 실제 주문은 **실제 주문·체결**, 서버·시세·신호 로그는 **서버·감시 로그**에서 본다.
- 파일 검사는 감시 중에도 가능하다. 연결 설정 변경은 **감시·주문 중지** 후 진행한다. 검사 중 경로가 바뀌면 이전 검사 결과를 폐기한다.

화면 없이 순위와 차트만 갱신하려면 프로젝트 루트에서:

```powershell
uv run python scripts/export_demo_watchlist.py --days 30 --output .dockdack/exchange/charts.json
```

이 스크립트는 주문/계좌 변경 경로와 실전 도메인을 차단한 조회 전용 연결을 사용한다.
기존 종목의 N은 보존하므로 `--days`는 새 종목에만 적용된다.

## 순위의 범위

- 국내: KRX 코스피·코스닥 일반 기업 보통주, 관리종목 제외. 키움 종목 마스터의 시장·회사분류 및 상품별 목록으로 ETF/ETN/펀드/우선주/리츠/스팩 등을 제외한다.
- 미국: NASDAQ/NYSE/AMEX의 일반 기업 보통주. 키움 종목/ETF·ETN 목록과 한국투자증권 공개 종목 마스터, Nasdaq 종목 디렉터리·업종 정보를 함께 확인한다. ETF/ETN/펀드/우선주/리츠/스팩/예탁증서와 분류 불명 종목은 제외한다.
- 미국 공개 분류는 완전성을 보증하는 통합 증권유형 원장이 아니다. 서로 충돌하거나 정보가 부족한 종목은 보수적으로 제외하므로 일부 일반 기업 주식도 빠질 수 있다. 외부 분류 조회 실패 시 검증되지 않은 종목을 허용하지 않는다.
- 당일 누적 거래대금 순위 API를 사용한다. 휴장/장 시작 전에는 API가 제공하는 최근 값일 수 있다.
- 국내 원문 백만원 → **KRW**, 미국 원문 천달러 → **USD**로 변환한다. 두 통화의 금액을 합산해서 순위를 만들지 않는다.
- TOP100 버튼은 즉시 추가/갱신한다. 별도의 **장중 개장·매 정시 TOP100 재선정**은 감시 시작 후 시장별 거래일 일정에 따라 자동 갱신한다.
  자동등록된 순위 이탈 종목만 비활성화하며 수동/구버전 관심종목, 보유·미체결·미확정 종목, 대기 수동 규칙은 보존한다.
- 상품을 제외한 뒤 100개가 모일 때까지 순위 연속조회를 진행한다. 분류된 종목이 100개 미만이거나 연속조회에 문제가 있으면 임의 종목으로 채우지 않고 갱신을 실패시킨다.
- 종목 분류는 시장 현지 날짜별로 캐시한다. 자동주문 직전에도 같은 보통주 정책을 적용해 기존 목록이나 새 외부 신호가 필터를 우회하지 못하게 한다.
- 관심종목은 최대 500개다. 이전에 등록한 종목이 순위 밖이면 전체 관심목록은 200개보다 많을 수 있다.
- 키움의 클래스 구분 코드 `BRKb`처럼 끝의 소문자가 중요한 종목은 그대로 보존한다. 외부 코드도 내보낸 `symbol`을 그대로 되돌려 보내야 한다.

근거: [국내 거래대금 순위 ka10032](https://openapi.kiwoom.com/guide/apiGuideContents/05/ka10032),
[미국 거래대금 순위 usa20540](https://openapi.kiwoom.com/guide/apiGuideContents/35/usa20540).

분류 자료: [키움 국내 종목 마스터](https://github.com/Kiwoom-Securities/Kiwoom-REST-API/blob/main/examples/국내주식/종목정보/list_domestic_stocks.py),
[한국투자증권 공개 해외 종목 마스터 정의](https://github.com/koreainvestment/open-trading-api/blob/main/stocks_info/overseas_stock_code.py),
[Nasdaq 종목 디렉터리 정의](https://www.nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs).
일정, 재시작, 실패 재시도 및 구버전 목록 보존은 [지속 감시 운영 안내](server-mode.md)를 참고한다.

## 차트 JSON 계약 (schema_version 1)

최상위 필드는 `schema_version`, `export_id`, `created_at`, `source`, `adjusted_prices`,
`history_cache_max_seconds`, `intraday_history_refresh_seconds`, `completed_bars_persisted`, `scope`,
`quote_time_is_fetch_time`, `stocks`다. 기존 버전 1의 필드는 유지하고 설명 필드를 추가했다.
`scope`는 전체 목록 `watchlist` 또는 종목별 업데이트 `instrument`다.

각 `stocks` 항목:

| 필드 | 의미 |
| --- | --- |
| `watch_id`, `market`, `symbol`, `exchange`, `currency`, `name` | 종목 식별. market은 `domestic`/`us`, exchange는 `KRX`/`ND`/`NY`/`NA` |
| `turnover_rank`, `turnover`, `ranking_fetched_at` | TOP100 등록 종목의 시장별 순위, KRW/USD 거래대금 문자열, 순위 수신시각 |
| `requested_days`, `available_days`, `complete` | 요청/실제 거래일 수, 충분한 일봉이 있는지 |
| `status` | `ok`, `missing`, `error`. 실패 이유는 `error` 필드 |
| `price`, `quote_fetched_at`, `quote_age_seconds`, `quote_stale` | 현재가 문자열, 조회 시각/경과초, 조회 후 15초 초과 여부 |
| `history_fetched_at` | 마지막 실제 일봉 조회 시각. 구버전 표시용 캐시만 있으면 `null`. 현재가 시각과 다름 |
| `bars` | 과거→최근 순서의 `date`, `open`, `high`, `low`, `close`, `volume`, `is_current_day` |

가격·거래량·거래대금은 부동소수점 정밀도 손실을 피하기 위한 **숫자 문자열**이다.
일봉 날짜는 시장 현지 날짜이며 수정주가다. 당일 봉은 아직 변할 수 있다.
완료 과거 봉은 DB에서 재사용한다. 두 `*_seconds` 값 300은 당일 일봉의 재조회 간격을 뜻하며,
과거 일봉을 5분마다 모두 다운로드한다는 의미가 아니다. 다음 순회까지의 지연은 별도로 발생한다.
조회 시각은 거래소 마지막 체결시각이 아니며 시세 자체가 지연될 수 있다.
순차 조회여서 앞쪽 종목은 내보낼 때 이미 `quote_stale=true`일 수 있다. 동시 스냅샷으로 간주하면 안 된다.
24시간 넘은 캐시나 조회 실패를 정상 데이터로 내보내지 않는다. 일봉 개수가 부족해도 만들어 채우지 않는다.
모델은 `status`, `complete`, 시각, 당일 봉 포함 여부를 확인해 사용할 데이터를 선택해야 한다.
주문 가격은 이 파일의 가격을 그대로 쓰지 않고 **주문 직전 새로 조회**한다.

## 외부 신호 JSON 계약 (schema_version 1)

다음은 구조 설명용 예시다. `export_id`와 시각은 자신의 새 차트/추론 결과로 설정해야 한다.
수량과 금액은 사용자가 정한 정책 범위 안이어야 한다.

```json
{
  "schema_version": 1,
  "source_id": "external-model",
  "signals": [
    {
      "signal_id": "model-run-001-samsung",
      "export_id": "COPY_ACTUAL_EXPORT_ID",
      "market": "domestic",
      "symbol": "005930",
      "exchange": "KRX",
      "action": "buy",
      "quantity": 1,
      "max_notional": "300000",
      "generated_at": "2026-09-15T01:00:00+00:00",
      "expires_at": "2026-09-15T01:02:00+00:00"
    }
  ]
}
```

- `action`: `buy` / `sell` / `hold`. `hold`에는 수량·금액·주문유형·매도조건 필드를 넣지 않는다.
- `quantity`: 양의 정수 주식 수. 비율, 소수 주식, 공매도 수량이 아니다.
- `max_notional`: 해당 시장 통화의 **주문 총액 상한** 문자열. 지정가 단가가 아니다.
- `order_type`: 생략 시 `limit`(주문 직전 현재가 지정가). `market`은 국내만 가능하며 `ExternalPolicy.allow_market=True`가 필요하다.
  GUI는 내장 테스트 신호기를 선택한 경우에만 국내 시장가를 허용한다. 미국 시장가는 항상 거부한다.
  시장가의 `max_notional`은 최신 현재가로 추정한 금액 제한이지 실제 체결 총액의 보장은 아니다.
- `min_sell_price`: 매도에만 사용할 수 있는 양의 가격 문자열. 전송 직전 현재가가 이 가격 이상이어야 한다.
- `cost_profit_pct`: 매도에만 사용할 수 있는 0 초과 100 이하 비율 문자열. `"1"`이면 전송 직전 잔고의
  수량 가중 평균 매입가 대비 현재가가 1% 이상 높아야 한다. 시장가의 체결가격이나 순수익은 보장하지 않는다.
- `cost_loss_pct`: 매도에만 사용할 수 있는 **0 초과 100 미만** 손실 비율 문자열.
  `"0.8"`이면 전송 직전 실제 잔고의 수량 가중 평균 매입가 대비 현재가가 -0.8% 이하이어야 한다.
  양수 문자열로 손실 폭을 전달하며 `"-0.8"`, 숫자 `0.8`, `"0"`, `"100"`, 비정상 숫자는 거부한다.
  `cost_profit_pct` 또는 `min_sell_price`와 함께 넣을 수 없으며 `buy`/`hold`에도 넣을 수 없다.
  가격이 회복되어 조건이 사라지면 주문을 보류한다. 지정가 체결이나 실제 손실을 -0.8% 이내로 제한하는 보장은 아니다.
- `generated_at`, `expires_at`: 시간대가 있는 ISO 8601. 유효기간은 생성 후 최대 10분이며 최초 수신 시 생성 후 5분 이내여야 한다.
- `export_id`: 같은 DB에서 24시간 이내 내보낸 유효한 종목 데이터와 연결되어야 한다.
- `signal_id`: 같은 판단을 재전달할 때 동일 ID/내용/시각을 사용한다. 같은 출처와 ID의 동일 내용은 무시하며, 내용 변경은 오류다.
- 한 파일에는 종목별 최대 1개, 전체 최대 500개 신호, 파일 크기 최대 2MB. 알 수 없는 필드/중복 JSON 필드/비정상 숫자를 거부한다.
- 파일 전체를 검사하고 한 트랜잭션으로 반영한다. 일부만 반영한 뒤 나머지가 실패하는 방식이 아니다.
- 같은 출처/종목의 더 새로운 판단은 아직 대기 중인 이전 신호를 대체한다. 오래된/같은 생성시각의 다른 ID는 실행하지 않는다.
- **HOLD는 대기 신호를 중단할 뿐, 이미 접수한 주문을 취소하거나 보유 주식을 청산하지 않는다.** 파일에서 종목을 생략하는 것도 취소가 아니다.
- 새 ID를 붙여 기존 실패/거절 주문을 재시도하지 않는다. 접수 불명확 상태는 기록/주문 내역에서 수동 확인한다.
- `source_id`는 출처 식별자이지 암호학적 인증이 아니다. 신뢰하는 단일 프로세스만 입력 파일을 쓸 수 있는 로컬 경로를 사용한다.

파일은 임시 파일에 완전히 쓴 뒤 `os.replace()`로 교체한다. 쓰는 중인 파일을 읽는 문제를 피하려면
`dockdack.signal_bridge.atomic_json(path, payload)`를 사용한다. 신호 파일이 잘못되면 자동주문을 끄며,
내용 수정 후에도 사용자가 다시 활성화해야 한다. 최초 입력 파일이 아직 없는 것은 정상 대기 상태다.

## 외부 모델 연결 예제

`examples/external_signal_producer.py`의 `build_signals()`는 모델 결과를 신호 JSON으로 변환한다.
직접 실행하면 모든 유효 종목을 **HOLD**로 만들고 기본 입력과 다른 `signals.example.json`에 저장한다.

```powershell
uv run python examples/external_signal_producer.py
```

모델 쪽에서는 `decisions`에 `{watch_id: {"action": "buy" 또는 "sell", "quantity": 정수, "max_notional": 문자열}}`을 전달한다.
판단이 없는 종목은 HOLD다. 모델의 동일 판단을 재전달할 때 `decision_id`와 `generated_at`을 유지한다.
이 예제는 전략을 선택하거나 수익성을 판단하지 않는다. 필요한 전략 로직은 별도 모델 코드에서 작성한다.
GUI의 내장 `RandomDemoSignals`는 별도 실행 없이 같은 파일 계약으로 연결되는 테스트 전략이다.
미보유 10% 매수, +1% 익절 / -0.8% 손절이며 기본 수량은 각 1주다. 익절은 기존 `min_sell_price`와
`cost_profit_pct="1"`, 손절은 `cost_loss_pct="0.8"`만 사용한다. 두 판단 모두 실제 평균 매입가 기준이다.
설정 및 정확한 확률·가격 조건의 의미는 [테스트 신호기](server-mode.md#내장-모의-테스트-신호기)를 참고한다.

GUI 없이 연결할 때:

```python
from decimal import Decimal
from dockdack.autotrade import AutoTrader
from dockdack.gui_service import TradingService
from dockdack.signal_bridge import ExternalPolicy, SignalFileReader
from dockdack.watchlist import default_store

store = default_store()
engine = AutoTrader(TradingService(), store)
engine.external_only = True
engine.external_policy = ExternalPolicy(
    source_id="external-model", max_quantity=int(input("주문당 최대 수량: ")),
    max_krw=Decimal(input("주문당 KRW 상한: ")), max_usd=Decimal(input("주문당 USD 상한: ")),
)
engine.external_reader = SignalFileReader(store, ".dockdack/exchange/signals.json", engine.external_policy)
engine.poll()  # 수신/조회만, 주문 OFF
# 별도 사용자 확인을 받은 경우에만 engine.enable_orders("DEMO_AUTOTRADE")
# for results in engine.run_forever(30): ...
# 다른 제어 스레드에서 engine.stop()
```

상한은 **주문당** 제한이며 일일 누적 손실/회전율 한도는 아니다. 새 신호별 주문이 발생할 수 있다.
선택한 거래 환경·주문 권한 확인, 정규장 시간, 신호 만료/중복, 미체결, 보유/매도가능 수량, 통화, 자금과 주문금액,
현재가 재조회, 전송 의도 기록을 모두 통과해야 주문한다. 접수는 체결이 아니며 재시도/자동 정정은 하지 않는다.
상세 차단/해제 규칙은 [자동매매 사용법](autotrading.md)을 참고한다.

## 확인한 범위

2026-09-15 모의 조회 API로 국내 100 + 미국 100종목과 현재가/일봉 내보내기를 확인했다.
199종목은 30거래일, 스카이랩스(386380)는 API 제공 이력 8거래일로 `complete=false`다.
버크셔 B의 대소문자 구분(`BRKb`)을 종목 마스터와 조회 API에서 확인하고 정상화 과정의 손실을 수정했다.
실제 API 주문은 전송하지 않았다. 외부 신호의 주문 연동은 가짜 브로커/HTTP 및 임시 DB로 검증한다.
