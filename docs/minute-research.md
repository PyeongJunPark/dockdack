# 분봉·인버스 헤지 연구 (2026-09-30)

이 문서는 서로 다른 두 분봉 실험을 구분한다. 2026-09-29의 **일봉 사전학습→실제 5분봉 재학습** 가중치와 인버스 헤지 비용 평가는 오프라인 연구 결과이며 주문에 연결하지 않았다. 2026-09-30의 Mark1.29–1.37은 국내 완료 **일봉만** 새로 학습하고 실제 완료 5분봉을 재학습 없이 추론 입력으로 쓰는 모의투자 전용 실험이다. 둘의 번들·검증 결과를 서로 바꿔 쓰지 않는다. Mark1.38–1.44는 인버스 헤지 연구 상태를 AI 화면에 표시할 뿐 두 다리 주문을 만들지 않는다. 일봉을 실제 분봉이라고 주장하지 않으며, 어느 경로도 분봉 수익성을 입증하지 않았다.

## 데이터와 수집 경계

- 완료 일봉: 기존 정제 SQLite를 읽기 전용으로 열어 종목·기간을 지정해 JSONL로 추출한다. 시간은 각 거래일의 공식 정규장 종료 시각으로 붙이고, 품질 표시가 있는 봉·미완료 세션은 제외한다. 원본 DB를 바꾸지 않는다.
- 완료 분봉: 키움 모의 REST의 [국내 주식 `ka10080`](https://github.com/Kiwoom-Securities/Kiwoom-REST-API/blob/main/examples/%EA%B5%AD%EB%82%B4%EC%A3%BC%EC%8B%9D/%EC%B0%A8%ED%8A%B8/get_domestic_stock_minute_chart.py), [국내 업종지수 `ka20005`](https://github.com/Kiwoom-Securities/Kiwoom-REST-API/blob/main/examples/%EA%B5%AD%EB%82%B4%EC%A3%BC%EC%8B%9D/%EC%B0%A8%ED%8A%B8/get_domestic_sector_minute_chart.py)를 **조회만** 한다. 키움의 [미국 주식 `usa06011`](https://github.com/Kiwoom-Securities/Kiwoom-REST-API/blob/main/examples/%EB%AF%B8%EA%B5%AD%EC%A3%BC%EC%8B%9D/%EC%B0%A8%ED%8A%B8/get_overseas_stock_minute_chart.py) API는 존재하지만, 2026-09-29 AAPL 모의 응답 일부가 `20260928240000`처럼 `24시`를 나타내고 전체 시각의 시간대 의미가 공식적으로 확인되지 않아 **앱의 미국 분봉 조회는 네트워크 호출 전 차단**한다. 별도의 `--allow-network` 없이는 국내 수집도 호출하지 않는다. 시각의 시작/끝 의미가 명시되지 않아 현재 진행 중일 수 있는 봉과 정규장 양쪽 경계 봉을 보수적으로 버린다. 실제 휴장·반일장은 학습기/평가기에서 거래소 달력으로 다시 검사한다. 봉 결손을 가짜 가격으로 채우지 않는다.
- 수집 파일과 영수증은 `outputs/`에 새 이름으로만 저장하고, 해시·조회 페이지 수·잘린 과거 범위·삭제한 중복/경계 봉 수를 남긴다. 5페이지 조회는 **가장 최근 일부**일 뿐 전 기간 데이터가 아니다. 수집 영수증의 해시는 파일 일치 여부를 확인하며, 거래소 원장의 정확성을 증명하지 않는다.
- 국내 지수 `201`은 KOSPI200 **현물 지수 포인트**다. `ka20005` 가격의 부호 표기와 100배 정수 스케일을 모의 응답과 공식 명세에 맞춰 정규화한다. 지수는 거래 가능한 주식이 아니다.
- 국내 인버스 사례 `114800`은 KODEX 인버스다. 운용사의 [상품 자료](https://m.samsungfund.com/sheet/20260106/2ETF20_20251230.pdf)는 F-KOSPI200 *선물*지수 **일간** 수익률의 −1배 목표라고 설명한다. 따라서 여기의 현물 KOSPI200 `201`과 ETF 간 차이는 기초지수 차이, 일간 재설정, 비용 등이 섞인 관측 괴리이며 ETF 자체 추적오차라고 단정하지 않는다. −2배 인버스 상품은 현재 비교 경로에서 거부한다.

2026-09-29 모의 조회의 로컬 연구 자료(원본 파일은 Git에서 제외):

| 입력 | 범위·수량 | SHA-256 |
| --- | --- | --- |
| 삼성전자 `005930` 완료 일봉 | 2023-01-01–2026-08-21, 887봉 | `73c02832237e768778491ae464d25028e25efeb203d881bff70f98bc6c2e3206` |
| 삼성전자 5분봉 | 2026-07-13–2026-09-29, 정규장 3,859봉·53거래일 | `b679df32ed4e89cff90e25df7cd8e8755c553c96eae53c9d570691240ca37b9b` |
| KODEX 인버스 `114800` 5분봉 | 2026-07-03–2026-09-29, 4,267봉·59거래일 | `baf2f332bcc59078c7d7154bbe7ce64d48a8719b79925ac02441500dd7e921d7` |
| KOSPI200 지수 `201` 5분봉 | 2026-07-03–2026-09-29, 4,325봉·59거래일 | `1ac7633c5630e7a0971746b75348b9bba222dd4263947d45303e25ff7fb41861` |

현재 장중 수집분이 포함되어 있어 실제 평가기는 미완료 최근 세션을 별도로 제외하거나 완료된 과거 구간만 사용해야 한다. 한 종목과 약 두 달의 선택 표본은 시장 전체 또는 미래의 효용을 대표하지 않는다.

## 일봉 사전학습 → 실제 5분봉 적응

`dockdack/research/minute_transfer.py`의 후보는 선형, 시간축 합성곱, GRU 세 구조다. 일봉과 분봉을 합쳐 하나의 시간축으로 위장하지 않는다. 각각 완료 OHLCV의 20/30봉 창을 독립적으로 만들고 일봉에서 사전학습한 뒤, 실제 분봉의 훈련 구간으로 재학습한다. 분봉은 동일 종목·거래소·세션에서 정확히 연속된 5분봉만 사용한다. 날짜순 훈련/검증/테스트 사이 세 거래 세션을 비우고 정규화는 훈련 구간만으로 구한다. **일봉 사전학습의 마지막 라벨 종료가 분봉 적응학습의 첫 진입보다 앞서야 하며**, 역전되면 입력을 거부한다. 이번 삼성전자 자료는 각각 2025-07-22 09:00과 2026-07-14 11:00(KST)였다. 검증 구간만 후보 신호 기준을 선택하고 테스트 결과로 기준을 다시 고르지 않는다.

분봉 라벨이 시작/종료 어느 쪽인지 확정되지 않았기 때문에, 완료 신호봉 뒤 두 개 봉을 비우고 세 번째 후속 봉의 **과거 관측 시가**를 가상 진입 가격으로 쓴다. 청산도 후속 봉 시가를 사용한다. 이는 실제 체결 보장이 아니며 매수·매도 각각 수수료 2bp와 슬리피지 8bp를 차감한다. 학습 목표는 지정가 체결이나 장중 고가/저가 장벽의 선후관계가 아니라 이 지연 시가 간 비용 후 수익의 양수 여부다.

창/보유시간 세 설정 × 세 구조, 총 9개 후보의 실자료 탐색 결과다. 뒤의 두 설정은 첫 실행 결과를 본 뒤 추가했으므로 사전등록된 단일 검증 실험으로 취급하지 않는다.

| 입력창 / 보유시간 | 분봉 훈련·검증·테스트 표본 | 세 모델의 검증 신호 기준 | 테스트 매수 후보 |
| --- | --- | --- | --- |
| 20봉 / 3봉 | 1,569 / 98 / 224 | 모두 보류 | 모두 0 |
| 20봉 / 6봉 | 1,467 / 92 / 209 | 모두 보류 | 모두 0 |
| 30봉 / 12봉 | 942 / 60 / 129 | 모두 보류 | 모두 0 |

기준을 통과한 후보가 없었으므로 이 가중치 9개는 `outputs/minute-research/transfer-*`의 **연구 번들**로만 남겼다. `models/`의 운영 번들이나 Mark1.n prototype으로 승격하지 않았고, 화면·신호기·자동주문에도 연결하지 않았다. 이 결과는 위 비용·기간·종목의 보수적 실험 결과이지 다른 종목/기간에서 유효한 후보가 절대 없다는 증거는 아니다.

`examples.paper_minute_signal`은 이 연구 번들의 **다음 거래 세션**에 수집한 최근 5분봉과 영수증을 받아 세 구조의 확률 대리값을 오프라인 보고서로 쓰는 종이 신호 모드다. 현재 진행 중인 정규장, 15분 이내 수집 영수증·봉, 현재 세션의 연속된 입력창, 훈련에 쓰지 않은 새 자료만 허용한다. 분봉 적응학습에 쓴 종목·거래소도 번들에 기록해 다른 종목/거래소로 바꿔 넣으면 거부한다. 최종 안전 계약으로 재학습한 로컬 번들은 `outputs/minute-research/transfer-005930-causal-20260929/`이며, 이전 번들은 종이 신호 경로에서 거부한다. 신호 기준이 `None`인 지금 번들은 매수 후보를 낼 수 없다. 브로커·계좌·주문을 불러오지 않으며 미래 세션의 체결/수익 검증도 아직 수행하지 않았다.

## 거래량 종목 + 시장지수/인버스 헤지

`dockdack/minute_hedge_research.py`는 완료 분봉으로만 종목 매수와 지수 헤지의 가상 손익을 비교한다. **인버스 ETF 분봉을 넣으면 ETF 실제 관측가격으로 가상 매수**를 평가하고, 넣지 않으면 명시적인 합성 지수 숏 참조 계산만 한다. 어느 쪽도 브로커 숏·ETF 주문이 아니다. 지수 종가를 보고 난 뒤의 시가를 같은 순간 체결로 취급하지 않도록 신호와 청산을 각각 세 번째 후속 봉 시가로 지연하고, 거래량 조건은 관측된 과거 봉만 쓴다. 각 후보의 진입 문턱은 이전 훈련 날짜만으로 고르는 expanding walk-forward이며, 결과가 없는 후보는 보류한다. 이 연구의 +1% 익절/−1.5% 손절 판단은 **완료봉 종가 관측 기준**이다. 봉 중 고가/저가의 첫 도달이나 지정가 체결로 해석하지 않는다.

일곱 독립 가설은 시장 대비 잔차 Z점수, 업종 대비 상대 낙폭, 실제 거래대금 VWAP, ATR 정규화 낙폭, 전일 갭 이후 회복, 거래량 급증 후 상대 낙폭, 과거 가격범위의 저점 이탈·회복이다. 마지막 두 가설은 현행 OHLCV와 지수 자료만으로 평가할 수 있게 했다. 후보 가설·문턱·비용 설정을 같은 자료를 보면서 다듬었으므로 이번 테스트는 **독립 블라인드 검증이 아니다**.

삼성전자·KOSPI200 현물지수·KODEX 인버스 실수집 분봉을 시각별로 맞추고 잘린 가장 오래된 세션과 수집 중이던 최근 세션을 제외한 **51개 연속 세션(2026-07-14–09-28)**으로 실행했다. 각 다리 가상 명목 100만 원, 과거 12봉을 본 뒤 최대 6봉 보유, 이전 10세션 이상·훈련 거래 3건 이상에서만 문턱을 골랐다. 조건별 결과는 다음과 같다.

| 비용 가정(매수·매도 각 다리, 편도) | 평가 가능한 테스트 거래 | 결정 |
| --- | --- | --- |
| 수수료 2bp + 슬리피지 8bp | 모든 7개 후보 0건 | 전부 보류. 비용 후 기준을 통과한 학습 선택/테스트 거래가 없음. |
| 민감도: 수수료 1bp + 슬리피지 2bp | 거래량 급증 후 반전 1건, 가격범위 회복 1건 | 각각 비용 후 약 **−2,824원**, **−5,020원**. 평가 가능했다는 뜻일 뿐 수익성 통과 아님. |

시장 잔차와 ATR 후보는 학습 선택 또는 테스트 거래 수가 부족했다. 업종·실제 거래대금/VWAP·전일 갭 후보는 필요한 별도 데이터가 없어 보류했다. 둘 다 손실인 민감도 결과와 보수 비용의 전원 보류를 고려해, 어느 후보도 자동주문·모의주문으로 승격하지 않았다. 최종 연구 보고서는 `outputs/minute-research/hedge-research-final-conservative-20260929.json`과 `hedge-research-final-low-cost-20260929.json`에 저장돼 있고 Git에는 포함되지 않는다.

잔여 데이터 제약: 현재 키움 주식 분봉의 OHLCV에는 실제 분당 거래대금이 없어 거래대금/VWAP 후보를 평가할 수 없다. 종목 업종 분봉이나 공식 전일 종가가 없으면 업종 상대/갭 후보도 보류한다. 종가×거래량을 실제 VWAP이나 거래대금으로 속여 쓰지 않는다. 지수 현물과 인버스 ETF 상품의 기초가 다르며, 종목과 ETF의 같은 명목금액도 종목 베타를 중립화한다는 뜻이 아니다. 명목금액 기반 가상 계산에는 정수 주식, 호가 스프레드·깊이·대기열·실제 체결 가능성이 반영되지 않는다.

Mark1.38–1.44의 AI 화면 연결은 위 일곱 가설의 **고정된 주문 보류 상태**다. 선택하거나 조회해도 과거 보고서를 다시 평가하거나 현재 분봉으로 종목·ETF 비중을 산출하지 않는다. 모의·실전 모두 주문 신호는 HOLD로만 남긴다.

## Mark1.29–1.37: 일봉 전용 학습 → 실제 5분봉 모의 추론 (2026-09-30)

이 9개 모델은 위의 2026-09-29 분봉 적응 번들을 사용하지 않는다. 국내 거래대금으로 고른 40개 종목의 완료 일봉을 `2026-08-21`까지만 사용했다. 기업행사 전후를 잇지 않도록 종목당 한 개의 연속 구간만 남기고 품질 문제 266행을 제외한 **53,807개 완료 일봉**을 학습·검증·일봉 테스트 원천으로 삼았다. 완료 봉 20/30개의 상대 OHLCV를 일반 순서열로 취급하며, 최종 특징 스키마 `range-normalized-relative-ohlcv5-v2`는 입력창의 중앙 고가·저가 범위로 가격 변화를 정규화한다. 분봉 가중치 재학습, 분봉 정규화 재적합, 분봉 임계값 조정은 하지 않는다.

| 모델 | 일봉 학습 입력 / 대리 보유 구간 | 구조 | 모의 확인 lot의 시한 |
| --- | --- | --- | --- |
| Mark1.29–1.31 | 20봉 / 3개 봉 | 선형망·시간축 합성곱·GRU | 체결 후 15분 |
| Mark1.32–1.34 | 20봉 / 6개 봉 | 선형망·시간축 합성곱·GRU | 체결 후 30분 |
| Mark1.35–1.37 | 30봉 / 12개 봉 | 선형망·시간축 합성곱·GRU | 체결 후 60분 |

일봉 정답은 완료된 신호봉 뒤 세 번째 후속 일봉의 과거 관측 시가에 진입하고, 각 보유 구간 뒤 관측 시가에 청산한 비용 후 수익 방향이다. 학습·검증·테스트는 날짜순으로 나누고 세 거래 세션을 비웠다. 검증 상위 20% 임계값은 **후보 빈도 조건**으로 정한 것이며 수익성 최적화나 실제 5분봉 승률 보정이 아니다. 40종목을 2026년 최근 거래대금으로 고른 탓에 2021–25년 과거 구간에는 미래 정보에 의한 종목 선정·생존 편향이 있다. 일봉 테스트의 대리 성과도 전체 시장 재현 수익이나 분봉 수익이 아니다.

실행 시에는 키움 **모의투자**의 국내 KRX 일반 주식에서, 학습한 40종목에 대해서만 실제 완료 5분봉을 요청한다. 학습 대상 밖 종목은 분봉 요청 없이 보류한다. 9개 모델은 한 종목·완료 슬롯의 조회 결과를 해시 영수증이 있는 공유 캐시에서 재사용하고, 현재 장의 최신 연속 봉 20/30개와 신호 뒤 두 개의 추가 완료 봉을 검사한다. 최신 페이지와 저장 봉을 합쳐 필요한 연속 구간이 부족할 때만 조회 페이지 상한을 1→2→5로 늘린다. 공식 API에 없는 시각 커서를 사용하지 않는다. 봉 게시 지연·조회 실패는 같은 완료 슬롯에서 20초 이상 간격으로 최대 3회 시도하고, 확인될 때까지 보류한다. 신호봉 기준 최소 15분 뒤에만 후보가 유효하며, 학습 종목·번들 정체성·시각·장 경계·남은 보유 시한이 맞지 않거나 캐시/네트워크 검증이 실패하면 보류한다. 상단 자동주문 ON 요청 후 전체 조회 검증과 감시 시작을 거쳐 주문이 허용된 상태에서 일반 주문 직전 검사도 통과해야 전송된다. 확인된 자기 매수 lot에만 +3% 익절·−2% 손절, 체결 관측 후 3/6/12개 5분봉 시한 및 장마감 5분 전 청산 후보를 적용한다. 개장 전·장 종료 후·휴장일에 처음 체결을 확인하면 이전 체결시각을 추정하지 않고 관측 시각부터 시한이 지난 이후 첫 정규장 시점부터 기간 매도 후보를 낸다. 보유 화면은 예정 시각을 거래소 현지 `YYYY-MM-DD HH:MM`으로 표시한다. 모의 모델의 확인된 보유분 매도에서 1주 가격이 양수인 주문금액 상한을 넘으면 1주만 예외로 허용하고, 여러 주는 기존 상한으로 계산한다. 상한 0·실전·수동 주문에는 이 예외를 적용하지 않는다. API 조회 사이에 지나간 가격이나 실제 주문 체결을 보장하지 않는다.

최종 봉인 번들은 `models/mark1_minute/daily-proxy-domestic-{20x3,20x6,30x12}-segment-safe-20260930/`의 세 폴더에 있다. 각 manifest에는 세 구조의 가중치 해시와 로컬 일봉 원본·영수증 검사 결과가 기록된다. 이 이름은 세 폴더의 공통 형식을 나타내며 셸 경로 확장 문법이 아니다. 2026-09-29 13:36에 수집한 삼성전자 `005930` 5분봉을 13:30 기준 완료봉까지만 재생한 로딩·추론 검사에서 9개 모두 유한한 점수를 반환했고 Mark1.34·35·36이 BUY 후보였다. 보고서는 `outputs/minute-research/all-nine-daily-proxy-load-smoke-20260930.json`이다. 사후 수집 자료의 입력 연결 **동작 확인**이며 독립적인 전진 성과, 실제 체결, 분봉 수익성 검증으로 해석하지 않는다. 새 모의 계정에서 9개는 선택 상태로 시작할 수 있으나 감시·주문은 OFF다.

## 재현 명령

`uv sync --extra research` 이후, 새 출력 경로를 지정한다. `outputs/`의 원본·영수증은 Git에서 제외되므로 먼저 로컬 파일 존재와 해시 일치를 확인한다. 명령의 새 출력 폴더는 예시이며 기존 봉인 번들을 덮어쓰지 않는다.

### 2026-09-30 일봉 전용 40종목 학습

아래 입력은 `daily-proxy-40-single-segment-stocks-2021-2026.jsonl`과 같은 이름의 `.receipt.json` 영수증이다. 이 파일은 40종목의 단일 연속 구간·품질 제외·2026-08-21 컷오프를 적용해 별도로 내보낸 로컬 자료다. `train_daily_proxy_minute`는 일봉과 영수증을 확인하고 세 구조를 학습하며 **분봉 파일을 받지 않는다**. 다시 학습한 결과를 앱의 `models/`에 자동 설치하지 않는다.

```powershell
uv run --no-sync python -m examples.train_daily_proxy_minute --market domestic --daily outputs/minute-research/daily-proxy-40-single-segment-stocks-2021-2026.jsonl --lookback 20 --horizon 3 --first-model-number 29 --output outputs/minute-research/recheck-daily-proxy-20x3
uv run --no-sync python -m examples.train_daily_proxy_minute --market domestic --daily outputs/minute-research/daily-proxy-40-single-segment-stocks-2021-2026.jsonl --lookback 20 --horizon 6 --first-model-number 32 --output outputs/minute-research/recheck-daily-proxy-20x6
uv run --no-sync python -m examples.train_daily_proxy_minute --market domestic --daily outputs/minute-research/daily-proxy-40-single-segment-stocks-2021-2026.jsonl --lookback 30 --horizon 12 --first-model-number 35 --output outputs/minute-research/recheck-daily-proxy-30x12
```

### 2026-09-29 분봉 재학습·헤지 오프라인 연구

아래의 삼성전자 1종목 분봉 재학습은 앞의 40종목 일봉 전용 모델과 다른 연구 경로다.

```powershell
uv run --no-sync python -m examples.export_daily_research_bars --database data/kiwoom_daily/clean-20260916-v1/domestic_daily_clean.sqlite3 --market domestic --exchange KRX --symbols 005930 --from-date 2023-01-01 --through-date 2026-08-21 --output outputs/minute-research/new-daily.jsonl
uv run --no-sync python -m examples.collect_minute_research --market domestic --exchange KRX --symbols 005930 --interval-minutes 5 --max-pages 5 --output outputs/minute-research/new-stock-5m.jsonl --allow-network
uv run --no-sync python -m examples.collect_minute_research --market domestic --exchange INDEX --index-code 201 --interval-minutes 5 --max-pages 5 --output outputs/minute-research/new-index-5m.jsonl --allow-network
uv run --no-sync python -m examples.collect_minute_research --market domestic --exchange KRX --symbols 114800 --interval-minutes 5 --max-pages 5 --output outputs/minute-research/new-etf-5m.jsonl --allow-network
uv run --no-sync python -m examples.train_daily_to_minute --market domestic --daily outputs/minute-research/new-daily.jsonl --minute outputs/minute-research/new-stock-5m.jsonl --output outputs/minute-research/new-transfer-bundle
```

훈련이 끝난 다음 거래 세션에 새 분봉 파일을 수집했을 때만 종이 신호를 조회할 수 있다. 아래 `YYYY-MM-DD`는 실행 시점의 현지 거래일이다. 이미 학습에 쓴 분봉 파일이나 과거 세션을 넣으면 거부한다.

```powershell
uv run --no-sync python -m examples.paper_minute_signal --bundle outputs/minute-research/new-transfer-bundle --minute outputs/minute-research/newer-stock-5m.jsonl --market domestic --exchange KRX --symbol 005930 --session YYYY-MM-DD --output outputs/minute-research/new-paper-signal.json
```

인버스 연구 보고서는 다음처럼 비용·명목금액·상품의 −1배 사실 확인을 명시해 별도로 만든다. `--confirm-inverse-etf-minus-one`은 운용사 상품 설명을 사람이 확인했다는 표시이지 프로그램이 상품 성격을 자동 인증한다는 뜻은 아니다.

```powershell
uv run --no-sync python -m examples.run_minute_hedge_research --market domestic --stock-file outputs/minute-research/new-stock-5m.jsonl --index-file outputs/minute-research/new-index-5m.jsonl --stock-symbol 005930 --index-code 201 --inverse-etf-file outputs/minute-research/new-etf-5m.jsonl --inverse-etf-symbol 114800 --confirm-inverse-etf-minus-one --commission-bps 2 --slippage-bps 8 --synthetic-index-borrow-bps-annual 500 --notional-per-leg 1000000 --min-prior-volume 10000 --min-train-sessions 10 --min-train-trades 3 --output outputs/minute-research/new-hedge-report.json
```

이 절의 분봉 재학습·헤지 연구 명령이 작성한 결과는 Mark1.29–1.37의 동결 번들이 아니며 주문 정책에 자동 반영되지 않는다. 일봉 전용 9개 모델의 모의 후보도 시각 라벨·가격 경로·체결 가정에 큰 불확실성이 남아 있다. 실전 계좌나 인버스 두 다리 주문으로 연결하지 않는다.
