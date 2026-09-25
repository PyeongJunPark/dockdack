# MK1.2 재사용 역사 백테스트

이 결과는 2025년 이후 이미 연구에서 확인한 과거 자료의 재평가입니다. 독립적인 새 테스트나 실전 체결 결과가 아닙니다.

## 실제 시가 진입 · 왕복 20bp 비용

| 시장 | 방식 | 신호 수 | 신호 성공률 | 체결 가정 거래 수 | 순수익률 | 최대낙폭 | 불확실 경로 거래 |
|---|---|---:|---:|---:|---:|---:|---:|
| domestic | carry | 473 | 37.6321% | 407 | -5.8640% | -6.0372% | 0 |
| domestic | eod | 473 | 37.6321% | 409 | -6.0665% | -6.1970% | 0 |
| us | carry | 1712 | 36.1565% | 1192 | -6.9049% | -7.5023% | 8 |
| us | eod | 1712 | 36.1565% | 1238 | -5.0422% | -5.4850% | 0 |

신호 성공률은 일봉 장벽 라벨 기준이며, 포트폴리오 승률과 다릅니다. 자금·동시보유·유동성 제한 때문에 신호가 모두 거래가 되지는 않습니다.

## 학습 구조 비교 · 시드 42의 두 개발 구간

아래는 2022/2024 개발 평가이며 위의 2025년 이후 재사용 백테스트와 다릅니다. ‘선택(진단용)’은 선택되었더라도 두 구간 앙상블 자격을 통과하지 못한 모델입니다. 합격도 실전 배포 승인이 아닙니다.

| 시장 | 구조 | 구간 | 매개변수 | 최적 epoch | 신호 | 성공률 | 20bp 차감 신호 평균 | Brier | 해당 구간 자격 | 선택 상태 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| domestic | mlp | walk_2022 | 378,820 | 2 | 1649 | 41.6616% | -0.2071% | 0.196386 | 미달 | 미선택 |
| domestic | rnn | walk_2022 | 69,604 | 2 | 1673 | 39.4501% | -0.2390% | 0.196930 | 미달 | 미선택 |
| domestic | lstm | walk_2022 | 158,884 | 2 | 1724 | 39.5592% | -0.2466% | 0.196805 | 미달 | 미선택 |
| domestic | gru | walk_2022 | 129,124 | 7 | 2706 | 43.4590% | -0.2021% | 0.196113 | 미달 | 미선택 |
| domestic | cnn | walk_2022 | 169,252 | 5 | 1909 | 44.6307% | -0.1742% | 0.195937 | 미달 | 선택(진단용) |
| domestic | resnet18 | walk_2022 | 2,335,204 | 1 | 1913 | 43.2305% | -0.1951% | 0.197062 | 미달 | 미선택 |
| domestic | resnet34 | walk_2022 | 4,234,660 | 1 | 1820 | 41.3736% | -0.2262% | 0.197212 | 미달 | 미선택 |
| domestic | mlp | walk_2024 | 378,820 | 2 | 861 | 40.8827% | -0.2443% | 0.181427 | 미달 | 미선택 |
| domestic | rnn | walk_2024 | 69,604 | 7 | 983 | 45.6765% | -0.1589% | 0.180817 | 미달 | 미선택 |
| domestic | lstm | walk_2024 | 158,884 | 7 | 960 | 46.5625% | -0.1452% | 0.180682 | 미달 | 미선택 |
| domestic | gru | walk_2024 | 129,124 | 7 | 1260 | 45.1587% | -0.1748% | 0.180603 | 미달 | 미선택 |
| domestic | cnn | walk_2024 | 169,252 | 2 | 1117 | 43.5989% | -0.2250% | 0.181362 | 미달 | 선택(진단용) |
| domestic | resnet18 | walk_2024 | 2,335,204 | 1 | 409 | 41.5648% | -0.2482% | 0.181931 | 미달 | 미선택 |
| domestic | resnet34 | walk_2024 | 4,234,660 | 1 | 418 | 37.5598% | -0.3409% | 0.181850 | 미달 | 미선택 |
| us | mlp | walk_2022 | 378,820 | 2 | 239 | 27.6151% | -0.4136% | 0.207736 | 미달 | 미선택 |
| us | rnn | walk_2022 | 69,604 | 8 | 622 | 29.4212% | -0.3453% | 0.206277 | 미달 | 미선택 |
| us | lstm | walk_2022 | 158,884 | 8 | 743 | 32.1669% | -0.3136% | 0.206371 | 미달 | 선택(진단용) |
| us | gru | walk_2022 | 129,124 | 8 | 850 | 33.6471% | -0.2982% | 0.206200 | 미달 | 미선택 |
| us | cnn | walk_2022 | 169,252 | 2 | 338 | 29.2899% | -0.3971% | 0.207311 | 미달 | 미선택 |
| us | resnet18 | walk_2022 | 2,335,204 | 4 | 202 | 23.7624% | -0.5239% | 0.207344 | 미달 | 미선택 |
| us | resnet34 | walk_2022 | 4,234,660 | 4 | 429 | 28.4382% | -0.4273% | 0.207189 | 미달 | 미선택 |
| us | mlp | walk_2024 | 378,820 | 1 | 172 | 13.9535% | -0.5076% | 0.198702 | 미달 | 미선택 |
| us | rnn | walk_2024 | 69,604 | 8 | 760 | 28.2895% | -0.2917% | 0.197961 | 미달 | 미선택 |
| us | lstm | walk_2024 | 158,884 | 17 | 747 | 28.7818% | -0.2427% | 0.198399 | 미달 | 선택(진단용) |
| us | gru | walk_2024 | 129,124 | 6 | 1013 | 27.9368% | -0.2964% | 0.198361 | 미달 | 미선택 |
| us | cnn | walk_2024 | 169,252 | 1 | 171 | 18.7135% | -0.4498% | 0.198734 | 미달 | 미선택 |
| us | resnet18 | walk_2024 | 2,335,204 | 1 | 487 | 21.5606% | -0.3775% | 0.198765 | 미달 | 미선택 |
| us | resnet34 | walk_2024 | 4,234,660 | 1 | 817 | 21.9094% | -0.3470% | 0.198724 | 미달 | 미선택 |

## 기간·표본·신뢰구간

- domestic: 2025-02-19~2026-08-21, 368거래일, 261,385개 기본 사건. 신호 성공률 95% 구간 31.1383%~45.3219%. 연구 자격 통과: False.
- us: 2025-02-18~2026-09-11, 394거래일, 1,076,163개 기본 사건. 신호 성공률 95% 구간 31.9751%~41.6255%. 연구 자격 통과: False.

구간은 날짜를 묶은 10거래일 이동 블록·2,000회·시드 42 방식이며, 표본 부족 시 산출하지 않습니다. 여러 실험을 반복해 본 영향이나 데이터 선택 편향을 보정한 구간이 아니며 자동매매 승인을 뜻하지 않습니다.

## 비용 민감도

| 시장 | 방식 | 왕복 비용(bp) | 순수익률 | 최대낙폭 |
|---|---|---:|---:|---:|
| domestic | carry | 0 | -2.2023% | -3.0217% |
| domestic | carry | 10 | -4.0497% | -4.4542% |
| domestic | carry | 20 | -5.8640% | -6.0372% |
| domestic | carry | 40 | -9.4632% | -9.5245% |
| domestic | eod | 0 | -2.3754% | -3.2293% |
| domestic | eod | 10 | -4.2316% | -4.6609% |
| domestic | eod | 20 | -6.0665% | -6.1970% |
| domestic | eod | 40 | -9.6770% | -9.7305% |
| us | carry | 0 | 4.0747% | -1.0549% |
| us | carry | 10 | -1.5880% | -3.1100% |
| us | carry | 20 | -6.9049% | -7.5023% |
| us | carry | 40 | -16.6548% | -16.7430% |
| us | eod | 0 | 6.6275% | -0.6518% |
| us | eod | 10 | 0.5943% | -1.8807% |
| us | eod | 20 | -5.0422% | -5.4850% |
| us | eod | 40 | -15.3865% | -15.3985% |

## 가상 후보가격 민감도 · 실거래 아님

시장마다 최대 20,000개 기본 사건을 결과와 무관하게 균등 추출(시드 42)하고 같은 행에 다섯 가격을 적용했습니다. 이는 독립 관측이나 실제 매매 횟수를 다섯 배 늘린 결과가 아닙니다. 가격이 당일 실제로 체결 가능했는지도 보장하지 않습니다.

| 시장 | 시가 배율 | 기본 사건 수 | 가상 매수신호 | 신호 비율 | 신호 성공률 |
|---|---:|---:|---:|---:|---:|
| domestic | 0.99 | 20,000 | 315 | 1.5750% | 83.1746% |
| domestic | 0.995 | 20,000 | 132 | 0.6600% | 63.6364% |
| domestic | 1 | 20,000 | 46 | 0.2300% | 34.7826% |
| domestic | 1.005 | 20,000 | 24 | 0.1200% | 12.5000% |
| domestic | 1.01 | 20,000 | 17 | 0.0850% | 0.0000% |
| us | 0.99 | 20,000 | 776 | 3.8800% | 94.7165% |
| us | 0.995 | 20,000 | 166 | 0.8300% | 62.0482% |
| us | 1 | 20,000 | 41 | 0.2050% | 24.3902% |
| us | 1.005 | 20,000 | 10 | 0.0500% | 10.0000% |
| us | 1.01 | 20,000 | 6 | 0.0300% | 0.0000% |

## 고정 가정·한계

초기자금 국내 1천만 원/미국 1만 달러, 최대 20종목, 시작일 자산의 5% 목표, 과거 20일 중앙 거래량의 0.1% 정수 주 한도입니다. 익절 +1%·손절 -0.9%, 동시 도달 시 손절 우선입니다.

- 2025+ history was inspected in prior research; this is reused history, not pristine out-of-sample evidence.
- Actual OPEN assumes immediate fills; no spread, impact, auction queue, partial fills, limits or latency is modeled.
- Daily OHLC cannot order both barriers: +1%/-0.9% both-touch is stop-first failure, not a verified intraday path.
- Zero-volume bars cannot fill; missing held-price paths lock capital and create explicitly uncertain accounting.
- 20bp is a hypothetical roundtrip rate, half charged to each side's actual notional, not actual broker fees/tax.
- Same-day intraday sales cannot finance that day's OPEN purchases; no leverage or short selling.
- EOD CLOSE liquidation is not an app order five minutes before the close; final-session liquidation is a convention.
- Four known US symbols are excluded for all periods; other corporate-action issues and survivorship bias may remain.
- Price sensitivity uses the same sampled base events repeatedly: counterfactual queries are not independent real trades.
- No threshold, architecture, calibration or trading permission is changed by this evaluation.

전체 예측·개별 모의 거래·고정 입력 해시는 `C:/Users/user/Desktop/dockdack/outputs/mark1/mark1-2-backtest-20260924-v1`의 시장별 JSON/NPZ 및 completed.json에 보관됩니다.
