# mark_1 정밀도 우선 선별 모델: 고정 연구 계획

작성: 2026-09-16. 실험 전 계획이며 결과 보고가 아니다. 이전 ResNet/Inception 실험은 보존한다. 사용자 요청에 따라 신호 빈도보다 실제 매수 신호의 성공 비율을 우선한다. 기존 운영 모델·GUI·주문 설정은 바꾸지 않는다.

## 유지할 조건과 이번 변경

- 과거 완료된 30봉 OHLCV + 당일 후보 매수가. 당일 최종 고가·저가·종가·거래량은 입력하지 않는다.
- 당일 고가 +1% 도달, 저가 −0.9% 미도달이면 성공. 두 경계가 닿으면 손절 우선/실패다.
- 평가 진입가는 실제 시가이며, +1% 익절/−0.9% 손절은 그대로다.
- **p>50%는 최소 조건**이고 더 높은 보류 임계값을 별도 과거 구간에서 선택한다. 기존의 모든 p>50% 매수 정책과 구분한다.
- 이번 비교는 실제 시가 관측만 학습한다. 기존 가상가격 증강에서 생기는 기계적 정답 편향을 피하기 위한 명시적 변경이다. 임의 장중 후보가격 입력은 가능하지만 그 진입 이후의 실제 경로는 검증되지 않았다.
- 가격을 바꾸거나 목표일의 고저 범위로 후보를 사후 선별해 새로운 실제 관측을 만든 것처럼 세지 않는다.

## 문헌과 구현 범위

[문헌 검토](mark1-selective-literature.md)의 거절/보류 분류, 금융 비선형 특징 상호작용, 별도 확률 보정과 시간순 검증을 적용한다. 일반적인 accepted accuracy가 아니라 **BUY precision**을 평가한다. SelectiveNet의 end-to-end 세 head, TRA 라우터, out-of-time meta-labeling은 이번 첫 비교의 구현이라고 주장하지 않는다.

Gu–Kelly–Xiu의 금융 예측 비교는 모멘텀·유동성·변동성 및 비선형 모델 비교의 참고다. 논문의 월별 자산가격 예측과 현재 당일 장벽 문제는 다르며 수익 수치를 차용하지 않는다. [저자 논문](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf)

## 184개 과거 특징

고정 창 5/10/20/30일의 수익률·상하방 변동성·고저 범위·로그 true-range, 몸통·꼬리·봉 내부 종가 위치, 거래량 수준/변화, 완료된 과거 봉의 익절만/손절만/양쪽/미도달 비율을 사용한다. 수익률 lag 1/2/3/5/10/20/29, 최근 봉 형태, 후보가와 직전 종가·과거 고저/평균의 거리도 포함한다.

특징은 각 표본의 과거만으로 계산하며 전 기간 통계·종목 ID·목표 날짜를 입력하지 않는다. 가격×거래량은 과거의 대용 지표일 뿐 실제 거래대금이나 유통주식수가 아니다. 고정 [-20,20] clipping, float64 중간 계산과 float32 출력을 사용한다. 상세 순서는 `mark1_selective_features.FEATURE_NAMES`로 저장한다.

## 모델과 학습

| 후보 | 학습 대상 | 학습 종료 선택 |
|---|---|---|
| CatBoost depth 6 | 성공/실패 Logloss | 이후 tune 연도 PR-AUC |
| CatBoost depth 8 | 같은 조건에서 복잡도 비교 | 이후 tune 연도 PR-AUC |
| CatBoost joint depth 6 | 익절만/손절만/양쪽/미도달 MultiClass | 이후 tune 연도 MultiClass loss |
| LightGBM 31 leaves | 성공/실패 binary loss | 이후 tune 연도 average precision |

CatBoost GPU Plain boosting, lr .03, L2 10, border 128, Bernoulli subsample .8. CatBoost의 ordered boosting을 그대로 재현했다고 부르지 않는다. LightGBM은 Windows CPU 12 threads, lr .03, 최소 leaf 200개, L2 10, feature fraction .85, bagging .8, max bin 127. 최대 3,000회, 개선 없는 150회 후 종료하며 최적 iteration을 보관한다. 클래스 가중치·양성 oversampling은 하지 않는다.

4후보 × 2구간 × 2시장 × seed42 =16회 비교한다. 구조를 잠근 뒤 **두 구간 모두**에 선정 구조 seed43·44를 더 학습한다. 총24회이며 좋은 seed만 고르지 않는다. raw success logits를 평균한 앙상블을 다시 별도 구간에서 확률 보정하고 보류 임계값을 정한다.

## 다섯 단계 시간 분리

| 구간 | 가중치 학습 | 종료 검증 | 확률 보정 | 매수 정책 선택 | 다음 연도 감사 |
|---|---|---|---|---|---|
| walk_2022 | 2012–2019 | 2020 | 2021 상반기 | 2021 하반기 | 2022 |
| walk_2024 | 2014–2021 | 2022 | 2023 상반기 | 2023 하반기 | 2024 |

학습 뒤 연도 경계 및 확률 보정/정책 선택의 반기 경계마다 30 **거래일**을 제거한다. 뒤 구간 입력의 첫 봉이 이전 구간 마지막 날 이후여야 한다. 학습 종목군은 해당 학습 기간의 승인 표본으로만 정한다. 학습 최대150만, 종료 검증 최대15만은 고정 seed42로 정답과 무관하게 균등 추출한다. 다른 구간은 적격 표본 전체다. 한 표본의 여러 종목/시점 관계를 독립 관측 수로 과장하지 않는다.

상반기에는 unweighted monotone Platt 확률 보정을 한다. joint 모델의 손절 확률은 `stop_only+both` logit을 별도로 보정한다. 독립인 것처럼 P(take)×P(no stop)을 곱하지 않는다. 두 이진 보정 확률을 다시 합이1인 4-class 분포라고 주장하지 않는다.

하반기에 임계값 `{.50,.55,.60,.65,.70,.75,.80,.85,.90,.95}`를 비교한다. joint 모델만 손절 위험 제한 없음/25% 이하 두 조건을 비교한다. 성공 확률은 경계보다 **엄격히 커야** 하고 손절 확률 상한은 이하이다. 높은 임계값이 반드시 더 높은 실제 정밀도를 뜻하지 않는다.

## 드문 신호의 증거 기준

희귀 신호 요청을 반영해 이전 200건/50일 조건 대신, 이번 실행 **전부터** 최소50신호·20신호일·10종목으로 고정한다. 신호가 적어도 되는 것과 표본 5건의 100%를 성공으로 인정하는 것은 다르다.

모든 평가 날짜를 포함해 10거래일 moving-block을 2,000회 재표집한다(seed42). 위 최소 증거량이 부족하면 구간을 표시하지 않는다. 각 날짜의 종목은 함께 재표집한다. 95% precision/net proxy 구간을 보고하되, 시장 의존성과 반복 선택을 완전히 보정하는 IID/conformal 보장이라고 부르지 않는다.

정책 선택은 충분한 증거가 있는 후보 중 정밀도 구간 하한, 순수익 구간 하한, 신호 수 순으로 고른다. 동률이면 손절 제한 없는 단순 정책/낮은 문턱을 우선한다. 충분한 후보가 없으면 p>.5를 **진단용**으로 기록하고 미통과 처리한다. 무거래는 100% 성공이 아니다.

연구 통과는 다음을 모두 요구한다.

- 관측 BUY precision ≥65%, 95% 하한 >57.9%.
- 왕복20bp를 뺀 평균 일봉 proxy 수익의 95% 하한 >0.
- 최소50신호·20일·10종목, 올바른 block/cost metadata.
- 최종 앙상블의 정책 선택 구간과 다음 연도 감사 구간 모두, 두 시간 구간 모두 통과.

구조 순위는 두 감사 기간 모두 증거가 있는 모델 중 **더 나쁜 기간의 정밀도 하한**을 우선한다. 다음은 평균 순수익 하한/총 신호 수다. 전부 증거가 부족하면 전체 Brier 순위로 진단 대상을 고르지만 목표 달성으로 판정하지 않는다. 여러 구조 선택에 쓰인 감사 연도는 완전히 독립된 최종 시험이 아니다.

## 이후 백테스트와 보호

양 시장 학습·구조·정책이 모두 잠긴 뒤에만 2025년 이후를 계산한다. 이미 이전 연구에서 본 자료이므로 **재사용 역사 평가**라고 부른다. 이 결과에 맞춰 문턱·모델·최소 표본 수를 바꾸지 않는다.

기존 MLP와 새 모델을 동일 승인 색인에서 비교한다. 자본 국내1천만원/미국1만달러, 최대20종목, 한 종목5%, 과거20일 거래량 중앙값0.1% 한도, 정수수량, 왕복0/10/20/40bp를 적용한다. 익절/손절까지 보유와 미도달 시 당일 종가 청산을 둘 다 기록한다. 기존 엔진의 갭 가격·양쪽 손절 우선·시가 매수에 오후 매도금 사용 금지·누락 경로 잠정 평가를 유지한다.

`outputs/mark1/selective-20260916`에 설정/코드 hash, 모델, raw 예측, 확률 보정, 문턱, 전체 후보 결과를 보관한다. 원본·정제 DB와 `models/mark1/*.pt`를 덮어쓰지 않는다. 환경 추가 라이브러리는 Git 제외 연구용 폴더에만 설치했다. 자동주문·커밋·푸시·병합은 하지 않는다.

```powershell
$env:PYTHONPATH='C:/Users/user/Desktop/dockdack-mark_1/outputs/mark1/selective-deps'
& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.train_mark1_selective
```

실제 일봉의 관측/생존/정제 편향과 임의 장중 진입 경로 부재는 그대로다. 신호 품질이 개선되더라도 미래 미사용 기간의 고정 예측 검증과 실제 비용/체결 검증 전에는 배포하지 않는다.

## 실행 결과와 연구용 추론

위 내용은 실행 전에 고정한 계획이다. 아래는 실행 후 추가한 사용 안내이며 계획/모델/문턱을 바꾸지 않는다. [실제 결과와 그래프](../reports/mark1-selective-20260916/REPORT.md)에서 전체 후보와 실패 결과도 확인할 수 있다. 국내는 `cat_joint6`, 미국은 `cat_binary8`의 3-seed 앙상블이 선정됐지만 **양 시장 모두 연구 기준 미통과**다. 최종 선별 정책은 기본 p>50%와 동일하므로 신호를 더 줄이는 데 성공한 결과가 아니다.

저장 위치:

- `outputs/mark1/selective-20260916`: 고정 계획, 24회 학습 기록, native 모델, 확률 보정/정책 선택, 개발 감사 예측, 시장별 CPU 추론 감사. 모델은 기존 `models/mark1`에 덮어쓰지 않았다.
- `outputs/mark1/selective-backtest-20260916`: 재사용 역사 예측과 64개 포트폴리오 장부, 독립 장부 감사.
- `reports/mark1-selective-20260916`: 사람이 읽는 보고서, 요약 JSON, 후보/신호 빈도/수익/비용 그래프 네 장.

기본값으로 완료한 실험의 백테스트와 보고서를 실행하는 명령은 다음과 같다. 재실행은 계산 시간이 들고 해당 결과 폴더를 갱신할 수 있으므로 단순 확인에는 저장 보고서를 연다. 선택적인 라이브러리는 Python 3.13용 격리 폴더에 설치됐으므로 다른 Python 버전에 그 폴더를 연결하지 않는다.

```powershell
$env:PYTHONPATH='C:/Users/user/Desktop/dockdack-mark_1/outputs/mark1/selective-deps'
& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.backtest_mark1_selective
& 'C:/Users/user/Desktop/dockdack/.venv-ml-cuda/Scripts/python.exe' -m examples.report_mark1_selective
```

위 작업 폴더에서 같은 Python 환경으로 다음 함수를 사용할 수 있다. `completed_ohlcv`는 오래된 순서의 **완료 30봉**, 열 순서는 시가·고가·저가·종가·거래량이다. `candidate_price`에는 같은 통화/가격 조정 기준의 양수 가격을 넘긴다.

```python
from dockdack.mark1_selective_inference import SelectivePredictor

predictor = SelectivePredictor("outputs/mark1/selective-20260916/domestic")

def inspect_candidate(completed_ohlcv, candidate_price):
    return predictor.predict(completed_ohlcv, candidate_price)
```

미국은 경로의 `domestic`을 `us`로 바꾼다. 실제 당일 시가를 입력한 경우에만 `entry_is_session_open=True`를 명시할 수 있다. 기본값은 `False`이며 임의 장중 진입 이후 경로의 확률로 해석하지 않는다. 이 플래그는 사용자 선언이지 데이터 수집/시가 여부 자동 인증이 아니다.

반환값은 `probability_success`, joint 모델의 `probability_stop`, 고정 `policy_threshold`, `selected_research`, 후보 가격 기준 `candidate_take_price`(+1%)와 `candidate_stop_price`(−0.9%)를 포함한다. 기존 보유분의 실제 평균 매입가를 대신 계산하지 않는다. 항상 `research_only=True`, `deployment_allowed=False`, `intraday_path_verified=False`이며 이번 실행의 `research_qualified`도 `False`다. `selected_research=True`는 연구용 마스크이지 매수 허가가 아니다. GUI·브로커·자동주문과 연결하지 않았다.
