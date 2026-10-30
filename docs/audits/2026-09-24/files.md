# 기준 저장소 391개 파일 전수 목록

[종합 보고서](README.md) · 기준 `6e3ace7` · 2026-09-24

## 이 목록을 읽는 방법

- 감사 시작 시 Git 추적 파일 **391개 전부**를 읽어 SHA-256을 기록했다. 이번에 새로 작성한 감사 문서는 기준 목록에 포함하지 않는다.
- Python 260개는 모두 AST 구문 분석했다. 전체 함수/메서드 정의 3,856개 중 테스트 밖 정의는 1,378개다. 모든 함수가 실행되거나 모든 입력 분기가 검증되었다는 뜻은 아니다.
- `주요 경로 검토`는 해당 영역 소스의 위험·계약 중심 검토다. 줄별 완전한 안전성 증명을 뜻하지 않는다. `심층 검토 제한`과 바이너리 메타데이터 검토를 숨기지 않았다.
- 모든 파일은 이번 단계에서 **유지**했다. 아래 마지막 열은 다음 구현 단계의 제안일 뿐 이동·삭제 완료 표시가 아니다.
- JSON 44개, TOML/lock 2개, SVG 1개는 파싱 오류가 없었다. 원본 391개 파일의 감사 전후 해시 변화는 0개였다.
- 전체 해시와 클래스/함수 개수: [inventory.json](../../../outputs/optimization-audit/inventory.json). 함수별 위치·길이 목록: [functions.json](../../../outputs/optimization-audit/functions.json). 이 두 증거 파일은 로컬 ignored 산출물이다.

## root — 11개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [.env.example](../../../.env.example) | 23 / — | 설정 예시・해시 목록 | 유지; 비밀값 없는 설정 예제 |
| [.gitattributes](../../../.gitattributes) | 29 / — | 동결 해시・줄바꿈 계약 검토 | R-03 해결 후 전체 봉인 소스 정책 보강 |
| [.gitignore](../../../.gitignore) | 226 / — | 저장소 경계 검토 | 유지; data/state/cache/비밀값 경계 명시 |
| [DEVLOG.md](../../../DEVLOG.md) | 937 / — | 기록/경로 참조 인벤토리 | 과거 기록 보존; 당시 경로를 현재 경로로 덮어쓰지 않음 |
| [DockDack.vbs](../../../DockDack.vbs) | 25 / — | 실행 경로・런처 연결 검토 | Windows 기본 진입점 유지; 설치 후 경로 계약 보강 |
| [README.md](../../../README.md) | 158 / — | 사용/개발 안내 및 진입점 검토 | 테스트 명령・단일 진입점・배포 제약 안내 보강 |
| [pyproject.toml](../../../pyproject.toml) | 43 / — | TOML・wheel 빌드/격리 실행 검증 | PKG-01/QA-01/R-05 해결 후 src 전환 |
| [requirements-ml-cuda.txt](../../../requirements-ml-cuda.txt) | 10 / — | 환경 역할・참조 검토 | CUDA 환경을 분리 재현 가능하게 유지 |
| [requirements-prototype-gui.txt](../../../requirements-prototype-gui.txt) | 4 / — | 환경 역할・참조 검토 | 일반 GUI와 prototype 최소 의존성 유지 |
| [requirements-selective.txt](../../../requirements-selective.txt) | 5 / — | 환경 역할・잘못된 안내 참조 확인 | R-05: 없는 requirements-ml.txt 안내 수정 |
| [uv.lock](../../../uv.lock) | 1872 / — | TOML・uv lock --check | 유지; 새 research extra와 환경 역할 정합성 확인 |

## dockdack — 88개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [dockdack/__init__.py](../../../dockdack/__init__.py) | 59 / 0 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/__main__.py](../../../dockdack/__main__.py) | 5 / 0 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/activity_snapshot.py](../../../dockdack/activity_snapshot.py) | 67 / 4 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/assets/dockdack-mark.svg](../../../dockdack/assets/dockdack-mark.svg) | 13 / — | XML 파싱・해시 목록 | 배포 자산 유지 |
| [dockdack/assets/dockdack.ico](../../../dockdack/assets/dockdack.ico) | 바이너리 / — | 바이너리 해시 목록・기존 브랜딩 회귀 | 배포 자산 유지 |
| [dockdack/autotrade.py](../../../dockdack/autotrade.py) | 1051 / 41 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/branding.py](../../../dockdack/branding.py) | 28 / 3 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/clean_daily_dataset.py](../../../dockdack/clean_daily_dataset.py) | 362 / 9 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/cli.py](../../../dockdack/cli.py) | 258 / 12 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/conditions.py](../../../dockdack/conditions.py) | 235 / 16 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/config.py](../../../dockdack/config.py) | 125 / 8 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/daily_dataset.py](../../../dockdack/daily_dataset.py) | 690 / 24 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/dashboard_theme.py](../../../dockdack/dashboard_theme.py) | 61 / 0 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/dataset_identity.py](../../../dockdack/dataset_identity.py) | 212 / 6 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/dataset_quality.py](../../../dockdack/dataset_quality.py) | 307 / 9 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/demo_session.py](../../../dockdack/demo_session.py) | 163 / 7 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/environment_gui.py](../../../dockdack/environment_gui.py) | 52 / 4 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/environment_store.py](../../../dockdack/environment_store.py) | 56 / 5 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/equity_policy.py](../../../dockdack/equity_policy.py) | 70 / 2 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/equity_universe.py](../../../dockdack/equity_universe.py) | 192 / 4 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/exceptions.py](../../../dockdack/exceptions.py) | 40 / 1 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/execution_policy.py](../../../dockdack/execution_policy.py) | 130 / 4 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/fill_recovery.py](../../../dockdack/fill_recovery.py) | 291 / 13 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/gui.py](../../../dockdack/gui.py) | 683 / 35 | AST・담당 영역 위험/주요 경로 검토 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/gui_service.py](../../../dockdack/gui_service.py) | 309 / 33 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/history.py](../../../dockdack/history.py) | 117 / 6 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/history_cache.py](../../../dockdack/history_cache.py) | 111 / 7 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/http.py](../../../dockdack/http.py) | 550 / 27 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/kiwoom.py](../../../dockdack/kiwoom.py) | 1617 / 76 | AST・담당 영역 위험/주요 경로 검토 | 외부 인터페이스 유지; 국내/미국/계좌/주문 파서 분리 |
| [dockdack/local_data_paths.py](../../../dockdack/local_data_paths.py) | 18 / 1 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/lstm30_adapter.py](../../../dockdack/lstm30_adapter.py) | 471 / 20 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/lstm30_close.py](../../../dockdack/lstm30_close.py) | 504 / 22 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/lstm30_gui.py](../../../dockdack/lstm30_gui.py) | 698 / 44 | AST・담당 영역 위험/주요 경로 검토 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/lstm30_rejections.py](../../../dockdack/lstm30_rejections.py) | 29 / 1 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/lstm30_runtime.py](../../../dockdack/lstm30_runtime.py) | 462 / 21 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/lstm30_universe.py](../../../dockdack/lstm30_universe.py) | 202 / 15 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/manual_orders.py](../../../dockdack/manual_orders.py) | 214 / 8 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/mark1_0504_backtest.py](../../../dockdack/mark1_0504_backtest.py) | 370 / 6 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_0504_data.py](../../../dockdack/mark1_0504_data.py) | 145 / 5 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_0504_inference.py](../../../dockdack/mark1_0504_inference.py) | 281 / 10 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_1_prototype_inference.py](../../../dockdack/mark1_1_prototype_inference.py) | 94 / 4 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_1_trigger.py](../../../dockdack/mark1_1_trigger.py) | 67 / 2 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/mark1_adapter.py](../../../dockdack/mark1_adapter.py) | 98 / 4 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/mark1_backtest.py](../../../dockdack/mark1_backtest.py) | 355 / 6 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_backtest_data.py](../../../dockdack/mark1_backtest_data.py) | 149 / 2 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_data.py](../../../dockdack/mark1_data.py) | 295 / 4 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_deep_data.py](../../../dockdack/mark1_deep_data.py) | 163 / 7 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_deep_inference.py](../../../dockdack/mark1_deep_inference.py) | 243 / 7 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_deep_models.py](../../../dockdack/mark1_deep_models.py) | 302 / 24 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_deep_validation.py](../../../dockdack/mark1_deep_validation.py) | 211 / 5 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_gui.py](../../../dockdack/mark1_gui.py) | 255 / 10 | AST・담당 영역 위험/주요 경로 검토 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/mark1_inference.py](../../../dockdack/mark1_inference.py) | 101 / 2 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_metrics.py](../../../dockdack/mark1_metrics.py) | 317 / 13 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_models.py](../../../dockdack/mark1_models.py) | 236 / 19 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_prototype_adapter.py](../../../dockdack/mark1_prototype_adapter.py) | 229 / 27 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/mark1_prototype_gui.py](../../../dockdack/mark1_prototype_gui.py) | 250 / 14 | AST・담당 영역 위험/주요 경로 검토 | 호환/관찰 역할 확인 후 별도 영역; 즉시 삭제 금지 |
| [dockdack/mark1_prototype_inference.py](../../../dockdack/mark1_prototype_inference.py) | 268 / 10 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_selective_features.py](../../../dockdack/mark1_selective_features.py) | 184 / 3 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_selective_inference.py](../../../dockdack/mark1_selective_inference.py) | 359 / 13 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_selective_models.py](../../../dockdack/mark1_selective_models.py) | 341 / 17 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_selective_policy.py](../../../dockdack/mark1_selective_policy.py) | 324 / 14 | AST・담당 영역 위험/주요 경로 검토 | 보존 우선; 동결 계약 확인 후 새 버전만 추출 |
| [dockdack/mark1_trigger.py](../../../dockdack/mark1_trigger.py) | 294 / 14 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/market_schedule.py](../../../dockdack/market_schedule.py) | 225 / 22 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/market_status.py](../../../dockdack/market_status.py) | 84 / 3 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/ml30.py](../../../dockdack/ml30.py) | 201 / 8 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/models.py](../../../dockdack/models.py) | 243 / 1 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/operations_gui.py](../../../dockdack/operations_gui.py) | 318 / 17 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/order_prices.py](../../../dockdack/order_prices.py) | 106 / 8 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/performance.py](../../../dockdack/performance.py) | 314 / 8 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/portfolio.py](../../../dockdack/portfolio.py) | 109 / 7 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/portfolio_gui.py](../../../dockdack/portfolio_gui.py) | 432 / 18 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/prototype_external.py](../../../dockdack/prototype_external.py) | 408 / 26 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/signal_bridge.py](../../../dockdack/signal_bridge.py) | 512 / 24 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/signal_connection_gui.py](../../../dockdack/signal_connection_gui.py) | 217 / 5 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/signal_status.py](../../../dockdack/signal_status.py) | 94 / 1 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/signals.py](../../../dockdack/signals.py) | 80 / 3 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/strategy_lots.py](../../../dockdack/strategy_lots.py) | 173 / 5 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/symbols.py](../../../dockdack/symbols.py) | 21 / 2 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/test_strategy.py](../../../dockdack/test_strategy.py) | 113 / 4 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 모의 진단으로 격리; 실사용 경로 확인 전 삭제 금지 |
| [dockdack/trade_journal.py](../../../dockdack/trade_journal.py) | 173 / 4 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/trade_journal_gui.py](../../../dockdack/trade_journal_gui.py) | 289 / 9 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/universe.py](../../../dockdack/universe.py) | 137 / 2 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/us_equity_universe.py](../../../dockdack/us_equity_universe.py) | 286 / 14 | AST・호출/테스트 연결 인벤토리; 심층 검토 제한 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/v00_app.py](../../../dockdack/v00_app.py) | 522 / 32 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/v00_widgets.py](../../../dockdack/v00_widgets.py) | 113 / 9 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |
| [dockdack/watch_gui.py](../../../dockdack/watch_gui.py) | 1920 / 89 | AST・담당 영역 위험/주요 경로 검토 | 제어 유지; 상태/연결/설정 뷰·작업자 분리 |
| [dockdack/watchlist.py](../../../dockdack/watchlist.py) | 1112 / 63 | AST・담당 영역 위험/주요 경로 검토 | 장부 보존; SQL 필터와 저장소별 책임 분리 |
| [dockdack/window_controls.py](../../../dockdack/window_controls.py) | 84 / 7 | AST・담당 영역 위험/주요 경로 검토 | 유지; 역할별 패키지로 단계 분리 |

## examples — 39개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [examples/audit_mark1_deep.py](../../../examples/audit_mark1_deep.py) | 139 / 6 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/audit_mark1_us_corporate.py](../../../examples/audit_mark1_us_corporate.py) | 159 / 1 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/audit_mark1_us_data.py](../../../examples/audit_mark1_us_data.py) | 283 / 6 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/audit_mark1_us_training.py](../../../examples/audit_mark1_us_training.py) | 134 / 3 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/backtest_mark1.py](../../../examples/backtest_mark1.py) | 222 / 6 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/backtest_mark1_0504.py](../../../examples/backtest_mark1_0504.py) | 304 / 6 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/backtest_mark1_deep.py](../../../examples/backtest_mark1_deep.py) | 359 / 10 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/backtest_mark1_selective.py](../../../examples/backtest_mark1_selective.py) | 411 / 11 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/collect_kiwoom_daily.py](../../../examples/collect_kiwoom_daily.py) | 7 / 0 | AST・참조 인벤토리; 심층 검토 제한 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/diagnose_mark1_us_probabilities.py](../../../examples/diagnose_mark1_us_probabilities.py) | 256 / 7 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/emit_lstm_signal.py](../../../examples/emit_lstm_signal.py) | 212 / 8 | AST・참조 인벤토리; 심층 검토 제한 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/export_mark1_0504.py](../../../examples/export_mark1_0504.py) | 266 / 5 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/export_mark1_1_prototype.py](../../../examples/export_mark1_1_prototype.py) | 148 / 6 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/export_mark1_prototype.py](../../../examples/export_mark1_prototype.py) | 224 / 5 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/external_signal_producer.py](../../../examples/external_signal_producer.py) | 63 / 2 | AST・참조 인벤토리; 심층 검토 제한 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/kiwoom_basic.py](../../../examples/kiwoom_basic.py) | 19 / 1 | AST・참조 인벤토리; 심층 검토 제한 | 최소 API 사용 예제로 유지 |
| [examples/preview_dual_prototype_provenance.py](../../../examples/preview_dual_prototype_provenance.py) | 125 / 2 | AST・참조 인벤토리; 심층 검토 제한 | 개발용 미리보기로 분리; 호출/테스트 의존 확인 후 삭제 검토 |
| [examples/preview_mark1_dual_gui.py](../../../examples/preview_mark1_dual_gui.py) | 254 / 7 | AST・참조 인벤토리; 심층 검토 제한 | 개발용 미리보기로 분리; 호출/테스트 의존 확인 후 삭제 검토 |
| [examples/preview_mark1_gui.py](../../../examples/preview_mark1_gui.py) | 108 / 3 | AST・참조 인벤토리; 심층 검토 제한 | 개발용 미리보기로 분리; 호출/테스트 의존 확인 후 삭제 검토 |
| [examples/preview_mark1_normal_gui.py](../../../examples/preview_mark1_normal_gui.py) | 34 / 1 | AST・참조 인벤토리; 심층 검토 제한 | 개발용 미리보기로 분리; 호출/테스트 의존 확인 후 삭제 검토 |
| [examples/preview_mark1_prototype_gui.py](../../../examples/preview_mark1_prototype_gui.py) | 180 / 7 | AST・참조 인벤토리; 심층 검토 제한 | 개발용 미리보기로 분리; 호출/테스트 의존 확인 후 삭제 검토 |
| [examples/publish_lstm30.py](../../../examples/publish_lstm30.py) | 105 / 4 | AST・참조 인벤토리; 심층 검토 제한 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/report_mark1.py](../../../examples/report_mark1.py) | 181 / 4 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/report_mark1_0504.py](../../../examples/report_mark1_0504.py) | 341 / 9 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/report_mark1_backtest.py](../../../examples/report_mark1_backtest.py) | 169 / 3 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/report_mark1_deep.py](../../../examples/report_mark1_deep.py) | 333 / 5 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/report_mark1_selective.py](../../../examples/report_mark1_selective.py) | 620 / 20 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/run_desktop_gui.py](../../../examples/run_desktop_gui.py) | 54 / 3 | AST・주요 연구/실행 계약 검토 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/run_lstm30_demo.py](../../../examples/run_lstm30_demo.py) | 7 / 0 | AST・참조 인벤토리; 심층 검토 제한 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/run_lstm30_gui.py](../../../examples/run_lstm30_gui.py) | 7 / 0 | AST・참조 인벤토리; 심층 검토 제한 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/run_mark1_gui.py](../../../examples/run_mark1_gui.py) | 7 / 0 | AST・참조 인벤토리; 심층 검토 제한 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/run_mark1_prototype_gui.py](../../../examples/run_mark1_prototype_gui.py) | 137 / 4 | AST・참조 인벤토리; 심층 검토 제한 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/run_prototype_signal.py](../../../examples/run_prototype_signal.py) | 65 / 2 | AST・주요 연구/실행 계약 검토 | 운영 사용 진입점은 패키지 CLI/worker로 이전; 옛 명령 호환 |
| [examples/train_lstm30.py](../../../examples/train_lstm30.py) | 739 / 20 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/train_lstm_daily.py](../../../examples/train_lstm_daily.py) | 374 / 18 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/train_mark1.py](../../../examples/train_mark1.py) | 340 / 13 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/train_mark1_0504.py](../../../examples/train_mark1_0504.py) | 292 / 9 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/train_mark1_deep.py](../../../examples/train_mark1_deep.py) | 341 / 11 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |
| [examples/train_mark1_selective.py](../../../examples/train_mark1_selective.py) | 349 / 15 | AST・주요 연구/실행 계약 검토 | 연구/진단으로 분리; 동결 코드 원본 보존 |

## scripts — 6개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [scripts/build_app_icon.py](../../../scripts/build_app_icon.py) | 41 / 1 | AST・참조/개발 도구 목록 | 개발/운영 보조 도구 구분; 검증 후 유지/이동 |
| [scripts/export_demo_watchlist.py](../../../scripts/export_demo_watchlist.py) | 56 / 2 | AST・참조/개발 도구 목록 | 개발/운영 보조 도구 구분; 검증 후 유지/이동 |
| [scripts/package_daily_snapshot.py](../../../scripts/package_daily_snapshot.py) | 152 / 4 | AST・참조/개발 도구 목록 | 개발/운영 보조 도구 구분; 검증 후 유지/이동 |
| [scripts/report_lstm30_training.py](../../../scripts/report_lstm30_training.py) | 92 / 3 | AST・참조/개발 도구 목록 | 개발/운영 보조 도구 구분; 검증 후 유지/이동 |
| [scripts/verify_lstm30_bridge.py](../../../scripts/verify_lstm30_bridge.py) | 134 / 3 | AST・참조/개발 도구 목록 | 개발/운영 보조 도구 구분; 검증 후 유지/이동 |
| [scripts/warm_history_cache.py](../../../scripts/warm_history_cache.py) | 22 / 1 | AST・참조/개발 도구 목록 | 개발/운영 보조 도구 구분; 검증 후 유지/이동 |

## tests — 129개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [tests/test_activity_snapshot.py](../../../tests/test_activity_snapshot.py) | 150 / 10 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_audit_mark1_us_training.py](../../../tests/test_audit_mark1_us_training.py) | 54 / 4 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_auto_integration.py](../../../tests/test_auto_integration.py) | 153 / 7 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_auto_order_resilience.py](../../../tests/test_auto_order_resilience.py) | 132 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_autotrade.py](../../../tests/test_autotrade.py) | 348 / 40 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_autotrade_domestic_price.py](../../../tests/test_autotrade_domestic_price.py) | 81 / 9 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_autotrade_us_price.py](../../../tests/test_autotrade_us_price.py) | 73 / 8 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_backtest_mark1_deep.py](../../../tests/test_backtest_mark1_deep.py) | 313 / 20 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_backtest_mark1_runner.py](../../../tests/test_backtest_mark1_runner.py) | 174 / 12 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_backtest_mark1_selective.py](../../../tests/test_backtest_mark1_selective.py) | 349 / 24 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_branding.py](../../../tests/test_branding.py) | 36 / 3 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_calendar_threading.py](../../../tests/test_calendar_threading.py) | 107 / 7 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_clean_daily_dataset.py](../../../tests/test_clean_daily_dataset.py) | 302 / 20 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_cli.py](../../../tests/test_cli.py) | 293 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_common_order_gate.py](../../../tests/test_common_order_gate.py) | 131 / 9 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_common_watchlist.py](../../../tests/test_common_watchlist.py) | 187 / 12 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_concurrent_prototype_feeds.py](../../../tests/test_concurrent_prototype_feeds.py) | 609 / 53 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_conditions.py](../../../tests/test_conditions.py) | 103 / 8 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_config.py](../../../tests/test_config.py) | 45 / 2 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_current_price_orders.py](../../../tests/test_current_price_orders.py) | 90 / 7 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_daily_dataset.py](../../../tests/test_daily_dataset.py) | 163 / 7 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_dashboard_integration.py](../../../tests/test_dashboard_integration.py) | 213 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_dashboard_performance.py](../../../tests/test_dashboard_performance.py) | 178 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_dataset_identity.py](../../../tests/test_dataset_identity.py) | 173 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_dataset_quality.py](../../../tests/test_dataset_quality.py) | 330 / 29 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_deferred_orders.py](../../../tests/test_deferred_orders.py) | 359 / 31 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_desktop_launcher.py](../../../tests/test_desktop_launcher.py) | 54 / 5 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_environment_gui.py](../../../tests/test_environment_gui.py) | 432 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_environment_store.py](../../../tests/test_environment_store.py) | 67 / 7 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_equity_policy.py](../../../tests/test_equity_policy.py) | 147 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_equity_universe.py](../../../tests/test_equity_universe.py) | 169 / 18 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_execution_history.py](../../../tests/test_execution_history.py) | 247 / 26 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_external_error_state.py](../../../tests/test_external_error_state.py) | 67 / 5 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_external_roundtrip_v00.py](../../../tests/test_external_roundtrip_v00.py) | 188 / 9 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_fill_recovery.py](../../../tests/test_fill_recovery.py) | 541 / 49 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_gui.py](../../../tests/test_gui.py) | 167 / 21 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_gui_service.py](../../../tests/test_gui_service.py) | 120 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_history.py](../../../tests/test_history.py) | 93 / 9 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_history_cache.py](../../../tests/test_history_cache.py) | 109 / 10 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_holding_progress.py](../../../tests/test_holding_progress.py) | 197 / 22 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_http.py](../../../tests/test_http.py) | 588 / 56 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_journal_categories.py](../../../tests/test_journal_categories.py) | 155 / 8 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_journal_performance.py](../../../tests/test_journal_performance.py) | 155 / 6 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_kiwoom.py](../../../tests/test_kiwoom.py) | 360 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_local_data_paths.py](../../../tests/test_local_data_paths.py) | 91 / 11 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm30_adapter.py](../../../tests/test_lstm30_adapter.py) | 423 / 39 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm30_close.py](../../../tests/test_lstm30_close.py) | 731 / 66 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm30_gui.py](../../../tests/test_lstm30_gui.py) | 648 / 43 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm30_rejections.py](../../../tests/test_lstm30_rejections.py) | 241 / 22 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm30_runtime.py](../../../tests/test_lstm30_runtime.py) | 327 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm30_universe.py](../../../tests/test_lstm30_universe.py) | 251 / 21 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm_example.py](../../../tests/test_lstm_example.py) | 135 / 9 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_lstm_signals.py](../../../tests/test_lstm_signals.py) | 138 / 12 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_manual_orders.py](../../../tests/test_manual_orders.py) | 318 / 40 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_0504_backtest.py](../../../tests/test_mark1_0504_backtest.py) | 150 / 15 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_0504_data.py](../../../tests/test_mark1_0504_data.py) | 190 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_0504_evaluation.py](../../../tests/test_mark1_0504_evaluation.py) | 95 / 11 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_0504_inference.py](../../../tests/test_mark1_0504_inference.py) | 367 / 31 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_0504_report.py](../../../tests/test_mark1_0504_report.py) | 96 / 6 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_1_prototype_inference.py](../../../tests/test_mark1_1_prototype_inference.py) | 148 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_1_trigger.py](../../../tests/test_mark1_1_trigger.py) | 217 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_adapter.py](../../../tests/test_mark1_adapter.py) | 152 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_backtest.py](../../../tests/test_mark1_backtest.py) | 261 / 29 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_backtest_data.py](../../../tests/test_mark1_backtest_data.py) | 94 / 5 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_data.py](../../../tests/test_mark1_data.py) | 211 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_deep_data.py](../../../tests/test_mark1_deep_data.py) | 169 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_deep_inference.py](../../../tests/test_mark1_deep_inference.py) | 326 / 29 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_deep_models.py](../../../tests/test_mark1_deep_models.py) | 309 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_deep_validation.py](../../../tests/test_mark1_deep_validation.py) | 165 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_gui.py](../../../tests/test_mark1_gui.py) | 319 / 26 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_inference.py](../../../tests/test_mark1_inference.py) | 236 / 19 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_metrics.py](../../../tests/test_mark1_metrics.py) | 193 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_model_provenance.py](../../../tests/test_mark1_model_provenance.py) | 245 / 25 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_models.py](../../../tests/test_mark1_models.py) | 179 / 15 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_paper_execution.py](../../../tests/test_mark1_paper_execution.py) | 336 / 31 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_prototype_adapter.py](../../../tests/test_mark1_prototype_adapter.py) | 184 / 12 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_prototype_gui.py](../../../tests/test_mark1_prototype_gui.py) | 251 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_prototype_inference.py](../../../tests/test_mark1_prototype_inference.py) | 234 / 22 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_prototype_launcher.py](../../../tests/test_mark1_prototype_launcher.py) | 35 / 4 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_prototype_safety.py](../../../tests/test_mark1_prototype_safety.py) | 333 / 30 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_selective_features.py](../../../tests/test_mark1_selective_features.py) | 172 / 18 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_selective_inference.py](../../../tests/test_mark1_selective_inference.py) | 377 / 22 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_selective_models.py](../../../tests/test_mark1_selective_models.py) | 206 / 15 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_selective_policy.py](../../../tests/test_mark1_selective_policy.py) | 303 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_mark1_trigger.py](../../../tests/test_mark1_trigger.py) | 247 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_market_schedule.py](../../../tests/test_market_schedule.py) | 146 / 12 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_market_status.py](../../../tests/test_market_status.py) | 133 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_ml30.py](../../../tests/test_ml30.py) | 292 / 27 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_monitor_status_gui.py](../../../tests/test_monitor_status_gui.py) | 136 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_operations_gui.py](../../../tests/test_operations_gui.py) | 207 / 15 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_order_acknowledgements.py](../../../tests/test_order_acknowledgements.py) | 141 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_order_prices.py](../../../tests/test_order_prices.py) | 170 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_order_status.py](../../../tests/test_order_status.py) | 296 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_paced_orders.py](../../../tests/test_paced_orders.py) | 190 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_performance.py](../../../tests/test_performance.py) | 381 / 39 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_portfolio.py](../../../tests/test_portfolio.py) | 133 / 14 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_portfolio_gui.py](../../../tests/test_portfolio_gui.py) | 200 / 13 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_prototype_external.py](../../../tests/test_prototype_external.py) | 395 / 37 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_random_strategy.py](../../../tests/test_random_strategy.py) | 164 / 15 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_report_mark1_deep.py](../../../tests/test_report_mark1_deep.py) | 312 / 18 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_report_mark1_selective.py](../../../tests/test_report_mark1_selective.py) | 358 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_session_resumption.py](../../../tests/test_session_resumption.py) | 261 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_settlement_cash.py](../../../tests/test_settlement_cash.py) | 188 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_signal_bridge.py](../../../tests/test_signal_bridge.py) | 280 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_signal_status.py](../../../tests/test_signal_status.py) | 205 / 22 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_signals.py](../../../tests/test_signals.py) | 69 / 10 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_strategy_lot_performance.py](../../../tests/test_strategy_lot_performance.py) | 65 / 8 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_strategy_lot_portfolio_gui.py](../../../tests/test_strategy_lot_portfolio_gui.py) | 87 / 8 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_strategy_lots.py](../../../tests/test_strategy_lots.py) | 327 / 35 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_strategy_stop_loss.py](../../../tests/test_strategy_stop_loss.py) | 166 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_trade_journal.py](../../../tests/test_trade_journal.py) | 157 / 18 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_trade_journal_gui.py](../../../tests/test_trade_journal_gui.py) | 123 / 11 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_trading_environment.py](../../../tests/test_trading_environment.py) | 300 / 29 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_train_mark1.py](../../../tests/test_train_mark1.py) | 295 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_train_mark1_0504.py](../../../tests/test_train_mark1_0504.py) | 64 / 4 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_train_mark1_deep.py](../../../tests/test_train_mark1_deep.py) | 219 / 20 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_train_mark1_selective.py](../../../tests/test_train_mark1_selective.py) | 257 / 23 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_training30.py](../../../tests/test_training30.py) | 479 / 41 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_universe.py](../../../tests/test_universe.py) | 156 / 15 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_us_equity_universe.py](../../../tests/test_us_equity_universe.py) | 180 / 16 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_us_position_exchange.py](../../../tests/test_us_position_exchange.py) | 112 / 8 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_v00_execution.py](../../../tests/test_v00_execution.py) | 512 / 56 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_v00_gui.py](../../../tests/test_v00_gui.py) | 396 / 27 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_v00_mark11_launcher.py](../../../tests/test_v00_mark11_launcher.py) | 49 / 4 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_v00_mark1_trigger.py](../../../tests/test_v00_mark1_trigger.py) | 398 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_v00_runtime_performance.py](../../../tests/test_v00_runtime_performance.py) | 150 / 10 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_volume_schedule_v00.py](../../../tests/test_volume_schedule_v00.py) | 253 / 26 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_watch_gui.py](../../../tests/test_watch_gui.py) | 404 / 28 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |
| [tests/test_window_controls.py](../../../tests/test_window_controls.py) | 335 / 20 | AST・전체 회귀 실행 대상 | 유지; 추후 unit/integration/contracts/performance로 분류 |

## models — 60개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [models/lstm30/README.md](../../../models/lstm30/README.md) | 30 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/lstm30/domestic.pt](../../../models/lstm30/domestic.pt) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/lstm30/us.pt](../../../models/lstm30/us.pt) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1/README.md](../../../models/mark1/README.md) | 16 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1/domestic.pt](../../../models/mark1/domestic.pt) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1/us.pt](../../../models/mark1/us.pt) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/manifest.json](../../../models/mark1_0504/domestic/manifest.json) | 301 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/seed42.cbm](../../../models/mark1_0504/domestic/seed42.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/seed42.cbm.json](../../../models/mark1_0504/domestic/seed42.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/seed43.cbm](../../../models/mark1_0504/domestic/seed43.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/seed43.cbm.json](../../../models/mark1_0504/domestic/seed43.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/seed44.cbm](../../../models/mark1_0504/domestic/seed44.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/domestic/seed44.cbm.json](../../../models/mark1_0504/domestic/seed44.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/export-validation.json](../../../models/mark1_0504/export-validation.json) | 417 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/manifest.json](../../../models/mark1_0504/manifest.json) | 278 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/manifest.sha256](../../../models/mark1_0504/manifest.sha256) | 바이너리 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/manifest.json](../../../models/mark1_0504/us/manifest.json) | 305 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/seed42.cbm](../../../models/mark1_0504/us/seed42.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/seed42.cbm.json](../../../models/mark1_0504/us/seed42.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/seed43.cbm](../../../models/mark1_0504/us/seed43.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/seed43.cbm.json](../../../models/mark1_0504/us/seed43.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/seed44.cbm](../../../models/mark1_0504/us/seed44.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_0504/us/seed44.cbm.json](../../../models/mark1_0504/us/seed44.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/alias-validation.json](../../../models/mark1_1_prototype/alias-validation.json) | 209 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/alias.json](../../../models/mark1_1_prototype/alias.json) | 70 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/alias.sha256](../../../models/mark1_1_prototype/alias.sha256) | 바이너리 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/manifest.json](../../../models/mark1_1_prototype/domestic/manifest.json) | 301 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/seed42.cbm](../../../models/mark1_1_prototype/domestic/seed42.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/seed42.cbm.json](../../../models/mark1_1_prototype/domestic/seed42.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/seed43.cbm](../../../models/mark1_1_prototype/domestic/seed43.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/seed43.cbm.json](../../../models/mark1_1_prototype/domestic/seed43.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/seed44.cbm](../../../models/mark1_1_prototype/domestic/seed44.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/domestic/seed44.cbm.json](../../../models/mark1_1_prototype/domestic/seed44.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/export-validation.json](../../../models/mark1_1_prototype/export-validation.json) | 417 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/manifest.json](../../../models/mark1_1_prototype/manifest.json) | 278 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/manifest.sha256](../../../models/mark1_1_prototype/manifest.sha256) | 바이너리 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/manifest.json](../../../models/mark1_1_prototype/us/manifest.json) | 305 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/seed42.cbm](../../../models/mark1_1_prototype/us/seed42.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/seed42.cbm.json](../../../models/mark1_1_prototype/us/seed42.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/seed43.cbm](../../../models/mark1_1_prototype/us/seed43.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/seed43.cbm.json](../../../models/mark1_1_prototype/us/seed43.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/seed44.cbm](../../../models/mark1_1_prototype/us/seed44.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_1_prototype/us/seed44.cbm.json](../../../models/mark1_1_prototype/us/seed44.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/manifest.json](../../../models/mark1_prototype/domestic/manifest.json) | 359 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/seed42.cbm](../../../models/mark1_prototype/domestic/seed42.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/seed42.cbm.json](../../../models/mark1_prototype/domestic/seed42.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/seed43.cbm](../../../models/mark1_prototype/domestic/seed43.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/seed43.cbm.json](../../../models/mark1_prototype/domestic/seed43.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/seed44.cbm](../../../models/mark1_prototype/domestic/seed44.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/domestic/seed44.cbm.json](../../../models/mark1_prototype/domestic/seed44.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/export-validation.json](../../../models/mark1_prototype/export-validation.json) | 377 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/manifest.json](../../../models/mark1_prototype/manifest.json) | 275 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/manifest.sha256](../../../models/mark1_prototype/manifest.sha256) | 바이너리 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/manifest.json](../../../models/mark1_prototype/us/manifest.json) | 356 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/seed42.cbm](../../../models/mark1_prototype/us/seed42.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/seed42.cbm.json](../../../models/mark1_prototype/us/seed42.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/seed43.cbm](../../../models/mark1_prototype/us/seed43.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/seed43.cbm.json](../../../models/mark1_prototype/us/seed43.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/seed44.cbm](../../../models/mark1_prototype/us/seed44.cbm) | 바이너리 / — | 바이너리 해시 목록; 일부 bundle 계약검사 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |
| [models/mark1_prototype/us/seed44.cbm.json](../../../models/mark1_prototype/us/seed44.cbm.json) | 13 / — | 형식/계약 확인 대상; 해시 목록 | 원본 유지; artifact ID·별도 배포 구조 검증 후 이동 |

## reports — 27개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [reports/mark1-0504-20260920/REPORT.md](../../../reports/mark1-0504-20260920/REPORT.md) | 115 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-0504-20260920/comparison.png](../../../reports/mark1-0504-20260920/comparison.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-0504-20260920/report-inputs.json](../../../reports/mark1-0504-20260920/report-inputs.json) | 43 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-20260916/REPORT.md](../../../reports/mark1-20260916/REPORT.md) | 108 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-20260916/comparison.png](../../../reports/mark1-20260916/comparison.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-20260916/experiment.json](../../../reports/mark1-20260916/experiment.json) | 256 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-20260916/learning-calibration.png](../../../reports/mark1-20260916/learning-calibration.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-20260916/results.json](../../../reports/mark1-20260916/results.json) | 3424 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/REPORT.md](../../../reports/mark1-backtest-20260916/REPORT.md) | 137 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/cost-sensitivity.png](../../../reports/mark1-backtest-20260916/cost-sensitivity.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/domestic-selected-carry-trades.json](../../../reports/mark1-backtest-20260916/domestic-selected-carry-trades.json) | 6944 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/domestic-selected-eod-trades.json](../../../reports/mark1-backtest-20260916/domestic-selected-eod-trades.json) | 7048 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/equity-drawdown.png](../../../reports/mark1-backtest-20260916/equity-drawdown.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/results.json](../../../reports/mark1-backtest-20260916/results.json) | 3560 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/us-selected-carry-trades.json](../../../reports/mark1-backtest-20260916/us-selected-carry-trades.json) | 730 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-backtest-20260916/us-selected-eod-trades.json](../../../reports/mark1-backtest-20260916/us-selected-eod-trades.json) | 782 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-deep-20260916/REPORT.md](../../../reports/mark1-deep-20260916/REPORT.md) | 160 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-deep-20260916/equity-comparison.png](../../../reports/mark1-deep-20260916/equity-comparison.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-deep-20260916/learning-curves.png](../../../reports/mark1-deep-20260916/learning-curves.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-deep-20260916/model-comparison.png](../../../reports/mark1-deep-20260916/model-comparison.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-deep-20260916/results.json](../../../reports/mark1-deep-20260916/results.json) | 62203 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-selective-20260916/REPORT.md](../../../reports/mark1-selective-20260916/REPORT.md) | 248 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-selective-20260916/audit-comparison.png](../../../reports/mark1-selective-20260916/audit-comparison.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-selective-20260916/cost-sensitivity.png](../../../reports/mark1-selective-20260916/cost-sensitivity.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-selective-20260916/equity-comparison.png](../../../reports/mark1-selective-20260916/equity-comparison.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-selective-20260916/policy-risk-coverage.png](../../../reports/mark1-selective-20260916/policy-risk-coverage.png) | 바이너리 / — | 바이너리 해시 목록; 이미지 의미 전수 미검증 | 과거 실험 증거 보존; 최신 안내와 구분 |
| [reports/mark1-selective-20260916/results.json](../../../reports/mark1-selective-20260916/results.json) | 195101 / — | 문서/JSON 형식·주요 입력 계약 검토 | 과거 실험 증거 보존; 최신 안내와 구분 |

## docs — 31개

| 파일 | 줄 / 함수 정의 | 검토 깊이 | 다음 단계 판단 |
|---|---:|---|---|
| [docs/MARK1_0504.md](../../../docs/MARK1_0504.md) | 71 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/MARK1_1_PROTOTYPE.md](../../../docs/MARK1_1_PROTOTYPE.md) | 82 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/MARK1_GUI.md](../../../docs/MARK1_GUI.md) | 92 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/MARK1_PROTOTYPE.md](../../../docs/MARK1_PROTOTYPE.md) | 86 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/VER_0_0.md](../../../docs/VER_0_0.md) | 174 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/api-rate-limits.md](../../../docs/api-rate-limits.md) | 37 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/autotrading.md](../../../docs/autotrading.md) | 147 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/cleanup-20260924.md](../../../docs/cleanup-20260924.md) | 30 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/daily-dataset.md](../../../docs/daily-dataset.md) | 108 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/dataset-cleaning.md](../../../docs/dataset-cleaning.md) | 95 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/datasets/2026-09-14.json](../../../docs/datasets/2026-09-14.json) | 198 / — | JSON 파싱・해시 목록 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/datasets/clean-2026-09-16.json](../../../docs/datasets/clean-2026-09-16.json) | 87 / — | JSON 파싱・해시 목록 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/external-signals.md](../../../docs/external-signals.md) | 271 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/gui.md](../../../docs/gui.md) | 198 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/kiwoom-broker.md](../../../docs/kiwoom-broker.md) | 176 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/lstm-example.md](../../../docs/lstm-example.md) | 78 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/lstm30-demo-runtime.md](../../../docs/lstm30-demo-runtime.md) | 122 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/lstm30-strategy.md](../../../docs/lstm30-strategy.md) | 145 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-deep-audit.md](../../../docs/mark1-deep-audit.md) | 84 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-deep-protocol.md](../../../docs/mark1-deep-protocol.md) | 183 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-deep-research.md](../../../docs/mark1-deep-research.md) | 80 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-selective-literature.md](../../../docs/mark1-selective-literature.md) | 83 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-selective-protocol.md](../../../docs/mark1-selective-protocol.md) | 115 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-strategy.md](../../../docs/mark1-strategy.md) | 84 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mark1-us-signal-diagnosis.md](../../../docs/mark1-us-signal-diagnosis.md) | 94 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/mock-trading-hours.md](../../../docs/mock-trading-hours.md) | 68 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/server-mode.md](../../../docs/server-mode.md) | 171 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/terminal.md](../../../docs/terminal.md) | 152 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/trade-journal.md](../../../docs/trade-journal.md) | 26 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/trading-modes.md](../../../docs/trading-modes.md) | 60 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |
| [docs/trading-signals.md](../../../docs/trading-signals.md) | 73 / — | 문서 참조·경로·주요 내용 검토 대상 | 유지; user/developer/research로 재분류, 과거 기록 보존 |

## Git 밖의 로컬 폴더

| 영역 | 이번 확인 | 판단 |
|---|---|---|
| `data/` | 파일 수·용량과 주요 데이터 계약/경로; 원본 전 행 스캔 미실시 | 원본·정제 DB 유지; 수정주가 세대/응답 검증부터 설계 |
| `outputs/mark1/` | 저장 실험/캐시 헤더/manifest·연구 재현 입력 | 실제 실행 의존성도 있어 전체 삭제 금지 |
| `.dockdack/` | 디렉터리 메타데이터만; 운영 장부/비밀값 내용 읽기 제외 | 계정 상태·원매수 장부·복구 bundle 보호 |
| `.env` | 내용 감사/공개 제외 | 비밀값 유지·Git 제외 |
| `.venv/`, `.venv-ml-cuda/` | 디렉터리 메타데이터와 감사용 해석기 확인 | 재생성 환경이나 현재 의존성 분리 검증 전 삭제 금지 |
| `.git/`, `.idea/` | 저장소/IDE 영역 식별 | 사용자 이력·설정 보호; 최적화 삭제 대상 아님 |
| `__pycache__`, egg-info, 감사 임시 출력 | 재생성 가능한 후보 식별 | 참조/사용 프로세스 확인 후 정리 가능; 이번에는 삭제 안 함 |

서드파티 가상환경 전체 소스나 모든 로컬 DB 레코드를 검증했다고 주장하지 않는다. 추적 파일의 전수 인벤토리와 제한된 실행/계약 검증을 구분한다.
