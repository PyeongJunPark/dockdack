# DockDack 적대적 검증 — 해결 방안과 실행 기록

2026-09-28 · [발견 사항](findings.md)에 대응한다. 완료 표시는 코드를 고치고 오프라인 회귀를 통과한 뒤에만 확정한다. 주문 허용, 운영 DB 이관, 모델 재학습은 해결 수단으로 사용하지 않는다.

| ID | 해결 방안 | 적용·확인할 기준 |
|---|---|---|
| A-01 | 봉인된 각 번들과 해시 대상 원본 소스를 `-text`로 고정하고, CRLF로 봉인된 두 manifest의 정확한 원시 바이트를 Git에 재등록한다. 봉인값이나 모델 수식은 바꾸지 않는다. | Windows/LF 체크아웃을 모사해 Git blob·작업본·manifest SHA가 모두 일치하는지 테스트. |
| A-02 | 모델 선택과 매수 비중을 기존 외부 설정 패널의 안전한 편집 잠금 상태에 묶는다. 감시/worker/확인 중 및 실전 모드에서는 변경하지 못하게 한다. | 작업 중 체크박스 비활성, 안전한 대기 상태에서만 활성. |
| A-03 | 1초 타이머에서 전 모델·전 종목 전체 갱신을 제거한다. 진입 시 즉시 표시하고 느린 표시 타이머에서 현재 보이는 부분만 갱신한다. | 숨김 상태에서 반복 전체 재생성 없음; 선택 전환 시 값 표시 유지. |
| A-04 | TOP100 순위 교체에서 불필요한 보호종목 계좌 조회를 없앤다. 매도·주문 경로의 필요한 fresh account 확인은 유지한다. | 정기·LSTM30·수동 경로에서 `protected_symbols()`가 실패해도 순위 적용; 주문 안전 검사는 별도 유지. |
| A-05 | 공통 탭 너비 정책을 뷰포트에 맞추고 표의 자체 스크롤만 남긴다. | 800×520의 모든 주탭 바깥 가로 스크롤 0; 내부 표 접근 유지. |
| A-06 | 숫자를 ‘연결’이 아닌 ‘선택’으로 표기한다. 실제 공급자 첫 상태줄에 실패/불가/오류가 있으면 작은 점검 배지와 접근성 이름·툴팁에 실패 수를 표시한다. | 첫 조회 전 13개 선택을 연결 성공으로 오인하지 않음; 실패 상태 노출. |
| A-07 | 환율 OFF는 USD 표시 전환만 수행하고 마지막 유효 환율을 유지한다. 다시 ON 하면 즉시 캐시 표시, ‘환율 갱신’만 명시적으로 재조회한다. | ON→OFF→ON 왕복에서 불필요한 추가 조회 0회; 강제 갱신 정상. |
| A-08 | 테스트가 새로 만든 창의 서비스 객체를 식별하도록 수정한다. | 격리/분할 전체 회귀에서 재현성 확보. |
| A-09 | 현재 사용법의 0.1/과거 모델 기본값/옛 메뉴 위치를 실제 0.2 화면과 맞추고, 역사적 설명은 당시 기준임을 표시한다. | README가 링크한 문서의 현재 절차와 `docs/FEATURES.md`가 서로 모순되지 않음. |
| A-10 | 두 FX 화면에서 양수·유한 환율과 올바른 고시일을 검증한 뒤만 새 표시/캐시를 갱신한다. 잘못된 응답은 USD 원본 또는 이전 유효 캐시를 유지한다. | NaN/0/잘못된 날짜에 UI 예외와 `NaN KRW` 없음; 정상 고시일은 표시. |
| A-11 | 수동 주문을 위한 순위 밖 신규 종목은 주문/체결 이력 행으로 저장하되 TOP100 활성 관심·일반 감시에 자동 편입하지 않는다. 순위가 아직 없는 초기 상태는 기존 수동 주문 가능성을 보존한다. | 완전 순위 100개 뒤 수동 claim에도 활성 100개, 주문 이력/미확정 안전 확인 유지. |
| A-12 | CI에서 빌드 wheel을 체크아웃 밖으로 복사하고 명령 진입점·GUI 자산·wheel 모듈 import를 계좌/모델 실행 없이 검사한다. | 로컬 격리 경로 smoke 성공, CI 구성 반영. 실제 별도 가상환경 설치·13모델 추론은 여전히 별도 수동 검사다. |
| A-13 | CI의 회귀를 `scripts/check.py --shard 1/4`~`4/4`의 독립 작업으로 나누고, wheel 빌드·빠른 import 검사도 별도 작업으로 실행한다. 각 분할은 0건 테스트를 거부하며 전부 성공해야 전체 CI가 성공이다. | 최종 원격 실행에서 5개 작업 모두 성공, 가장 긴 회귀 작업도 30분 미만. |
| A-14 | 테스트의 파일 동일성·포함 관계를 실제 경로 기준으로 비교한다. 연구/CLI 테스트의 임시 루트도 정규화하되 원래 검증 대상 파일·경계는 유지한다. | 짧은/긴 Windows 경로 별칭에서도 같은 파일은 통과하고 다른 파일은 실패; 관련 8개 파일 135개 로컬 테스트 통과. |
| A-15 | 두 pytest 전용 모듈을 공식 `unittest` 탐색이 수집하는 클래스로 변환한다. 매개변수 조합은 `subTest`, 예외 문구 검증은 `assertRaisesRegex`로 유지한다. | pytest 설치 없이도 공식 검사에서 두 모듈 6개+2개가 수집·통과. |
| A-16 | Mark1.2 export 테스트의 임시 작업 폴더를 항상 존재하는 저장소 루트 아래에 만든다. 테스트가 만든 폴더만 자동 정리한다. | `outputs/`가 없는 깨끗한 체크아웃에서도 임시 폴더 생성; Mark1.2 backtest 테스트 11개 통과. |
| A-17 | 공식 Node 24 런타임을 선언한 `actions/checkout@v5`·`astral-sh/setup-uv@v7.5.0`으로 CI action을 올린다. | 최종 원격 실행에서 checkout·uv 설치·네 회귀·wheel 검사 성공, 기존 Node 20 강제 실행 경고 제거. |

## 적용 위치

이번 표의 A-01~A-17은 작업본에 반영했다. 첫 4분할 원격 실행은 각 작업이 30분 전에 끝났고 wheel 검사가 통과했지만, 새로 드러난 A-14~A-16 탓에 전체 실패했다. 해당 수정 후 [최종 원격 실행](https://github.com/PyeongJunPark/dockdack/actions/runs/36340292560)에서는 회귀 4분할과 wheel 검사 5개 작업이 모두 통과했다. 핵심 변경과 그 회귀 근거는 다음과 같다.

- A-01: `.gitattributes`, 두 `models/mark1_{series,horizons}/manifest.json`, `tests/test_sealed_checkout_bytes.py`. 모델 가중치·봉인 문자열은 변경하지 않았다.
- A-02·A-03·A-06: `dockdack/ui/v00_app.py`, `tests/test_v00_gui.py`, `tests/test_v00_runtime_performance.py`. 상태 숫자는 선택 개수로 표시하고, 숨긴 모델 표의 초당 전체 갱신과 작업 중 재선택을 막았다.
- A-04·A-11: `dockdack/market_schedule.py`, `dockdack/lstm30_universe.py`, `dockdack/ui/watch_gui.py`, `dockdack/persistence/watchlist.py`, `dockdack/trading/manual_orders.py`, 해당 순위·수동 주문 테스트. 계좌 조회 제거는 TOP100 선택에만 적용했고 주문 직전 검사는 유지했다.
- A-05: `dockdack/ui/watch_gui.py`, `tests/test_portfolio_layout_gui.py`. 800×520 가짜 화면에서 주탭의 바깥 가로 스크롤 331→0px을 확인했다.
- A-07·A-10: `dockdack/ui/portfolio_gui.py`, `dockdack/ui/trade_journal_gui.py`, 환율 GUI 테스트. 잘못된 새 고시는 캐시에 덮어쓰지 않는다.
- A-08: `tests/test_startup_portfolio.py`에서 생성한 서비스의 창을 정확히 선택한다.
- A-09: 현재 사용 안내 문서 10곳과 `README.md`, `docs/FEATURES.md`, `docs/gui.md`의 0.2 화면 위치·기본값을 맞췄다. 연구 당시의 결과와 역사적 인벤토리는 수정하지 않았다.
- A-12: `.github/workflows/check.yml`과 `scripts/smoke_wheel_imports.py`에 계좌·모델·주문을 실행하지 않는 wheel 검사 경로를 추가했다.
- A-13: 단일 job의 30분 제한에 막힌 원격 실행을 [실패 기록](https://github.com/PyeongJunPark/dockdack/actions/runs/36336777175)으로 보존하고, 회귀 분할과 wheel job을 분리한다.
- A-14: `tests/test_environment_gui.py`, `tests/test_v00_gui.py`, `tests/test_mark1_2_data.py`, `tests/test_local_data_paths.py`, `tests/test_research_tools.py`, `tests/test_mark1_2_backtest.py`, `tests/test_mark1_2_finish.py`, `tests/test_desktop_launcher_scope.py`의 경로 비교를 실경로 기준으로 맞췄다.
- A-15: `tests/test_mark1_8_allocation.py`, `tests/test_mark1_special_inference.py`의 기존 함수형 사례를 공식 검사기의 `unittest` 수집 대상으로 옮겼다.
- A-16: `tests/test_mark1_2_backtest.py`의 export 임시 디렉터리 부모를 `outputs/`에서 저장소 루트로 바꿨다.
- A-17: `.github/workflows/check.yml`의 두 action을 Node 24 선언 버전으로 올렸다. 이 변경은 원격에서 다시 확인한다.

## 검증·공개 기록

첫 게시본에서 `scripts/check.py --shard 1/4`부터 `4/4`까지 **559/640/576/467개, 총 2,242개 실행·실패 0·건너뜀 7**이었다. 추가 회귀 8개를 공식 검사에 수집하도록 바꾼 최종 작업본은 **538/611/576/525개, 총 2,250개 실행·실패 0·건너뜀 7**을 다시 확인했다. 이 명령은 `dockdack/`, `tests/`, `scripts/`, `examples/`의 Python을 AST 파싱하고 0건 테스트를 거부한다. 분담 검증의 GUI 114개, TOP100 54개, 수동 주문 관련 90개, 봉인 checkout 1개는 전체 회귀와 중복될 수 있으므로 합산하지 않는다. 모델 의존성 `python -m examples.run_desktop_gui --check`는 13개 모델을 오프라인으로 확인했으며 계좌·주문·네트워크를 사용하지 않았다.

`uv lock --check --offline`, `uv build --wheel --offline`, 체크아웃 밖 wheel archive의 5개 명령·2개 GUI 자산 import smoke, `git diff --check`가 통과했다. 두 CRLF manifest의 현재 작업본 SHA-256은 기존 `manifest.sha256`과 일치하고, `git diff --cached --ignore-space-at-eol --exit-code`로 내용 변경 없음도 확인했다. 이 wheel 검사는 실제 별도 가상환경 설치나 외부 모델 추론을 대체하지 않는다. 첫 원격 CI는 30분 시간 초과, [두 번째 원격 CI](https://github.com/PyeongJunPark/dockdack/actions/runs/36338869523)는 wheel 성공·네 분할 종료 후 A-14~A-16 테스트 실패였다. [세 번째 원격 CI](https://github.com/PyeongJunPark/dockdack/actions/runs/36340292560)는 전체 5개 작업이 성공했다. 운영 기능 실패나 실계좌 결과로 해석하지 않는다.

남은 표시 한계: 오래 실행한 앱의 보유·일지 환율 캐시는 자동 TTL 재조회하지 않으므로 사용자가 고시일 툴팁을 확인하고 `환율 갱신`을 눌러야 한다. 외부 모델의 작은 점검 배지는 상태 첫 줄의 알려진 실패 문구를 기준으로 하므로 모든 실패 유형을 검출한다고 보장하지 않는다.

비밀키·`.env`·운영 DB·IDE 설정·임시 빌드 산출물은 게시 대상에서 제외했다. 구현·CI·회귀 커밋 `78eb499`까지 `mark_1` 로컬·원격 HEAD 일치를 확인했다. `main` 병합은 사용자가 직접 검토·결정할 때까지 수행하지 않는다. 모의·실전 주문 성공이나 실제 장중 성능은 오프라인 테스트로 주장하지 않는다.
