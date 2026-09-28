# 실행·개발·유지보수 안내

## 병합과 변경 기록

병합 작업에는 별도 요청이 없어도 `docs/FEATURES.md`와 `DEVLOG.md` 갱신을 포함한다. 기능서는 현재 사용법·화면 위치·제한을 유지하고, 개발일지는 날짜·출발/대상 브랜치·변경 내용·실제 검증 결과·남은 제한을 기록한다. 기능 변경이 없는 경우에도 기능서 일치 여부를 확인한다. 완료하지 않은 검증·병합·푸시는 완료로 쓰지 않는다. 작업 지침은 저장소 루트의 `AGENTS.md`에 둔다.

## 일반 실행과 모델 파일

저장소에서는 `uv sync --locked --extra gui --extra prototype` 후 `uv run dockdack-gui` 또는 기존 `DockDack.vbs`를 사용한다. 저장된 선택이 없는 새 모의 계정은 Mark1.0–1.28 전체를 초기 선택하지만 시작 시 감시·자동주문 OFF다. 기존 계정의 선택은 복원한다. 모의계좌 전체의 마감 전량매도 기능은 제거했다. 모델별 목표·기간 매도는 확인된 해당 모델 매수분에만 적용되며, 감시·모의 자동주문 ON과 주문별 검사가 필요하다.

현재 주탭은 관심종목·보유종목·매매일지·주문·체결·AI 추론 모델·고급설정이다. 상단 자동주문 **상태 버튼 하나**로 ON/OFF를 전환하고 OFF는 반복 감시도 중단한다. 모델별 `1회 매수` 비중(0.01–100%)은 AI 추론 모델 → 모델 선택 카드에, 기타 외부/구형 신호의 기본 비중과 시장별 주문 한도는 고급설정 → 공통 주문·연결에 있다. 계정별 v4 화면 설정은 모델별 값을 저장하며 과거 v1–v3 공통 값은 최초 모델별 값으로 승계한다. 화면 변경만으로 주문 권한이 켜지지 않는다.

wheel에는 코드/worker/UI 자산이 들어가고 모델·API 키·거래 DB는 포함하지 않는다. 설치본은 `DOCKDACK_HOME`을 데이터 홈으로 사용하며, 미지정 시 Windows `%LOCALAPPDATA%/DockDack`이다. 저장소 실행은 기존 저장소 홈을 유지한다. `DOCKDACK_MODEL_ROOT`로 **models의 상위 폴더가 아니라 models 폴더 자체**를 지정한다. 그 아래 기존 장중 `mark1_prototype`, `mark1_1_prototype`, `mark1_2_prototype`, 일봉 대리 `mark1_intraday`(Mark1.13–1.17), `mark1_intraday_extra`(Mark1.18–1.22), `mark1_target_horizon_v1`(Mark1.23–1.28), 장전 `mark1_3`, `mark1_4`, `mark1_series`, `mark1_8`, `mark1_9`, `mark1_10`, `mark1_horizons` 묶음을 상대 구조 그대로 둔다. manifest와 runtime 소스 검증을 통과한 모델만 읽는다. 새 세 묶음은 국내·미국 합계 32개 독립 체크포인트를 담으며 분봉/틱 기반 장중 경로 모델이 아니다. [당일 대리모델](mark1-intraday-proxy.md) · [목표가·보유기간 대리모델](mark1-target-horizon-methods.md)

```powershell
$env:DOCKDACK_HOME = 'C:/DockDack'
$env:DOCKDACK_MODEL_ROOT = 'C:/DockDack/models'
dockdack-gui
dockdack-prototype-worker --help
```

추론 worker는 계좌키를 전달받지 않으며 네트워크 연결을 방어적으로 차단한다. OS 수준 샌드박스는 아니다. 처음 기동/모델 import/입력 전달도 통신 제한시간에 포함된다.

## 계정별 장부와 기존 기록

모의/실전 키의 해시 범위별로 장부를 분리한다. 비밀키 원문은 경로에 넣지 않는다. 모의계좌를 증권사에서 리셋한 경우 **앱을 종료하고** `.env`의 `DOCKDACK_DEMO_GENERATION`을 새로운 식별자로 바꾸면 새 세대 장부를 사용한다. 같은 계정/세대는 같은 값을 유지한다.

기존 공용 모의 장부가 있으면 최초 시작에서 현재 계정·리셋 세대의 기록인지 확인한 뒤 복사할지, 새 장부로 시작할지, 취소할지 선택한다. 기본은 취소다. 기존 장부를 자동 귀속하거나 기존 대상 파일을 덮어쓰지 않는다. 소유가 불명확하면 복사하지 말아야 한다. 새 장부는 과거 모델별 원가/주문을 알지 못한다.

명시적 관리 명령도 제공한다. 다음 명령의 경로와 scope는 실제 선택된 계정의 값이어야 하며, 앱이 종료되고 **대상 파일이 아직 없는 경우**만 실행한다. source 원본과 미확정 주문은 보존된다.

```powershell
python -m dockdack.maintenance migrate-demo --source <기존장부> --destination <새계정장부> --scope <64자리계정범위> --confirm CONFIRM_LEGACY_DEMO_OWNERSHIP
```

## 감시 로그 보존

자동 삭제는 하지 않는다. 기본 90일을 넘은 `monitor` 텍스트만 동일 DB의 SHA 검증 압축 묶음으로 보관할 수 있다. 주문·체결·신호/HOLD·차트 멤버십·재전송 거부 근거는 그대로 유지한다. 앱을 닫고 먼저 dry-run으로 확인한다.

```powershell
python -m dockdack.maintenance archive-monitor --db <장부경로>
python -m dockdack.maintenance archive-monitor --db <장부경로> --days 90 --limit 2000 --apply
python -m dockdack.maintenance read-archive --db <장부경로> --id 1
```

아카이브는 디스크 공간을 즉시 반환하는 VACUUM이 아니며 거래 이력 증가를 없애지 않는다. 본 작업에서는 운영 장부에 실행하지 않았다.

## 검증 진입점

```powershell
uv sync --locked --python 3.13 --extra gui --extra prototype --extra conditions --extra research --extra dev
uv run python scripts/check.py
uv run python -m unittest discover -v
uv lock --check
uv build --wheel
```

`scripts/check.py`는 소스 구문 검사와 전체 unittest를 실행하고, 발견된 테스트가 0개면 실패한다. 새 테스트도 `unittest`가 수집할 수 있게 작성한다. GitHub Windows 단일 작업의 30분 제한에 걸린 뒤 CI는 `--shard 1/4`부터 `4/4`까지 독립 작업으로 실행하도록 바꿨다. 각 분할도 0건이면 실패하며 네 작업이 모두 통과해야 회귀가 완료된다. GUI는 가상 화면에서 검증한다. 실제 API 키나 주문 권한은 필요하지 않다. 연구/CUDA 설치는 [연구 도구](research-tools.md)와 `requirements-ml-cuda.txt`를 따르며, 운영 GUI에 GPU 학습 환경을 강제하지 않는다.

별도 환경에 wheel과 gui/prototype 의존성을 설치한 뒤 **저장소 밖 작업 폴더**에서 해당 Python으로 `scripts/smoke_installed.py --model-root <모델폴더>`를 실행하면 체크아웃 import 여부, 일반 GUI import, 제공 모델의 독립 worker와 국내/미국 저장 추론을 확인한다. 이 검사는 계좌·주문을 만들지 않는다. 원격 CI의 별도 wheel 작업은 빠른 `scripts/smoke_wheel_imports.py`로 빌드 산출물의 5개 명령 진입점·GUI 자산·소스 밖 import만 확인한다. 이 빠른 검사는 별도 설치 환경이나 모델 추론의 대체가 아니다. 로컬 통과와 원격 실행 완료는 구분한다.

## 구조와 호환

실제 구현은 `dockdack/broker`, `trading`, `persistence`, `application`, `signals`, `ui`로 분리한다. 옛 `dockdack.autotrade` 등 공개 import는 같은 모듈 객체를 가리키는 얇은 호환 경로다. 모델의 봉인된 소스·과거 연구 기록은 원래 위치/내용을 보존한다. 역할 분리 자체가 수익성이나 모든 실행시간 향상을 증명하지는 않는다.
