# DATA / 연구 감사 수정 결과

2026-09-24. 이 문서는 원래 [감사 기록](research.md)을 덮지 않고 실제 구현·검증·남은 한계를 분리한 후속 기록이다. [기계 판독 검증 영수증](research-fix-verification.json)과 [사용 명령](../../research-tools.md)을 함께 본다.

| 항목 | 구현 | 검증과 범위 |
|---|---|---|
| DATA-01 수정주가 혼합 | `daily_generations.py`에 전체 이력 staging, 원자적 공개, 이전 원본 행 보존, 재시작/공개 anchor, 소유 lease/fencing, 공개 DB fingerprint 추가. 이전 마지막 날짜를 만났다는 이유로 중단하지 않음 | 가짜 DB로 분할·역분할, 중간 종료·네트워크 실패, 재개 시 revision, 공개 중 강제 실패 rollback, 동시 소유권/별도 writer 회귀. 실제 raw/clean DB 갱신은 하지 않음 |
| DATA-02 누락 필드의 빈 성공 처리 | 필수 목록 키/타입, 각 객체·날짜·유한 OHLCV·범위·정수 거래량·정렬·중복·연속조회 검사 | 누락 목록·잘못된 행은 공개/완료 금지. 명시된 최초 빈 이력은 허용, 기존 공개 이력을 빈 응답으로 삭제하지 않음 |
| DATA-03 잘못된 종목 저장 | 응답/행에 있는 종목과 거래소를 요청 identity와 대조 | 국내 A 접두사와 방향 가격 부호 호환; 다른 종목·미국 거래소 거부. 공급자가 식별자를 아예 주지 않으면 독립 identity 검증은 불가능 |
| R-01 옛 절대 경로 | `research_artifacts` 내용 확인 resolver, `research_compat` 원계약 캐시 API, `research_tools`의 deep/selective/half backtest 및 prototype/half export gateway | 실제 이전 경로 DB·NPZ의 원해시와 원 캐시 키 확인. 양 시장의 원래 학습 artifact/comparator 검증 PASS. 새 전체 backtest/export 수치 계산은 재실행하지 않음 |
| R-02 half 보고서 키 | 원본 reader를 독립 모듈로 로드하고 확인된 summary 해시의 물리 경로 별칭만 메모리상 추가 | 실제 canonical half training/backtest/bundle 세트가 원래 모든 report 입력 검사를 통과. 원본 JSON·보고서 수정 없음. 새 렌더는 선택형 |
| R-03 deep 줄바꿈 봉인 | `mark1_deep_models.py`의 원래 LF 바이트 복원, `.gitattributes -text` 추가 | SHA `50ae22…d16d7` 일치. deep 7/7, selective 10/10, half 13/13 소스 해시 PASS. 현재 Git attribute도 text unset. 새 Windows 클론 전체 설치는 별도 검증 필요 |
| R-04 큰 배열 메모리 | 원 NPZ를 스트리밍하여 새 NPY mmap store로 옮기는 선택형 도구; 원 특징 함수를 그대로 배치 호출하여 디스크에 바로 쓰는 공용 도구 | 작은 fixture에서 두 frozen 특징 함수의 전체 출력과 정확히 같음, 배치 2/2/1 관측, 중단 결과 공개 금지. 기존 봉인 trainer의 packed loading/GPU bank 및 반복 해시 I/O는 그대로며 전시장 속도/peak-RSS 배수는 주장하지 않음 |
| R-05 연구 의존성 | `requirements-selective.txt`에 관측 NumPy/CatBoost/LightGBM 버전과 Torch GPU 분리 조건 명시; research/dev extra와 lock 추가 | 별도 설치본의 Torch/LightGBM/CatBoost import 및 5개 gateway help 성공. GPU fitting·전체 수치 재실행은 안 함; 플랫폼 독립 학습 재현 보장 아님 |
| R-06 동결 복제의 유지보수 | 새 JSON/내용 해시/DB 확인/이식/완료 기록·배열 I/O를 공용 모듈로 모으고 원 frozen 계산 모듈은 보존 | 세 bundle의 모델/sidecar/runtime/alias 해시 확인. 기존 서로 다른 목표의 장부 엔진을 강제 통합하지 않으며 모델 수식·가중치 수정 없음 |

## 검증 기록

- `python -m unittest test_daily_dataset test_daily_generations test_research_tools -q`: **42개 PASS**, 실제 DB 대신 임시 SQLite/배열/가짜 응답 사용.
- Python compile 검사와 수정 파일 whitespace 검사 통과.
- deep/selective/half 봉인 소스: **7/7, 10/10, 13/13**.
- `models/mark1_prototype`, `models/mark1_0504`, `models/mark1_1_prototype`: 체크섬 확인. mark1.1의 identity alias 목록·wrapper·원본 파일 및 검증 영수증도 확인.
- 실제 국내/미국 clean DB와 원 NPZ는 읽기 전후 전체 SHA-256이 과거 기록과 일치. 큰 배열 내용은 메모리에 올리지 않고 헤더만 조회했다.
- 실제 deep/selective/half 양 시장의 원래 산출물 검증과 selective comparator provenance 검증 통과.
- 5개 runner gateway는 가짜 main dispatch 테스트, 실제 parser help 확인. 전체 모델 추론·백테스트·내보내기·학습은 실행하지 않았다.
- 독립 경계 검토에서 발견한 출력 옵션 약어 우회, cwd 복구 실패 시 전역 복구 누락, pack-cache 외부 목적지 문제를 수정하고 회귀 검사했다. `--mmap-root`도 실제 gateway context에서 가짜 DB/NPZ → 완료 mmap → 읽기 전용 원 dataclass까지 확인했다.

## 남는 운영 조건

새 수집기의 전체 세대 조회는 요청 수와 저장 공간이 늘어난다. 이전·중단 세대를 자동 지우지 않는다. 공급자 snapshot ID가 없으므로 첫 페이지에 드러나지 않는 임의 과거 revision을 완전히 배제할 수 없다. 이번 수정이 기존 데이터의 모든 기업행위 오류를 제거한 것은 아니다. [수집 공개 단위 안내](../../daily-collection-generations.md)에 상세 제약이 있다.

연구 gateway는 소스 checkout이 있는 별도 프로세스 전용이다. 기존 파일 덮어쓰기를 막고 새 출력 위치를 요구한다. source 계약·원 캐시 키·원 수식은 유지하지만, 전시장 수치 재실행은 아직 하지 않았으므로 향후 전체 재실행 결과와 원 결과의 대조가 추가 검증 단계다. 저장 모델의 연구 미통과·재사용 평가·장중 경로 미검증 표시는 변경하지 않는다.
