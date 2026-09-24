# 동결 연구 자료의 위치 호환과 새 저장 도구

2026-09-24 감사 수정. 저장 모델의 가중치·특징 수식·확률·보정·매매 조건은 바꾸지 않는다. 학습, 실제 주문, 기존 DB 수정은 이 작업에서 수행하지 않았다. 이 문서는 과거 결과가 좋아졌다는 의미가 아니다. 연구 미통과, 재사용 역사 평가, 장중 경로 미검증 표시는 그대로다.

## 현재 위치와 원래 기록

현재 작업 폴더는 `C:/Users/user/Desktop/dockdack`이다. 정제 DB는 `data/kiwoom_daily/clean-20260916-v1`, 원래 캐시는 `outputs/mark1/cache`, 실험별 결과는 `outputs/mark1`에 있다. 과거 `dockdack-data-collection` / `dockdack-mark_1` 절대 경로는 학습 계약과 캐시 식별자의 일부이므로 **JSON 안에서 찾아 바꾸지 않는다**.

`ArtifactResolver`는 명시된 과거 루트만 현재 작업 폴더로 연결하고 SHA-256을 대조한다. 이름이 같다는 이유로 임의 파일을 고르지 않는다. 다른 경로는 Python API의 명시적 `relocations` 설정이 필요하다. DB는 읽기 전용으로 열며 시장·정제 완료·필수 테이블·비어 있지 않은 WAL을 검사한다. 읽기 후 내용 해시도 다시 확인한다. 큰 DB의 전체 해시 검사는 시간이 걸리지만 mtime만 믿도록 약화하지 않는다.

## 먼저 해볼 읽기 전용 점검

아래 명령은 작업 폴더에서 실행한다. 예시의 `python`은 연구 의존성이 설치된 환경을 뜻한다.

```powershell
python -m dockdack.research_tools --workspace . preflight --protocol outputs/mark1/deep-20260916/protocol.json --protocol outputs/mark1/selective-20260916/protocol.json --protocol outputs/mark1/half-20260920/protocol.json
python -m dockdack.research_tools --workspace . bundle-check --bundle models/mark1_prototype
python -m dockdack.research_tools --workspace . bundle-check --bundle models/mark1_0504
python -m dockdack.research_tools --workspace . bundle-check --bundle models/mark1_1_prototype
python -m dockdack.research_tools --workspace . report-half --training outputs/mark1/half-20260920 --backtest outputs/mark1/half-backtest-20260920 --bundle models/mark1_0504
```

마지막 명령은 보고서 입력만 검증한다. 새 보고서를 만들려면 존재하지 않는 `--output reports/새폴더`를 추가한다. 기존 report 함수의 완료·목표·배포 금지·모델 해시·시장별 일치 검사를 모두 유지한다. 단, 옛 요약파일 절대 경로로 기록된 체크섬 키에 대해서만 실제 내용이 같은 현재 경로를 메모리상 별칭으로 추가한다. 원래 보고서·학습 결과·JSON 파일은 수정하지 않으며 새 보고서에는 이 호환 처리 영수증을 함께 남긴다.

## 과거 backtest/export 실행 진입점

새 gateway가 원래 연구 모듈의 `main`을 호출한다. 먼저 `--help`만 보면 계산을 시작하지 않는다.

운영 앱 wheel에는 과거 `examples` 전체를 넣지 않는다. 이 연구 명령은 `--workspace`에 **원래 연구 소스 checkout과 해당 산출물이 있는 폴더**를 명시해야 한다. 설치된 CLI도 이 소스 위치를 해당 실행 동안만 Python 검색 경로에 추가하고 정상/실패 모두 복구한다. 다른 checkout의 examples가 이미 로드됐거나 설치본/소스의 코드 바이트가 다르면 차단하므로 새 CLI 프로세스에서 실행한다. 동결 원본 코드 해시는 원래 runner의 검증을 계속 적용한다.

```powershell
python -m dockdack.research_tools --workspace . backtest-deep -- --help
python -m dockdack.research_tools --workspace . backtest-selective -- --help
python -m dockdack.research_tools --workspace . backtest-half -- --help
python -m dockdack.research_tools --workspace . export-prototype -- --help
python -m dockdack.research_tools --workspace . export-half -- --help
```

실제로 실행할 때 backtest는 `--output-dir`, export는 `--output`으로 **새 폴더를 반드시 지정**한다. 생략하거나 기존 폴더를 주면 중단한다. 기존 스크립트를 직접 호출할 때의 옛 기본 경로 문제를 숨기지 않기 위한 조치다. 자동 학습 명령은 지원하지 않는다. 기존 연구 코드 파일은 그대로 두고 이 단일 목적 프로세스 안에서만 다음을 연결한다.

긴 옵션 이름의 약어는 허용하지 않는다. 예를 들어 `--output-di`는 `--output-dir` 대신 쓸 수 없다. 입력·출력 경로는 지정 workspace 범위 안이어야 하며 `..`, 드라이브 상대 경로, 외부 루트·링크 경유를 거부한다. 캐시 변환의 새 목적지도 같은 범위를 적용한다.

- 명시된 과거 절대 경로 → 현재 물리적 경로.
- 내용이 검증된 체크섬 사전의 논리 경로 키 → 현재 경로 키. 원본 source 계약 문자열과 캐시 키는 유지.
- 기존 frozen-cache 로더 → 원래 export 영수증의 NPZ 전체 해시 및 DB 해시를 확인하는 호환 로더.

수식·모델·시드·앙상블·거래 장부 계산은 원래 함수 그대로다. 끝나면 새 출력에 `relocation-compatibility.json`이 생긴다. Python API의 `compatibility_context`는 전역 참조를 임시 연결하므로 **GUI나 동시 추론 프로세스 안에서 쓰지 말고 별도 CLI 프로세스에서만 사용**한다. 정상/예외 종료 모두 원래 모듈 참조를 복원한다.

검증 수준: 실제 저장자료의 deep/selective/half 양 시장 학습 산출물, selective 비교 모델 provenance, 세 portable 묶음과 alias, 양 시장 DB·NPZ 내용 해시를 읽기 전용으로 확인했다. 5개 CLI는 help와 작은 가짜 main dispatch를 검증했다. **전체 backtest/export 수치 계산을 다시 돌린 것은 아니며**, 과거 손익이나 성공률의 새 검증을 주장하지 않는다. 이후 전체 재실행에서는 이식 영수증과 기존 결과 수치를 별도로 대조해야 한다.

## 큰 배열: 새 선택형 경로

`research_arrays.unpack_frozen_cache` / gateway `pack-cache`는 원래 NPZ 각 멤버를 1 MiB씩 새 NPY 폴더에 복사한다. 원래 캐시를 변경·삭제하지 않는다. `--cache-sha256`에는 원래 export 영수증에 기록된 값을 명시해야 한다. 바이트와 계약을 확인한 뒤 완료 표시가 있는 새 폴더를 공개한다. 오류/중단 시 `.이름-building-*`를 남겨 조사할 수 있으며 성공 산출물로 취급하지 않는다.

```powershell
python -m dockdack.research_tools --workspace . pack-cache --source outputs/mark1/selective-20260916/domestic/source.json --cache-dir outputs/mark1/cache --cache-sha256 5c6df7b0d79785f2c8b48ebbc53cb647fd5529322c093f810ac70923842f7af0 --destination outputs/mark1/cache/domestic-fb9a4f1e01e682f0
```

이 예시는 선택적으로 **추가 디스크 공간을 쓰는** 변환 명령이지 이번에 실제 대형 캐시를 변환했다는 뜻이 아니다. `load_frozen_cache_compatible(..., mmap_directory=...)`로 읽으면 숫자 배열이 읽기 전용 mmap이다. 원래 NPZ만 주면 기존과 같이 전체 배열을 메모리에 올린다.

실제 backtest/export gateway에서도 command 앞에 `--mmap-root`를 지정하면 사용할 수 있다. 이 폴더 아래에 정확히 `{market}-{원캐시키}`라는 이름의 완료된 mmap 폴더가 있어야 한다. 예를 들어 국내만 처리하는 deep 실행은 다음과 같다.

```powershell
python -m dockdack.research_tools --workspace . --mmap-root outputs/mark1/cache backtest-deep -- --market domestic --output-dir outputs/mark1/deep-backtest-mmap-new
```

양 시장을 처리하는 명령에는 미국 `us-7bc645867ce3c050` 완료 폴더도 필요하다. 누락/미완료/다른 source 또는 NPZ 해시/변경된 NPY이면 메모리 모드로 조용히 되돌아가지 않고 중단한다. 원 NPZ 및 DB 전체 검증은 그대로 수행하므로 이 옵션은 주로 host 메모리 개선 경로이며 해시 I/O를 없애지 않는다. 모델/배치 함수가 자기 GPU bank나 표본 필터 사본을 만드는 비용은 남는다.

`write_feature_chunks`는 원래 특징 함수를 주어진 배치로 호출해 바로 float32 NPY에 쓴다. 전체 `표본수×184` 출력 행렬을 RAM에 먼저 만들지 않는다. 작은 고정 입력에서 selective와 half 두 원래 함수의 결과와 완전히 같음을 검사했다. 완료 파일에 입력 identity, 색인 해시, 파일 해시를 남긴다. 원래 봉인된 trainer가 자동으로 이 도구를 사용하도록 바꾸지는 않았다. 기존 전체 packed DB 로딩, 반복 전체 해시, GPU bank, 학습 함수 자체의 메모리 비용은 여전히 남으며 전시장 peak-RSS나 속도 개선 배수를 측정하지 않았다.

## 의존성·봉인·중복 코드

`requirements-selective.txt`는 관측된 연구 CPU 라이브러리 버전(NumPy 2.5.3, CatBoost 1.2.10, LightGBM 4.6.0 등)을 고정한다. Torch GPU 빌드는 플랫폼에 맞게 별도 선택한다. 과거 기록은 Torch `2.11.0+cu128`, RTX 5080이다. 같은 버전/시드가 다른 GPU 드라이버에서 비트 단위 동일 학습을 보장하지 않는다. GUI 런타임에 모든 연구 의존성을 넣을 필요는 없다.

`mark1_deep_models.py`는 바뀐 CRLF를 기록 당시 LF로만 복원했다. SHA-256은 `50ae22f70278a9cd4ca0fb276d20260a2421c9874ac8f35846a10529d15d16d7`이며 `.gitattributes`가 추가 줄바꿈 변환을 막는다. 모델 수식은 그대로다.

새 JSON/해시/내용 확인/이식·저장 기능은 `research_artifacts`, `research_arrays`, `research_compat`, `research_tools`에 공용화했다. 과거 backtest 두 버전은 목표와 원래 봉인을 재현하는 자료이므로 보존한다. 이를 억지로 하나의 새 손익 엔진으로 합치거나 저장 모델의 추론 코드를 리팩터링하지 않는다. 향후 연구는 새 공용 I/O를 재사용하고 숫자 계산 변경은 별도 버전·golden parity 검증 후에만 채택한다.
