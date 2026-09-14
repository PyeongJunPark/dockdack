# 키움 전 종목 일봉 데이터셋

`dockdack.daily_dataset`은 키움 REST API에서 현재 조회 가능한 종목 목록을 받은 뒤, 각 종목의 수정주가 일봉을 연속조회가 끝날 때까지 저장한다.

수집 항목은 거래일, 시가, 고가, 저가, 종가, 거래량, 거래대금, 전일대비, 등락률, 수정주가 정보다. 국내 가격은 KRW, 미국 가격은 USD로 저장한다.

## 수집된 데이터 다운로드 (2026-09-14 스냅샷)

[GitHub 데이터 Release](https://github.com/PyeongJunPark/dockdack/releases/tag/daily-data-2026-09-14)에서 국내·미국 SQLite DB 압축본을 받을 수 있다.

| 시장 | 종목 목록 | 완료 상태 | 조회 오류 | 일봉 행 수 |
| --- | ---: | ---: | ---: | ---: |
| 국내 | 7,374 | 7,374 | 0 | 14,772,418 |
| 미국 | 12,742 | 12,718 | 24 | 26,100,170 |
| 합계 | 20,116 | 20,092 | 24 | 40,872,588 |

완료 상태는 API 연속조회가 끝났다는 의미이며, 빈 응답 종목도 포함한다. 실제 일봉이 있는 종목 수, 날짜 범위, 오류 종목과 파일 SHA-256은 [스냅샷 명세](datasets/2026-09-14.json)를 참고한다. 각 종목의 수집일이 다르므로 모든 종목이 2026-09-14까지 갱신됐다는 뜻은 아니다.

실제 일봉을 보유한 종목은 국내 7,039개, 미국 12,716개다. 전체 DB의 날짜 범위는 국내 1985-01-04~2026-08-21, 미국 1970-01-02~2026-09-11이며 각 종목의 시작일·종료일은 다르다. 두 ZIP의 압축 후 합계 크기는 약 1.71GB다.

다운로드 파일:

- [국내 DB ZIP](https://github.com/PyeongJunPark/dockdack/releases/download/daily-data-2026-09-14/domestic_daily.daily-data-2026-09-14.zip)
- [미국 DB ZIP](https://github.com/PyeongJunPark/dockdack/releases/download/daily-data-2026-09-14/us_daily.daily-data-2026-09-14.zip)
- [파일 체크섬](https://github.com/PyeongJunPark/dockdack/releases/download/daily-data-2026-09-14/SHA256SUMS.txt)

ZIP을 풀면 각각 `domestic_daily.sqlite3`, `us_daily.sqlite3`가 나온다. 프로젝트의 `data/kiwoom_daily/` 폴더에 두고 아래 Python 예제로 읽을 수 있다. 이미 수집한 DB가 있다면 다른 폴더에 풀어 확인한다.

DB 원본 합계는 약 7.58GB다. [GitHub 일반 Git의 파일 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) 때문에 DB 압축본은 Release 첨부 파일로 보관한다. `git clone`은 코드와 명세를 내려받으며, DB는 위 링크에서 별도로 다운로드한다. ZIP에는 DB만 포함된다.

압축본은 SQLite 백업 API로 일관된 스냅샷을 만든 뒤 `PRAGMA quick_check`와 압축 해제 SHA-256 일치 여부를 검증한다. 다음 명령으로 새로운 스냅샷을 만들 수 있다. 출력 디렉터리는 기존 파일 덮어쓰기를 방지하기 위해 새 경로여야 한다.

```powershell
.\.venv\Scripts\python.exe scripts/package_daily_snapshot.py --output-dir data/kiwoom_daily/release-20260914 --tag daily-data-2026-09-14
```

일반적인 다운로드 파일 검증은 `Get-FileHash .\us_daily.daily-data-2026-09-14.zip -Algorithm SHA256` 결과를 `SHA256SUMS.txt`와 비교한다.

## 전체 수집

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
.\.venv\Scripts\python.exe -m dockdack.daily_dataset --market domestic
.\.venv\Scripts\python.exe -m dockdack.daily_dataset --market us
```

두 시장을 한 프로세스에서 순서대로 수집하려면 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe -m dockdack.daily_dataset --market all
```

기본 저장 위치는 다음과 같다.

- `data/kiwoom_daily/domestic_daily.sqlite3`
- `data/kiwoom_daily/us_daily.sqlite3`

페이지를 받을 때마다 SQLite에 커밋하고 다음 연속조회 키를 기록한다. 실행이 중단되어도 같은 명령을 다시 실행하면 완료 종목은 건너뛰고 미완료 종목은 저장된 다음 페이지부터 이어간다.

최신 거래일을 추가하려면 완료 종목의 첫 구간만 다시 확인한다.

```powershell
.\.venv\Scripts\python.exe -m dockdack.daily_dataset --market all --refresh-complete
```

## 수집 범위

- 국내: KOSPI, KOSDAQ, K-OTC, KONEX, ETF, ETN, 리츠 등 `ka10099`의 주식 API 대상 카탈로그. 금현물은 별도 금 차트 TR을 사용하므로 제외한다.
- 미국: `usa06012`가 공식 지원하는 AMEX(`NA`), NASDAQ(`ND`), NYSE(`NY`). 종목 목록에 함께 반환되는 Pink Sheet(`NP`)는 이 일봉 TR의 지원 거래소가 아니므로 제외한다.
- 기간: 국내 `ka10081`, 미국 `usa06012`의 연속조회가 끝날 때까지다. 종목별 실제 최초 제공일은 상장일과 키움 보유 이력에 따라 다르다.
- 가격: `수정주가구분=1`. 미국은 환율을 적용하지 않은 USD 원가격이다.

현재 종목 목록을 기준으로 수집하므로 상장폐지 종목이 빠지는 생존편향이 있다. 과거 시점별 전체 종목 구성까지 필요한 연구에서는 별도 상장·상폐 이력 데이터와 결합해야 한다.

## 테이블

- `instruments`: 종목코드, 거래소, 종목명, ETF 여부와 원본 카탈로그 정보
- `daily_bars`: 종목별 일봉 OHLCV와 부가 필드
- `collection_progress`: 종목별 완료/부분/오류 상태와 연속조회 키
- `metadata`: 데이터 소스, 시장, 스키마 버전

예를 들어 Python 표준 라이브러리로 삼성전자 일봉을 읽을 수 있다.

```python
import sqlite3

connection = sqlite3.connect("data/kiwoom_daily/domestic_daily.sqlite3")
rows = connection.execute(
    """
    SELECT trade_date, open, high, low, close, volume
    FROM daily_bars
    WHERE symbol = ?
    ORDER BY trade_date
    """,
    ("005930",),
).fetchall()
```

빠른 검증 실행은 종목 수와 종목당 페이지 수를 제한한다.

```powershell
.\.venv\Scripts\python.exe -m dockdack.daily_dataset --market all --max-symbols 2 --max-pages-per-symbol 2
```

전체 수집에서는 두 제한을 지정하지 않는다. 모의투자 호출 제한 때문에 전체 완료에는 장시간이 걸릴 수 있다.
