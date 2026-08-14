# Development Log

## 2026-08-06

### 진행 내용

- DockDack 프로젝트 개발 시작
- 프로젝트 계획서 작성
- GitHub 저장소 생성
- README 작성
- Development Log 작성

## 2026-08-13

### 진행 내용

- Codex와 GitHub 연동
- `feat/broker-api` 브랜치 분기 및 [이슈 #2](https://github.com/PyeongJunPark/dockdack/issues/2) 생성
- `feat/data-collection` 브랜치 분기 및 [이슈 #1](https://github.com/PyeongJunPark/dockdack/issues/1) 생성

## 2026-08-14

### 결정 사항

- 증권사 API는 키움증권 REST API를 사용하기로 결정

### 결정 사유

- 국내주식과 미국주식을 모두 지원
- 국내주식과 미국주식 모의투자를 지원
- 초빈도매매에 최적화된 API는 아니지만, WebSocket 실시간 시세를 활용한 자동매매 구현이 가능

### 진행 내용

- 키움증권 API 연동 모듈 개발
- 키움증권 API App Key/Secret 발급 완료
