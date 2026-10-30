# 일반 데스크 GUI 런타임 컨트롤 전수 목록

기준: `codex/optimization-adversarial-audit`, HEAD `6e3ace7`. 임시 DB·가짜 서비스·두 prototype 체크 상태·주문/감시 OFF. 계좌/API/주문 호출 0. QWidget 하위 버튼, 체크박스, 입력, 날짜, 탭, 텍스트 256개를 열거했다. Qt 자체 내부 입력도 포함한다. 동적 종목 행/로그의 모든 가능한 문자열을 열거한 것은 아니다.

상태: 기본 보유종목 탭에서 표시/활성 여부와 관련 탭 전환 뒤 보이는지 기록. 숨김은 불필요함과 같지 않다. 표준 Qt 내부 위젯은 삭제하지 않는다. 각 항목의 전체 tooltip, 선택지, 접근성 이름, 삭제/유지 사유는 [JSON](../../../outputs/optimization-audit/gui-inventory.json)에 있다. 결함 근거와 개선안은 [GUI 검토](gui.md).

| ID | 위젯 | 속성/이름 | 소속 탭 | 텍스트/선택값 | 초기표시/활성/발견 | 연결 동작 | 판정 |
|---:|---|---|---|---|---|---|---|
| 1 | QPushButton | WindowControls.fullscreen_button | 공통 상단/하단 | 전체화면 · F11 | 표시/활성/탭 이동 시 표시 | toggle_fullscreen | 유지 |
| 2 | QLabel | V00Window.brand_mark | 공통 상단/하단 |  | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 3 | QLabel | heading | 공통 상단/하단 | DOCKDACK  ver 0.0 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 4 | QLabel | V00Window.environment_caption | 공통 상단/하단 | 모의 / 실전 · 기본 주문 OFF | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 5 | QPushButton | (익명) | 공통 상단/하단 | 모의투자 | 표시/비활성/탭 이동 시 표시 | EnvironmentSelector.requested -> request_environment | 유지 |
| 6 | QPushButton | (익명) | 공통 상단/하단 | 실전투자 | 표시/비활성/탭 이동 시 표시 | EnvironmentSelector.requested -> request_environment | 유지 |
| 7 | QLabel | EnvironmentSelector.badge | 공통 상단/하단 | 모의 · 가상 자금 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 8 | QLabel | V00Window.mode_label | 공통 상단/하단 | 자동주문 OFF · 주문 차단 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 9 | QLabel | V00Window.environment_notice | 공통 상단/하단 | 모의 · 가상 자금 사용 / 실전과 잔고·규칙·매매일지 분리 | 숨김/활성/계속 숨김 |  | 유지·중복 문구 축약 검토 |
| 10 | QLabel | V00Window.monitoring_label | 공통 상단/하단 | 감시 중지 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 11 | QLabel | V00Window.order_status_detail | 공통 상단/하단 | 시세 감시와 자동주문이 중지되어 있습니다. | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 12 | QLabel | connectionMode | 공통 상단/하단 | 한국 · 휴장 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 13 | QLabel | connectionMode | 공통 상단/하단 | 미국 · 장전 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 14 | QLabel | muted | 공통 상단/하단 | 정규장 기준 · 거래 시간은 마우스를 올려 확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 15 | QLabel | V00Window.health_label | 공통 상단/하단 | 앱 응답 16:50:28  ·  감시 중지  ·  시세 오류 0 / 잔고 오류 0건 | 숨김/활성/계속 숨김 |  | 유지·중복 문구 축약 검토 |
| 16 | QLabel | muted | 공통 상단/하단 | 순회 후 대기 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 17 | QSpinBox | V00Window.interval | 공통 상단/하단 | 30 초 | 표시/활성/탭 이동 시 표시 |  | 유지 |
| 18 | QLineEdit | qt_spinbox_lineedit | 공통 상단/하단 | 30 초 | 표시/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 19 | QPushButton | V00Window.refresh_button | 공통 상단/하단 | 전체 1회 조회 | 표시/활성/탭 이동 시 표시 | refresh_all (normal window) / request_refresh (portfolio, order history) / synchronous refresh(force=True) (journal) | 유지 |
| 20 | QPushButton | V00Window.start_button | 공통 상단/하단 | 감시 시작 (조회만) | 표시/활성/탭 이동 시 표시 | start_monitoring | 유지 |
| 21 | QPushButton | V00Window.arm_button | 공통 상단/하단 | 자동주문 켜기 (ON) | 표시/활성/탭 이동 시 표시 | enable_auto_orders | 유지 |
| 22 | QPushButton | V00Window.disarm_button | 공통 상단/하단 | 자동주문 끄기 (OFF) | 표시/비활성/탭 이동 시 표시 | disable_auto_orders | 유지 |
| 23 | QPushButton | V00Window.stop_button | 공통 상단/하단 | 감시·주문 중지 | 표시/비활성/탭 이동 시 표시 | stop_monitoring | 유지 |
| 24 | QLabel | V00Window.connection_summary | 공통 상단/하단 | 외부 AI 2개 · 감시 OFF · 주문 OFF · 모델별 연결 상태는 매매 설정에서 확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 25 | QPushButton | V00Window.connection_shortcut | 공통 상단/하단 | 신호 연결 확인  → | 표시/활성/탭 이동 시 표시 | navigate to signal_connection_page | 탐색 통합 후보 |
| 26 | QTabWidget | V00Window.workspace_tabs | 공통 상단/하단 | 매매 설정 | 표시/활성/탭 이동 시 표시 |  | 유지 |
| 27 | QTabWidget | V00Window.tabs | 매매 설정 | 트리거 규칙 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 28 | QCheckBox | V00Window.random_demo | 매매 설정 / 모의 테스트 신호기 | 내장 모의 테스트 신호기 · 미보유 10% 매수 / +1% 익절 · -0.8% 손절 | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 고급/호환 기능으로 이동 |
| 29 | QLabel | muted | 매매 설정 / 모의 테스트 신호기 | 미국 처리 방식 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 30 | QComboBox | V00Window.random_us | 매매 설정 / 모의 테스트 신호기 | 미국 주문 차단 (모의 시장가 미지원) | 숨김/활성/탭 이동 시 표시 |  | 고급/호환 기능으로 이동 |
| 31 | QLabel | muted | 매매 설정 / 모의 테스트 신호기 | 국내 매수는 시장가 · 매수 수량은 설정한 평가자산 비중 또는 고정 수량 적용<br>보유분 매도는 상방·하방 가격을 별도로 확인하며 금액 상한과 매도 가능 수량 적용<br>같은 입력은 다시 추첨하지 않습니다. 이미 보유하면 추가매수하지 않습니다.<br>실제 평균 매입가 대비 +1% 이상 익절 / -0.8% 이하 손절 신호입니다.<br>호출 지연·가격 변동·미체결로 해당 수익률에서의 체결은 보장되지 않습니다 (수수료·세금 별도).<br>외부 신호 탭에서 KRW/USD 상한 설정 → 자동주문 켜기 확인 → 전체 조회 성공 후 ON. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 32 | QCheckBox | V00Window.external_mode | 매매 설정 / 외부 신호 연결 | 외부 신호 모드 (수동 트리거 실행 안 함) | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 유지 |
| 33 | QLabel | muted | 매매 설정 / 외부 신호 연결 | source_id | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 34 | QLineEdit | V00Window.external_source | 매매 설정 / 외부 신호 연결 | external-model | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 35 | QSpinBox | V00Window.external_quantity | 매매 설정 / 외부 신호 연결 | 999999999 주 상한 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 36 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 외부 신호 연결 | 999999999 주 상한 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 37 | QDoubleSpinBox | V00Window.external_krw | 매매 설정 / 외부 신호 연결 | 10,000,000.00 KRW / 주문 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 38 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 외부 신호 연결 | 10,000,000.00 KRW / 주문 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 39 | QDoubleSpinBox | V00Window.external_usd | 매매 설정 / 외부 신호 연결 | 10,000.00 USD / 주문 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 40 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 외부 신호 연결 | 10,000.00 USD / 주문 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 41 | QLabel | muted | 매매 설정 / 외부 신호 연결 | 신호 JSON 입력 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 42 | QLineEdit | V00Window.signal_path | 매매 설정 / 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\signals.json | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 43 | QLabel | muted | 매매 설정 / 외부 신호 연결 | 차트 JSON 출력 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 44 | QLineEdit | V00Window.chart_path | 매매 설정 / 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\charts.json | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 45 | QPushButton | V00Window.read_signals_button | 매매 설정 / 외부 신호 연결 | 신호 파일 1회 읽기 (주문 안 함) | 숨김/활성/탭 이동 시 표시 | read_signals (ingests candidates, orders remain OFF) | 고급/호환 기능으로 이동 |
| 46 | QLabel | muted | 매매 설정 / 외부 신호 연결 | 외부 모드: 종목별 즉시 전송 + 순회 후 전체 파일 갱신 · 과거 일봉 DB 재사용 / 당일 봉 증분 조회<br>금액 0은 해당 시장 차단 · 파일 읽기는 주문 활성화가 아님 · 입력/상한은 이번 창에서만 유지 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 47 | QCheckBox | V00Window.percent_sizing | 매매 설정 / 외부 신호 연결 | 평가자산 비중으로 매수 | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 유지 |
| 48 | QDoubleSpinBox | V00Window.buy_percent | 매매 설정 / 외부 신호 연결 | 10.00 % / 1회 매수 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 49 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 외부 신호 연결 | 10.00 % / 1회 매수 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 50 | QCheckBox | V00Window.order_popups | 매매 설정 / 외부 신호 연결 | 매수·매도 주문 팝업 알림 | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 유지 |
| 51 | QLabel | muted | 매매 설정 / 외부 신호 연결 | 시장별 현금 + 보유 평가금액 기준 · 국내 음수 예수금은 검증된 D+2 기준 · 가용액/상한 이내 정수 주<br>보유종목 별도 매도 순회: 전략별 목표 우선 · 목표 없는 기존 보유분만 기본 +1% / -0.8% · 일반 오류는 해당 종목만 보류 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 52 | QLabel | (익명) | 매매 설정 / 외부 신호 연결 | 추가 매수 신호기 · 각 source_id와 JSON 입력은 서로 달라야 합니다 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 53 | QAbstractButton | qt_tableview_cornerbutton | 매매 설정 / 외부 신호 연결 |  | 숨김/활성/탭 이동 시 표시 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 54 | QPushButton | (익명) | 매매 설정 / 외부 신호 연결 | 신호기 추가 | 숨김/활성/탭 이동 시 표시 | SourceList.add_row / currentRow remove | 고급 기능으로 이동 |
| 55 | QPushButton | (익명) | 매매 설정 / 외부 신호 연결 | 선택 신호기 제거 | 숨김/활성/탭 이동 시 표시 | SourceList.add_row / currentRow remove | 고급 기능으로 이동 |
| 56 | QLabel | V00Window.source_status | 매매 설정 / 외부 신호 연결 | external-model: missing · mark1-prototype-demo-trigger: ok · mark1-1-prototype-demo-trigger: ok | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 57 | QLabel | (익명) | 매매 설정 / 외부 신호 연결 | 외부 AI 매수 신호기 · 두 모델 동시 연결 가능 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 58 | QCheckBox | external-mark1-prototype | 매매 설정 / 외부 신호 연결 | mark1 prototype · +1% / −0.9% · 외부 프로세스 | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 유지 |
| 59 | QLabel | (익명) | 매매 설정 / 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\external-models\mark1-prototype\signals.json | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 60 | QCheckBox | external-mark1-1-prototype | 매매 설정 / 외부 신호 연결 | mark1.1 prototype · +0.5% / −0.4% · 외부 프로세스 | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 유지 |
| 61 | QLabel | (익명) | 매매 설정 / 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\external-models\mark1-1-prototype\signals.json | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 62 | QLabel | (익명) | 매매 설정 / 외부 신호 연결 | 기존 내장 트리거 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 63 | QComboBox | V00Window.model_trigger | 매매 설정 / 외부 신호 연결 | 내장 트리거 없음 · 외부 신호만 사용 | 숨김/활성/탭 이동 시 표시 |  | 고급/호환 기능으로 이동 |
| 64 | QLabel | V00Window.model_status | 매매 설정 / 외부 신호 연결 | mark1-prototype · 외부 연결 대기<br>mark1-1-prototype · 외부 연결 대기 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 65 | QLabel | V00Window.model_notice | 매매 설정 / 외부 신호 연결 | mark1 prototype · 모의투자 전용 · 과거 30일봉 + 현재가 · 성공확률 50% 초과일 때 매수<br>이 트리거의 새 매수 목표: +1% 익절 / −0.9% 손절 · 기존 보유분의 저장된 목표는 유지<br>연구 검증 미통과 · 주식분할 가격단위 데이터 문제 확인 · 미국 과거 검증 매수 신호 0건 · 수익 보장 없음<br><br>mark1.1 prototype · 모의투자 전용 · 과거 30일봉 + 현재가 · 성공확률 50% 초과일 때 매수<br>이 트리거의 새 매수 목표: +0.5% 익절 / −0.4% 손절 · 기존 보유분의 저장된 목표는 유지<br>연구 검증 미통과 · 미국 과거 시가 검증 매수 신호 123건 · 비용 반영 손실 · 주식분할 가격단위 문제와 장중 진입 성과 미검증 · 수익 보장 없음 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 66 | QLabel | V00Window.rule_label | 매매 설정 / 트리거 규칙 | 005930 · KRW · 1회성 규칙 등록 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 67 | QLabel | muted | 매매 설정 / 트리거 규칙 | 조건 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 68 | QComboBox | V00Window.trigger | 매매 설정 / 트리거 규칙 | 현재가 ≥ 지정 가격 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 69 | QLabel | muted | 매매 설정 / 트리거 규칙 | 기준 가격 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 70 | QDoubleSpinBox | V00Window.threshold | 매매 설정 / 트리거 규칙 | 0.0000 KRW | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 71 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 트리거 규칙 | 0.0000 KRW | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 72 | QLabel | muted | 매매 설정 / 트리거 규칙 | 이동평균 N | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 73 | QSpinBox | V00Window.period | 매매 설정 / 트리거 규칙 | 20 일 | 숨김/비활성/탭 이동 시 표시 |  | 유지 |
| 74 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 트리거 규칙 | 20 일 | 숨김/비활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 75 | QLabel | muted | 매매 설정 / 트리거 규칙 | 방향 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 76 | QComboBox | V00Window.side | 매매 설정 / 트리거 규칙 | 매수 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 77 | QLabel | muted | 매매 설정 / 트리거 규칙 | 수량 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 78 | QSpinBox | V00Window.quantity | 매매 설정 / 트리거 규칙 | 1 주 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 79 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 트리거 규칙 | 1 주 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 80 | QLabel | muted | 매매 설정 / 트리거 규칙 | 주문금액 상한 (필수) | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 81 | QDoubleSpinBox | V00Window.max_notional | 매매 설정 / 트리거 규칙 | 0.00 KRW | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 82 | QLineEdit | qt_spinbox_lineedit | 매매 설정 / 트리거 규칙 | 0.00 KRW | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 83 | QLabel | muted | 매매 설정 / 트리거 규칙 |  | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 84 | QPushButton | V00Window.rule_button | 매매 설정 / 트리거 규칙 | 규칙 등록 | 숨김/활성/탭 이동 시 표시 | add_rule | 유지 |
| 85 | QAbstractButton | qt_tableview_cornerbutton | 매매 설정 / 트리거 규칙 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 86 | QPushButton | V00Window.pause_button | 매매 설정 / 트리거 규칙 | 선택 규칙 비활성화 | 숨김/활성/탭 이동 시 표시 | pause_rule | 유지 |
| 87 | QPushButton | V00Window.review_button | 매매 설정 / 트리거 규칙 | 내역 확인 후 종목 차단 해제 | 숨김/활성/탭 이동 시 표시 | review_attempt | 유지 |
| 88 | QToolButton | ScrollLeftButton | 매매 설정 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 89 | QToolButton | ScrollRightButton | 매매 설정 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 90 | QLabel | section | 외부 신호 연결 | 자동매매 신호 연결 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 91 | QPushButton | SignalConnectionPanel.settings_button | 외부 신호 연결 | 연결 설정 | 숨김/활성/탭 이동 시 표시 | request_settings -> open_connection_settings | 탐색 통합 후보 |
| 92 | QPushButton | SignalConnectionPanel.inspect_button | 외부 신호 연결 | 파일 검사 (주문 없음) | 숨김/활성/탭 이동 시 표시 | request_inspect -> inspect_signals (format only) | 유지 |
| 93 | QPushButton | SignalConnectionPanel.folder_button | 외부 신호 연결 | 연결 폴더 | 숨김/활성/탭 이동 시 표시 | request_folder -> open_connection_folder | 유지 |
| 94 | QLabel | SignalConnectionPanel.source_label | 외부 신호 연결 | 외부 코드 · JSON 파일 교환 · source_id: external-model | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 95 | QLabel | SignalConnectionPanel.mode_label | 외부 신호 연결 | 설정됨 · 수신기 비활성 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 96 | QLabel | muted | 외부 신호 연결 | 종목별 즉시 출력 → 계속 확인할 폴더 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 97 | QLineEdit | SignalConnectionPanel.update_path | 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\charts_updates | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 98 | QLabel | muted | 외부 신호 연결 | 전체 차트 출력 → 순회 완료 후 저장 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 99 | QLineEdit | SignalConnectionPanel.output_path | 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\charts.json | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 100 | QLabel | muted | 외부 신호 연결 | 신호 입력 ← 신호기가 쓸 파일 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 101 | QLineEdit | SignalConnectionPanel.input_path | 외부 신호 연결 | C:\Users\user\AppData\Local\Temp\tmpmeou4edq\exchange\signals.json | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 102 | QLabel | section | 외부 신호 연결 | 1  차트 전달 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 103 | QLabel | (익명) | 외부 신호 연결 | 종목별 즉시 파일 · 저장 성공 기록 없음<br>전체 차트 파일 (순회 완료 후) · 저장 성공 기록 없음<br>저장 완료는 외부 코드가 읽었다는 확인이 아닙니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 104 | QLabel | section | 외부 신호 연결 | 2  신호 입력 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 105 | QLabel | (익명) | 외부 신호 연결 | 수신 중지 · 아래 경로/설정은 자동으로 적용되지 않을 수 있습니다. · 최근 읽기 — | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 106 | QLabel | section | 외부 신호 연결 | 3  검증·수신 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 107 | QLabel | (익명) | 외부 신호 연결 | 접수 확인 없음 · 형식 검사와 실제 신호 접수는 별개입니다.<br>후보/접수는 주문·체결이 아닙니다. HOLD는 매매하지 않음입니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 108 | QLabel | section | 외부 신호 연결 | 4  주문 허용 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 109 | QLabel | (익명) | 외부 신호 연결 | OFF · 새 자동주문 차단 / 감시와 신호 수신은 별도 동작<br>한국 · 휴장 · 미국 · 장전 · 최근 차단/대기 사유: 사용자 주문 허용 OFF | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 110 | QLabel | SignalConnectionPanel.inspection_label | 외부 신호 연결 | ‘파일 검사’는 JSON 형식·설정 상한만 확인합니다. 신호 접수·규칙 생성·주문은 하지 않습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 111 | QLabel | muted | 외부 신호 연결 | 실시간 연결은 종목별 출력 폴더를 확인하세요. 전체 차트 파일은 전체 순회가 끝난 뒤 갱신됩니다.<br>고유 signal_id와 읽은 차트의 export_id로 신호 JSON을 원자적으로 교체하세요.<br>파일 존재만으로 연결 성공을 판정하지 않으며, 실제 매수·매도는 ‘실제 주문·체결’ 탭에서 확인합니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 112 | QPushButton | V00Window.ranking_button | 관심종목·차트 | 현재 선정 가능한 시장 · 거래량 TOP100 | 숨김/활성/탭 이동 시 표시 | add_top100 | 유지 |
| 113 | QPushButton | V00Window.export_button | 관심종목·차트 | 차트 JSON 내보내기 | 숨김/활성/탭 이동 시 표시 | export_json | 유지 |
| 114 | QCheckBox | V00Window.hourly_ranking | 관심종목·차트 | 국내·미국 거래량 TOP100 · 개장 10분 전 / 장중 매 정시 | 숨김/활성/탭 이동 시 표시 | toggled/currentChanged setting; see owning class | 유지 |
| 115 | QLineEdit | V00Window.symbol_input | 관심종목·차트 |  | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 116 | QComboBox | V00Window.exchange_input | 관심종목·차트 | 거래소 자동 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 117 | QSpinBox | V00Window.days_input | 관심종목·차트 | 31 거래일 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 118 | QLineEdit | qt_spinbox_lineedit | 관심종목·차트 | 31 거래일 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 119 | QPushButton | V00Window.add_button | 관심종목·차트 | 관심종목 추가 | 숨김/활성/탭 이동 시 표시 | add_item | 유지 |
| 120 | QPushButton | V00Window.days_button | 관심종목·차트 | 선택 종목 기간 적용 | 숨김/활성/탭 이동 시 표시 | apply_days | 유지 |
| 121 | QPushButton | V00Window.remove_button | 관심종목·차트 | 제외 | 숨김/활성/탭 이동 시 표시 | remove_item | 유지 |
| 122 | QLabel | V00Window.chart_title | 관심종목·차트 | 종목을 조회해 주세요 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 123 | QTabWidget | (익명) | 관심종목·차트 | 일봉 차트 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 124 | QAbstractButton | qt_tableview_cornerbutton | 관심종목·차트 / 일봉 데이터 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 125 | QToolButton | ScrollLeftButton | 관심종목·차트 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 126 | QToolButton | ScrollRightButton | 관심종목·차트 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 127 | QTabWidget | V00Window.watch_market_tabs | 관심종목·차트 | 한국 · KRW (1) | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 128 | QAbstractButton | qt_tableview_cornerbutton | 관심종목·차트 / 미국 · USD (0) |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 129 | QAbstractButton | qt_tableview_cornerbutton | 관심종목·차트 / 한국 · KRW (1) |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 130 | QToolButton | ScrollLeftButton | 관심종목·차트 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 131 | QToolButton | ScrollRightButton | 관심종목·차트 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 132 | QLabel | V00Window.watch_market_hint | 관심종목·차트 | 한국 관심종목 1개 · 탭 전환과 무관하게 양쪽 시장 감시 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 133 | QLabel | section | 서버·감시 로그 | 서버 상태와 데이터 흐름 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 134 | QLabel | OperationsPanel.runtime | 서버·감시 로그 | 앱 응답 16:50:28  ·  감시 중지  ·  시세 오류 0 / 잔고 오류 0건<br>마지막 작업 응답 —  ·  마지막 작업 종료 —<br>자동주문 OFF · 주문 차단 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 135 | QLabel | OperationsPanel.flow | 서버·감시 로그 | 최근 감시 기록 —  ·  최근 신호 기록 — | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 136 | QTabWidget | OperationsPanel.tabs | 서버·감시 로그 | 서버·오류 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 137 | QLabel | muted | 서버·감시 로그 / 매매 신호 (주문 아님) | BUY / SELL은 매매 제안이며 주문·체결이 아닙니다. HOLD는 매매하지 않음입니다. 자동주문 OFF여도 신호 수신은 계속됩니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 138 | QLabel | muted | 서버·감시 로그 / 매매 신호 (주문 아님) | 아직 이 분류의 기록이 없습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 139 | QAbstractButton | qt_tableview_cornerbutton | 서버·감시 로그 / 매매 신호 (주문 아님) |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 140 | QLabel | muted | 서버·감시 로그 / 시세·차트 감시 | 종목별 조회 성공·실패, 일봉 캐시와 차트 전달 기록입니다. 매수·매도 내역이 아닙니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 141 | QLabel | muted | 서버·감시 로그 / 시세·차트 감시 | 아직 이 분류의 기록이 없습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 142 | QAbstractButton | qt_tableview_cornerbutton | 서버·감시 로그 / 시세·차트 감시 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 143 | QLabel | muted | 서버·감시 로그 / 서버·오류 | 시작·중지, 순회 상태, 자동주문 허용 상태와 운영 오류입니다. 앱 응답 표시만으로 API 정상 여부를 보장하지 않습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 144 | QLabel | muted | 서버·감시 로그 / 서버·오류 | 아직 이 분류의 기록이 없습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 145 | QAbstractButton | qt_tableview_cornerbutton | 서버·감시 로그 / 서버·오류 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 146 | QToolButton | ScrollLeftButton | 서버·감시 로그 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 147 | QToolButton | ScrollRightButton | 서버·감시 로그 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 148 | QLabel | DailyTradeJournalPanel.title | 매매일지 | 매매일지 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 149 | QLabel | DailyTradeJournalPanel.mode_badge | 매매일지 | 모의투자 · DEMO | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 150 | QPushButton | DailyTradeJournalPanel.refresh_button | 매매일지 | 일지 새로고침 | 숨김/활성/탭 이동 시 표시 | refresh_all (normal window) / request_refresh (portfolio, order history) / synchronous refresh(force=True) (journal) | 유지 |
| 151 | QLabel | muted | 매매일지 | 주문일 기준: 한국은 서울, 미국은 뉴욕 날짜 · 정확한 체결일별 집계가 아닙니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 152 | QLabel | muted | 매매일지 | 매도 실현 수익률 = 확인된 실현손익 ÷ 해당 매수원가 · 모델 매도는 지정 매수분, 일반 매도는 FIFO · 계좌 전체 일수익률 아님 · 수수료·세금 제외 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 153 | QTabWidget | DailyTradeJournalPanel.market_tabs | 매매일지 | 국내 · KRW | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 154 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 주문일 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 155 | QDateEdit | (익명) | 매매일지 / 해외 (미국) · USD | 2026-09-24 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 156 | QLineEdit | qt_spinbox_lineedit | 매매일지 / 해외 (미국) · USD | 2026-09-24 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 157 | QAbstractButton | qt_tableview_cornerbutton | 공통 상단/하단 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 158 | QToolButton | qt_calendar_prevmonth | 공통 상단/하단 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 159 | QToolButton | qt_calendar_nextmonth | 공통 상단/하단 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 160 | QToolButton | qt_calendar_monthbutton | 공통 상단/하단 | 9월 | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 161 | QToolButton | qt_calendar_yearbutton | 공통 상단/하단 | 2026 | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 162 | QSpinBox | qt_calendar_yearedit | 공통 상단/하단 | 2026 | 숨김/활성/계속 숨김 |  | 유지 (Qt 내부) |
| 163 | QLineEdit | qt_spinbox_lineedit | 공통 상단/하단 | 2026 | 숨김/활성/계속 숨김 |  | 유지 (Qt 내부) |
| 164 | QPushButton | (익명) | 매매일지 / 해외 (미국) · USD | 오늘 | 숨김/활성/탭 이동 시 표시 | market-specific select_today | 유지 |
| 165 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 뉴욕 날짜 · 서머타임 반영 · 아래로 스크롤하면 상세내역 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 166 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 체결 매수금액 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 167 | QLabel | metric | 매매일지 / 해외 (미국) · USD | 0.00 USD | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 168 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 체결 매도금액 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 169 | QLabel | metric | 매매일지 / 해외 (미국) · USD | 0.00 USD | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 170 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 매도 실현손익 (세전) | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 171 | QLabel | metric | 매매일지 / 해외 (미국) · USD | 매도 체결 없음 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 172 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 매도 실현 수익률 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 173 | QLabel | metric | 매매일지 / 해외 (미국) · USD | — | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 174 | QLabel | muted | 매매일지 / 해외 (미국) · USD | 매수 체결 0건 · 매도 체결 0건  \|  매도 손익 확인 0건 / 미확인 0건<br>접수·확인 대기 0건 · 거절 0건 · 취소/기타 0건 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 175 | QAbstractButton | qt_tableview_cornerbutton | 매매일지 / 해외 (미국) · USD |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 176 | QLabel | muted | 매매일지 / 국내 · KRW | 주문일 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 177 | QDateEdit | (익명) | 매매일지 / 국내 · KRW | 2026-09-24 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 178 | QLineEdit | qt_spinbox_lineedit | 매매일지 / 국내 · KRW | 2026-09-24 | 숨김/활성/탭 이동 시 표시 |  | 유지 (Qt 내부) |
| 179 | QAbstractButton | qt_tableview_cornerbutton | 공통 상단/하단 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 180 | QToolButton | qt_calendar_prevmonth | 공통 상단/하단 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 181 | QToolButton | qt_calendar_nextmonth | 공통 상단/하단 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 182 | QToolButton | qt_calendar_monthbutton | 공통 상단/하단 | 9월 | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 183 | QToolButton | qt_calendar_yearbutton | 공통 상단/하단 | 2026 | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 184 | QSpinBox | qt_calendar_yearedit | 공통 상단/하단 | 2026 | 숨김/활성/계속 숨김 |  | 유지 (Qt 내부) |
| 185 | QLineEdit | qt_spinbox_lineedit | 공통 상단/하단 | 2026 | 숨김/활성/계속 숨김 |  | 유지 (Qt 내부) |
| 186 | QPushButton | (익명) | 매매일지 / 국내 · KRW | 오늘 | 숨김/활성/탭 이동 시 표시 | market-specific select_today | 유지 |
| 187 | QLabel | muted | 매매일지 / 국내 · KRW | 서울 날짜 · 아래로 스크롤하면 상세내역 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 188 | QLabel | muted | 매매일지 / 국내 · KRW | 체결 매수금액 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 189 | QLabel | metric | 매매일지 / 국내 · KRW | 0 KRW | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 190 | QLabel | muted | 매매일지 / 국내 · KRW | 체결 매도금액 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 191 | QLabel | metric | 매매일지 / 국내 · KRW | 0 KRW | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 192 | QLabel | muted | 매매일지 / 국내 · KRW | 매도 실현손익 (세전) | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 193 | QLabel | metric | 매매일지 / 국내 · KRW | 매도 체결 없음 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 194 | QLabel | muted | 매매일지 / 국내 · KRW | 매도 실현 수익률 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 195 | QLabel | metric | 매매일지 / 국내 · KRW | — | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 196 | QLabel | muted | 매매일지 / 국내 · KRW | 매수 체결 0건 · 매도 체결 0건  \|  매도 손익 확인 0건 / 미확인 0건<br>접수·확인 대기 0건 · 거절 0건 · 취소/기타 0건 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 197 | QAbstractButton | qt_tableview_cornerbutton | 매매일지 / 국내 · KRW |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 198 | QToolButton | ScrollLeftButton | 매매일지 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 199 | QToolButton | ScrollRightButton | 매매일지 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 200 | QLabel | DailyTradeJournalPanel.warning | 매매일지 |  | 숨김/활성/계속 숨김 |  | 유지·중복 문구 축약 검토 |
| 201 | QLabel | muted | 매매일지 | 앱이 기록한 주문만 포함 · 접수/미체결 금액은 제외 · 다른 앱의 거래·입출금·평가손익은 미포함 · KRW와 USD는 합산하지 않습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 202 | QLabel | OrderHistoryPanel.heading | 실제 주문·체결 | 실제 주문·체결 내역 · 모의투자 계좌 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 203 | QPushButton | OrderHistoryPanel.refresh_button | 실제 주문·체결 | 체결가 다시 확인 | 숨김/활성/탭 이동 시 표시 | refresh_all (normal window) / request_refresh (portfolio, order history) / synchronous refresh(force=True) (journal) | 유지 |
| 204 | QLabel | muted | 실제 주문·체결 | 이 앱이 기록한 자동·수동 주문을 표시합니다. HOLD·신호는 제외하며 ‘접수’는 체결이 아닙니다. 체결가는 증권사 확인값만 사용합니다. 기록 시작 전이나 다른 앱의 주문은 포함되지 않습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 205 | QLabel | OrderHistoryPanel.summary | 실제 주문·체결 | 체결 확인  매수 0건 / 매도 0건    ·    접수·확인 대기 0건 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 206 | QLabel | OrderHistoryPanel.performance_label | 실제 주문·체결 | 한국 · 매도 기록 없음  /  미국 · 매도 기록 없음 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 207 | QLabel | OrderHistoryPanel.recovery_label | 실제 주문·체결 | 체결가 확인 전 · 현재가나 주문가로 체결가를 대신 채우지 않습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 208 | QTabWidget | OrderHistoryPanel.tabs | 실제 주문·체결 | 주문 장부 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 209 | QLabel | muted | 실제 주문·체결 / 주문 처리 로그 | 주문 전송 의도·접수·체결 확인의 상태 변화 기록입니다. ‘전송 의도’나 ‘접수 여부 확인 필요’를 체결로 해석하지 마세요. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 210 | QLabel | muted | 실제 주문·체결 / 주문 처리 로그 | 아직 이 분류의 기록이 없습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 211 | QAbstractButton | qt_tableview_cornerbutton | 실제 주문·체결 / 주문 처리 로그 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 212 | QComboBox | OrderHistoryPanel.filter | 실제 주문·체결 / 주문 장부 | 전체 주문 | 숨김/활성/탭 이동 시 표시 |  | 유지 |
| 213 | QLabel | OrderHistoryPanel.count_label | 실제 주문·체결 / 주문 장부 | 아직 자동주문 기록이 없습니다. 신호만 수신해도 여기는 늘어나지 않습니다. | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 214 | QAbstractButton | qt_tableview_cornerbutton | 실제 주문·체결 / 주문 장부 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 215 | QToolButton | ScrollLeftButton | 실제 주문·체결 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 216 | QToolButton | ScrollRightButton | 실제 주문·체결 |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 217 | QLabel | muted | 실제 주문·체결 | 전체 앱 주문 장부의 주문순서 FIFO · 수수료·세금 제외 · 계좌 전체 수익률 아님 · 매수 기록이 없으면 원가/손익 미확인 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 218 | QLabel | PortfolioPanel.heading | 보유종목 | 현재 보유종목 · 모의투자 계좌 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 219 | QPushButton | PortfolioPanel.refresh_button | 보유종목 | 보유종목 새로고침 | 표시/활성/탭 이동 시 표시 | refresh_all (normal window) / request_refresh (portfolio, order history) / synchronous refresh(force=True) (journal) | 유지 |
| 220 | QTabWidget | PortfolioPanel.market_tabs | 보유종목 | 한국 · KRW | 표시/활성/탭 이동 시 표시 |  | 유지 |
| 221 | QLabel | section | 보유종목 / 미국 · USD | 미국 · USD | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 222 | QLabel | portfolioStatus | 보유종목 / 미국 · USD | 미확인 · 조회 전 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 223 | QLabel | portfolioHoldings | 보유종목 / 미국 · USD | 보유종목 미확인 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 224 | QLabel | muted | 보유종목 / 미국 · USD | 총 평가금액 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 225 | QLabel | portfolioValue | 보유종목 / 미국 · USD | 미확인 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 226 | QLabel | muted | 보유종목 / 미국 · USD | 평가손익 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 227 | QLabel | portfolioValue | 보유종목 / 미국 · USD | 미확인 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 228 | QLabel | muted | 보유종목 / 미국 · USD | 예수금 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 229 | QLabel | portfolioValue | 보유종목 / 미국 · USD | 미확인 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 230 | QLabel | muted | 보유종목 / 미국 · USD | 주문가능금액 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 231 | QLabel | portfolioValue | 보유종목 / 미국 · USD | 미확인 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 232 | QLabel | muted | 보유종목 / 미국 · USD | 잔고 기준 조회 전 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 233 | QLabel | muted | 보유종목 / 미국 · USD | 미국: 미확인 · 조회 전 | 숨김/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 234 | QAbstractButton | qt_tableview_cornerbutton | 보유종목 / 미국 · USD |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 235 | QLabel | section | 보유종목 / 한국 · KRW | 한국 · KRW | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 236 | QLabel | portfolioStatus | 보유종목 / 한국 · KRW | 미확인 · 조회 전 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 237 | QLabel | portfolioHoldings | 보유종목 / 한국 · KRW | 보유종목 미확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 238 | QLabel | muted | 보유종목 / 한국 · KRW | 총 평가금액 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 239 | QLabel | portfolioValue | 보유종목 / 한국 · KRW | 미확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 240 | QLabel | muted | 보유종목 / 한국 · KRW | 평가손익 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 241 | QLabel | portfolioValue | 보유종목 / 한국 · KRW | 미확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 242 | QLabel | muted | 보유종목 / 한국 · KRW | 예수금 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 243 | QLabel | portfolioValue | 보유종목 / 한국 · KRW | 미확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 244 | QLabel | muted | 보유종목 / 한국 · KRW | 주문가능금액 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 245 | QLabel | portfolioValue | 보유종목 / 한국 · KRW | 미확인 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 246 | QLabel | muted | 보유종목 / 한국 · KRW | 잔고 기준 조회 전 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 247 | QLabel | muted | 보유종목 / 한국 · KRW | 한국: 미확인 · 조회 전 | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 248 | QAbstractButton | qt_tableview_cornerbutton | 보유종목 / 한국 · KRW |  | 숨김/활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 249 | QToolButton | ScrollLeftButton | 보유종목 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 250 | QToolButton | ScrollRightButton | 보유종목 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 251 | QLabel | muted | 보유종목 | 예수금 ≠ 주문가능금액 ≠ 주문당 상한 · KRW와 USD는 합산하지 않습니다. | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 252 | QToolButton | ScrollLeftButton | 공통 상단/하단 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 253 | QToolButton | ScrollRightButton | 공통 상단/하단 |  | 숨김/비활성/계속 숨김 | Qt internal or not independently resolved | 유지 (Qt 내부) |
| 254 | QLabel | V00Window.message | 공통 상단/하단 | ver 0.0 · 감시·자동주문 OFF · 거래 설정에서 10% 비중과 연결을 확인하세요. | 표시/활성/탭 이동 시 표시 |  | 유지·중복 문구 축약 검토 |
| 255 | QLabel | (익명) | 공통 상단/하단 |  | 숨김/활성/계속 숨김 |  | 유지·중복 문구 축약 검토 |
| 256 | QCheckBox | V00Window.builtin_lstm | 공통 상단/하단 | 내장 LSTM30 매수 신호기 사용 · 다른 신호기와 동시 연결 가능 | 숨김/활성/계속 숨김 | toggled/currentChanged setting; see owning class | 호환 API 정리 후 제거 후보 |

---

[종합 보고서로 돌아가기](README.md)
