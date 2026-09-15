"""Cheap, explicit signal pipeline status; this widget performs no I/O.

The owner feeds ``set_status`` snapshots from completed background work and
SignalFileReader.status(). File presence is never presented as a connection.
"""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from dockdack.gui import label


def _time(value):
    if not value:
        return "—"
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%m/%d %H:%M:%S")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "시각 확인 필요"


def _age(value, now):
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return (now - parsed).total_seconds()
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def connection_view(status: dict) -> dict:
    """Pure presentation of observed state, not guessed health or market state.

    Required booleans are configured/active/monitoring/orders_enabled/pending_arm.
    Additional keys: producer ('random-demo', 'external-file', 'manual'), source_id,
    input_path/output_path/update_path, reader status() fields, last_export_at
    (whole-watchlist file only), last_update_at (per-symbol update file only), market_summary,
    block_reason, configuration_pending, inspection. ``now`` may be supplied for
    deterministic freshness rendering; omitting it uses the current UTC time.
    """
    configured = bool(status.get("configured"))
    active = bool(status.get("active")) and bool(status.get("monitoring"))
    producer = status.get("producer", "manual")
    source = str(status.get("source_id") or "미설정")
    is_demo = producer == "random-demo"
    if is_demo:
        source_text = f"내장 모의 신호기 · {source} (외부 프로그램 연결 아님)"
    elif producer == "external-file":
        source_text = f"외부 코드 · JSON 파일 교환 · source_id: {source}"
    else:
        source_text = "수동 트리거 모드 · 외부 신호 실행 안 함"
    mode = "감시 중 · 신호 수신기 활성" if active and status.get("monitoring") else "설정됨 · 수신기 비활성" if configured else "연결 미설정"
    if status.get("configuration_pending"):
        mode += " · 변경 설정 적용 전"
    last_export, last_update = status.get("last_export_at"), status.get("last_update_at")
    update_text = f"저장 성공 {_time(last_update)}" if last_update else "저장 성공 기록 없음"
    full_text = f"저장 성공 {_time(last_export)}" if last_export else "저장 성공 기록 없음"
    export = (f"종목별 즉시 파일 · {update_text}\n전체 차트 파일 (순회 완료 후) · {full_text}\n"
              "저장 완료는 외부 코드가 읽었다는 확인이 아닙니다.")

    state = status.get("reader_state", "waiting")
    error = str(status.get("reader_error") or "")
    last_read, accepted = status.get("last_read_at"), status.get("last_accepted_at")
    if not active:
        incoming = "수신 중지 · 아래 경로/설정은 자동으로 적용되지 않을 수 있습니다."
    elif error or state == "error":
        incoming = f"수신 오류 · {error or '입력 파일을 확인하세요.'}"
    elif state == "missing":
        incoming = "입력 파일 없음 · 첫 발행 대기 / 경로 확인 필요"
    elif state == "unchanged":
        incoming = "새 파일 대기 · 이전 파일을 중복 처리하지 않습니다."
    elif last_read:
        incoming = "JSON 읽기 성공 · 파일 발행 관측 (외부 프로세스 생존 여부는 미확인)" if not is_demo else "내장 테스트 파일 읽기 성공"
    else:
        incoming = "첫 신호 파일 수신 대기 · 연결 성공은 아직 미확인"
    incoming += f" · 최근 읽기 {_time(last_read)}"
    counts = status.get("received_counts") or {}
    if accepted:
        accepted_text = (f"최근 접수 {_time(accepted)} · 매수/매도 후보 {counts.get('queued', 0)} · "
                         f"HOLD {counts.get('hold', 0)} · 중복 {counts.get('duplicates', 0)} · 만료 {counts.get('expired', 0)}")
    else:
        accepted_text = "접수 확인 없음 · 형식 검사와 실제 신호 접수는 별개입니다."
    now = status.get("now") or datetime.now(timezone.utc)
    age = _age(accepted, now)
    if active and age is not None and age > 300:
        accepted_text += " · 최근 접수 5분 경과 (현재 유효 신호 여부는 개별 검사)"
    accepted_text += "\n후보/접수는 주문·체결이 아닙니다. HOLD는 매매하지 않음입니다."
    if status.get("orders_enabled"):
        gate = "ON · 유효 신호 + 정규장 + 잔고·수량·금액 검사 통과 시에만 모의주문"
    elif status.get("pending_arm"):
        gate = "OFF · 전체 조회 검증 후 ON 예약됨 · OFF 버튼으로 예약 취소 가능"
    else:
        gate = "OFF · 새 자동주문 차단 / 감시와 신호 수신은 별도 동작"
    market = str(status.get("market_summary") or "시장 개장 상태 미확인 · 주문 직전 별도 검사")
    block = str(status.get("block_reason") or "")
    gate += f"\n{market}"
    if block:
        gate += f" · 최근 차단/대기 사유: {block}"
    inspection = status.get("inspection") or {}
    if inspection:
        inspection_text = f"파일 검사 {_time(inspection.get('checked_at'))} · {inspection.get('summary', '결과 없음')}"
        inspection_counts = inspection.get("counts") or {}
        if inspection.get("state") == "format_ok":
            inspection_text += (f"\nBUY {inspection_counts.get('buy', 0)} / SELL {inspection_counts.get('sell', 0)} / "
                                f"HOLD {inspection_counts.get('hold', 0)} · 이 중 만료 {inspection_counts.get('expired', 0)}")
    else:
        inspection_text = "‘파일 검사’는 JSON 형식·설정 상한만 확인합니다. 신호 접수·규칙 생성·주문은 하지 않습니다."
    tone = "error" if error or state == "error" else "active" if active and status.get("monitoring") else "idle"
    return {"source": source_text, "mode": mode, "tone": tone, "export": export, "incoming": incoming,
            "accepted": accepted_text, "gate": gate, "inspection": inspection_text,
            "input_path": str(status.get("input_path") or ""), "output_path": str(status.get("output_path") or ""),
            "update_path": str(status.get("update_path") or ""),
            "inspect_enabled": configured and bool(status.get("input_path")) and not status.get("inspection_busy", False),
            "folder_enabled": bool(status.get("update_path") or status.get("output_path") or status.get("input_path"))}


class SignalConnectionPanel(QWidget):
    request_settings = Signal()
    request_inspect = Signal()
    request_folder = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        header = QHBoxLayout()
        header.addWidget(label("자동매매 신호 연결", "section"), 1)
        self.settings_button = QPushButton("연결 설정")
        self.inspect_button = QPushButton("파일 검사 (주문 없음)")
        self.folder_button = QPushButton("연결 폴더")
        for button, request in ((self.settings_button, self.request_settings),
                                (self.inspect_button, self.request_inspect), (self.folder_button, self.request_folder)):
            button.setAutoDefault(False)
            button.clicked.connect(request.emit)
            header.addWidget(button)
        self.settings_button.setToolTip("신호기 종류, source_id, 입력·출력 경로와 주문 상한 설정으로 이동합니다.")
        self.inspect_button.setToolTip("파일 형식과 설정 상한만 확인합니다. 신호를 접수하거나 주문 규칙을 만들지 않습니다.")
        self.folder_button.setToolTip("연결 파일이 놓이는 로컬 폴더를 엽니다. 외부 코드를 실행하지 않습니다.")
        layout.addLayout(header)
        self.source_label = label("", "section", wrap=True)
        self.mode_label = label("", "connectionMode", wrap=True)
        layout.addWidget(self.source_label)
        layout.addWidget(self.mode_label)

        paths = QGridLayout()
        paths.setVerticalSpacing(5)
        self.update_path = QLineEdit()
        self.output_path = QLineEdit()
        self.input_path = QLineEdit()
        for row, (title, field) in enumerate((("종목별 즉시 출력 → 계속 확인할 폴더", self.update_path),
                                             ("전체 차트 출력 → 순회 완료 후 저장", self.output_path),
                                             ("신호 입력 ← 신호기가 쓸 파일", self.input_path))):
            field.setReadOnly(True)
            field.setObjectName("connectionPath")
            field.setPlaceholderText("연결 설정에서 경로를 지정하세요.")
            paths.addWidget(label(title, "muted"), row, 0)
            paths.addWidget(field, row, 1)
        paths.setColumnStretch(1, 1)
        layout.addLayout(paths)

        self.pipeline_card = QFrame()
        self.pipeline_card.setObjectName("card")
        pipeline = QGridLayout(self.pipeline_card)
        pipeline.setContentsMargins(14, 12, 14, 12)
        pipeline.setVerticalSpacing(12)
        self.step_labels = {}
        for row, (key, title) in enumerate((("export", "1  차트 전달"), ("incoming", "2  신호 입력"),
                                           ("accepted", "3  검증·수신"), ("gate", "4  주문 허용"))):
            heading = label(title, "section")
            heading.setAlignment(Qt.AlignmentFlag.AlignTop)
            detail = label("", wrap=True)
            detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            pipeline.addWidget(heading, row, 0)
            pipeline.addWidget(detail, row, 1)
            self.step_labels[key] = detail
        pipeline.setColumnStretch(1, 1)
        layout.addWidget(self.pipeline_card)
        self.inspection_label = label("", "muted", wrap=True)
        self.inspection_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.inspection_label)
        layout.addWidget(label("실시간 연결은 종목별 출력 폴더를 확인하세요. 전체 차트 파일은 전체 순회가 끝난 뒤 갱신됩니다.\n"
                               "고유 signal_id와 읽은 차트의 export_id로 신호 JSON을 원자적으로 교체하세요.\n"
                               "파일 존재만으로 연결 성공을 판정하지 않으며, 실제 매수·매도는 ‘실제 주문·체결’ 탭에서 확인합니다.", "muted", wrap=True))
        layout.addStretch(1)
        self._view = None
        self.set_status({})

    def set_status(self, status: dict):
        view = connection_view(status)
        if view == self._view:
            return False
        previous = self._view or {}
        self._view = view
        labels = {"source": self.source_label, "mode": self.mode_label, "inspection": self.inspection_label,
                  **self.step_labels}
        for key, field in labels.items():
            if previous.get(key) != view[key]:
                field.setText(view[key])
        for key, field in (("input_path", self.input_path), ("output_path", self.output_path),
                           ("update_path", self.update_path)):
            if previous.get(key) != view[key]:
                field.setText(view[key])
                field.setToolTip(view[key])
                field.setCursorPosition(0)
        self.inspect_button.setEnabled(view["inspect_enabled"])
        self.folder_button.setEnabled(view["folder_enabled"])
        if previous.get("tone") != view["tone"]:
            self.mode_label.setProperty("tone", view["tone"])
            self.mode_label.style().unpolish(self.mode_label)
            self.mode_label.style().polish(self.mode_label)
        return True
