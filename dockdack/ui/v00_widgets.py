"""Small bounded GUI controls; never authenticate, query or submit orders."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter
from time import monotonic
from PySide6.QtWidgets import (QFrame, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar)


class ActivityProgressBar(QProgressBar):
    """Animate unknown-duration reads without inventing completed work.

    Qt hides busy-mode text on some Windows styles; draw that label explicitly.
    Elapsed wait is a heartbeat, not a claim that a broker response succeeded.
    """
    def set_activity(self, text, *, completed=0, total=0, waiting=False):
        self._activity_text = text
        self._activity_started = monotonic()
        self._activity_waiting = waiting
        self.setRange(0, 0 if waiting else max(1, total))
        if not waiting:
            self.setValue(min(completed, max(1, total)))
        self.setFormat(text)
        self.setAccessibleDescription(text)

    def refresh_wait(self, suffix=''):
        if not getattr(self, '_activity_waiting', False):
            return
        elapsed = max(0, int(monotonic() - self._activity_started))
        detail = suffix or f'응답 대기 {elapsed}초'
        if elapsed >= 120:
            detail = f'응답 지연 확인 필요 · {elapsed}초'
        self.setFormat(f'{self._activity_text} · {detail}')

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.minimum() == self.maximum() == 0 and self.isTextVisible():
            painter = QPainter(self)
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter, self.format())


class SourceList(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('추가 매수 신호기 · 각 source_id와 JSON 입력은 서로 달라야 합니다'))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['source_id', '신호 JSON 입력 경로'])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(145)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        add = self.add_button = QPushButton('신호기 추가')
        remove = self.remove_button = QPushButton('선택 신호기 제거')
        add.clicked.connect(self.add_row)
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()) if self.table.currentRow() >= 0 else None)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.limit_label = QLabel('추가 신호기 0 / 16개')
        layout.addWidget(self.limit_label)
        self.table.setAccessibleName('추가 외부 신호기 · 출처와 JSON 입력 경로')
        self.table.model().rowsInserted.connect(self._update_limit)
        self.table.model().rowsRemoved.connect(self._update_limit)
        self._update_limit()

    def _update_limit(self, *_):
        count = self.table.rowCount()
        self.add_button.setEnabled(count < 16)
        self.limit_label.setText(f'추가 신호기 {count} / 16개' + (' · 최대 수에 도달했습니다' if count >= 16 else ''))
        self.add_button.setToolTip('추가 신호기는 최대 16개입니다.' if count >= 16 else '서로 다른 출처와 파일 경로를 추가합니다.')

    def add_row(self, checked=False, *, source='', path=''):
        if self.table.rowCount() >= 16:
            self._update_limit()
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((source, path)):
            self.table.setItem(row, column, QTableWidgetItem(value))

    def sources(self):
        result = []
        for row in range(self.table.rowCount()):
            values = tuple(self.table.item(row, col).text().strip() if self.table.item(row, col) else '' for col in range(2))
            if any(values):
                if not all(values):
                    raise ValueError('추가 신호기의 source_id와 파일 경로를 모두 입력하세요.')
                result.append(values)
        return tuple(result)

    def raw_sources(self):
        return tuple(tuple(self.table.item(row, col).text().strip() if self.table.item(row, col) else ''
                           for col in range(2)) for row in range(self.table.rowCount()))


class OrderToast(QFrame):
    """One reusable nonmodal notification; no popup queue or focus stealing."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('orderToast')
        self.setStyleSheet('#orderToast {background:#203b4d;border:1px solid #69d7bc;border-radius:8px;}')
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QVBoxLayout(self)
        self.message = QLabel()
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.setFixedWidth(460)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.hide()

    def notify(self, lines):
        if not lines:
            return
        lines = tuple(lines)
        self.message.setText('매수·매도 주문 알림\n' + '\n'.join(str(line)[:240] for line in lines[-3:])
                             + (f'\n외 {len(lines)-3}건 · 전체 내용은 주문 로그' if len(lines) > 3 else ''))
        self.adjustSize()
        self.move(max(12, self.parentWidget().width()-self.width()-24), 92)
        self.show()
        self.raise_()
        self.timer.start(7000)
