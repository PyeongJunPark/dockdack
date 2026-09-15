"""Explicit mode controls; selection is never automatic-order activation."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton
from dockdack.models import TradingMode


def environment_name(mode):
    return '실전투자' if TradingMode(mode) is TradingMode.REAL else '모의투자'


def confirm_environment(parent, mode):
    real = TradingMode(mode) is TradingMode.REAL
    title = '실전투자 전환 경고' if real else '모의투자로 전환'
    text = ('실전에서는 실제 계좌의 돈으로 주문되며 원금 손실이 발생할 수 있습니다.\n'
            '실전 API 키와 DOCKDACK_ALLOW_LIVE_ORDERS=true 설정이 필요합니다.\n'
            '내장 랜덤 모의 신호기는 실전에서 사용할 수 없습니다.\n\n' if real else '')
    text += ('현재 감시·자동주문을 중지하고 진행 중인 응답이 끝난 뒤 전환합니다.\n'
             '이미 접수된 주문은 취소되지 않습니다. 이전 환경의 미체결을 별도로 확인하세요.\n'
             '모의·실전 기록은 분리되고 전환 후 자동주문은 OFF입니다. 계속할까요?')
    return QMessageBox.warning(parent, title, text,
                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                              QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


class EnvironmentSelector(QFrame):
    requested = Signal(object)

    def __init__(self, mode=TradingMode.DEMO, parent=None):
        super().__init__(parent)
        self.setObjectName('environmentSelector')
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.buttons = {}
        for value in TradingMode:
            button = QPushButton(environment_name(value))
            button.setAutoDefault(False)
            button.clicked.connect(lambda checked=False, selected=value: self.requested.emit(selected))
            self.buttons[value] = button
            row.addWidget(button)
        self.badge = QLabel()
        row.addWidget(self.badge)
        self.apply(mode)

    def apply(self, mode, *, pending=False):
        mode = TradingMode(mode)
        for value, button in self.buttons.items():
            selected = value is mode
            button.setEnabled(not pending and not selected)
            color = '#ffad9d' if value is TradingMode.REAL else '#8ee4ce'
            button.setStyleSheet(f'color: {color}; border: 1px solid {color}; font-weight: 700;' if selected else '')
        self.badge.setText('전환 대기 · 주문 OFF' if pending else ('실전 · 실제 자금' if mode is TradingMode.REAL else '모의 · 가상 자금'))
        self.badge.setStyleSheet('color: #ffad9d; font-weight: 700;' if mode is TradingMode.REAL else 'color: #8ee4ce;')
