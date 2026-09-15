"""Presentation-only window controls; never start or stop trading workers."""

from PySide6.QtCore import QEvent, QObject, Qt, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QPushButton


class WindowControls(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._return_maximized = False
        self._return_geometry = None
        # QDialog does not request minimize/maximize controls by default.
        # Configure these before show(), preserving ownership and modality.
        window.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        window.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        window.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        window.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self.fullscreen_button = self._button("전체화면 · F11", "F11: 전체화면 전환 · Esc: 전체화면 해제. 감시·자동매매는 계속됩니다.", self.toggle_fullscreen)
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), window)
        self.escape_shortcut = QShortcut(QKeySequence("Esc"), window)
        for shortcut in (self.fullscreen_shortcut, self.escape_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(False)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        # Consume Escape even in windowed mode: QDialog's default reject()
        # would otherwise unexpectedly stop the running watchlist monitor.
        self.escape_shortcut.activated.connect(self.leave_fullscreen)
        window.installEventFilter(self)
        self._sync()

    def _button(self, text, tooltip, callback):
        button = QPushButton(text, self.window)
        button.setAutoDefault(False)
        button.setDefault(False)
        button.setToolTip(tooltip)
        button.setAccessibleName(text)
        button.clicked.connect(callback)
        return button

    def add_to(self, layout):
        # Minimize/maximize remain on the native titlebar, without duplicates.
        layout.addWidget(self.fullscreen_button)

    @Slot()
    def toggle_fullscreen(self):
        if self.window.isFullScreen():
            self.leave_fullscreen()
            return
        self._return_maximized = self.window.isMaximized()
        self._return_geometry = self.window.normalGeometry() if self._return_maximized else self.window.geometry()
        self.window.showFullScreen()
        self._sync()

    @Slot()
    def leave_fullscreen(self):
        if self.window.isFullScreen():
            if self._return_maximized:
                self.window.showMaximized()
            else:
                self.window.showNormal()
                if self._return_geometry is not None and self._return_geometry.isValid():
                    self.window.setGeometry(self._return_geometry)
        self._sync()

    def _sync(self):
        fullscreen = self.window.isFullScreen()
        self.fullscreen_button.setText("창모드 · F11" if fullscreen else "전체화면 · F11")
        self.fullscreen_button.setAccessibleName(self.fullscreen_button.text())

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Type.WindowStateChange:
            self._sync()
        return super().eventFilter(watched, event)
