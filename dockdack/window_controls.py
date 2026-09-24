"""Presentation-only window controls; never start or stop trading workers."""

from PySide6.QtCore import QEvent, QObject, Qt, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QPushButton


class WindowControls(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._return_geometry = None
        # QDialog does not request minimize/maximize controls by default.
        # Configure these before show(), preserving ownership and modality.
        window.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        window.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        window.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)
        window.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self.fullscreen_button = self._button("전체화면 · F11", "F11: 전체화면 전환 · Esc: 전체화면/최대화 해제 후 이전 크기의 창으로 복원. 감시·자동매매는 계속됩니다.", self.toggle_fullscreen)
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
        self._return_geometry = self.window.normalGeometry()
        self.window.showFullScreen()
        self._sync()

    @Slot()
    def leave_fullscreen(self):
        fullscreen = self.window.isFullScreen()
        if fullscreen or self.window.isMaximized():
            # "Windowed" must leave a resizable normal window, not return to
            # maximized and appear to have done nothing. Native maximize is
            # still available on the titlebar; Escape also restores it.
            geometry = self._return_geometry if fullscreen else self.window.normalGeometry()
            self.window.showNormal()
            if geometry is not None and geometry.isValid():
                self.window.setGeometry(geometry)
            self._return_geometry = None
        self._sync()

    def _sync(self):
        fullscreen = self.window.isFullScreen()
        self.fullscreen_button.setText("창모드 · F11" if fullscreen else "전체화면 · F11")
        self.fullscreen_button.setAccessibleName(self.fullscreen_button.text())

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Type.WindowStateChange:
            # Also handle fullscreen entered outside our button/shortcut.
            # A minimize/taskbar restore must not overwrite the saved size.
            if (self.window.isFullScreen()
                    and not event.oldState() & Qt.WindowState.WindowFullScreen):
                geometry = self.window.normalGeometry()
                if geometry.isValid():
                    self._return_geometry = geometry
            self._sync()
        return super().eventFilter(watched, event)
