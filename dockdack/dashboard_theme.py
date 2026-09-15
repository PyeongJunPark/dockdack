"""Windows-friendly, restrained dashboard styling (no custom title bar)."""

DASHBOARD_STYLE = """
QDialog#watchDashboard { background: #0d1421; }
QWidget#signalConnectionPanel, QScrollArea { background: #111c2e; }
QWidget { font-family: 'Segoe UI', 'Malgun Gothic'; font-size: 13px; }
QLabel#heading { color: #f2f6ff; font-size: 27px; font-weight: 700; }
QLabel#eyebrow { color: #8294af; font-size: 10px; letter-spacing: 1px; }
QLabel#section { color: #e5edfa; font-size: 16px; font-weight: 600; }
QLabel#muted { color: #94a6c1; }
QFrame#controlBar { background: #152033; border: 1px solid #263650; border-radius: 12px; }
QFrame#card { background: #172338; border: 1px solid #2a3a55; border-radius: 12px; }
QFrame#portfolioMetric { background: #111e31; border: 1px solid #2b3d57; border-radius: 9px; }
QLabel#portfolioValue { font-size: 20px; font-weight: 600; color: #edf3ff; }
QLabel#portfolioStatus { color: #83dfc4; padding: 4px 8px; }
QLabel#portfolioHoldings { color: #c5d6f0; background: #263952; border-radius: 6px; padding: 4px 9px; }
QLabel#metric { font-size: 21px; font-weight: 600; color: #edf3ff; }
QPushButton { background: #1c2b42; border: 1px solid #334761; border-radius: 7px; padding: 8px 11px; }
QPushButton:hover { background: #293d59; border-color: #718aa9; }
QPushButton:focus { border: 1px solid #9ab9ff; }
QPushButton#primary { background: #253e5c; color: #c3d9ff; border: 1px solid #416086; }
QPushButton#primary:hover { background: #345276; }
QPushButton#armOrders { background: #65dfc3; color: #062c27; border: 1px solid #65dfc3; font-weight: 700; }
QPushButton#armOrders:hover { background: #9aecd7; border-color: #b1f4e2; }
QPushButton#armOrders:pressed { background: #41bda3; }
QPushButton#disarmOrders { color: #f4c2c9; border-color: #6b4553; background: #302735; }
QPushButton#disarmOrders:hover { background: #483241; border-color: #b77e8b; }
QPushButton:disabled, QPushButton#primary:disabled, QPushButton#armOrders:disabled,
QPushButton#disarmOrders:disabled { background: #192437; border-color: #28354a; color: #6c7d96; }
QPushButton#linkButton { background: transparent; color: #8cb8ff; border: none; padding: 4px 6px; }
QPushButton#linkButton:hover { color: #c7ddff; background: #192a42; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #101b2d; border-color: #344963;
    padding: 7px 9px; border-radius: 7px; min-height: 20px; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #7789a4; background: #162133; border-color: #29364a; }
QLineEdit#connectionPath { font-family: 'Cascadia Mono', 'Consolas'; font-size: 12px; color: #adc9ef; }
QTabWidget::pane { background: #111c2e; border: 1px solid #2b3b54; border-radius: 10px; top: -1px; }
QTabBar::tab { background: transparent; color: #94a7c2; border: none; border-bottom: 2px solid transparent;
    padding: 10px 14px; margin: 0 2px 5px 0; }
QTabBar::tab:hover { background: #1b2b43; color: #d5e5ff; }
QTabBar::tab:selected { background: #21354c; color: #83e5ce; border-bottom: 2px solid #68d9bd; }
QTabWidget#workspaceTabs > QTabBar::tab { padding: 11px 18px; font-weight: 600; }
QTableWidget { background: #111c2e; alternate-background-color: #152238; border: none; outline: none;
    selection-background-color: #284663; selection-color: #ffffff; }
QHeaderView::section { background: #1b2b43; color: #a9bdd8; font-size: 12px; font-weight: 600;
    padding: 9px 7px; border-bottom: 1px solid #314662; }
QTableWidget::item { padding: 7px; border-bottom: 1px solid #1b2a40; }
QTableWidget::item:selected { background: #264562; color: #ffffff; }
QScrollBar:vertical { background: #111b2c; width: 10px; margin: 0; border: none; }
QScrollBar:horizontal { background: #111b2c; height: 10px; margin: 0; border: none; }
QScrollBar::handle:vertical { background: #3b4d68; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:horizontal { background: #3b4d68; border-radius: 4px; min-width: 30px; }
QScrollBar::handle:hover { background: #60799b; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QSplitter::handle { background: #26354b; width: 3px; }
QProgressBar#sweepProgress { border: 1px solid #486866; border-radius: 5px; background: #152e2d; color: #f2fff9; font-weight: 700; text-align: center; }
QProgressBar#sweepProgress::chunk { background: #216953; border-radius: 4px; }
QLabel#connectionMode { padding: 7px 10px; border-radius: 7px; background: #202e44; color: #adbed6; }
QLabel#connectionMode[tone="active"] { background: #19382f; color: #87e4c5; }
QLabel#connectionMode[tone="error"] { background: #403222; color: #ffda91; }
"""
