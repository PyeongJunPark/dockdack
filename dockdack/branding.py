"""The DOCKDACK contact-hitter mark, shared by Qt and the Windows taskbar."""

from pathlib import Path
import sys

from PySide6.QtGui import QIcon

ASSET_FOLDER = Path(__file__).with_name("assets")
APP_NAME = "DOCKDACK"
TAGLINE = "짧고 정확하게, 똑딱."


def app_icon() -> QIcon:
    icon = QIcon(str(ASSET_FOLDER / "dockdack.ico"))
    return QIcon(str(ASSET_FOLDER / "dockdack-mark.svg")) if icon.isNull() else icon


def set_windows_app_id() -> None:
    """Use our own taskbar identity rather than grouping under Python."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DockDack.DemoTrading.Desktop")


def apply_branding(app) -> None:
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setWindowIcon(app_icon())
