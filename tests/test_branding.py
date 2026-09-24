import importlib.util
import os
from pathlib import Path
import struct
import tomllib
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtSvg import QSvgRenderer
    from dockdack.branding import ASSET_FOLDER, APP_NAME, app_icon, apply_branding
from dockdack.version import APP_VERSION, APP_RELEASE


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class BrandingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_vector_mark_and_windows_icon_render_at_small_and_large_sizes(self):
        self.assertTrue(QSvgRenderer(str(ASSET_FOLDER / "dockdack-mark.svg")).isValid())
        icon = app_icon()
        self.assertFalse(icon.isNull())
        for size in (16, 32, 48, 256):
            self.assertFalse(icon.pixmap(size, size).isNull())
        data = (ASSET_FOLDER / "dockdack.ico").read_bytes()
        self.assertEqual(struct.unpack_from("<HHH", data), (0, 1, 9))

    def test_app_identity_and_icon_are_shared(self):
        apply_branding(self.app)
        self.assertEqual(self.app.applicationDisplayName(), APP_NAME)
        self.assertEqual(self.app.applicationVersion(), APP_VERSION)
        self.assertFalse(self.app.windowIcon().isNull())

    def test_release_matches_package_and_lock_without_changing_model_names(self):
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
        package = next(item for item in locked["package"] if item["name"] == "dockdack")
        self.assertEqual(APP_VERSION, "0.1.0")
        self.assertEqual(APP_RELEASE, "0.1 (MK1)")
        self.assertEqual(project["project"]["version"], APP_VERSION)
        self.assertEqual(package["version"], APP_VERSION)


if __name__ == "__main__":
    unittest.main()
