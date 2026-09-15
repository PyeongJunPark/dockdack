import importlib.util
import os
import struct
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtSvg import QSvgRenderer
    from dockdack.branding import ASSET_FOLDER, APP_NAME, app_icon, apply_branding


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
        self.assertFalse(self.app.windowIcon().isNull())


if __name__ == "__main__":
    unittest.main()
