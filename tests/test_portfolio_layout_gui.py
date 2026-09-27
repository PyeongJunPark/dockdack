from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication
    from dockdack.portfolio_gui import PortfolioPanel

from dockdack.models import Market


@unittest.skipUnless(HAS_QT, "Install the gui extra")
class PortfolioColumnLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = PortfolioPanel()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def _show_at_width(self, width: int) -> None:
        self.panel.resize(width, 720)
        self.panel.show()
        for _ in range(4):
            self.app.processEvents()

    def test_1280_window_retains_readable_widths_and_horizontal_scroll(self):
        self._show_at_width(1280)
        for index, market in enumerate(Market):
            self.panel.market_tabs.setCurrentIndex(index)
            for _ in range(3):
                self.app.processEvents()
            table = self.panel.tables[market]
            self.assertGreaterEqual(table.columnWidth(1), 220)
            self.assertGreaterEqual(table.columnWidth(12), 195)
            self.assertGreater(table.horizontalScrollBar().maximum(), 0)

    def test_1920_window_fills_table_viewport_in_both_markets(self):
        self._show_at_width(1920)
        for index, market in enumerate(Market):
            self.panel.market_tabs.setCurrentIndex(index)
            for _ in range(3):
                self.app.processEvents()
            table = self.panel.tables[market]
            visible_width = sum(table.columnWidth(column) for column in range(table.columnCount())
                                if not table.isColumnHidden(column))
            self.assertLessEqual(abs(visible_width - table.viewport().width()), 2)
            self.assertEqual(table.horizontalScrollBar().maximum(), 0)
            self.assertGreater(table.columnWidth(1), 220)

    def test_resizing_back_to_1280_restores_scrollable_base_widths(self):
        self._show_at_width(1920)
        table = self.panel.table
        self.assertGreater(table.columnWidth(1), 220)
        self.panel.resize(1280, 720)
        for _ in range(4):
            self.app.processEvents()
        self.assertEqual(table.columnWidth(1), 220)
        self.assertEqual(table.columnWidth(12), 195)
        self.assertGreater(table.horizontalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
