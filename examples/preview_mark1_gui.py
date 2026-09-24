"""Render the actual mark1 GUI with clearly labelled fake data, no network.

This source-checkout-only visual QA helper deliberately reuses test fixtures.
It never loads account configuration, real weights, or an existing runtime.
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtTest import QTest
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QLabel

from dockdack.gui_service import Instrument
from dockdack.mark1_gui import Mark1WatchlistDialog
from dockdack.models import Market
from dockdack.watchlist import WatchItem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/mark1/gui-preview.png"))
    args = parser.parse_args(argv)
    test_root = Path(__file__).resolve().parents[1] / "tests"
    sys.path.insert(0, str(test_root))
    from test_mark1_gui import CalendarFakeService, NOW

    app = QApplication.instance() or QApplication([])
    # Qt's Windows offscreen backend may not discover system fonts by itself.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError("The preview requires the installed Korean Malgun Gothic font")
    app.setFont(QFont(families[0], 10))
    service = CalendarFakeService()
    predictor = SimpleNamespace(
        metadata={"market": "domestic", "buy_threshold": .5, "model_name": "FAKE MODEL / 화면 예시"},
        buy_threshold=.5,
        predict=Mock(return_value={"probability_success": .7, "predicts_success": True, "buy_threshold": .5}),
    )

    def position(stock):
        return {**{key: stock[key] for key in ("market", "symbol", "exchange", "currency")},
                "quantity": "0", "sellable_quantity": "0", "average_price": None, "fetched_at": NOW.isoformat()}

    def idle(window):
        deadline = time.monotonic() + 30
        quiet = 0
        while quiet < 3:
            app.processEvents()
            quiet = 0 if window.worker or window._inspection_worker else quiet + 1
            QTest.qWait(10)
            if time.monotonic() > deadline:
                raise RuntimeError("Fake GUI worker did not finish")

    with tempfile.TemporaryDirectory(prefix="mark1-preview-") as folder, \
            patch("requests.sessions.Session.request", side_effect=AssertionError("Preview network forbidden")) as network:
        window = Mark1WatchlistDialog(
            service, runtime_dir=folder, predictors={"domestic": predictor}, quantity=1,
            max_krw="1000", max_usd="1000", clock=lambda: NOW, position_provider=position,
            items=[WatchItem(Instrument(Market.DOMESTIC, "005930", "KRX"), "가상 시세 / 삼성전자 예시", 31)],
        )
        try:
            window.setStyleSheet(window.styleSheet() + f"\nQWidget {{ font-family: '{families[0]}'; }}")
            banner = QLabel("가상 데이터 미리보기 · FAKE MODEL · 예시 확률 70% · 실제 모델 성능/시세/계좌 아님 · 주문 전송 없음")
            banner.setWordWrap(True)
            banner.setStyleSheet("background: #592529; color: white; font-size: 16px; font-weight: 700; padding: 12px;")
            window.layout().insertWidget(0, banner)
            window.resize(1600, 1000)
            window.show()  # The forced offscreen Qt platform creates no visible native window.
            window.start_session()  # Fake service only; automatic orders remain OFF.
            idle(window)
            window.stop_monitoring()
            idle(window)
            window.workspace_tabs.setCurrentIndex(window.workspace_tabs.count() - 1)
            window._update_mark1_table()
            window.message.setText("화면 검증용 가상 데이터입니다. 네트워크 연결과 주문 전송은 하지 않았습니다.")
            app.processEvents()
            if window.engine.orders_enabled or service.submitted:
                raise AssertionError("Preview must not enable or submit orders")
            if window.mark1_model_table.item(0, 3).text() != "buy":
                raise AssertionError(f"Preview fake signal missing: {window.lstm_bridge.diagnostics}; {window.errors}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.output.resolve())):
                raise RuntimeError("Unable to save GUI preview")
            network.assert_not_called()
            print(args.output.resolve())
        finally:
            window.stop_monitoring()
            idle(window)
            window.shutdown()
            window.close()
            window.deleteLater()
            app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
