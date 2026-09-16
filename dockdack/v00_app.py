"""Unified ver 0.0 desktop: optional LSTM plus multiple data-only signal feeds.

Always starts OFF. Models initialize lazily on the existing broker worker, not
the GUI thread. This replaces the old DEMO-only shortcut, not its ledger.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox, QLabel

from dockdack.branding import apply_branding, set_windows_app_id
from dockdack.environment_store import selected_mode
from dockdack.gui_service import Instrument, TradingService
from dockdack.models import Market, TradingMode
from dockdack.portfolio import PortfolioCache
from dockdack.signal_bridge import atomic_json
from dockdack.watch_gui import WatchlistDialog
from dockdack.watchlist import WatchStore


class DesktopModelBridge:
    """Worker-only model/position cache. No permission changes or order sends."""
    def __init__(self, window, *, predictors=None):
        self.window, self.predictors = window, predictors
        self.producer = None
        self.accounts = {}
        self.diagnostics = {}
        self.status = '내장 LSTM · 첫 장중 조회 시 모델 확인'
        self._load_error = ''

    def _publish_unavailable(self, chart):
        # Clear this source's previous actionable proposal without interfering
        # with independent feeds. No repeated native-library imports per stock.
        now = self.window.engine.clock()
        rows = [{**{key: stock[key] for key in ('market', 'symbol', 'exchange')},
                 'signal_id': uuid5(NAMESPACE_URL, f"lstm-unavailable:{chart['export_id']}:{stock['watch_id']}").hex,
                 'export_id': chart['export_id'], 'action': 'hold',
                 'generated_at': now.isoformat(), 'expires_at': (now + timedelta(seconds=120)).isoformat()}
                for stock in chart['stocks'] if stock.get('status') == 'ok']
        atomic_json(self.window.engine.external_reader.path, {'schema_version': 1, 'source_id': 'lstm30-mark0',
            'trading_mode': selected_mode(self.window.service).value, 'signals': rows})

    def _position(self, stock):
        window = self.window
        market = Market(stock['market'])
        now = window.engine.clock()
        cached = self.accounts.get(market)
        if cached is None or not 0 <= (now-cached[1]).total_seconds() < 10:
            inst = Instrument(market, stock['symbol'], stock['exchange'])
            account = window.service.safety_account(inst)
            PortfolioCache._validate(account, market)
            cached = (account, window.engine.clock())
            self.accounts[market] = cached
        account, fetched = cached
        positions = [p for p in account.positions if p.symbol == stock['symbol'] and p.exchange == stock['exchange']]
        quantity = sum((p.quantity for p in positions), Decimal(0))
        sellable = sum((p.sellable_quantity for p in positions), Decimal(0))
        cost = sum((p.quantity*p.average_price for p in positions), Decimal(0))
        return {**{key: stock[key] for key in ('market', 'symbol', 'exchange', 'currency')},
                'quantity': str(quantity), 'sellable_quantity': str(sellable),
                'average_price': str(cost/quantity) if quantity else None, 'fetched_at': fetched.isoformat()}

    def publish(self, chart):
        window = self.window
        if window.engine._stop.is_set():
            return
        if self._load_error:
            self._publish_unavailable(chart)
            return
        if self.producer is None:
            from dockdack.lstm30_adapter import LSTM30SignalProducer
            if self.predictors is None:
                try:
                    from dockdack.ml30 import Predictor
                    root = Path(__file__).resolve().parent.parent / 'models/lstm30'
                    self.predictors = {market: Predictor(root / f'{market}.pt', buy_threshold=0.4)
                                       for market in ('domestic', 'us')}
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    self._load_error = str(exc)
                    self.status = '내장 LSTM 실행 불가 · 이 신호기는 HOLD 처리 / 다른 외부 신호와 보유 매도 감시는 계속\n' + str(exc)[:500]
                    self._publish_unavailable(chart)
                    return
            policy = window.engine.external_policy
            self.producer = LSTM30SignalProducer(self.predictors, position_provider=self._position, quantity=1,
                max_krw=policy.max_krw, max_usd=policy.max_usd, clock=window.engine.clock,
                trading_mode=selected_mode(window.service).value,
                state_path=window.store.path.parent / 'exchange/model-decisions-v00.json')
        payload, diagnostics = self.producer(chart)
        self.status = '내장 LSTM 연결됨 · 완료된 일봉이 같으면 예측 재사용 / 주문 여부는 별도 검증'
        # Holdings exits belong to the independent bracket-price pass, never a
        # model's BUY/SELL classification or the rank-list membership.
        by_symbol = {(row['market'], row['exchange'], row['symbol']): row for row in chart['stocks']}
        for entry in payload['signals']:
            if entry['action'] == 'sell':
                for key in ('quantity', 'max_notional', 'min_sell_price', 'cost_profit_pct', 'cost_loss_pct'):
                    entry.pop(key, None)
                entry['action'] = 'hold'
            elif entry['action'] == 'buy':
                stock = by_symbol[(entry['market'], entry['exchange'], entry['symbol'])]
                price = Decimal(stock['price'])
                entry['take_profit_price'] = str(price * Decimal('1.01'))
                entry['stop_loss_price'] = str(price * Decimal('0.992'))
        for row in diagnostics:
            self.diagnostics[row['watch_id']] = row
        # Bound memory even if ranks change for days.
        while len(self.diagnostics) > 500:
            self.diagnostics.pop(next(iter(self.diagnostics)))
        atomic_json(window.engine.external_reader.path, payload)


class V00Window(WatchlistDialog):
    def __init__(self, service=None, store=None, *, builtin=True, predictors=None):
        self._model_predictors = predictors
        super().__init__(service or TradingService(), store)
        self.builtin_lstm = QCheckBox('내장 LSTM30 매수 신호기 사용 · 다른 신호기와 동시 연결 가능')
        self.builtin_lstm.setChecked(builtin)
        self.external_grid.addWidget(self.builtin_lstm, 7, 0, 1, 5)
        self.model_status = QLabel('내장 LSTM · 첫 장중 조회 시 모델 확인')
        self.model_status.setWordWrap(True)
        self.external_grid.addWidget(self.model_status, 9, 0, 1, 5)
        self.external_mode.setChecked(True)
        self.external_krw.setValue(10_000_000)
        self.external_usd.setValue(10_000)
        self.hourly_ranking.setChecked(True)
        self.hourly_ranking.setText('국내·미국 거래량 TOP100 · 개장 10분 전 / 장중 매 정시')
        self.ranking_button.setText('현재 선정 가능한 시장 · 거래량 TOP100')
        self.days_input.setValue(31)
        if builtin:
            with self.store.connection() as db:
                db.execute('UPDATE watchlist SET days=31 WHERE days<31')
            self.reload_tables()
        self.environment_caption.setText('모의 / 실전 · 기본 주문 OFF')
        self.message.setText('ver 0.0 · 감시·자동주문 OFF · 거래 설정에서 10% 비중과 연결을 확인하세요.')
        self._apply_execution_preferences()

    def _finish_environment_switch(self):
        before = self.service
        super()._finish_environment_switch()
        if self.service is not before:
            self.external_quantity.setValue(999999999)
            self._apply_execution_preferences()

    def update_controls(self):
        super().update_controls()
        # The entire external panel is disabled during a running sweep, so
        # selecting a model never changes a worker's active policy mid-order.

    def _update_connection(self):
        super()._update_connection()
        if hasattr(self, 'model_status'):
            producer = self.test_producer
            text = (producer.status if isinstance(producer, DesktopModelBridge) else '내장 LSTM · 첫 장중 조회 시 모델 확인') if self.builtin_lstm.isChecked() else '내장 LSTM 꺼짐 · 연결한 외부 신호 사용'
            self.model_status.setText(text)
            if isinstance(producer, DesktopModelBridge) and producer._load_error:
                self.connection_summary.setText('내장 LSTM 실행 불가 · 매매 설정에서 원인 확인 · 다른 연결은 별도 운영')

    def configure_external(self):
        if self.builtin_lstm.isChecked():
            if self.random_demo.isChecked():
                raise ValueError('내장 LSTM과 랜덤 모의 테스트기는 하나만 선택하세요.')
            self.external_mode.setChecked(True)
            self.external_source.setText('lstm30-mark0')
        result = super().configure_external()
        if self.builtin_lstm.isChecked():
            self.test_producer = DesktopModelBridge(self, predictors=self._model_predictors)
        return result

    def confirm_automation(self):
        if self.builtin_lstm.isChecked():
            if self.random_demo.isChecked():
                raise ValueError('내장 LSTM과 랜덤 모의 테스트기는 하나만 선택하세요.')
            self.external_mode.setChecked(True)
            self.external_source.setText('lstm30-mark0')
        return super().confirm_automation()


def default_desktop_path(root):
    # Reuse the user's latest dedicated LSTM ledger when present; never merge
    # different ledgers or silently discard prior fills/targets.
    runtime = root / '.dockdack/lstm30-demo'
    if not (runtime / 'watchlist.sqlite3').exists():
        runtime = root / '.dockdack'
    return runtime / 'watchlist.sqlite3'


def default_desktop_store(root):
    return WatchStore(default_desktop_path(root), seed_defaults=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-model', action='store_true')
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    from dockdack.lstm30_runtime import SessionLock
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / '.env', override=False)
    set_windows_app_id()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_branding(app)
    app.setStyle('Fusion')
    app.setFont(QFont('Malgun Gothic', 9))
    path = default_desktop_path(root)
    lock = SessionLock(path.parent / 'session.lock')
    try:
        lock.acquire()
    except RuntimeError as exc:
        QMessageBox.warning(None, 'DOCKDACK 이미 실행 중', str(exc))
        return 1
    try:
        store = WatchStore(path, seed_defaults=True)
        window = V00Window(store=store, builtin=not args.no_model)
        window.show()
        return app.exec()
    finally:
        lock.release()


if __name__ == '__main__':
    raise SystemExit(main())
