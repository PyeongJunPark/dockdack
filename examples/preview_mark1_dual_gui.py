"""Offline proof of simultaneous external models in the ordinary DockDack GUI.

Uses read-only 2024 daily windows, a temporary ledger and a synthetic flat
account. Credentials, HTTP, monitoring, order arming and submission are denied.
The screenshots show replayed historical quotes, not the user's current account.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

from dockdack.local_data_paths import default_clean_database_dir
from examples.preview_mark1_prototype_gui import historical_input
from examples.run_desktop_gui import ROOT, configure_local_dependencies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mark1-bundle', type=Path, default=ROOT / 'models/mark1_prototype')
    parser.add_argument('--mark11-bundle', type=Path, default=ROOT / 'models/mark1_1_prototype')
    parser.add_argument('--database-dir', type=Path,
                        default=default_clean_database_dir(ROOT))
    parser.add_argument('--day', type=date.fromisoformat, default=date(2024, 7, 15))
    parser.add_argument('--model', action='append', choices=('mark1-prototype', 'mark1-1-prototype'),
                        help='Optional single-model compatibility preview; default runs both together')
    parser.add_argument('--output-prefix', type=Path,
                        default=ROOT / 'outputs/mark1/external-dual-prototype-preview-20260924')
    args = parser.parse_args(argv)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    configure_local_dependencies()
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QLabel
    from dockdack.gui_service import Instrument
    from dockdack.history import DailyBar, DailyHistory
    from dockdack.lstm30_adapter import _market_calendar
    from dockdack.mark1_prototype_inference import PrototypePredictor
    from dockdack.mark1_1_prototype_inference import Mark11PrototypePredictor
    from dockdack.models import Market, Quote
    from dockdack.signal_bridge import atomic_json
    from dockdack.v00_app import MARK1_TRIGGER, MARK11_TRIGGER, PROTOTYPE_SOURCES, ExternalFeedGroup, V00Window
    from dockdack.watchlist import WatchItem, WatchStore
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_autotrade import FakeTradingService

    now = datetime(args.day.year, args.day.month, args.day.day, 14, tzinfo=timezone.utc)
    instruments = (Instrument(Market.DOMESTIC, '005930', 'KRX'), Instrument(Market.US, 'AAPL', 'ND'))
    inputs = {inst.market.value: historical_input(args.database_dir / f'{inst.market.value}_daily_clean.sqlite3',
                                                inst.symbol, inst.exchange, args.day) for inst in instruments}
    choices = ((MARK1_TRIGGER, PrototypePredictor, args.mark1_bundle),
               (MARK11_TRIGGER, Mark11PrototypePredictor, args.mark11_bundle))
    if args.model:
        choices = tuple(row for row in choices if row[0] in args.model)

    class HistoricalService(FakeTradingService):
        submit_calls = 0

        def quote(self, inst):
            self.quote_calls += 1
            return Quote(inst.market, inst.symbol, '과거 시가 재생', inst.exchange,
                         inputs[inst.market.value][1], inst.currency)

        def history(self, inst, days):
            self.history_calls += 1
            rows, price = inputs[inst.market.value]
            bars = [DailyBar(date.fromisoformat(row[0]), *(Decimal(str(value)) for value in row[1:])) for row in rows]
            # No future same-day high, low, close or volume is exposed to a model.
            bars.append(DailyBar(args.day, price, price, price, price, Decimal(0)))
            return DailyHistory(inst.market, inst.symbol, inst.exchange, inst.currency, days, tuple(bars))

        def execution_history(self, *args, **kwargs):
            return ()

        def submit(self, request):
            self.submit_calls += 1
            raise AssertionError('Offline proof must never submit an order')

    app = QApplication.instance() or QApplication([])
    # Offscreen Qt has no system-font inventory. Register a Latin/symbol
    # fallback as well so the actual U+2212 minus signs remain readable.
    QFontDatabase.addApplicationFont(str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/segoeui.ttf'))
    font_id = QFontDatabase.addApplicationFont(str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/malgun.ttf'))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if families:
        app.setFont(QFont(families[0], 10))

    def idle(window):
        deadline, quiet = time.monotonic() + 60, 0
        while quiet < 3:
            app.processEvents()
            quiet = 0 if (window.worker or window._inspection_worker or
                          getattr(window, '_activity_worker', None) or getattr(window, '_schedule_probe', None)) else quiet + 1
            time.sleep(.01)
            if time.monotonic() > deadline:
                raise RuntimeError('Historical dual-model GUI replay did not finish')

    def assert_off(window, service, store):
        if (window.monitoring or window.engine.orders_enabled or window.pending_auto_arm
                or service.submitted or service.submit_calls or store.attempts()):
            raise AssertionError('Offline proof must stay monitoring/order OFF with no attempts')

    with tempfile.TemporaryDirectory(prefix='mark1-dual-preview-') as folder, \
            patch('requests.sessions.Session.request', side_effect=AssertionError('Offline: HTTP forbidden')) as network, \
            patch('dockdack.gui_service.KiwoomConfig.from_env', side_effect=AssertionError('Offline: credentials forbidden')) as keys, \
            patch('dockdack.autotrade.AutoTrader.enable_orders', side_effect=AssertionError('Offline: arming forbidden')) as arm, \
            patch.object(V00Window, 'start_monitoring', side_effect=AssertionError('Offline: monitoring forbidden')) as monitor:
        expected = {}
        for trigger, predictor_class, bundle in choices:
            for inst in instruments:
                market = inst.market.value
                rows, price = inputs[market]
                model = predictor_class(bundle, market)
                expected[trigger, market] = model.predict([[float(value) for value in row[1:]] for row in rows],
                                                          current_price=price)
                _market_calendar(market, args.day.year)
        store = WatchStore(Path(folder) / 'watchlist.sqlite3')
        items = [WatchItem(inst, '과거 시가 재생 / 가상 미보유', 31) for inst in instruments]
        for item in items:
            store.save_item(item)
        service = HistoricalService()
        window = V00Window(service, store, trigger='none', external_models=tuple(row[0] for row in choices),
                           mark1_bundle=args.mark1_bundle, mark11_bundle=args.mark11_bundle)
        window.engine.clock = lambda: now
        window.portfolio.clock = lambda: now
        rows_out, screenshots, selected_states, processes = [], [], [], []
        proof_feeds, result = (), None
        try:
            for timer in window.findChildren(QTimer):
                timer.stop()
            if families:
                window.setStyleSheet(window.styleSheet() + f"\nQWidget {{ font-family: '{families[0]}'; }}")
                window.model_notice.setStyleSheet(window.model_notice.styleSheet() + " font-family: 'Segoe UI'; font-size: 11px;")
                window.model_status.setStyleSheet("font-size: 11px;")
            selection_text = ('mark1 prototype (+1% / −0.9%) + mark1.1 prototype (+0.5% / −0.4%)'
                              if len(choices) == 2 else 'mark1 prototype (+1% / −0.9%)'
                              if choices[0][0] == MARK1_TRIGGER else 'mark1.1 prototype (+0.5% / −0.4%)')
            banner = QLabel(f'저장 모델 연결 검증 · {args.day} 과거 시가 / 가상 미보유 · 현재 시세·계좌 아님\n'
                            f'외부 AI 연결: {selection_text} · 주문 없음')
            banner.setWordWrap(True)
            banner.setStyleSheet('background: #203c57; color: white; font-size: 15px; font-weight: 700; padding: 12px;')
            window.layout().insertWidget(0, banner)
            window.resize(1600, 1500)
            window.show()
            idle(window)
            window.hourly_ranking.setChecked(False)
            assert_off(window, service, store)
            window.configure_external()
            proof_feeds = tuple(window._prototype_feeds.values())
            if not isinstance(window.test_producer, ExternalFeedGroup):
                raise AssertionError('Normal GUI did not connect the external feed group')
            expected_sources = {PROTOTYPE_SOURCES[model] for model, _, _ in choices}
            if set(window.engine.source_validators) != expected_sources:
                raise AssertionError('Both external model validators must be registered together')
            if window.model_trigger.currentData() != 'none':
                raise AssertionError('Prototype models must not use the built-in selector')
            window.engine.session_only_poll = False  # Replay both historical markets at one instant.
            window.refresh_all()  # One fake read only; no monitoring/arming controls.
            idle(window)
            for trigger, _, _ in choices:
                bridge = window._prototype_feeds[trigger]
                if not bridge.client.is_alive:
                    raise AssertionError(f'External model process is not alive: {trigger}; {bridge.status}; {window.errors}')
                health = bridge.client.request('health', start=False)
                if health['pid'] == os.getpid():
                    raise AssertionError('Inference must run outside the GUI process')
                processes.append({'model_id': trigger, 'pid': health['pid'], 'source_id': bridge.source_id,
                                  'output_path': str(bridge.output_path),
                                  'state_path': str(bridge.client.state_path)})
                result_rows = []
                for item in items:
                    market = item.instrument.market.value
                    diagnostic = bridge.diagnostics.get(item.id, {})
                    actual = diagnostic.get('prediction', {}).get('probability_success')
                    reference = expected[trigger, market]['probability_success']
                    if actual is None or abs(actual - reference) > 1e-12:
                        raise AssertionError(f'Bridge/direct mismatch {trigger}/{market}: {diagnostic}; all={bridge.diagnostics}; status={bridge.status}; {window.errors}')
                    if diagnostic.get('title') != bridge.title or diagnostic.get('strategy_id') != trigger:
                        raise AssertionError(f'Wrong diagnostic model provenance: {diagnostic}')
                    result_rows.append({'trigger': trigger, 'model_title': bridge.title,
                                        'source_id': bridge.source_id, 'market': market, 'symbol': item.instrument.symbol,
                                        'probability_success': actual, 'direct_inference_probability': reference,
                                        'absolute_error': abs(actual - reference),
                                        'reason': diagnostic.get('reason')})
                with store.connection() as db:
                    received = [dict(row) for row in db.execute(
                        'SELECT source_id,watch_id,status FROM external_signals WHERE source_id=? ORDER BY watch_id',
                        (PROTOTYPE_SOURCES[trigger],))]
                if {row['watch_id'] for row in received} != {item.id for item in items}:
                    raise AssertionError(f'Both market signals were not ingested: {received}')
                assert_off(window, service, store)
                selected_states.append({'external_model': trigger, 'checkbox_enabled': window.external_model_checks[trigger].isChecked(),
                                        'source_id': bridge.source_id, 'separate_process_pid': health['pid'],
                                        'monitoring': window.monitoring, 'orders_enabled': window.engine.orders_enabled,
                                        'auto_arm_pending': window.pending_auto_arm, 'received_signals': received})
                rows_out.extend(result_rows)
            if len({row['pid'] for row in processes}) != len(choices):
                raise AssertionError('Each selected external model must own a distinct process')
            window.workspace_tabs.setCurrentWidget(window.tabs)
            window.tabs.setCurrentWidget(window.external_panel)
            window._update_connection()
            values = ' · '.join(f"{row['model_title']} {row['symbol']} {row['probability_success']:.2%}" for row in rows_out)
            window.message.setText(f'{values} · 직접 추론과 일치 · 감시 OFF / 자동주문 OFF / 주문 0건')
            app.processEvents()
            output = args.output_prefix.with_suffix('.png')
            output.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(output.resolve())):
                raise RuntimeError('Could not save ordinary external GUI screenshot')
            screenshots.append(str(output.resolve()))
            for denied in (network, keys, monitor, arm):
                denied.assert_not_called()
            if len(rows_out) != len(choices) * 2:
                raise AssertionError('Expected both models times both markets')
            result = {'title': 'Ordinary GUI: simultaneous external mark1 + mark1.1 historical integration proof',
                      'day': args.day.isoformat(), 'gui_class': type(window).__name__,
                      'not_live_quotes': True, 'synthetic_flat_positions': True, 'temporary_ledger': True,
                      'read_only_source_databases': True, 'actual_saved_models': True,
                      'models_enabled_simultaneously': len(choices) > 1, 'separate_child_processes': True,
                      'builtin_trigger': window.model_trigger.currentData(), 'gui_pid': os.getpid(),
                      'processes': processes, 'comparison_count': len(rows_out),
                      'all_probabilities_match': True, 'network_calls': network.call_count,
                      'credential_loads': keys.call_count, 'monitoring_starts': monitor.call_count,
                      'arming_calls': arm.call_count, 'orders': service.submit_calls,
                      'order_attempts': len(store.attempts()), 'states': selected_states,
                      'rows': rows_out, 'screenshots': screenshots}
            verification = args.output_prefix.with_suffix('.json')
            atomic_json(verification, result)
            print(verification.resolve())
            print(result)
        finally:
            window._pending_environment = None
            window.stop_monitoring()
            for name in ('pool', 'inspection_pool', 'activity_pool'):
                getattr(window, name).waitForDone(10000)
            idle(window)
            window.close()
            window.deleteLater()
            app.processEvents()
            if any(feed.client.is_alive for feed in proof_feeds):
                raise AssertionError('Offline proof leaked a model child process')
            if result is not None:
                result['child_processes_closed'] = True
                atomic_json(args.output_prefix.with_suffix('.json'), result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
