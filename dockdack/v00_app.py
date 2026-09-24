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
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QMessageBox, QLabel

from dockdack.branding import apply_branding, set_windows_app_id
from dockdack.environment_store import selected_mode
from dockdack.gui_service import Instrument, TradingService
from dockdack.models import Market, TradingMode
from dockdack.portfolio import PortfolioCache
from dockdack.signal_bridge import atomic_json, ExternalPolicy, SignalFileReader
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


MARK1_TRIGGER = 'mark1-prototype'
MARK11_TRIGGER = 'mark1-1-prototype'
MARK1_NOTICE = (
    'mark1 prototype · 모의투자 전용 · 과거 30일봉 + 현재가 · 성공확률 50% 초과일 때 매수\n'
    '이 트리거의 새 매수 목표: +1% 익절 / −0.9% 손절 · 기존 보유분의 저장된 목표는 유지\n'
    '연구 검증 미통과 · 주식분할 가격단위 데이터 문제 확인 · 미국 과거 검증 매수 신호 0건 · 수익 보장 없음'
)
MARK11_NOTICE = (
    'mark1.1 prototype · 모의투자 전용 · 과거 30일봉 + 현재가 · 성공확률 50% 초과일 때 매수\n'
    '이 트리거의 새 매수 목표: +0.5% 익절 / −0.4% 손절 · 기존 보유분의 저장된 목표는 유지\n'
    '연구 검증 미통과 · 미국 과거 시가 검증 매수 신호 123건 · 비용 반영 손실 · '
    '주식분할 가격단위 문제와 장중 진입 성과 미검증 · 수익 보장 없음'
)
PROTOTYPE_NOTICES = {MARK1_TRIGGER: MARK1_NOTICE, MARK11_TRIGGER: MARK11_NOTICE}
PROTOTYPE_TITLES = {MARK1_TRIGGER: 'mark1 prototype', MARK11_TRIGGER: 'mark1.1 prototype'}
PROTOTYPE_SOURCES = {MARK1_TRIGGER: 'mark1-prototype-demo-trigger',
                     MARK11_TRIGGER: 'mark1-1-prototype-demo-trigger'}


class ExternalFeedGroup:
    """Independent file producers; execution stays in the one normal engine."""
    def __init__(self, feeds, builtin=None):
        self.feeds = dict(feeds)
        self.builtin = builtin

    def publish(self, chart):
        if self.builtin is not None:
            try:
                self.builtin.publish(chart)
            except Exception as exc:
                self.builtin.status = f'내장 신호 전달 실패: {exc}'
        for feed in self.feeds.values():
            # Each external adapter publishes HOLD on its own failures. It
            # cannot overwrite another source's file or account for an order.
            try:
                feed.publish(chart)
            except Exception as exc:
                # Even a failed HOLD-file write cannot leave that model's
                # validator usable or starve the other independent feed.
                feed.close()
                feed.status = f'{feed.source_id} 외부 전달 실패 · 이 모델 차단: {exc}'

    @property
    def status(self):
        return '\n'.join(feed.status for feed in self.feeds.values())

    @property
    def diagnostics(self):
        return {f'{model}:{key}': value for model, feed in self.feeds.items()
                for key, value in feed.diagnostics.items()}

    def close(self):
        for feed in self.feeds.values():
            feed.close()


class V00Window(WatchlistDialog):
    def __init__(self, service=None, store=None, *, builtin=True, predictors=None,
                 trigger=None, mark1_predictors=None, mark1_bundle=None,
                 mark11_predictors=None, mark11_bundle=None,
                 external_models=None, external_feed_factory=None):
        choice = trigger if trigger is not None else 'lstm30' if builtin else 'none'
        if choice not in {'none', 'lstm30', *PROTOTYPE_NOTICES}:
            raise ValueError('알 수 없는 내장 매수 트리거입니다.')
        enabled_models = tuple(external_models or ())
        if choice in PROTOTYPE_NOTICES:
            # Old CLI links migrate to an external feed, never back into the
            # mutually exclusive builtin selector.
            enabled_models = tuple(dict.fromkeys((*enabled_models, choice)))
            choice = 'none'
        if set(enabled_models) - set(PROTOTYPE_NOTICES):
            raise ValueError('알 수 없는 외부 AI 신호기입니다.')
        if enabled_models and service is not None and selected_mode(service) is not TradingMode.DEMO:
            raise ValueError('prototype 외부 신호기는 모의투자에서만 연결할 수 있습니다.')
        if mark1_predictors is not None or mark11_predictors is not None:
            raise ValueError('모델은 별도 외부 프로세스에서 실행합니다. 테스트에는 external_feed_factory를 사용하세요.')
        self._model_predictors = predictors
        self._mark1_bundle = mark1_bundle
        self._mark11_bundle = mark11_bundle
        self._external_feed_factory = external_feed_factory
        self._prototype_feeds = {}
        super().__init__(service or TradingService(), store)
        # A disabled BUY feed must not change the ownership or exit policy of
        # already filled model lots. REAL use is separately blocked below.
        self.engine.prototype_lots_enabled = True
        # Retain mark0 compatibility; prototypes are independent external feeds.
        self.builtin_lstm = QCheckBox('내장 LSTM30 매수 신호기 사용 · 다른 신호기와 동시 연결 가능')
        self.builtin_lstm.setChecked(choice == 'lstm30')
        self.builtin_lstm.setParent(self)
        self.builtin_lstm.hide()
        settings_items = []
        while self.external_grid.count():
            position = self.external_grid.getItemPosition(0)
            settings_items.append((self.external_grid.takeAt(0), position))
        for item, (row, column, row_span, column_span) in settings_items:
            self.external_grid.addItem(item, row + 6, column, row_span, column_span)
        self.external_grid.addWidget(QLabel('외부 AI 매수 신호기 · 두 모델 동시 연결 가능'), 0, 0, 1, 5)
        self.external_model_checks = {}
        for row, model in enumerate(PROTOTYPE_NOTICES, 1):
            targets = '+1% / −0.9%' if model == MARK1_TRIGGER else '+0.5% / −0.4%'
            checkbox = QCheckBox(f'{PROTOTYPE_TITLES[model]} · {targets} · 외부 프로세스')
            checkbox.setObjectName('external-' + model)
            checkbox.setChecked(model in enabled_models)
            self.external_model_checks[model] = checkbox
            self.external_grid.addWidget(checkbox, row, 0, 1, 2)
            path = self._prototype_output_path(model)
            label = QLabel(str(path))
            label.setWordWrap(True)
            label.setToolTip(f'{PROTOTYPE_SOURCES[model]}\n{path}')
            self.external_grid.addWidget(label, row, 2, 1, 3)
        self.model_trigger = QComboBox()
        self.model_trigger.setObjectName('builtinTradingTrigger')
        self.model_trigger.addItem('내장 트리거 없음 · 외부 신호만 사용', 'none')
        self.model_trigger.addItem('LSTM30 mark0 · 기존 내장 매수 트리거', 'lstm30')
        self.model_trigger.setCurrentIndex(self.model_trigger.findData(choice))
        self.external_grid.addWidget(QLabel('기존 내장 트리거'), 5, 0)
        self.external_grid.addWidget(self.model_trigger, 5, 1, 1, 4)
        self.model_status = QLabel('내장 LSTM · 첫 장중 조회 시 모델 확인')
        self.model_status.setWordWrap(True)
        self.external_grid.addWidget(self.model_status, 4, 0, 1, 5)
        self.model_notice = QLabel(MARK1_NOTICE)
        self.model_notice.setWordWrap(True)
        self.model_notice.setStyleSheet('color: #ffda91;')
        self.external_grid.addWidget(self.model_notice, 3, 0, 1, 5)
        self.model_trigger.currentIndexChanged.connect(self._model_trigger_changed)
        for checkbox in self.external_model_checks.values():
            checkbox.toggled.connect(self._model_trigger_changed)
        self.builtin_lstm.toggled.connect(self._legacy_lstm_changed)
        self.external_mode.setChecked(True)
        self.external_krw.setValue(10_000_000)
        self.external_usd.setValue(10_000)
        self.hourly_ranking.setChecked(True)
        self.hourly_ranking.setText('국내·미국 거래량 TOP100 · 개장 10분 전 / 장중 매 정시')
        self.ranking_button.setText('현재 선정 가능한 시장 · 거래량 TOP100')
        self.days_input.setValue(31)
        self.environment_caption.setText('모의 / 실전 · 기본 주문 OFF')
        self.message.setText('ver 0.0 · 감시·자동주문 OFF · 거래 설정에서 10% 비중과 연결을 확인하세요.')
        self._apply_execution_preferences()
        self._model_trigger_changed()

    def _chosen_trigger(self):
        return self.model_trigger.currentData() if hasattr(self, 'model_trigger') else 'none'

    def _chosen_external_models(self):
        return tuple(model for model, check in getattr(self, 'external_model_checks', {}).items()
                     if check.isChecked())

    def _prototype_output_path(self, model):
        return self.store.path.parent / 'exchange' / 'external-models' / model / 'signals.json'

    def _close_external_feeds(self):
        for feed in self._prototype_feeds.values():
            feed.close()
        self._prototype_feeds = {}

    @property
    def builtin_confirmation_notice(self):
        return '\n\n'.join(PROTOTYPE_NOTICES[model] for model in self._chosen_external_models())

    def _legacy_lstm_changed(self, checked):
        choice = 'lstm30' if checked else 'none'
        if checked or self._chosen_trigger() == 'lstm30':
            self.model_trigger.setCurrentIndex(self.model_trigger.findData(choice))

    def _model_trigger_changed(self, *_):
        choice = self._chosen_trigger()
        enabled_models = self._chosen_external_models()
        if enabled_models and selected_mode(self.service) is not TradingMode.DEMO:
            for check in self.external_model_checks.values():
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
            self.engine.disarm()
            self.message.setText('prototype 외부 신호기는 모의투자에서만 사용할 수 있습니다. 실전 주문에 연결하지 않았습니다.')
            return
        # Programmatic changes cannot swap a running worker's active strategy.
        if self.monitoring or self.worker is not None or self.pending_auto_arm:
            self.stop_monitoring()
        self.engine.disarm()
        # A validator belongs to exactly one selected model. Removing the old
        # callback also releases its cached predictor before the next lazy load.
        self.engine.configure_source_validators({})
        self._close_external_feeds()
        self.builtin_lstm.blockSignals(True)
        self.builtin_lstm.setChecked(choice == 'lstm30')
        self.builtin_lstm.blockSignals(False)
        self.test_producer = None
        if choice != 'none' or enabled_models:
            # Thirty completed sessions plus today's quote need a 31-day query;
            # selecting a model must also upgrade pre-existing 30-day rows.
            self.days_input.setValue(max(31, self.days_input.value()))
            with self.store.connection() as db:
                db.execute('UPDATE watchlist SET days=31 WHERE days<31')
            self.reload_tables()
        if choice == 'lstm30':
            self.external_mode.setChecked(True)
            self.external_source.setText('lstm30-mark0')
        elif self.external_source.text().strip() in {'lstm30-mark0', *PROTOTYPE_SOURCES.values()}:
            self.external_source.setText('external-model')
        if enabled_models:
            self.external_mode.setChecked(True)
        self.model_notice.setText(self.builtin_confirmation_notice)
        self.model_notice.setVisible(bool(enabled_models))
        self._sync_model_mode_controls()
        self._update_connection()

    def _sync_model_mode_controls(self):
        if not hasattr(self, 'model_trigger'):
            return
        if self._chosen_external_models():
            self.environment_selector.buttons[TradingMode.REAL].setEnabled(False)
            self.environment_selector.buttons[TradingMode.REAL].setToolTip('prototype 외부 신호기는 모의투자 전용입니다. 두 연결을 끄면 환경 전환할 수 있습니다.')
        else:
            self.environment_selector.apply(selected_mode(self.service), pending=self._pending_environment is not None)

    def request_environment(self, mode):
        if TradingMode(mode) is TradingMode.REAL and self._chosen_external_models():
            self.engine.disarm()
            self.message.setText('prototype 외부 신호기는 모의투자 전용입니다. 실전 전환 전에 두 연결을 끄세요.')
            return
        return super().request_environment(mode)

    def _sync_environment(self):
        super()._sync_environment()
        self._sync_model_mode_controls()

    def _external_config_key(self):
        return (*super()._external_config_key(), self._chosen_trigger(), self._chosen_external_models())

    def _finish_environment_switch(self):
        if (self._chosen_external_models() and self._pending_environment
                and selected_mode(self._pending_environment[0]) is TradingMode.REAL):
            self._pending_environment = None
            self.environment_timer.stop()
            self.engine.disarm()
            self._sync_environment()
            self.message.setText('prototype 외부 연결로 실전 환경 전환을 취소했습니다. 모의·주문 OFF 유지')
            self.update_controls()
            return
        before = self.service
        super()._finish_environment_switch()
        if self.service is not before:
            self.external_quantity.setValue(999999999)
            self._apply_execution_preferences()

    def update_controls(self):
        super().update_controls()
        # The entire external panel is disabled during a running sweep, so
        # selecting a model never changes a worker's active policy mid-order.
        self._sync_model_mode_controls()

    def _update_connection(self):
        super()._update_connection()
        if hasattr(self, 'model_status'):
            producer = self.test_producer
            choice = self._chosen_trigger()
            text = '\n'.join(
                getattr(self._prototype_feeds.get(model), 'status',
                        f'{PROTOTYPE_TITLES[model]} · 외부 연결 선택됨 · 첫 조회 시 별도 프로세스 시작')
                for model in self._chosen_external_models())
            if not text:
                text = ((producer.status if isinstance(producer, DesktopModelBridge) else '내장 LSTM · 첫 장중 조회 시 모델 확인')
                        if choice == 'lstm30' else '외부 AI 연결 꺼짐 · 직접 연결한 외부 JSON은 별도 사용')
            self.model_status.setToolTip(text)
            if self._chosen_external_models():
                text = '\n'.join(getattr(self._prototype_feeds.get(model), 'status',
                    f'{PROTOTYPE_TITLES[model]} · 외부 연결 선택됨 · 첫 조회 시 별도 프로세스 시작').splitlines()[0]
                    for model in self._chosen_external_models())
                state = '감시 중' if self.monitoring else '감시 OFF'
                orders = '주문 ON' if self.engine.orders_enabled else '주문 OFF'
                self.connection_summary.setText(f'외부 AI {len(self._chosen_external_models())}개 · {state} · {orders} · 모델별 연결 상태는 매매 설정에서 확인')
            self.model_status.setText(text)

    def _prepare_builtin(self):
        choice = self._chosen_trigger()
        models = self._chosen_external_models()
        if choice == 'none' and not models:
            return
        if self.random_demo.isChecked():
            raise ValueError('AI 신호 연결과 랜덤 모의 테스트기는 하나만 선택하세요.')
        if models and selected_mode(self.service) is not TradingMode.DEMO:
            raise ValueError('prototype 외부 신호기는 모의투자 전용입니다.')
        self.external_mode.setChecked(True)
        if choice == 'lstm30':
            self.external_source.setText('lstm30-mark0')

    def configure_external(self):
        self._prepare_builtin()
        self._close_external_feeds()
        result = super().configure_external()
        self.engine.prototype_lots_enabled = True
        self.engine.configure_source_validators({})
        builtin = (DesktopModelBridge(self, predictors=self._model_predictors)
                   if self._chosen_trigger() == 'lstm30' else self.test_producer)
        sources = list(self.engine.external_sources.values())
        seen_sources = set(self.engine.external_sources)
        seen_paths = {Path(reader.path).resolve() for _, reader in sources}
        seen_paths.update((Path(self.chart_path.text()).resolve(), self.store.path.resolve()))
        if self._chosen_external_models():
            from dockdack.prototype_external import ExternalPrototypeFeed
            factory = self._external_feed_factory or ExternalPrototypeFeed
            policy = result[2]
            try:
                for model in self._chosen_external_models():
                    source, path = PROTOTYPE_SOURCES[model], self._prototype_output_path(model)
                    if source in seen_sources or path.resolve() in seen_paths:
                        raise ValueError('외부 AI 신호 출처/파일이 다른 연결과 중복됩니다.')
                    extra = ExternalPolicy(source, policy.max_quantity, policy.max_krw, policy.max_usd)
                    bundle = self._mark1_bundle if model == MARK1_TRIGGER else self._mark11_bundle
                    feed = factory(self, model, extra, path, bundle_root=bundle)
                    self._prototype_feeds[model] = feed
                    sources.append((extra, SignalFileReader(self.store, path, extra, self.engine.clock)))
                    seen_sources.add(source)
                    seen_paths.add(path.resolve())
                self.engine.configure_external_sources(sources)
                self.engine.configure_source_validators({feed.source_id: feed.validate_execution
                                                        for feed in self._prototype_feeds.values()})
            except Exception:
                self._close_external_feeds()
                self.engine.configure_source_validators({})
                raise
        self.test_producer = (ExternalFeedGroup(self._prototype_feeds, builtin) if self._prototype_feeds else builtin)
        self._update_connection()
        return result

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted():
            self._close_external_feeds()

    def confirm_automation(self):
        self._prepare_builtin()
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


def desktop_model_choices(trigger, no_model, external_models):
    models = list(external_models)
    if trigger is None and not no_model and not models:
        return 'none', list(PROTOTYPE_NOTICES)
    if trigger is None and models:
        return 'none', models
    return trigger, models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    trigger_options = parser.add_mutually_exclusive_group()
    trigger_options.add_argument('--no-model', action='store_true')
    trigger_options.add_argument('--trigger', choices=('none', 'lstm30', MARK1_TRIGGER, MARK11_TRIGGER))
    parser.add_argument('--mark1-bundle', type=Path, help='mark1 prototype 저장 모델 폴더')
    parser.add_argument('--mark11-bundle', type=Path, help='mark1.1 prototype 저장 모델 폴더')
    parser.add_argument('--external-model', action='append', choices=tuple(PROTOTYPE_NOTICES), default=[],
                        help='외부 AI 신호기. 두 모델을 함께 쓰려면 각각 지정 (자동주문은 OFF)')
    parser.add_argument('--store', type=Path, help='기존 통합 GUI 장부 경로 (복사하지 않음)')
    parser.add_argument('--env-file', type=Path, help='기존 키 설정 파일 경로 (복사하지 않음)')
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    from dockdack.lstm30_runtime import SessionLock
    root = Path(__file__).resolve().parent.parent
    load_dotenv(args.env_file.expanduser() if args.env_file else root / '.env', override=False)
    set_windows_app_id()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_branding(app)
    app.setStyle('Fusion')
    app.setFont(QFont('Malgun Gothic', 9))
    path = args.store.expanduser().resolve() if args.store else default_desktop_path(root)
    lock = SessionLock(path.parent / 'session.lock')
    try:
        lock.acquire()
    except RuntimeError as exc:
        QMessageBox.warning(None, 'DOCKDACK 이미 실행 중', str(exc))
        return 1
    try:
        store = WatchStore(path, seed_defaults=True)
        trigger, models = desktop_model_choices(args.trigger, args.no_model, args.external_model)
        window = V00Window(store=store, builtin=not args.no_model,
                           trigger=trigger, mark1_bundle=args.mark1_bundle,
                           mark11_bundle=args.mark11_bundle, external_models=models)
        window.show()
        return app.exec()
    finally:
        lock.release()


if __name__ == '__main__':
    raise SystemExit(main())
