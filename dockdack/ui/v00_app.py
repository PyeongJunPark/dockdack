"""Unified desktop: optional LSTM plus multiple data-only signal feeds.

Always starts OFF. Models initialize lazily on the existing broker worker, not
the GUI thread. This replaces the old DEMO-only shortcut, not its ledger.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys
from time import monotonic
from uuid import NAMESPACE_URL, uuid5

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHeaderView, QHBoxLayout, QMessageBox, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from dockdack.branding import apply_branding, set_windows_app_id
from dockdack.version import APP_RELEASE
from dockdack.environment_store import selected_mode, scoped_store_path, store_for_service
from dockdack.gui_service import Instrument, TradingService
from dockdack.lstm30_adapter import MAX_QUOTE_AGE
from dockdack.models import Market, TradingMode
from dockdack.portfolio import PortfolioCache
from dockdack.signal_bridge import atomic_json, ExternalPolicy, SignalFileReader
from dockdack.watch_gui import WatchlistDialog
from dockdack.watchlist import WatchStore, utc_now


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
                    from dockdack.runtime_paths import model_bundle
                    root = model_bundle('lstm30')
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
MARK12_TRIGGER = 'mark1-2-prototype'
MARK1_NOTICE = (
    'mark1 prototype · 모의투자 전용 · 과거 30일봉 + 현재가 · 모델 추정 확률 50% 초과일 때 매수\n'
    '이 트리거의 새 매수 목표: +1% 익절 / −0.9% 손절 · 기존 보유분의 저장된 목표는 유지\n'
    '연구 검증 미통과 · 주식분할 가격단위 데이터 문제 확인 · 미국 과거 검증 매수 신호 0건 · 수익 보장 없음'
)
MARK11_NOTICE = (
    'mark1.1 prototype · 모의투자 전용 · 과거 30일봉 + 현재가 · 모델 추정 확률 50% 초과일 때 매수\n'
    '이 트리거의 새 매수 목표: +0.5% 익절 / −0.4% 손절 · 기존 보유분의 저장된 목표는 유지\n'
    '연구 검증 미통과 · 미국 과거 시가 검증 매수 신호 123건 · 비용 반영 손실 · '
    '주식분할 가격단위 문제와 장중 진입 성과 미검증 · 수익 보장 없음'
)
MARK12_NOTICE = (
    'mark1.2 prototype · 모의투자 전용 · 과거 완료 30일봉 + 현재가 · 모델 추정 확률 50% 초과일 때 매수\n'
    '이 트리거의 새 매수 목표: +1% 익절 / −0.9% 손절 · 기존 보유분의 저장된 목표는 유지\n'
    '딥러닝 연구 검증 미통과 · 재사용 과거 평가 국내 473신호/미국 1,712신호 · '
    '왕복 20bp 가정 비용 후 보유형 국내 −5.86%/미국 −6.90% · '
    '장중 선후관계·실제 체결 미검증 · 수익 보장 없음'
)
PROTOTYPE_NOTICES = {MARK1_TRIGGER: MARK1_NOTICE, MARK11_TRIGGER: MARK11_NOTICE,
                     MARK12_TRIGGER: MARK12_NOTICE}
PROTOTYPE_TITLES = {MARK1_TRIGGER: 'mark1 prototype', MARK11_TRIGGER: 'mark1.1 prototype',
                    MARK12_TRIGGER: 'mark1.2 prototype'}
PROTOTYPE_TARGETS = {MARK1_TRIGGER: '+1% / −0.9%', MARK11_TRIGGER: '+0.5% / −0.4%',
                     MARK12_TRIGGER: '+1% / −0.9%'}
PROTOTYPE_SOURCES = {MARK1_TRIGGER: 'mark1-prototype-demo-trigger',
                     MARK11_TRIGGER: 'mark1-1-prototype-demo-trigger',
                     MARK12_TRIGGER: 'mark1-2-prototype-demo-trigger'}


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
            if isinstance(chart, dict) and isinstance(chart.get('stocks'), list):
                for stock in chart['stocks']:
                    if isinstance(stock, dict) and stock.get('status') == 'ok' and isinstance(stock.get('watch_id'), str):
                        feed.diagnostics.pop(stock['watch_id'], None)
            try:
                feed.publish(chart)
            except Exception as exc:
                # Even a failed HOLD-file write cannot leave that model's
                # validator usable or starve the other independent feed.
                feed.close()
                feed.status = f'{feed.source_id} 외부 전달 실패 · 이 모델 차단: {exc}'
                if isinstance(chart, dict) and isinstance(chart.get('stocks'), list):
                    for stock in chart['stocks']:
                        if isinstance(stock, dict) and stock.get('status') == 'ok' and isinstance(stock.get('watch_id'), str):
                            feed.diagnostics[stock['watch_id']] = {
                                'watch_id': stock['watch_id'], 'reason': 'MODEL_DELIVERY_UNAVAILABLE', 'error': str(exc)}
            # Display provenance is intentionally separate from the order
            # JSON. A score may only be paired with the exact quote used for
            # this model's latest completed decision.
            if isinstance(chart, dict) and isinstance(chart.get('stocks'), list):
                for stock in chart['stocks']:
                    if not isinstance(stock, dict) or stock.get('status') != 'ok':
                        continue
                    key = stock.get('watch_id')
                    if not isinstance(key, str):
                        continue
                    diagnostic = feed.diagnostics.get(key)
                    if diagnostic is None or not getattr(feed, '_ready', True):
                        diagnostic = {'watch_id': key, 'reason': 'MODEL_RESULT_UNAVAILABLE',
                                      'error': str(getattr(feed, 'status', '모델 판단 실패'))}
                    feed.diagnostics[key] = {
                        **diagnostic, '_display_quote_fetched_at': stock.get('quote_fetched_at'),
                        '_display_price': stock.get('price')}

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
                 mark12_bundle=None,
                 external_models=None, external_feed_factory=None, session_lock=None):
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
        self._mark12_bundle = mark12_bundle
        self._external_feed_factory = external_feed_factory
        self._prototype_feeds = {}
        # Display history only. Neither external JSON nor order validation
        # reads this cache; old probabilities cannot authorize a trade.
        self._last_model_scores = {}
        self._last_queried_watch_id = None
        super().__init__(service or TradingService(), store, defer_workspace=True, session_lock=session_lock)
        for view in self.watch_tables.values():
            view.setColumnCount(7)
            view.setHorizontalHeaderLabels(('종목', '현재가', 'N일', '상태/조회시각',
                                            'MK1 추정확률', 'MK1.1 추정확률', 'MK1.2 추정확률'))
            view.setMinimumWidth(610)
            for column, width in ((0, 110), (1, 100), (2, 40), (3, 115), (4, 140), (5, 140), (6, 140)):
                view.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                view.setColumnWidth(column, width)
        self.model_score_summary = QLabel('모델 판단 전 · 추정확률은 실제 수익률이나 검증된 적중률이 아닙니다.')
        self.model_score_summary.setObjectName('modelScoreSummary')
        self.model_score_summary.setWordWrap(True)
        self.model_score_summary.setToolTip('MK1과 MK1.2는 +1%/−0.9%, MK1.1은 +0.5%/−0.4% 일봉 조건의 모델 추정값입니다. 모두 연구 검증 미통과.')
        self.chart_title.parentWidget().layout().insertWidget(1, self.model_score_summary)
        self.latest_model_summary = QLabel('최근 조회 종목 — · 활성 외부 AI 모델 평균 대기')
        self.latest_model_summary.setObjectName('latestModelSummary')
        self.latest_model_summary.setAccessibleName('마지막 조회 종목과 활성 외부 AI 모델 추정확률 평균')
        self.latest_model_summary.setWordWrap(True)
        self.latest_model_summary.setStyleSheet('color: #aee4d4; font-weight: 600;')
        self.latest_model_summary.setToolTip('같은 종목·같은 조회 시세에서 활성화된 모든 외부 prototype 모델이 판단했을 때만 산술평균을 표시합니다. 모델별 목표 조건이 달라 매매 판단에 쓰지 않습니다.')
        self.layout().insertWidget(self.layout().indexOf(self.sweep_progress), self.latest_model_summary)
        self.health_timer.timeout.connect(self._refresh_model_scores)
        self._refresh_model_scores()
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
            self.external_grid.addItem(item, row + 9, column, row_span, column_span)
        self.external_grid.addWidget(QLabel('외부 AI 매수 신호기 · 세 모델 동시 연결 가능'), 0, 0, 1, 5)
        self.external_model_checks = {}
        for row, model in enumerate(PROTOTYPE_NOTICES, 1):
            checkbox = QCheckBox(f'{PROTOTYPE_TITLES[model]} · {PROTOTYPE_TARGETS[model]} · 외부 프로세스')
            checkbox.setObjectName('external-' + model)
            checkbox.setChecked(model in enabled_models)
            self.external_model_checks[model] = checkbox
            self.external_grid.addWidget(checkbox, row, 0, 1, 2)
            path = self._prototype_output_path(model)
            label = QLabel('JSON 연결 · 전체 경로는 도움말/상태·검사에서 확인')
            label.setWordWrap(True)
            label.setToolTip(f'{PROTOTYPE_SOURCES[model]}\n{path}')
            self.external_grid.addWidget(label, row, 2, 1, 3)
        self.model_trigger = QComboBox()
        self.model_trigger.setObjectName('builtinTradingTrigger')
        self.model_trigger.addItem('내장 트리거 없음 · 외부 신호만 사용', 'none')
        self.model_trigger.addItem('LSTM30 mark0 · 기존 내장 매수 트리거', 'lstm30')
        self.model_trigger.setCurrentIndex(self.model_trigger.findData(choice))
        self.legacy_trigger_label = QLabel('기존 내장 트리거')
        self.external_grid.addWidget(self.legacy_trigger_label, 6, 0)
        self.external_grid.addWidget(self.model_trigger, 6, 1, 1, 4)
        self.model_status = QLabel('내장 LSTM · 첫 장중 조회 시 모델 확인')
        self.model_status.setWordWrap(True)
        self.external_grid.addWidget(self.model_status, 5, 0, 1, 5)
        self.model_notice = QLabel(MARK1_NOTICE)
        self.model_notice.setWordWrap(True)
        self.model_notice.setStyleSheet('color: #ffda91;')
        self.external_grid.addWidget(self.model_notice, 4, 0, 1, 5)
        self.close_all_at_market_end = QCheckBox('장마감 5분 전 모의계좌 전체 매도')
        self.close_all_at_market_end.setChecked(selected_mode(self.service) is TradingMode.DEMO)
        self.close_all_at_market_end.setToolTip(
            '기존 보유분을 포함한 국내·미국 전체 매도가능 수량 대상. 마감 5분 전 신규 매수를 차단하며, '
            '마감 매도에만 평소 수량·금액 한도를 적용하지 않습니다. 주문 ON 확인이 필요하며 체결은 보장되지 않습니다.')
        self.external_grid.addWidget(self.close_all_at_market_end, 7, 0, 1, 5)
        self.close_liquidation_label = QLabel('마감 청산 대기 · 모의계좌 전체 · 감시 및 주문 ON 필요 · 체결 보장 없음')
        self.close_liquidation_label.setWordWrap(True)
        self.external_grid.addWidget(self.close_liquidation_label, 8, 0, 1, 5)
        self.close_all_at_market_end.toggled.connect(self._close_policy_changed)
        self.model_trigger.currentIndexChanged.connect(self._model_trigger_changed)
        for checkbox in self.external_model_checks.values():
            checkbox.toggled.connect(self._model_trigger_changed)
        self.builtin_lstm.toggled.connect(self._legacy_lstm_changed)
        self.external_mode.setChecked(True)
        self.external_krw.setValue(10_000_000)
        self.external_usd.setValue(10_000)
        self.hourly_ranking.setChecked(True)
        self.hourly_ranking.setText('국내·미국 거래량 TOP100 · 개장 10분 전 / 개장 / 매 정시')
        self.ranking_button.setText('현재 선정 가능한 시장 · 거래량 TOP100')
        self.days_input.setValue(31)
        self.environment_caption.setText('모의 / 실전 · 기본 주문 OFF')
        self.message.setText(f'ver {APP_RELEASE} · 감시·자동주문 OFF · 신호 연결에서 10% 비중과 연결을 확인하세요.')
        self._apply_execution_preferences()
        self._configure_close_policy()
        self._label_inputs()
        self._move_legacy_settings()
        self._model_trigger_changed()
        self._poll_after_maintenance = False
        self.close_maintenance_timer = QTimer(self)
        self.close_maintenance_timer.setInterval(5000)
        self.close_maintenance_timer.timeout.connect(self._close_wakeup)
        self.close_maintenance_timer.start()

    def _current_model_score(self, model, item):
        """Accept only a valid display diagnostic for this exact quote."""
        feed = self._prototype_feeds.get(model)
        snapshot = self.snapshots.get(item.id)
        if (feed is None or not getattr(feed, '_ready', True) or snapshot is None
                or item.id in self.errors):
            return None
        row = feed.diagnostics.get(item.id)
        if (not isinstance(row, dict) or row.get('error')
                or row.get('_display_quote_fetched_at') != snapshot.fetched_at.isoformat()
                or row.get('_display_price') != str(snapshot.quote.price)):
            return None
        reason = row.get('reason')
        if reason not in {'PREDICTED_DAILY_BARRIER_SUCCESS', 'BELOW_OR_EQUAL_BUY_THRESHOLD',
                          'USER_QUANTITY_OR_NOTIONAL_CAP', 'QUOTE_EXPIRED_DURING_INFERENCE'}:
            return None
        prediction = row.get('prediction')
        if not isinstance(prediction, dict):
            return None
        try:
            value = prediction.get('probability_success')
            if isinstance(value, bool):
                return None
            probability = Decimal(str(value))
            reference_price = Decimal(str(row.get('reference_price')))
            if (not probability.is_finite() or not 0 <= probability <= 1
                    or reference_price != snapshot.quote.price
                    or (reason == 'PREDICTED_DAILY_BARRIER_SUCCESS' and probability <= Decimal('0.5'))
                    or (reason == 'BELOW_OR_EQUAL_BUY_THRESHOLD' and probability > Decimal('0.5'))):
                return None
        except (InvalidOperation, TypeError, ValueError):
            return None
        score = {'probability': probability, 'price': reference_price,
                 'fetched_at': snapshot.fetched_at, 'reason': reason}
        self._last_model_scores[(model, item.id)] = score
        return score

    def _model_score_display(self, model, item):
        """Keep the last valid display score; execution still uses fresh quotes."""
        title = PROTOTYPE_TITLES[model]
        target = PROTOTYPE_TARGETS[model]
        note = (f'{title} · {target} · 일봉 기반 모델 추정확률이며 실제 적중률·수익률 보장 아님. '
                '연구 검증 미통과. 과거 표시값은 매매 판단에 사용하지 않고 주문 직전 새 시세로 재판단합니다.')
        checks = getattr(self, 'external_model_checks', {})
        if model not in checks or not checks[model].isChecked():
            return '연결 꺼짐', note + '\n이 모델은 연결되어 있지 않습니다.', '꺼짐'
        current = self._current_model_score(model, item)
        snapshot = self.snapshots.get(item.id)
        if current is not None and snapshot is not None:
            age = (self.engine.clock() - snapshot.fetched_at).total_seconds()
            if (0 <= age <= MAX_QUOTE_AGE and item.id in self.fresh_ids
                    and current['reason'] != 'QUOTE_EXPIRED_DURING_INFERENCE'):
                decision = '매수 판정' if current['reason'] == 'PREDICTED_DAILY_BARRIER_SUCCESS' else '대기'
                pct = f"{current['probability']:.1%}"
                context = (f"\n판단 기준 현재가 {current['price']} {item.instrument.currency}"
                           f" · 조회 {current['fetched_at'].astimezone():%m/%d %H:%M:%S}")
                return (f'{decision}\n{pct}',
                        note + context + f"\n{decision} · 추정 성공확률 {pct} · 사유 {current['reason']}",
                        f'{decision} {pct}')
        saved = self._last_model_scores.get((model, item.id))
        if saved is not None:
            pct = f"{saved['probability']:.1%}"
            prior_time = saved['fetched_at'].astimezone().strftime('%m/%d %H:%M:%S')
            context = (f"마지막 판단 {prior_time}"
                       f" · 당시 현재가 {saved['price']} {item.instrument.currency}")
            return (f'최근 추정\n{pct}', note + '\n' + context
                    + '\n이전 조회값 · 현재 시세의 매수 판정이나 주문 허가가 아닙니다.',
                    f"최근 {pct} ({saved['price']} {item.instrument.currency} · {prior_time} 기준)")
        if item.id in self.errors:
            return '시세 오류\n확률 —', note + '\n현재 시세 조회 오류: ' + str(self.errors[item.id]), '시세 오류'
        if snapshot is None:
            return '현재가 없음\n확률 —', note + '\n현재가 조회 전입니다.', '현재가 없음'
        row = getattr(self._prototype_feeds.get(model), 'diagnostics', {}).get(item.id)
        state = ('보유 중' if isinstance(row, dict) and row.get('reason') == 'POSITION_EXIT_MANAGED_BY_GUI'
                 else '판단 불가' if isinstance(row, dict) and (row.get('error') or row.get('prediction'))
                 else '판단 전')
        return f'{state}\n확률 —', note + f'\n현재가 {snapshot.quote.price} {item.instrument.currency} · 이 시세의 유효한 모델 확률이 없습니다.', state

    def _watch_values(self, item):
        return (*super()._watch_values(item),
                *(self._model_score_display(model, item)[0] for model in PROTOTYPE_NOTICES))

    def _update_model_score_row(self, key):
        item = self._items_by_id.get(key)
        row = self._watch_rows.get(key)
        if item is None or row is None:
            return
        view = self.watch_tables[item.instrument.market]
        for column, model in enumerate(PROTOTYPE_NOTICES, 4):
            cell = view.item(row, column)
            if cell is None:
                continue
            value, tip, _ = self._model_score_display(model, item)
            if cell.text() != value:
                cell.setText(value)
            if cell.toolTip() != tip:
                cell.setToolTip(tip)

    def _refresh_model_scores(self):
        if not hasattr(self, 'model_score_summary'):
            return
        for key in tuple(getattr(self, '_watch_rows', ())):
            self._update_model_score_row(key)
        self._update_selected_model_scores()
        self._update_latest_model_summary()

    def _update_latest_model_summary(self):
        """Global display of the last queried symbol, never an order gate."""
        if not hasattr(self, 'latest_model_summary'):
            return
        item = getattr(self, '_items_by_id', {}).get(self._last_queried_watch_id)
        snapshot = self.snapshots.get(self._last_queried_watch_id)
        if item is None or snapshot is None:
            self.latest_model_summary.setText('최근 조회 종목 — · 활성 외부 AI 모델 평균 대기')
            return
        stamp = snapshot.fetched_at.astimezone().strftime('%m/%d %H:%M:%S')
        prefix = (f'마지막 조회 {item.name or item.instrument.symbol} ({item.instrument.symbol}) · {stamp}'
                  f' · 당시가 {snapshot.quote.price} {item.instrument.currency}')
        models = self._chosen_external_models()
        if not models:
            self.latest_model_summary.setText(prefix + ' · 활성 외부 AI 모델 없음')
            return
        scores = [self._current_model_score(model, item) for model in models]
        available = [score for score in scores if score is not None]
        if len(available) != len(models):
            self.latest_model_summary.setText(
                prefix + f' · 모델 단순 평균 대기 ({len(available)}/{len(models)}개 결과)')
            return
        average = sum((score['probability'] for score in available), Decimal(0)) / Decimal(len(models))
        self.latest_model_summary.setText(
            prefix + f' · 활성 모델 단순 평균 {average:.1%} ({len(models)}/{len(models)})'
            + ' · 목표 조건 상이·매매 판단에 사용 안 함')

    def _update_selected_model_scores(self):
        if not hasattr(self, 'model_score_summary'):
            return
        item = self.selected_item()
        if item is None:
            self.model_score_summary.setText('종목을 선택하면 모델별 매수 판정과 추정확률이 표시됩니다. 실제 수익률·적중률 보장 아님.')
            return
        snapshot = self.snapshots.get(item.id)
        quote = (f'{snapshot.quote.price} {item.instrument.currency} · {snapshot.fetched_at.astimezone():%m/%d %H:%M:%S}'
                 if snapshot is not None and item.id not in self.errors else '현재가 확인 불가')
        scores = ' · '.join(f'{PROTOTYPE_TITLES[model]} {self._model_score_display(model, item)[2]}'
                            for model in PROTOTYPE_NOTICES)
        self.model_score_summary.setText(
            f'{item.instrument.symbol} · 현재가 {quote}\n{scores} · 조회 시점 추정확률 / 주문 직전 재판단 · 실제 적중률·수익률 아님')

    def select_item(self, *_):
        super().select_item(*_)
        self._update_selected_model_scores()

    def _update_watch_row(self, key):
        super()._update_watch_row(key)
        self._update_model_score_row(key)
        self._update_selected_model_scores()

    def reload_tables(self, *, items=None, rules=None):
        super().reload_tables(items=items, rules=rules)
        valid_ids = set(getattr(self, '_items_by_id', {}))
        self._last_model_scores = {key: score for key, score in self._last_model_scores.items()
                                   if key[1] in valid_ids}
        if self._last_queried_watch_id not in valid_ids:
            self._last_queried_watch_id = None
        self._refresh_model_scores()

    def _move_legacy_settings(self):
        """Move controls only; opening this area never changes trading policy."""
        self.advanced_mode_panel = QWidget()
        options = QVBoxLayout(self.advanced_mode_panel)
        self.advanced_mode_note = QLabel(
            '일반 AI 자동매매는 신호 연결에서 모델을 선택하세요.\n'
            '수동 가격·이동평균 규칙은 AI 모델과 구형 신호기를 모두 끈 뒤 '
            '아래 AI·외부 신호 사용을 해제해야 실행됩니다. 자동주문 ON은 별도 확인이 필요합니다.')
        self.advanced_mode_note.setWordWrap(True)
        options.addWidget(self.advanced_mode_note)
        for widget in (self.external_mode, self.legacy_trigger_label, self.model_trigger):
            self.external_grid.removeWidget(widget)
            options.addWidget(widget)
        self.external_mode.setText('AI·외부 신호 사용 (해제하면 수동 규칙 사용)')
        self.external_mode.setToolTip('AI 모델이나 구형 신호기가 선택되어 있으면 자동으로 켜집니다. 감시·주문 중에는 변경할 수 없습니다.')
        options.addStretch(1)
        self.tabs.insertTab(0, self.advanced_mode_panel, '신호 방식·구형 모델')
        self.tabs.setCurrentWidget(self.advanced_mode_panel)
        index = self.workspace_tabs.indexOf(self.tabs)
        self.workspace_tabs.setTabText(index, '기타·고급')
        self.workspace_tabs.setTabVisible(index, False)
        footer = QHBoxLayout()
        footer.addStretch(1)
        self.advanced_settings_button = QPushButton('기타·고급 설정')
        self.advanced_settings_button.setCheckable(True)
        self.advanced_settings_button.setAutoDefault(False)
        self.advanced_settings_button.setObjectName('linkButton')
        self.advanced_settings_button.setToolTip('수동 가격·이동평균 규칙, 구형 LSTM, 모의 테스트 신호기. 열고 닫아도 실행 설정은 변하지 않습니다.')
        self.advanced_settings_button.toggled.connect(self._show_advanced_settings)
        footer.addWidget(self.advanced_settings_button)
        self.layout().addLayout(footer)
        self.external_mode.toggled.connect(self._update_connection)

    def _show_advanced_settings(self, visible):
        index = self.workspace_tabs.indexOf(self.tabs)
        if not visible and self.workspace_tabs.currentWidget() is self.tabs:
            self.workspace_tabs.setCurrentWidget(self.signal_connection_page)
        self.workspace_tabs.setTabVisible(index, visible)
        if visible:
            self.workspace_tabs.setCurrentWidget(self.tabs)
        self.advanced_settings_button.setText('고급 설정 닫기' if visible else '기타·고급 설정')

    def _close_wakeup(self):
        """A cheap Qt wakeup; all calendar/account work stays on the broker worker."""
        close = self.engine.close_liquidator
        if (self._close_when_idle or self._pending_environment or self._confirming_environment
                or self._confirming_orders or not self.monitoring or self.worker is not None
                or selected_mode(self.service) is not TradingMode.DEMO
                or not self.engine.orders_enabled or self.engine._stop.is_set()
                or close is None or not close.enabled):
            return
        engine = self.engine
        self._poll_after_maintenance = False
        def maintain(progress):
            try:
                engine.maintenance_checkpoint()
                progress(('close_liquidation', engine.close_liquidation_status()))
            except Exception as exc:
                engine.disarm()
                try:
                    engine.store.event('SYSTEM', f'마감 청산 점검 실패 · 자동주문 OFF: {exc}', category='system')
                except Exception:
                    pass  # A failed ledger must not prevent immediate OFF.
                raise
            return {}
        self._run(maintain, streaming=True, job_kind='close-maintenance',
                  done='마감 청산 주기 점검 완료 · 주문 접수와 체결은 다릅니다.')

    def _completed(self, results, error):
        if self._worker_kind != 'close-maintenance':
            return super()._completed(results, error)
        self._worker_kind = ''
        self.worker = None
        self._last_completed_at = utc_now()
        self._last_activity = monotonic()
        poll_due, self._poll_after_maintenance = self._poll_after_maintenance, False
        if error is not None:
            self.engine.disarm()
            self.set_pending_auto_arm(False)
            self._last_worker_error = str(error)
            self.message.setText(f'마감 청산 점검 실패 · 자동주문 OFF: {error}')
        else:
            self.message.setText(self._done_message)
        self.sweep_progress.set_activity('마감 청산 점검 실패 · 자동주문 OFF' if error is not None
                                         else '마감 청산 점검 완료 · 다음 일반 조회 시각은 유지',
                                         completed=int(error is None), total=1)
        self.update_controls()
        if self._close_when_idle:
            QTimer.singleShot(0, self.close)
        elif self._pending_environment:
            self._finish_environment_switch()
        elif self._manual_arm_pending and self.pending_auto_arm:
            # Only a separately confirmed ON request may start its ordinary
            # warmup here; maintenance completion itself never arms orders.
            self._begin_manual_warmup()
        elif poll_due and self.monitoring and not self.engine._stop.is_set():
            # A single-shot ordinary poll may have expired while this job ran.
            # Run that overdue poll now, without changing any future deadline.
            self.refresh_all()
        self._update_health()

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
        self._last_model_scores.clear()

    @property
    def builtin_confirmation_notice(self):
        return '\n\n'.join(PROTOTYPE_NOTICES[model] for model in self._chosen_external_models())

    @property
    def closing_confirmation_notice(self):
        if not hasattr(self, 'close_all_at_market_end') or not self.close_all_at_market_end.isChecked():
            return ''
        return ('장마감 청산 사용: 각 정규장 종료 5분 전부터 신규 매수를 차단합니다.\n'
                '이 실행기 이전 보유분을 포함한 모의계좌의 모든 국내·미국 보유종목의 실제 매도가능 수량을 매도합니다.\n'
                '마감 청산에 한해서 평소 1회 수량·금액 한도를 적용하지 않습니다. 보유수량 초과 매도는 금지합니다.\n'
                '휴장·서머타임·조기 폐장을 반영하며, 주문 접수나 체결을 보장하지 않습니다. 감시와 주문 ON이 필요합니다.')

    def _configure_close_policy(self):
        if hasattr(self, 'close_all_at_market_end'):
            if selected_mode(self.service) is TradingMode.REAL and self.close_all_at_market_end.isChecked():
                self.close_all_at_market_end.blockSignals(True)
                self.close_all_at_market_end.setChecked(False)
                self.close_all_at_market_end.blockSignals(False)
            self.engine.configure_close_liquidation(
                enabled=selected_mode(self.service) is TradingMode.DEMO and self.close_all_at_market_end.isChecked(),
                minutes_before_close=5)
            if hasattr(self, 'close_liquidation_label'):
                self.close_liquidation_label.setText(
                    '마감 청산 대기 · 모의계좌 전체 · 감시 및 주문 ON 필요 · 체결 보장 없음'
                    if self.close_all_at_market_end.isChecked() and selected_mode(self.service) is TradingMode.DEMO
                    else '계좌 전체 마감 청산 사용 안 함 · 일반 보유 매도 조건은 별도')

    def _close_policy_changed(self, *_):
        self.stop_monitoring()
        self._configure_close_policy()
        self._update_connection()

    def _progress(self, data):
        if len(data) == 2 and data[0] == 'close_liquidation':
            status = data[1]
            active = ', '.join(status.get('closing_markets', ())) or '대기'
            self.close_liquidation_label.setText(
                f"마감 청산 {active} · 남은 보유 {len(status.get('unsold', ()))}건 · "
                f"주문 기록 {len(status.get('attempts', ()))}건 · 오류 {len(status.get('errors', ()))}건 · 접수는 체결이 아님")
            self.close_liquidation_label.setToolTip(str(status))
            return
        result = super()._progress(data)
        if len(data) == 4 and not isinstance(data[1], Exception):
            self._last_queried_watch_id = data[0]
            self._update_latest_model_summary()
        return result

    def _label_inputs(self):
        for name, text in {
            'symbol_input': '추가할 종목 코드', 'exchange_input': '종목 시장과 거래소', 'days_input': '조회할 일봉 수',
            'interval': '조회 간격 초', 'external_source': '직접 연결 신호 출처 ID', 'signal_path': '입력 신호 JSON 경로',
            'chart_path': '출력 차트 JSON 경로', 'external_quantity': '주문당 최대 수량', 'external_krw': '국내 주문 금액 상한 원',
            'external_usd': '미국 주문 금액 상한 달러', 'buy_percent': '매수 비중 퍼센트', 'model_trigger': '기존 내장 트리거',
            'trigger': '수동 규칙 조건', 'side': '수동 규칙 매수 또는 매도', 'quantity': '수동 규칙 수량',
            'max_notional': '수동 규칙 금액 상한', 'threshold': '수동 규칙 기준값', 'period': '수동 규칙 기간',
            'random_us': '미국 랜덤 모의 주문 방식',
        }.items():
            getattr(self, name).setAccessibleName(text)

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
            self.request_workspace_reload(minimum_days=31)
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
        self._refresh_model_scores()

    def _sync_model_mode_controls(self):
        if not hasattr(self, 'model_trigger'):
            return
        if hasattr(self, 'close_all_at_market_end'):
            self.close_all_at_market_end.setEnabled(selected_mode(self.service) is TradingMode.DEMO)
        if self._chosen_external_models():
            self.environment_selector.buttons[TradingMode.REAL].setEnabled(False)
            self.environment_selector.buttons[TradingMode.REAL].setToolTip('prototype 외부 신호기는 모의투자 전용입니다. 모든 모델 연결을 끄면 환경 전환할 수 있습니다.')
        else:
            self.environment_selector.apply(selected_mode(self.service), pending=self._pending_environment is not None)
        if hasattr(self, 'advanced_mode_panel'):
            # These widgets used to inherit the external panel's editing lock.
            self.advanced_mode_panel.setEnabled(self.external_panel.isEnabled())
            self.external_mode.setEnabled(not (self._chosen_external_models()
                                               or self._chosen_trigger() != 'none'
                                               or self.random_demo.isChecked()))

    def request_environment(self, mode):
        if TradingMode(mode) is TradingMode.REAL and self._chosen_external_models():
            self.engine.disarm()
            self.message.setText('prototype 외부 신호기는 모의투자 전용입니다. 실전 전환 전에 모든 모델 연결을 끄세요.')
            return
        return super().request_environment(mode)

    def _sync_environment(self):
        super()._sync_environment()
        self._sync_model_mode_controls()

    def _external_config_key(self):
        return (*super()._external_config_key(), self._chosen_trigger(), self._chosen_external_models(),
                bool(getattr(self, 'close_all_at_market_end', None) and self.close_all_at_market_end.isChecked()))

    def connection_source_options(self, config):
        options = super().connection_source_options(config)
        models = config[13] if len(config) > 13 else self._chosen_external_models()
        prototypes = [{'source_id': PROTOTYPE_SOURCES[model],
                       'input_path': str(self._prototype_output_path(model)),
                       'label': PROTOTYPE_TITLES[model] + ' · 외부 AI'} for model in models]
        if prototypes and options[0]['source_id'] == 'external-model':
            options[0]['label'] = '직접 연결 JSON · external-model (선택한 모델과 별도)'
        return prototypes + options

    def _finish_environment_switch(self):
        if (self._chosen_external_models() and self._pending_environment
                and selected_mode(self._pending_environment[0]) is TradingMode.REAL):
            self._cancel_pending_environment()
            self.engine.disarm()
            self._sync_environment()
            self.message.setText('prototype 외부 연결로 실전 환경 전환을 취소했습니다. 모의·주문 OFF 유지')
            self.update_controls()
            return
        before = self.service
        super()._finish_environment_switch()
        if self.service is not before:
            self._last_model_scores.clear()
            self._last_queried_watch_id = None
            self.external_quantity.setValue(999999999)
            self._apply_execution_preferences()
            self.close_all_at_market_end.blockSignals(True)
            self.close_all_at_market_end.setChecked(selected_mode(self.service) is TradingMode.DEMO)
            self.close_all_at_market_end.setEnabled(selected_mode(self.service) is TradingMode.DEMO)
            self.close_all_at_market_end.blockSignals(False)
            self._configure_close_policy()
            self._refresh_model_scores()

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
                self.connection_summary.setText(f'외부 AI {len(self._chosen_external_models())}개 · {state} · {orders} · 모델별 연결 상태는 신호 연결에서 확인')
            self.model_status.setText(text)
            builtin = producer.builtin if isinstance(producer, ExternalFeedGroup) else producer
            if isinstance(builtin, DesktopModelBridge) and builtin._load_error:
                warning = '내장 LSTM 실행 불가 · 신호 연결에서 원인 확인 · 다른 연결은 별도 운영'
                self.connection_summary.setText(
                    self.connection_summary.text() + '\n' + warning
                    if self._chosen_external_models() else warning)

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
        self._configure_close_policy()
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
            from dockdack.mark1_trigger import SharedPrototypeAccounts
            factory = self._external_feed_factory or ExternalPrototypeFeed
            shared = SharedPrototypeAccounts(self)
            policy = result[2]
            try:
                for model in self._chosen_external_models():
                    source, path = PROTOTYPE_SOURCES[model], self._prototype_output_path(model)
                    if source in seen_sources or path.resolve() in seen_paths:
                        raise ValueError('외부 AI 신호 출처/파일이 다른 연결과 중복됩니다.')
                    extra = ExternalPolicy(source, policy.max_quantity, policy.max_krw, policy.max_usd)
                    bundle = {MARK1_TRIGGER: self._mark1_bundle,
                              MARK11_TRIGGER: self._mark11_bundle,
                              MARK12_TRIGGER: self._mark12_bundle}[model]
                    kwargs = {'bundle_root': bundle}
                    if self._external_feed_factory is None:
                        kwargs['account_snapshots'] = shared
                    feed = factory(self, model, extra, path, **kwargs)
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
        self._refresh_model_scores()
        return result

    def closeEvent(self, event):
        if hasattr(self, 'close_maintenance_timer'):
            self.close_maintenance_timer.stop()
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
        # New research models are opt-in; the existing two-model startup
        # selection remains unchanged and still starts with orders OFF.
        return 'none', [MARK1_TRIGGER, MARK11_TRIGGER]
    if trigger is None and models:
        return 'none', models
    return trigger, models


def prepare_legacy_ledger(service, legacy, destination):
    """Before creating a new scoped ledger, obtain explicit copy/empty consent.

    Returns a visible notice, or None to cancel startup. Migration itself holds
    the legacy session lock and preserves its source; this never guesses ownership.
    """
    if destination.exists() or not legacy.exists() or legacy.resolve() == destination.resolve():
        return ''
    if selected_mode(service) is not TradingMode.DEMO:
        return ''
    scope = getattr(service, 'storage_scope', '')
    if len(scope) != 64 or any(char not in '0123456789abcdef' for char in scope):
        return '모의계정 인증 범위 미설정 · 기존 장부는 보존되며 계정 확인 전에는 이관할 수 없습니다.'
    answer = QMessageBox.question(None, '기존 모의 장부 소유 확인 · 자동주문 OFF',
        f'기존 장부: {legacy}\n새 계정 장부: {destination}\n\n'
        '예: 기존 장부가 현재 모의계정 및 같은 계좌 리셋 세대의 기록임을 직접 확인했습니다. '
        '체결·미확정 주문·모델별 보유 기록을 새 경로에 복사합니다. 원본은 보존합니다.\n\n'
        '아니오: 새 빈 장부로 시작합니다. 기존 주문·체결·모델별 원가 기록이 빠져 자동매도가 보류될 수 있습니다. '
        '이미 만든 대상 장부에는 나중에 덮어쓰기 이관할 수 없습니다.\n\n'
        '소유 여부나 계좌 초기화 여부가 불확실하면 취소하고 확인하세요. API 키만으로 장부 소유를 추정하지 않습니다.',
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel)
    if answer == QMessageBox.StandardButton.Cancel:
        return None
    if answer == QMessageBox.StandardButton.Yes:
        from dockdack.persistence.account_migration import migrate_legacy_demo, CONFIRM_LEGACY_DEMO
        migrate_legacy_demo(legacy, destination, scope, confirmation=CONFIRM_LEGACY_DEMO)
        return '사용자가 소유 확인한 기존 모의 장부를 복사했습니다. 원본 보존 · 계정별 장부 · 자동주문 OFF'
    return '사용자 선택으로 새 빈 계정 장부 사용 · 기존 장부 보존 · 이전 체결·보유 기록은 포함하지 않습니다.'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    trigger_options = parser.add_mutually_exclusive_group()
    trigger_options.add_argument('--no-model', action='store_true')
    trigger_options.add_argument('--trigger', choices=('none', 'lstm30', *PROTOTYPE_NOTICES))
    parser.add_argument('--mark1-bundle', type=Path, help='mark1 prototype 저장 모델 폴더')
    parser.add_argument('--mark11-bundle', type=Path, help='mark1.1 prototype 저장 모델 폴더')
    parser.add_argument('--mark12-bundle', type=Path, help='mark1.2 prototype 저장 모델 폴더')
    parser.add_argument('--external-model', action='append', choices=tuple(PROTOTYPE_NOTICES), default=[],
                        help='외부 AI 신호기. 세 모델을 함께 쓰려면 각각 지정 (자동주문은 OFF)')
    ledger_options = parser.add_mutually_exclusive_group()
    ledger_options.add_argument('--store', type=Path, help='현재 계정에 이미 귀속된 장부 경로 (범위 검사 유지, 이관하지 않음)')
    ledger_options.add_argument('--legacy-store', type=Path,
                                help='이전 공용 모의 장부 경로 (소유 확인 후 계정별 경로로 복사, 원본 보존)')
    parser.add_argument('--env-file', type=Path, help='기존 키 설정 파일 경로 (복사하지 않음)')
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    from dockdack.lstm30_runtime import SessionLock
    from dockdack.runtime_paths import app_home
    root = app_home()
    load_dotenv(args.env_file.expanduser() if args.env_file else root / '.env', override=False)
    set_windows_app_id()
    app = QApplication.instance() or QApplication(sys.argv)
    apply_branding(app)
    app.setStyle('Fusion')
    app.setFont(QFont('Malgun Gothic', 9))
    service = TradingService()
    legacy = args.legacy_store.expanduser().resolve() if args.legacy_store else default_desktop_path(root)
    if args.legacy_store and not legacy.is_file():
        QMessageBox.warning(None, '기존 장부 확인 실패 · 시작 취소', f'기존 장부 파일이 없습니다: {legacy}')
        return 1
    base_folder = legacy.parent
    path = args.store.expanduser().resolve() if args.store else scoped_store_path(service, base_folder=base_folder)
    lock = SessionLock(path.parent / 'session.lock')
    try:
        lock.acquire()
    except RuntimeError as exc:
        QMessageBox.warning(None, 'DOCKDACK 이미 실행 중', str(exc))
        return 1
    try:
        try:
            legacy_notice = prepare_legacy_ledger(service, legacy, path) if not args.store else ''
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(None, '기존 장부 이관 실패 · 시작 취소', str(exc))
            return 1
        if legacy_notice is None:
            return 0
        try:
            store = (WatchStore(path, seed_defaults=True, mode=selected_mode(service), storage_scope=service.storage_scope)
                     if args.store else store_for_service(service, base_folder=base_folder))
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(None, '계정 장부 열기 실패 · 시작 취소', str(exc))
            return 1
        trigger, models = desktop_model_choices(args.trigger, args.no_model, args.external_model)
        window = V00Window(service=service, store=store, builtin=not args.no_model,
                           trigger=trigger, mark1_bundle=args.mark1_bundle,
                           mark11_bundle=args.mark11_bundle, mark12_bundle=args.mark12_bundle,
                           external_models=models, session_lock=lock)
        if legacy_notice:
            window.message.setText(legacy_notice)
        window.show()
        return app.exec()
    finally:
        lock.release()


if __name__ == '__main__':
    raise SystemExit(main())
