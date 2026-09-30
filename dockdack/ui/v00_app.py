"""Unified desktop: optional LSTM plus multiple data-only signal feeds.

Always starts OFF. Models initialize lazily on the existing broker worker, not
the GUI thread. This replaces the old DEMO-only shortcut, not its ledger.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHeaderView,
    QHBoxLayout, QMessageBox, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from dockdack.branding import apply_branding, set_windows_app_id
from dockdack.version import APP_RELEASE
from dockdack.environment_store import selected_mode, scoped_store_path, store_for_service
from dockdack.gui_service import Instrument, TradingService
from dockdack.lstm30_adapter import MAX_QUOTE_AGE
from dockdack.minute_model_catalog import (MINUTE_HEDGE_IDS, MINUTE_RESEARCH_BY_ID,
                                           MINUTE_RESEARCH_IDS, MINUTE_TRANSFER_IDS)
from dockdack.models import Market, TradingMode
from dockdack.portfolio import PortfolioCache
from dockdack.signal_bridge import atomic_json, ExternalPolicy, SignalFileReader
from dockdack.signals.mark1_intraday_trigger import MODEL_IDS as INTRADAY_MODEL_IDS, RISK_NOTICE as INTRADAY_RISK_NOTICE
from dockdack.signals.mark1_target_horizon_trigger import (
    MODEL_IDS as TARGET_HORIZON_MODEL_IDS, MODEL_SPECS as TARGET_HORIZON_MODEL_SPECS,
    RISK_NOTICE as TARGET_HORIZON_RISK_NOTICE, SCORE_SCOPE as TARGET_HORIZON_SCORE_SCOPE,
)
from dockdack.signals.preopen_series import PREOPEN_MODELS
from dockdack.trading.model_exit_schedule import timed_exit_due
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
MARK14_TRIGGER = 'mark1-4-prototype'
BARRIER_MODELS = (MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER, *INTRADAY_MODEL_IDS)
QUOTE_SCORE_MODELS = (*BARRIER_MODELS, *TARGET_HORIZON_MODEL_IDS)
PREOPEN_MODEL_IDS = (MARK14_TRIGGER, *PREOPEN_MODELS)
ALL_MODEL_IDS = tuple(sorted((*QUOTE_SCORE_MODELS, *PREOPEN_MODEL_IDS, *MINUTE_RESEARCH_IDS),
                             key=lambda model: int(model.split('-')[1]) if model != MARK1_TRIGGER else 0))
WATCH_MODEL_IDS = (tuple(model for model in ALL_MODEL_IDS if model in QUOTE_SCORE_MODELS)
                   + tuple(model for model in ALL_MODEL_IDS if model in PREOPEN_MODEL_IDS))
TARGET_HORIZON_FAMILIES = dict(zip(TARGET_HORIZON_MODEL_IDS,
                                   ('GRU', 'MLP', 'GRU', '선형망', 'CNN', 'MLP')))
MARK1_NOTICE = (
    'mark1.0 prototype · 30일봉 특징을 CatBoost 3개 시드로 학습 · 현재가에서 +1%/−0.9% 선후 도달 확률 추정\n'
    '확률 50% 초과 시 모의 매수 후보 · 해당 매수분 +1% 익절/−0.9% 손절 · 기존 보유분 기준 유지'
)
MARK11_NOTICE = (
    'mark1.1 prototype · 30일봉 특징을 CatBoost 3개 시드로 별도 학습 · 현재가에서 +0.5%/−0.4% 선후 도달 확률 추정\n'
    '확률 50% 초과 시 모의 매수 후보 · 해당 매수분 +0.5% 익절/−0.4% 손절 · 기존 보유분 기준 유지'
)
MARK12_NOTICE = (
    'mark1.2 prototype · 가상 매수가를 증강한 30일봉 신경망 · 국내 CNN / 미국 LSTM 3개 시드 앙상블\n'
    '현재가에서 +1%/−0.9% 선후 도달 확률 50% 초과 시 모의 매수 후보 · 해당 매수분의 기준은 +1%/−0.9%'
)
MARK14_NOTICE = (
    'mark1.4 prototype · 국내: 완료 30일봉의 추세·변동성·거래량/유동성 압축 특징을 학습한 신경망. '
    '미국: 같은 날 100종목의 다음 거래일 상대 순수익 순위를 학습한 신경망.\n'
    '개장 10분 전 점수·후보 동결, 숫자 기준 초과 상위 10종목만 개장 후 5분 모의 매수 후보. 점수는 확률이 아닙니다.\n'
    '1회 매수 비중은 모델 선택 화면 설정과 주문 상한을 적용 · Mark1.4 실제 체결로 확인된 보유분만 해당 거래일 마감 5분 전부터 가격 무관 매도 시도. 계좌 전체 청산은 하지 않습니다.'
)
PROTOTYPE_NOTICES = {MARK1_TRIGGER: MARK1_NOTICE, MARK11_TRIGGER: MARK11_NOTICE,
                     MARK12_TRIGGER: MARK12_NOTICE, MARK14_TRIGGER: MARK14_NOTICE}
PROTOTYPE_NOTICES.update({model: spec.strategy_notice + '\n' + spec.risk_notice
                          for model, spec in PREOPEN_MODELS.items()})
PROTOTYPE_NOTICES.update({model: '완료 30일봉+현재가의 독립 일봉 대리점수 · 점수 0.5 초과 시에만 모의 매수 후보\n'
                          + INTRADAY_RISK_NOTICE for model in INTRADAY_MODEL_IDS})
PROTOTYPE_NOTICES.update({
    model: (f'완료 {lookback}일봉 + 현재가 · {TARGET_HORIZON_FAMILIES[model]} · '
            f'목표 +{target:g}% / {horizon}거래 세션 · 손절 없음\n'
            '다음 세션 관측 시가로 학습한 점수를 장중 현재가로 조회합니다. '
            f'목표 미도달 시 매수 체결일 포함 {horizon}번째 거래 세션 마감 5분 전부터 매도 시도합니다. '
            '목표 도달 확률·실제 지정가 체결·수익성은 검증되지 않았습니다.\n'
            + TARGET_HORIZON_RISK_NOTICE)
    for model, (lookback, horizon, target) in TARGET_HORIZON_MODEL_SPECS.items()})
PROTOTYPE_NOTICES.update({
    model: (f'{spec.title} · {spec.method} · {spec.output}. '
            '모의 매수·매도 후보는 완료된 분봉, 적격 모델 묶음과 주문 안전조건을 통과할 때만 발생합니다. '
            '실전 주문은 차단되며 분봉 수익성은 검증되지 않았습니다.')
    for model, spec in MINUTE_RESEARCH_BY_ID.items() if model in MINUTE_TRANSFER_IDS})
PROTOTYPE_NOTICES.update({
    model: (f'{spec.title} · {spec.method} · {spec.output}. '
            '연구 연결·주문 보류: 두 다리 실행 정책이 검증될 때까지 '
            '실전·모의 주문 모두 전송하지 않습니다.')
    for model, spec in MINUTE_RESEARCH_BY_ID.items() if model in MINUTE_HEDGE_IDS})
PROTOTYPE_TITLES = {MARK1_TRIGGER: 'mark1.0 prototype', MARK11_TRIGGER: 'mark1.1 prototype',
                    MARK12_TRIGGER: 'mark1.2 prototype', MARK14_TRIGGER: 'mark1.4 prototype'}
PROTOTYPE_TITLES.update({model: spec.title for model, spec in PREOPEN_MODELS.items()})
PROTOTYPE_TITLES.update({model: model.replace('mark1-', 'mark1.').replace('-prototype', ' prototype')
                         for model in INTRADAY_MODEL_IDS})
PROTOTYPE_TITLES.update({model: model.replace('mark1-', 'mark1.').replace('-prototype', ' prototype')
                         for model in TARGET_HORIZON_MODEL_IDS})
PROTOTYPE_TITLES.update({model: spec.title for model, spec in MINUTE_RESEARCH_BY_ID.items()})
PROTOTYPE_TARGETS = {MARK1_TRIGGER: '+1% / −0.9%', MARK11_TRIGGER: '+0.5% / −0.4%',
                     MARK12_TRIGGER: '+1% / −0.9%', MARK14_TRIGGER: '다음 날 시가→종가 점수'}
PROTOTYPE_TARGETS.update({model: spec.output for model, spec in PREOPEN_MODELS.items()})
PROTOTYPE_TARGETS.update({model: '일봉 전체 +1% / −0.9% 대리점수' for model in INTRADAY_MODEL_IDS})
PROTOTYPE_TARGETS.update({model: f'목표 +{target:g}% / {horizon}세션 시가 대리점수'
                          for model, (_, horizon, target) in TARGET_HORIZON_MODEL_SPECS.items()})
PROTOTYPE_TARGETS.update({model: spec.output for model, spec in MINUTE_RESEARCH_BY_ID.items()})
PROTOTYPE_SOURCES = {MARK1_TRIGGER: 'mark1-prototype-demo-trigger',
                     MARK11_TRIGGER: 'mark1-1-prototype-demo-trigger',
                     MARK12_TRIGGER: 'mark1-2-prototype-demo-trigger',
                     MARK14_TRIGGER: 'mark1-4-prototype-demo-trigger'}
PROTOTYPE_SOURCES.update({model: spec.source_id for model, spec in PREOPEN_MODELS.items()})
PROTOTYPE_SOURCES.update({model: model + '-demo-trigger' for model in INTRADAY_MODEL_IDS})
PROTOTYPE_SOURCES.update({model: model + '-demo-trigger' for model in TARGET_HORIZON_MODEL_IDS})
PROTOTYPE_SOURCES.update({model: spec.source_id for model, spec in MINUTE_RESEARCH_BY_ID.items()})
PROTOTYPE_METHODS = {
    MARK1_TRIGGER: '30일봉 특징 · CatBoost 3개 시드 앙상블',
    MARK11_TRIGGER: '30일봉 특징 · 별도 CatBoost 3개 시드 앙상블',
    MARK12_TRIGGER: '가상 매수가 증강 30일봉 · 국내 CNN / 미국 LSTM 3개 시드',
    MARK14_TRIGGER: ('국내: 추세·변동성·유동성 압축 특징 신경망 / '
                     '미국: 같은 날 100종목의 다음 날 상대 수익 순위 신경망'),
}
PROTOTYPE_METHODS.update({model: spec.method for model, spec in PREOPEN_MODELS.items()})
PROTOTYPE_METHODS.update(dict(zip(INTRADAY_MODEL_IDS, (
    '완료 30일봉 추세 지속 특징 · 독립 MLP',
    '완료 30일봉 과매도·반전 특징 · 독립 MLP',
    '완료 30일봉 고저점 돌파 특징 · 독립 MLP',
    '완료 30일봉 변동성 국면 특징 · 독립 MLP',
    '완료 30일봉 거래량 압력 특징 · 독립 MLP',
    '완료 일봉의 초단기 반전 특징 · 독립 SiLU MLP',
    '완료 일봉의 캔들 위치·갭 특징 · 독립 Tanh MLP',
    '완료 일봉의 추세 가속 특징 · 독립 GELU MLP',
    '완료 일봉의 변동성×거래량 특징 · 독립 게이트망',
    '완료 일봉의 압축·확장 특징 · 독립 ReLU 심층망',
))))
PROTOTYPE_METHODS.update({model: f'완료 {lookback}봉+현재가 6채널 · {TARGET_HORIZON_FAMILIES[model]}'
                          for model, (lookback, _, _) in TARGET_HORIZON_MODEL_SPECS.items()})
PROTOTYPE_METHODS.update({model: spec.method for model, spec in MINUTE_RESEARCH_BY_ID.items()})
PROTOTYPE_EXITS = {
    MARK1_TRIGGER: '매수분 +1% / −0.9%',
    MARK11_TRIGGER: '매수분 +0.5% / −0.4%',
    MARK12_TRIGGER: '매수분 +1% / −0.9%',
    MARK14_TRIGGER: '매수 체결 당일 장마감 5분 전',
}
PROTOTYPE_EXITS.update({model: spec.horizon for model, spec in PREOPEN_MODELS.items()})
PROTOTYPE_EXITS.update({model: '해당 모델 매수분 +1% / −0.9%' for model in INTRADAY_MODEL_IDS})
PROTOTYPE_EXITS.update({
    model: f'해당 매수분 +{target:g}% 목표 · 손절 없음 · {horizon}번째 거래 세션 마감 5분 전부터 매도 시도'
    for model, (_, horizon, target) in TARGET_HORIZON_MODEL_SPECS.items()})
PROTOTYPE_EXITS.update({
    model: f'체결 후 {MINUTE_RESEARCH_BY_ID[model].horizon_bars}개 5분봉 시한 · +3% / −2% · 장마감 전 청산'
    for model in MINUTE_TRANSFER_IDS})
PROTOTYPE_EXITS.update({model: '연구 연결·주문 보류' for model in MINUTE_HEDGE_IDS})


def _model_group(model):
    if model in MINUTE_TRANSFER_IDS:
        return 'minute_transfer'
    if model in MINUTE_HEDGE_IDS:
        return 'minute_hedge'
    if model in TARGET_HORIZON_MODEL_IDS:
        return 'target_horizon'
    if model in INTRADAY_MODEL_IDS:
        return 'proxy'
    if model in BARRIER_MODELS:
        return 'probability'
    return 'preopen'


class Mark14Panel(QWidget):
    """One model selection surface; pre-open scores retain their own units."""

    def __init__(self, enabled_models, parent=None, *, clock=None):
        super().__init__(parent)
        self._clock = clock
        self.setObjectName('mark14Panel')
        layout = QVBoxLayout(self)
        self.heading = QLabel('모델 선택 · 선택한 모델은 각자 독립 신호를 냅니다')
        self.heading.hide()
        selection_toolbar = QHBoxLayout()
        selection_toolbar.setContentsMargins(2, 0, 2, 3)
        selection_toolbar.setSpacing(8)
        selection_title = QLabel('모델 선택')
        selection_title.setStyleSheet('font-size: 16px; font-weight: 700; color: #e9f5ff;')
        selection_toolbar.addWidget(selection_title)
        self.selection_count = QLabel()
        self.selection_count.setObjectName('modelSelectionCount')
        self.selection_count.setStyleSheet('font-size: 12px; font-weight: 600; color: #9bdacb;')
        selection_toolbar.addWidget(self.selection_count)
        selection_toolbar.addStretch()
        self.select_all_button = QPushButton('전체 선택')
        self.select_all_button.setObjectName('selectAllModels')
        self.select_all_button.setAccessibleName('모든 AI 모델 선택')
        self.clear_all_button = QPushButton('선택 해제')
        self.clear_all_button.setObjectName('clearAllModels')
        self.clear_all_button.setAccessibleName('모든 AI 모델 선택 해제')
        for button in (self.select_all_button, self.clear_all_button):
            button.setMinimumHeight(30)
            button.setStyleSheet(
                'QPushButton { background: #1c3547; color: #dcf4f6; border: 1px solid #3a6271; '
                'border-radius: 5px; font-size: 12px; font-weight: 600; padding: 3px 11px; } '
                'QPushButton:hover { background: #285467; border-color: #65cbb5; } '
                'QPushButton:disabled { background: #1a2939; color: #748e9c; border-color: #2c4050; }')
            selection_toolbar.addWidget(button)
        layout.addLayout(selection_toolbar)
        sizing_toolbar = QHBoxLayout()
        sizing_toolbar.setContentsMargins(2, 0, 2, 3)
        sizing_toolbar.setSpacing(8)
        sizing_title = QLabel('1회 매수 일괄')
        sizing_title.setStyleSheet('font-size: 12px; font-weight: 600; color: #a9c9d8;')
        sizing_toolbar.addWidget(sizing_title)
        self.bulk_buy_percent = QDoubleSpinBox()
        self.bulk_buy_percent.setObjectName('bulkModelBuyPercent')
        self.bulk_buy_percent.setAccessibleName('모든 주문 모델의 1회 매수 비중 퍼센트')
        self.bulk_buy_percent.setRange(0.01, 100)
        self.bulk_buy_percent.setDecimals(2)
        self.bulk_buy_percent.setSingleStep(0.5)
        self.bulk_buy_percent.setKeyboardTracking(False)
        self.bulk_buy_percent.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.bulk_buy_percent.setValue(1)
        self.bulk_buy_percent.setSuffix(' %')
        self.bulk_buy_percent.setFixedSize(96, 30)
        self.bulk_buy_percent.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.bulk_buy_percent.setStyleSheet(
            'QDoubleSpinBox { background: #10283b; color: #e9f5ff; border: 1px solid #42657a; '
            'border-radius: 5px; font-size: 13px; font-weight: 600; padding-left: 4px; } '
            'QDoubleSpinBox:disabled { color: #748e9c; border-color: #2c4050; }')
        sizing_toolbar.addWidget(self.bulk_buy_percent)
        self.apply_bulk_buy_percent_button = QPushButton('일괄 적용')
        self.apply_bulk_buy_percent_button.setObjectName('applyBulkModelBuyPercent')
        self.apply_bulk_buy_percent_button.setAccessibleName('모든 주문 모델의 1회 매수 비중 일괄 적용')
        self.apply_bulk_buy_percent_button.setMinimumHeight(30)
        self.apply_bulk_buy_percent_button.setStyleSheet(self.select_all_button.styleSheet())
        bulk_notice = ('선택 여부와 관계없이 모든 주문 모델의 1회 매수 비중에 적용합니다. '
                       '주문 불가인 인버스 헤지 연구 모델은 제외합니다. 모델 선택을 바꾸거나 자동주문을 시작하지 않습니다.')
        self.bulk_buy_percent.setToolTip(bulk_notice)
        self.apply_bulk_buy_percent_button.setToolTip(bulk_notice)
        sizing_toolbar.addWidget(self.apply_bulk_buy_percent_button)
        sizing_toolbar.addStretch()
        layout.addLayout(sizing_toolbar)
        choices = QWidget()
        choices.setStyleSheet('background: #121b2a; color: #e9f5ff;')
        choice_stack = QVBoxLayout(choices)
        choice_stack.setContentsMargins(7, 7, 7, 7)
        choice_stack.setSpacing(6)
        self.model_checks = {}
        self.model_buy_percents = {}
        self.model_descriptions = {}
        self.model_detail_fields = {}
        self.model_sections = {}
        self._model_detail_rows = []
        self._detail_columns = None
        previous_group = None
        for model in ALL_MODEL_IDS:
            group = _model_group(model)
            if group != previous_group:
                count = sum(_model_group(item) == group for item in ALL_MODEL_IDS)
                heading_text = {'probability': '장중 · 확률 모델',
                                'preopen': '장전 · 점수 모델',
                                'proxy': '장중 · 일봉 대리 모델',
                                'target_horizon': '장중 · 목표가/보유기간 연구',
                                'minute_transfer': '장중 · 일봉 학습 → 5분봉 추론',
                                'minute_hedge': '장중 · 지수 인버스 헤지 연구'}[group]
                section = QLabel(f'{heading_text}  {count}개')
                section.setObjectName('modelSection-' + group)
                section.setStyleSheet('font-size: 12px; font-weight: 700; color: #9bdacb; '
                                      'padding: 7px 5px 2px; border-bottom: 1px solid #355365;')
                choice_stack.addWidget(section)
                self.model_sections[group] = section
                previous_group = group
            phase = ('5분' if model in MINUTE_RESEARCH_IDS else
                     '장중' if model in QUOTE_SCORE_MODELS else '장전')
            card = QFrame()
            card.setObjectName('modelChoiceCard')
            card.setStyleSheet('QFrame#modelChoiceCard { background: #1a293a; border: 1px solid #30465b; '
                               'border-radius: 6px; }')
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(11, 8, 11, 8)
            card_layout.setSpacing(4)
            card_header = QHBoxLayout()
            card_header.setSpacing(8)
            check = QCheckBox(PROTOTYPE_TITLES[model])
            check.setObjectName('external-' + model)
            check.setChecked(model in enabled_models)
            check.setToolTip(PROTOTYPE_NOTICES[model])
            check.setStyleSheet('QCheckBox { font-size: 15px; font-weight: 600; color: #e9f5ff; }')
            card_header.addWidget(check)
            card_header.addStretch()
            phase_label = QLabel(phase)
            phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            phase_label.setFixedWidth(44)
            phase_label.setStyleSheet('font-size: 12px; font-weight: 600; color: #aee4d4; '
                                      'background: #254653; border-radius: 4px; padding: 2px;')
            card_header.addWidget(phase_label)
            size_label = QLabel('1회 매수')
            size_label.setStyleSheet('font-size: 12px; color: #a9c9d8;')
            card_header.addWidget(size_label)
            buy_percent = QDoubleSpinBox(card)
            buy_percent.setObjectName('buy-percent-' + model)
            buy_percent.setAccessibleName(f'{PROTOTYPE_TITLES[model]} 1회 매수 비중 퍼센트')
            buy_percent.setToolTip(
                '해당 모델의 한 번의 매수 예산 상한입니다. 시장별 평가자산 기준이며 주문당 금액·가용액 상한도 적용합니다. '
                '모델별 비중은 독립적이므로 여러 모델이 같은 종목을 선택하면 총 노출이 합산될 수 있습니다.')
            buy_percent.setRange(0.01, 100)
            buy_percent.setDecimals(2)
            buy_percent.setSingleStep(0.5)
            buy_percent.setKeyboardTracking(False)
            buy_percent.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            buy_percent.setValue(1 if model in MINUTE_TRANSFER_IDS else 10)
            buy_percent.setSuffix(' %')
            buy_percent.setFixedSize(96, 27)
            buy_percent.setAlignment(Qt.AlignmentFlag.AlignRight)
            buy_percent.setStyleSheet(
                'QDoubleSpinBox { background: #10283b; color: #e9f5ff; '
                'border: 1px solid #42657a; border-radius: 5px; '
                'font-size: 13px; font-weight: 600; padding-left: 4px; } '
                'QDoubleSpinBox:disabled { color: #89a5b6; border-color: #30465b; }')
            card_header.addWidget(buy_percent)
            if model in MINUTE_HEDGE_IDS:
                # A two-leg hedge is research-only until both legs have a
                # verified execution policy. Do not suggest it has a usable
                # single-model buy allocation in this screen.
                size_label.hide()
                buy_percent.hide()
            card_layout.addLayout(card_header)
            description = QLabel(
                f'학습: {PROTOTYPE_METHODS[model]} · 출력: {PROTOTYPE_TARGETS[model]} · '
                f'매도: {PROTOTYPE_EXITS[model]}', card)
            description.setObjectName('method-' + model)
            description.setToolTip(PROTOTYPE_NOTICES[model])
            description.hide()  # Semantic text for accessibility and compatibility; visible fields are separate.
            detail_grid = QGridLayout()
            detail_grid.setContentsMargins(0, 2, 0, 0)
            detail_grid.setHorizontalSpacing(6)
            detail_grid.setVerticalSpacing(6)
            fields = []
            for caption, value in (('학습', PROTOTYPE_METHODS[model]),
                                   ('출력', PROTOTYPE_TARGETS[model]),
                                   ('매도', PROTOTYPE_EXITS[model])):
                field = QFrame()
                field.setObjectName('modelDetailField')
                field.setStyleSheet('QFrame#modelDetailField { background: #22364a; '
                                    'border: 1px solid #314e64; border-radius: 4px; }')
                field_layout = QVBoxLayout(field)
                field_layout.setContentsMargins(8, 5, 8, 6)
                field_layout.setSpacing(3)
                heading = QLabel(caption)
                heading.setStyleSheet('font-size: 11px; font-weight: 600; color: #80dbcd;')
                field_layout.addWidget(heading)
                value_label = QLabel(value)
                value_label.setWordWrap(True)
                value_label.setStyleSheet('font-size: 13px; color: #e9f5ff;')
                field_layout.addWidget(value_label)
                fields.append(field)
            card_layout.addLayout(detail_grid)
            self._model_detail_rows.append((detail_grid, fields))
            choice_stack.addWidget(card)
            self.model_checks[model] = check
            self.model_buy_percents[model] = buy_percent
            self.model_descriptions[model] = description
            self.model_detail_fields[model] = fields
        choice_stack.addStretch()
        for check in self.model_checks.values():
            check.toggled.connect(self._update_selection_count)
        self._update_selection_count()
        self._reflow_model_details()
        self.enable_check = self.model_checks[MARK14_TRIGGER]
        self.barrier_checks = {model: self.model_checks[model] for model in BARRIER_MODELS}
        self.extra_checks = {}
        for model in PREOPEN_MODELS:
            self.extra_checks[model] = self.model_checks[model]
        choice_list = QScrollArea()
        choice_list.setObjectName('allModelChoices')
        choice_list.setStyleSheet('QScrollArea { background: #121b2a; border: 1px solid #2a455c; border-radius: 5px; }')
        choice_list.setWidgetResizable(True)
        choice_list.setWidget(choices)
        choice_list.setMinimumHeight(180)
        self.choices_scroll = choice_list
        layout.addWidget(choice_list, 1)
        self.notice = QLabel(
            '모의계좌 모델 신호 · 실전 모델 주문 차단 · 자동주문 ON은 별도. '
            '장전 점수와 장중 추정확률은 단위가 달라 합산하지 않습니다.')
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet('color: #ffda91;')
        self.notice.setToolTip(MARK14_NOTICE)
        self.notice.hide()
        self.status = QLabel('—')
        self.status.setObjectName('mark14PreopenStatus')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('background: #183047; color: #c6e9f1; padding: 8px; border-radius: 5px;')
        layout.addWidget(self.status)
        self.status.hide()
        self.model_results = {}
        self.model_selector = QComboBox()
        self.model_selector.setObjectName('preopenModelResults')
        for model in ALL_MODEL_IDS:
            self.model_selector.addItem(PROTOTYPE_TITLES[model], model)
        self.model_selector.currentIndexChanged.connect(self._show_selected_model)
        layout.addWidget(self.model_selector)
        self.selected_result = QLabel('선택 종목 —')
        self.selected_result.setObjectName('selectedModelResult')
        self.selected_result.setWordWrap(True)
        self.selected_result.setStyleSheet('background: #193148; color: #e9f5ff; padding: 7px; border-radius: 6px;')
        layout.addWidget(self.selected_result)
        self.tables = {}
        self.market_tabs = QTabWidget()
        for market, title in (('domestic', '국내 · 예상 순수익률 점수'),
                              ('us', '미국 · 상대순위 점수')):
            table = QTableWidget(0, 5)
            table.setObjectName('mark14-' + market)
            table.setHorizontalHeaderLabels(('종목', '점수', '진입기준', '장전 판정', '학습 비중'))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.market_tabs.addTab(table, title)
            self.tables[market] = table
        layout.addWidget(self.market_tabs, 1)
        self.model_selector.setCurrentIndex(self.model_selector.findData(MARK14_TRIGGER))
        # A prepared plan is only actionable for its own exchange session.
        # Keep older scores for inspection, but never present them as today's candidates.
        self.session_display_timer = QTimer(self)
        self.session_display_timer.setInterval(60_000)
        self.session_display_timer.timeout.connect(self._show_selected_model)
        self.session_display_timer.start()

    def _update_selection_count(self, *_):
        count = sum(check.isChecked() for check in self.model_checks.values())
        self.selection_count.setText(f'{count} / {len(self.model_checks)}개 선택')

    def _reflow_model_details(self):
        columns = 3 if self.width() >= 1050 else 2 if self.width() >= 690 else 1
        if columns == self._detail_columns:
            return
        self._detail_columns = columns
        for grid, fields in self._model_detail_rows:
            while grid.count():
                grid.takeAt(0)
            for index, field in enumerate(fields):
                if columns == 2 and index == 2:
                    grid.addWidget(field, 1, 0, 1, 2)
                else:
                    grid.addWidget(field, index // columns, index % columns)
            for index in range(3):
                grid.setColumnStretch(index, 1 if index < columns else 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_model_detail_rows'):
            self._reflow_model_details()

    def _result_is_current_session(self, market, result):
        try:
            from dockdack.history import market_time
            from dockdack.market_schedule import session_on
            market_enum = Market(market)
            now = self._clock()
            session = session_on(market_enum, market_time(market_enum, now).date())
            opened = datetime.fromisoformat(str(result['session_open']))
            return session is not None and opened == session.opened
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def show_preopen(self, result):
        if not isinstance(result, dict):
            return
        model = result.get('model_id', MARK14_TRIGGER)
        if model not in PREOPEN_MODEL_IDS:
            return
        market = result.get('market')
        if market not in self.tables:
            return
        self.model_results[(model, market)] = result
        if self.model_selector.currentData() == model:
            self._show_selected_model()

    def _show_selected_model(self, *_):
        model = self.model_selector.currentData()
        self.notice.setToolTip(PROTOTYPE_NOTICES.get(model, ''))
        if model in QUOTE_SCORE_MODELS:
            measure = ('시가 기준 일봉 대리점수 · 장중 검증 전' if model in TARGET_HORIZON_MODEL_IDS else
                       '일봉 전체 대리점수' if model in INTRADAY_MODEL_IDS
                       else '추정확률')
            self.market_tabs.hide()
            self.status.setText(
                f'{PROTOTYPE_TITLES[model]} · 장중 추론\n'
                f'{PROTOTYPE_METHODS[model]}\n'
                f'현재가 기준 {PROTOTYPE_TARGETS[model]} {measure}는 관심종목 표와 상단 최근 조회에서 확인')
            return
        self.market_tabs.show()
        statuses = []
        for index, (market, table) in enumerate(self.tables.items()):
            result = self.model_results.get((model, market), {})
            name = '국내' if market == 'domestic' else '미국'
            if model == MARK14_TRIGGER:
                self.market_tabs.setTabText(index, '국내 · 예상 순수익률 점수' if market == 'domestic'
                                            else '미국 · 상대순위 점수')
            else:
                self.market_tabs.setTabText(index, f'{name} · {PROTOTYPE_TARGETS[model]}')
            if not result:
                statuses.append(f'{name} · —')
                table.setRowCount(0)
                continue
            state = str(result.get('state', '대기'))
            reason = str(result.get('reason') or '')
            opened = str(result.get('session_open') or '')
            prior = not self._result_is_current_session(market, result)
            statuses.append(f'{name} · {"이전 장 · " if prior else ""}{state} · 개장 {opened}'
                            + (f' · {reason}' if reason else ''))
            candidates = result.get('candidates')
            if not isinstance(candidates, list):
                table.setRowCount(0)
                continue
            table.setRowCount(min(len(candidates), 100))
            for row, candidate in enumerate(candidates[:100]):
                if not isinstance(candidate, dict):
                    continue
                score = candidate.get('score')
                threshold = candidate.get('threshold')
                decision = ('매수 후보' if candidate.get('selected') else '대기')
                if prior:
                    decision = '이전 장 · ' + decision
                if candidate.get('out_of_training_universe'):
                    decision += ' · 학습종목 밖'
                fraction = candidate.get('equity_fraction')
                fraction_text = (f'{fraction * 100:.3f}%' if isinstance(fraction, (int, float))
                                 else '—')
                for column, value in enumerate((candidate.get('symbol', ''),
                                                 f'{score:.6f}' if isinstance(score, (int, float)) else '—',
                                                 f'{threshold:.6f}' if isinstance(threshold, (int, float)) else '—',
                                                 decision, fraction_text)):
                    table.setItem(row, column, QTableWidgetItem(str(value)))
                table.item(row, 1).setToolTip(
                    f"{candidate.get('score_metric', '모델 점수')} · 단위 {candidate.get('score_unit', '모델별')} · 적중확률 아님")
        self.status.setText(PROTOTYPE_TITLES.get(model, '장전 모델') + '\n' + '\n'.join(statuses))


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
                 mark12_bundle=None, mark14_bundle=None,
                 external_models=None, external_feed_factory=None, session_lock=None,
                 restore_model_choices=True):
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
        self._mark14_bundle = mark14_bundle
        self._external_feed_factory = external_feed_factory
        self._restore_model_choices = restore_model_choices
        self._prototype_feeds = {}
        self._mark14_preopen_ready_sessions = set()
        self._mark14_preopen_attempts = {}
        # Display history only. Neither external JSON nor order validation
        # reads this cache; old probabilities cannot authorize a trade.
        self._last_model_scores = {}
        self._last_preopen_scores = {}
        self._last_queried_watch_id = None
        self._last_complete_model_summary = None
        super().__init__(service or TradingService(), store, defer_workspace=True, session_lock=session_lock)
        self._last_workspace_page = self.workspace_tabs.currentWidget()
        self.workspace_tabs.tabBar().tabBarClicked.connect(self._workspace_tab_clicked)
        self.mark14_panel = Mark14Panel(enabled_models, self, clock=lambda: self.engine.clock())
        self.mark14_panel.model_selector.currentIndexChanged.connect(self._update_model_detail)
        status_page = self.signal_connection_page.widget(0)
        self.signal_connection_page.removeTab(0)
        status_page.hide()  # Keep diagnostic internals, remove the status/inspect screen.
        self.signal_connection_page.removeTab(self.signal_connection_page.indexOf(self.external_panel))
        self.tabs.addTab(self.external_panel, '공통 주문·연결')
        self.workspace_tabs.removeTab(self.workspace_tabs.indexOf(self.model_performance_panel))
        self.signal_connection_page.addTab(self.mark14_panel, '모델 선택')
        self.signal_connection_page.addTab(self.model_performance_panel, '모델 성과')
        self.signal_connection_page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.signal_connection_page.currentChanged.connect(lambda _: self._reload_activity(visible_force=True))
        self.signal_connection_page.currentChanged.connect(self._refresh_visible_model_display)
        self.workspace_tabs.setTabText(self.workspace_tabs.indexOf(self.watch_page), '관심종목')
        self.workspace_tabs.setTabText(self.workspace_tabs.indexOf(self.signal_connection_page),
                                       'AI 추론 모델')
        self.workspace_tabs.setTabText(self.workspace_tabs.indexOf(self.order_history_panel),
                                       '주문·체결')
        for destination, page in enumerate((self.watch_page, self.portfolio_panel,
                                            self.trade_journal_panel, self.order_history_panel,
                                            self.signal_connection_page, self.tabs)):
            source = self.workspace_tabs.indexOf(page)
            if source != destination:
                self.workspace_tabs.tabBar().moveTab(source, destination)
        self.workspace_tabs.setCurrentWidget(self.watch_page)
        for view in self.watch_tables.values():
            view.setColumnCount(3 + len(WATCH_MODEL_IDS))
            view.setHorizontalHeaderLabels(('종목',
                                            *(PROTOTYPE_TITLES[model].replace('mark', 'MK').replace(' prototype', '')
                                              + (' 대리점수' if model in (*INTRADAY_MODEL_IDS, *TARGET_HORIZON_MODEL_IDS) else
                                                 ' 추정확률' if model in BARRIER_MODELS else ' 점수')
                                              for model in WATCH_MODEL_IDS),
                                            '현재가', '조회'))
            view.setMinimumWidth(285 if self._watch_compact else 680)
            for column, width in ((0, 110), (view.columnCount() - 2, 95),
                                  (view.columnCount() - 1, 110)):
                view.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                view.setColumnWidth(column, width)
            for column in range(1, view.columnCount() - 2):
                view.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                view.setColumnWidth(column, 126)
        self.model_score_summary = QLabel('현재가 —')
        self.model_score_summary.setObjectName('modelScoreSummary')
        self.model_score_summary.setWordWrap(False)
        self.model_score_summary.setFixedHeight(34)
        self.model_score_summary.setStyleSheet(
            'QLabel#modelScoreSummary { background: #193148; color: #e9f5ff; '
            'border: 1px solid #31536d; border-radius: 6px; padding: 4px 8px; font-weight: 600; }')
        self.price_change_summary = QLabel('전일 대비 —')
        self.price_change_summary.setObjectName('priceChangeSummary')
        self.price_change_summary.setFixedHeight(34)
        self.price_change_summary.setStyleSheet(
            'QLabel#priceChangeSummary { background: #193148; color: #e9f5ff; '
            'border: 1px solid #31536d; border-radius: 6px; padding: 4px 8px; font-weight: 600; }')
        self.price_strip = QWidget()
        price_layout = QHBoxLayout(self.price_strip)
        price_layout.setContentsMargins(0, 0, 0, 0)
        price_layout.setSpacing(6)
        price_layout.addWidget(self.model_score_summary, 1)
        price_layout.addWidget(self.price_change_summary, 1)
        self.chart_title.parentWidget().layout().insertWidget(1, self.price_strip)
        self.latest_model_summary = QLabel('최근 조회 종목 — · 모델 추정확률 —')
        self.latest_model_summary.setObjectName('latestModelSummary')
        self.latest_model_summary.setAccessibleName('최근 조회 종목과 활성 AI 모델별 매수 판단 및 점수')
        self.latest_model_summary.setWordWrap(False)
        self.latest_model_summary.setFixedHeight(42)
        self.latest_model_summary.setStyleSheet('QLabel#latestModelSummary { color: #e9f5ff; '
                                                'background: transparent; border: none; '
                                                'padding: 6px 8px; font-weight: 600; }')
        self.latest_model_summary.setToolTip('최근 완료 조회 기준. 장중 확률 평균은 확률 모델끼리만 계산하며, 장전 점수는 모델별 단위 그대로 표시합니다. 매매 판단에 합산하지 않습니다.')
        self.latest_model_summary_area = QScrollArea()
        self.latest_model_summary_area.setObjectName('latestModelSummaryArea')
        self.latest_model_summary_area.setWidgetResizable(False)
        self.latest_model_summary_area.setFixedHeight(44)
        self.latest_model_summary_area.setStyleSheet(
            'QScrollArea#latestModelSummaryArea { background: #193148; '
            'border: 1px solid #31536d; border-radius: 6px; }')
        self.latest_model_summary_area.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.latest_model_summary_area.setWidget(self.latest_model_summary)
        self.connection_flow.addWidget(self.latest_model_summary_area, 1)
        # Keep the plain-text status for existing diagnostics, but make the
        # compact visible card emphasize the model count instead of a sentence.
        self.connection_flow.removeWidget(self.connection_summary)
        self.connection_summary.setParent(self)
        self.connection_summary.hide()
        self.ai_connection_card = QFrame()
        self.ai_connection_card.setObjectName('aiConnectionCard')
        self.ai_connection_card.setFixedHeight(44)
        self.ai_connection_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.ai_connection_card.setStyleSheet(
            'QFrame#aiConnectionCard { background: #193148; border: 1px solid #31536d; border-radius: 6px; }')
        ai_row = QHBoxLayout(self.ai_connection_card)
        ai_row.setContentsMargins(10, 3, 9, 3)
        ai_row.setSpacing(3)
        ai_title = QLabel('AI 모델')
        ai_title.setStyleSheet('font-size: 11px; color: #b8cbdc;')
        ai_row.addWidget(ai_title)
        ai_row.addStretch()
        self.ai_model_count = QLabel('0')
        self.ai_model_count.setObjectName('aiModelCount')
        self.ai_model_count.setStyleSheet('font-size: 23px; font-weight: 700; color: #78e6c7;')
        ai_row.addWidget(self.ai_model_count)
        ai_suffix = QLabel('개 선택')
        ai_suffix.setStyleSheet('font-size: 11px; color: #b8cbdc;')
        ai_row.addWidget(ai_suffix)
        self.ai_connection_warning = QLabel('점검')
        self.ai_connection_warning.setObjectName('aiConnectionWarning')
        self.ai_connection_warning.setStyleSheet('font-size: 11px; font-weight: 600; color: #ffbf80;')
        self.ai_connection_warning.hide()
        ai_row.addWidget(self.ai_connection_warning)
        self.connection_flow.insertWidget(2, self.ai_connection_card, 1)
        # Health text is refreshed every second, but the 200-row x 13-model
        # watch grid is only time-sensitive while visible. Keep that work off
        # the health tick; quote/pre-open events still update their rows at once.
        self.model_display_timer = QTimer(self)
        self.model_display_timer.setInterval(30_000)
        self.model_display_timer.timeout.connect(self._refresh_visible_model_display)
        self.model_display_timer.start()
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
            self.external_grid.addItem(item, row + 7, column, row_span, column_span)
        self.external_grid.removeWidget(self.percent_sizing)
        self.percent_sizing.setParent(self)
        self.percent_sizing.setChecked(True)
        self.percent_sizing.hide()
        self.external_grid.removeWidget(self.buy_percent)
        self.buy_percent.setSuffix(' %')
        self.buy_percent.setFixedWidth(100)
        self.buy_percent.setToolTip('AI 모델별 비중이 아닌 기타 외부 신호의 기본값입니다. 주문당 금액·가용액 상한도 적용합니다.')
        self.external_grid.addWidget(QLabel('기타 신호 1회 매수 비중'), 4, 0, 1, 2)
        self.external_grid.addWidget(self.buy_percent, 4, 2)
        self.external_model_checks = self.mark14_panel.model_checks
        self.mark14_panel.select_all_button.clicked.connect(lambda: self._set_all_models_selected(True))
        self.mark14_panel.clear_all_button.clicked.connect(lambda: self._set_all_models_selected(False))
        self.mark14_panel.apply_bulk_buy_percent_button.clicked.connect(self._apply_bulk_model_buy_percent)
        self._sync_watch_model_columns()
        self.watch_market_tabs.currentChanged.connect(self._sync_watch_model_columns)
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
        self.model_status.hide()
        self.model_notice = QLabel(MARK1_NOTICE)
        self.model_notice.setWordWrap(True)
        self.model_notice.setStyleSheet('color: #ffda91;')
        self.model_notice.setParent(self)
        self.model_notice.hide()  # Full method/output/exit descriptions now live beside every model.
        self.model_trigger.currentIndexChanged.connect(self._model_trigger_changed)
        for checkbox in self.external_model_checks.values():
            checkbox.toggled.connect(self._model_trigger_changed)
        self.builtin_lstm.toggled.connect(self._legacy_lstm_changed)
        self.external_mode.setChecked(True)
        self.external_krw.setValue(10_000_000)
        self.external_usd.setValue(10_000)
        self.hourly_ranking.setChecked(True)
        self.hourly_ranking.setText('국내·미국 종합 TOP100 · 개장 10분 전 / 개장 / 매 정시')
        self.ranking_button.setText('현재 선정 가능한 시장 · 종합 TOP100')
        # Desktop lists are selected by the market ranking. Keep the common
        # history-length control but remove manual symbol add/delete affordances.
        for widget in (self.symbol_input, self.exchange_input, self.add_button, self.remove_button):
            widget.hide()
        self.days_button.setText('선정 종목 기간 적용')
        self.days_input.setValue(31)
        self.environment_caption.hide()
        self.message.setText('')
        self._apply_execution_preferences()
        self._label_inputs()
        self._move_legacy_settings()
        self._loading_preferences = False
        self._preferences_timer = QTimer(self)
        self._preferences_timer.setSingleShot(True)
        self._preferences_timer.timeout.connect(self._save_preferences)
        self._restore_preferences()
        self._connect_preference_changes()

    def _apply_execution_preferences(self):
        # This desktop exposes only percentage sizing. A legacy saved False
        # must never silently turn the hidden checkbox into fixed-quantity BUY.
        if hasattr(self, 'percent_sizing'):
            self.percent_sizing.setChecked(True)
        super()._apply_execution_preferences()
        if hasattr(self, 'mark14_panel'):
            self.engine.source_buy_percents = {
                PROTOTYPE_SOURCES[model]: Decimal(str(field.value()))
                for model, field in self.mark14_panel.model_buy_percents.items()
                if model not in MINUTE_HEDGE_IDS
            }

    def _capture_preferences(self):
        # The random test producer temporarily rewrites these three controls.
        # Keep the user's underlying connection choices, not that runtime mode.
        connection = (self.external_mode.isChecked(), self.external_source.text(), self.signal_path.text())
        if self.random_demo.isChecked() and hasattr(self, '_saved_external_config'):
            connection = self._saved_external_config
        geometry = self.normalGeometry() if self.isMaximized() else self.geometry()
        return {
            'version': 4,
            'interval': self.interval.value(),
            'hourly_ranking': self.hourly_ranking.isChecked(),
            'exchange': self.exchange_input.currentData(),
            'external_models': list(self._chosen_external_models()) if selected_mode(self.service) is TradingMode.DEMO else [],
            'model_trigger': self._chosen_trigger(),
            'external_mode': bool(connection[0]),
            'external_source': connection[1],
            'signal_path': connection[2],
            'chart_path': self.chart_path.text(),
            'external_quantity': self.external_quantity.value(),
            'external_krw': self.external_krw.value(),
            'external_usd': self.external_usd.value(),
            'percent_sizing': True,
            'buy_percent': self.buy_percent.value(),
            'model_buy_percents': {model: field.value()
                                   for model, field in self.mark14_panel.model_buy_percents.items()},
            'order_popups': self.order_popups.isChecked(),
            'additional_sources': [list(pair) for pair in self.additional_sources.raw_sources()],
            'random_us': self.random_us.currentData(),
            'watch_market_tab': self.watch_market_tabs.currentIndex(),
            'workspace_tab': self.workspace_tabs.currentIndex(),
            'workspace_tab_id': self._workspace_tab_id(),
            'ai_subtab': ('performance' if self.signal_connection_page.currentWidget() is self.model_performance_panel
                          else 'models'),
            'advanced_tab': self.tabs.currentIndex(),
            'advanced_visible': True,
            'window_width': geometry.width(),
            'window_height': geometry.height(),
        }

    def _save_preferences(self):
        if getattr(self, '_loading_preferences', False):
            return
        try:
            self.store.save_ui_preferences(self._capture_preferences())
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self.message.setText(f'화면 설정 저장 실패: {exc}')

    def _workspace_tab_id(self):
        pages = ((self.portfolio_panel, 'portfolio'), (self.order_history_panel, 'orders'),
                 (self.trade_journal_panel, 'journal'), (self.watch_page, 'watch'),
                 (self.signal_connection_page, 'ai'), (self.operations_panel, 'logs'),
                 (self.tabs, 'advanced'))
        return next((name for widget, name in pages
                     if self.workspace_tabs.currentWidget() is widget), 'portfolio')

    def _queue_save_preferences(self, *_):
        if not getattr(self, '_loading_preferences', False):
            self._preferences_timer.start(350)

    def _model_buy_percent_changed(self, *_):
        # Editing a model's order size changes its execution policy. Never
        # mutate a live worker's sizing while a send may already be pending.
        if self.monitoring or self.worker is not None or self.pending_auto_arm:
            self.stop_monitoring()
        self.engine.disarm()
        self._apply_execution_preferences()
        self._queue_save_preferences()

    def _apply_bulk_model_buy_percent(self):
        # A batch changes sizing once, including models currently unselected.
        # Check live state as well as widget locks for programmatic invocation.
        if (selected_mode(self.service) is not TradingMode.DEMO
                or not self.external_panel.isEnabled()
                or self.monitoring or self.worker is not None
                or self._workspace_worker is not None or self.pending_auto_arm
                or self._confirming_orders or self._pending_environment
                or self._confirming_environment):
            return
        panel = self.mark14_panel
        value = panel.bulk_buy_percent.value()
        fields = [field for model, field in panel.model_buy_percents.items()
                  if model not in MINUTE_HEDGE_IDS]
        if all(field.value() == value for field in fields):
            return
        offset = panel.choices_scroll.verticalScrollBar().value()
        for field in fields:
            blocked = field.blockSignals(True)
            try:
                field.setValue(value)
            finally:
                field.blockSignals(blocked)
        self.engine.disarm()
        self._apply_execution_preferences()
        self._queue_save_preferences()
        QTimer.singleShot(0, lambda area=panel.choices_scroll, saved=offset:
                          area.verticalScrollBar().setValue(saved))

    def _connect_preference_changes(self):
        for widget in (self.interval, self.external_quantity, self.external_krw,
                       self.external_usd, self.buy_percent):
            widget.valueChanged.connect(self._queue_save_preferences)
        for field in self.mark14_panel.model_buy_percents.values():
            field.valueChanged.connect(self._model_buy_percent_changed)
        for widget in (self.hourly_ranking, self.external_mode, self.percent_sizing,
                       self.order_popups, self.advanced_settings_button,
                       *self.external_model_checks.values()):
            widget.toggled.connect(self._queue_save_preferences)
        for widget in (self.external_source, self.signal_path, self.chart_path):
            widget.textChanged.connect(self._queue_save_preferences)
        for widget in (self.exchange_input, self.model_trigger, self.random_us):
            widget.currentIndexChanged.connect(self._queue_save_preferences)
        for tabs in (self.watch_market_tabs, self.workspace_tabs, self.signal_connection_page, self.tabs):
            tabs.currentChanged.connect(self._queue_save_preferences)
        table = self.additional_sources.table
        table.itemChanged.connect(self._queue_save_preferences)
        table.model().rowsInserted.connect(self._queue_save_preferences)
        table.model().rowsRemoved.connect(self._queue_save_preferences)

    def _restore_preferences(self, *, after_switch=False):
        try:
            saved = self.store.load_ui_preferences()
        except (OSError, sqlite3.Error) as exc:
            self.message.setText(f'화면 설정 불러오기 실패 · 기본값 사용: {exc}')
            saved = {}
        defaults = {}
        if after_switch:
            folder = self.store.path.parent / 'exchange'
            defaults = {
                'interval': 30, 'hourly_ranking': True, 'exchange': '',
                'external_models': (list(PROTOTYPE_NOTICES) if selected_mode(self.service) is TradingMode.DEMO
                                    else []), 'model_trigger': 'none',
                'external_mode': False, 'external_source': 'external-model',
                'signal_path': str(folder / 'signals.json'), 'chart_path': str(folder / 'charts.json'),
                'external_quantity': 999999999, 'external_krw': 0, 'external_usd': 0,
                'percent_sizing': True, 'buy_percent': 10, 'order_popups': True,
                'model_buy_percents': {},
                'additional_sources': [], 'random_us': 'blocked',
            }
        choices = {**defaults, **saved}
        self._loading_preferences = True
        controls = (self.interval, self.hourly_ranking, self.exchange_input,
                    self.model_trigger, self.external_mode, self.external_source, self.signal_path,
                    self.chart_path, self.external_quantity, self.external_krw, self.external_usd,
                    self.percent_sizing, self.buy_percent, self.order_popups, self.random_us,
                    *self.external_model_checks.values(),
                    *self.mark14_panel.model_buy_percents.values())
        previous = [(widget, widget.blockSignals(True)) for widget in controls]
        try:
            def restore_int(key, widget):
                value = choices.get(key)
                if type(value) is int and widget.minimum() <= value <= widget.maximum():
                    widget.setValue(value)

            def restore_number(key, widget):
                value = choices.get(key)
                if (type(value) in (int, float) and widget.minimum() <= value <= widget.maximum()
                        and isfinite(value)):
                    widget.setValue(value)

            def restore_bool(key, widget):
                value = choices.get(key)
                if type(value) is bool:
                    widget.setChecked(value)

            def restore_text(key, widget, limit):
                value = choices.get(key)
                if isinstance(value, str) and len(value) <= limit and not any(c in value for c in '\0\r\n'):
                    widget.setText(value)

            def restore_combo(key, widget):
                value = choices.get(key)
                if isinstance(value, str):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)

            for key, widget in (('interval', self.interval), ('external_quantity', self.external_quantity)):
                restore_int(key, widget)
            for key, widget in (('external_krw', self.external_krw), ('external_usd', self.external_usd),
                                ('buy_percent', self.buy_percent)):
                restore_number(key, widget)
            model_sizing = choices.get('model_buy_percents')
            if not isinstance(model_sizing, dict):
                model_sizing = {}
            for model, field in self.mark14_panel.model_buy_percents.items():
                default_percent = 1 if model in MINUTE_TRANSFER_IDS else self.buy_percent.value()
                value = model_sizing.get(model, default_percent)
                if (type(value) in (int, float) and field.minimum() <= value <= field.maximum()
                        and isfinite(value)):
                    field.setValue(value)
                else:
                    field.setValue(default_percent)
            for key, widget in (('hourly_ranking', self.hourly_ranking), ('external_mode', self.external_mode),
                                ('order_popups', self.order_popups)):
                restore_bool(key, widget)
            self.percent_sizing.setChecked(True)
            for key, widget, limit in (('external_source', self.external_source, 128),
                                       ('signal_path', self.signal_path, 4096),
                                       ('chart_path', self.chart_path, 4096)):
                restore_text(key, widget, limit)
            for key, widget in (('exchange', self.exchange_input), ('random_us', self.random_us)):
                restore_combo(key, widget)
            if self._restore_model_choices:
                restore_combo('model_trigger', self.model_trigger)
                enabled = choices.get('external_models')
                if isinstance(enabled, list):
                    allowed = set(enabled) & set(PROTOTYPE_NOTICES) if all(isinstance(v, str) for v in enabled) else set()
                    # These models did not exist when a v2 selection held
                    # older prototypes. Add them once without changing those
                    # older choices. Explicitly empty or corrupt selections
                    # remain OFF; v3 user choices are authoritative.
                    if saved.get('version') == 2 and enabled and all(
                            isinstance(value, str) and value in PROTOTYPE_NOTICES for value in enabled):
                        allowed.update(PREOPEN_MODELS)
                    for model, checkbox in self.external_model_checks.items():
                        checkbox.setChecked(selected_mode(self.service) is TradingMode.DEMO and model in allowed)
            if selected_mode(self.service) is TradingMode.REAL:
                for checkbox in self.external_model_checks.values():
                    checkbox.setChecked(False)
            sources = choices.get('additional_sources')
            if (isinstance(sources, list) and len(sources) <= 16
                    and all(isinstance(row, list) and len(row) == 2
                            and all(isinstance(value, str) and len(value) <= limit
                                    and not any(c in value for c in '\0\r\n')
                                    for value, limit in zip(row, (128, 4096))) for row in sources)):
                self.additional_sources.table.setRowCount(0)
                for source, path in sources:
                    self.additional_sources.add_row(source=source, path=path)
            for key, tabs in (('watch_market_tab', self.watch_market_tabs),
                              ('advanced_tab', self.tabs)):
                index = choices.get(key)
                if type(index) is int and 0 <= index < tabs.count():
                    tabs.setCurrentIndex(index)
            ai_subtab = choices.get('ai_subtab')
            if ai_subtab in ('models', 'performance'):
                self.signal_connection_page.setCurrentWidget(
                    self.model_performance_panel if ai_subtab == 'performance' else self.mark14_panel)
            pages = {'portfolio': self.portfolio_panel, 'orders': self.order_history_panel,
                     'journal': self.trade_journal_panel, 'watch': self.watch_page,
                     'performance': self.signal_connection_page,
                     'ai': self.signal_connection_page, 'logs': self.tabs,
                     'advanced': self.tabs}
            selected_page_id = choices.get('workspace_tab_id')
            page = pages.get(selected_page_id)
            if page is None and type(saved.get('workspace_tab')) is int:
                legacy = ('portfolio', 'orders', 'journal', 'logs', 'watch',
                          'performance', 'ai', 'advanced')
                old_index = saved['workspace_tab']
                selected_page_id = legacy[old_index] if 0 <= old_index < len(legacy) else None
                page = pages.get(selected_page_id)
            if selected_page_id == 'performance' and ai_subtab not in ('models', 'performance'):
                self.signal_connection_page.setCurrentWidget(self.model_performance_panel)
            if selected_page_id == 'logs':
                self.tabs.setCurrentWidget(self.operations_panel)
            if page is not None:
                index = self.workspace_tabs.indexOf(page)
                if index >= 0 and self.workspace_tabs.isTabVisible(index):
                    self.workspace_tabs.setCurrentWidget(page)
            width, height = choices.get('window_width'), choices.get('window_height')
            if type(width) is int and type(height) is int and width >= 800 and height >= 520:
                available = self.screen().availableGeometry() if self.screen() is not None else None
                if available is not None:
                    self.resize(min(width, max(self.minimumWidth(), available.width() - 24)),
                                min(height, max(self.minimumHeight(), available.height() - 48)))
        finally:
            for widget, blocked in previous:
                widget.blockSignals(blocked)
            self._loading_preferences = False
        self.mark14_panel._update_selection_count()
        self._model_trigger_changed()
        self._apply_execution_preferences()
        self._update_connection()
        self.update_controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_preferences_timer'):
            self._queue_save_preferences()

    def _adjust_watch_layout(self, *_):
        super()._adjust_watch_layout()
        if (hasattr(self, 'signal_connection_page') and hasattr(self, 'workspace_tabs')
                and self.workspace_tabs.currentWidget() is self.signal_connection_page):
            self.workspace_tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

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
                          'TARGET_HORIZON_CANDIDATE', 'TARGET_HORIZON_POLICY_NOT_MET',
                          'USER_QUANTITY_OR_NOTIONAL_CAP', 'QUOTE_EXPIRED_DURING_INFERENCE'}:
            return None
        prediction = row.get('prediction')
        if not isinstance(prediction, dict):
            return None
        if (model in INTRADAY_MODEL_IDS
                and prediction.get('score_scope') != 'daily_open_whole_session_proxy_not_intraday'):
            return None
        if (model in TARGET_HORIZON_MODEL_IDS
                and (prediction.get('score_scope') != TARGET_HORIZON_SCORE_SCOPE
                     or prediction.get('strategy_id') != model
                     or prediction.get('stop_loss_pct', False) is not None
                     or prediction.get('research_only') is not True
                     or prediction.get('deployment_allowed') is not False)):
            return None
        try:
            value = prediction.get('probability_success')
            if isinstance(value, bool):
                return None
            probability = Decimal(str(value))
            reference_price = Decimal(str(row.get('reference_price')))
            expected_net_return = (Decimal(str(prediction.get('expected_net_return')))
                                   if model in TARGET_HORIZON_MODEL_IDS else None)
            if (not probability.is_finite() or not 0 <= probability <= 1
                    or reference_price != snapshot.quote.price
                    or (expected_net_return is not None and not expected_net_return.is_finite())):
                return None
            if model in TARGET_HORIZON_MODEL_IDS:
                candidate = probability >= Decimal('0.5') and expected_net_return > 0
                if ((reason == 'TARGET_HORIZON_CANDIDATE' and not candidate)
                        or (reason == 'TARGET_HORIZON_POLICY_NOT_MET' and candidate)):
                    return None
            elif ((reason == 'PREDICTED_DAILY_BARRIER_SUCCESS' and probability <= Decimal('0.5'))
                  or (reason == 'BELOW_OR_EQUAL_BUY_THRESHOLD' and probability > Decimal('0.5'))):
                return None
        except (InvalidOperation, TypeError, ValueError):
            return None
        score = {'probability': probability, 'price': reference_price,
                 'fetched_at': snapshot.fetched_at, 'reason': reason,
                 'expected_net_return': expected_net_return}
        self._last_model_scores[(model, item.id)] = score
        return score

    def _model_score_display(self, model, item):
        """Keep the last valid display score; execution still uses fresh quotes."""
        title = PROTOTYPE_TITLES[model]
        target = PROTOTYPE_TARGETS[model]
        proxy = model in (*INTRADAY_MODEL_IDS, *TARGET_HORIZON_MODEL_IDS)
        measure = ('다음 날 관측 시가 학습·현재가 질의의 목표 도달 대리점수 · 장중 경로 미검증'
                   if model in TARGET_HORIZON_MODEL_IDS else
                   '일봉 전체 대리점수 · 장중 진입 이후 경로 미검증' if proxy else
                   '일봉 기반 모델 추정확률')
        note = (f'{title} · {target} · {measure} · 실제 적중률·수익률 보장 아님. '
                '과거 표시값은 매매 판단에 사용하지 않고 주문 직전 새 시세로 재판단합니다.'
                + (' 모의 연구 신호이며 목표가 지정가 체결이나 기한 내 매도 체결을 보장하지 않습니다.'
                   if model in TARGET_HORIZON_MODEL_IDS else ''))
        checks = getattr(self, 'external_model_checks', {})
        if model not in checks or not checks[model].isChecked():
            return '—', note + '\n이 모델은 연결되어 있지 않습니다.', '꺼짐'
        current = self._current_model_score(model, item)
        snapshot = self.snapshots.get(item.id)
        if current is not None and snapshot is not None:
            age = (self.engine.clock() - snapshot.fetched_at).total_seconds()
            if (0 <= age <= MAX_QUOTE_AGE and item.id in self.fresh_ids
                    and current['reason'] != 'QUOTE_EXPIRED_DURING_INFERENCE'):
                decision = ('연구 후보' if model in TARGET_HORIZON_MODEL_IDS
                            and current['reason'] == 'TARGET_HORIZON_CANDIDATE' else
                            '매수 판정' if current['reason'] == 'PREDICTED_DAILY_BARRIER_SUCCESS' else '대기')
                pct = f"{current['probability']:.1%}"
                context = (f"\n판단 기준 현재가 {current['price']} {item.instrument.currency}"
                           f" · 조회 {current['fetched_at'].astimezone():%m/%d %H:%M:%S}")
                if model in TARGET_HORIZON_MODEL_IDS:
                    context += f" · 예상 순수익 대리값 {current['expected_net_return']:.2%}"
                return (f'{decision}\n{pct}',
                        note + context + f"\n{decision} · {'대리점수' if proxy else '추정 성공확률'} {pct} · 사유 {current['reason']}",
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
            return '시세 오류', note + '\n현재 시세 조회 오류: ' + str(self.errors[item.id]), '시세 오류'
        if snapshot is None:
            return '—', note + '\n현재가 조회 전입니다.', '—'
        row = getattr(self._prototype_feeds.get(model), 'diagnostics', {}).get(item.id)
        state = ('보유 중' if isinstance(row, dict) and row.get('reason') == 'POSITION_EXIT_MANAGED_BY_GUI'
                 else '판단 불가' if isinstance(row, dict) and (row.get('error') or row.get('prediction'))
                 else '판단 전')
        value = state if state != '판단 전' else '—'
        return value, note + f"\n현재가 {snapshot.quote.price} {item.instrument.currency} · 이 시세의 유효한 {'대리점수' if proxy else '모델 확률'}가 없습니다.", value

    def _preopen_score_display(self, model, item):
        """Show today's frozen plan or clearly marked last display-only decision."""
        panel = getattr(self, 'mark14_panel', None)
        result = panel.model_results.get((model, item.instrument.market.value)) if panel is not None else None
        saved = self._last_preopen_scores.get((model, item.id))
        def saved_display():
            if saved is None:
                return '—'
            from dockdack.history import market_time
            from dockdack.market_schedule import session_on
            session = session_on(item.instrument.market,
                                 market_time(item.instrument.market, self.engine.clock()).date())
            prefix = '마지막 판단' if session is not None and saved[0] == session.opened else '이전 장'
            return f'{prefix} {saved[1]}'
        if not result or result.get('state') != 'prepared':
            return saved_display()
        try:
            from dockdack.history import market_time
            from dockdack.market_schedule import session_on
            now = self.engine.clock()
            session = session_on(item.instrument.market, market_time(item.instrument.market, now).date())
            opened = datetime.fromisoformat(str(result['session_open']))
            current = session is not None and opened == session.opened
        except (KeyError, TypeError, ValueError):
            return saved_display()
        candidates = result.get('candidates')
        if not isinstance(candidates, list):
            return saved_display()
        row = next((entry for entry in candidates if isinstance(entry, dict)
                    and entry.get('watch_id') == item.id), None)
        if row is None:
            return saved_display()
        score = row.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
            return saved_display()
        score_text = f'{score:.3f}%' if row.get('score_unit') == 'percent' else f'{score:.3f}점'
        decision = '매수 후보' if row.get('selected') is True else '관망'
        display = f'{decision}({score_text})'
        if current:
            self._last_preopen_scores[(model, item.id)] = (opened, display)
            return display
        return f'이전 장 {display}'

    def _model_name(self, model):
        return PROTOTYPE_TITLES[model].replace('mark', 'MK').replace(' prototype', '')

    def _minute_model_diagnostic(self, model, item):
        """Display only the minute verdict paired with this exact fresh quote."""
        feed = self._prototype_feeds.get(model)
        snapshot = self.snapshots.get(item.id)
        if (feed is None or not getattr(feed, '_ready', False) or snapshot is None
                or item.id not in self.fresh_ids or item.id in self.errors
                or not 0 <= (self.engine.clock() - snapshot.fetched_at).total_seconds() <= MAX_QUOTE_AGE):
            return None
        row = feed.diagnostics.get(item.id)
        if (not isinstance(row, dict)
                or row.get('_display_quote_fetched_at') != snapshot.fetched_at.isoformat()
                or row.get('_display_price') != str(snapshot.quote.price)):
            return None
        return row

    def _minute_buy_display(self, model, item):
        if model in MINUTE_HEDGE_IDS:
            return '연구 전용 · 주문 보류'
        row = self._minute_model_diagnostic(model, item)
        if row is None:
            return '5분봉 판단 대기'
        score = row.get('score')
        score_text = (f' · 점수 {score:.3f}'
                      if isinstance(score, (int, float)) and isfinite(score) else '')
        if row.get('reason') == 'MINUTE_BUY_CANDIDATE':
            return '매수 후보' + score_text
        if row.get('reason') == 'MINUTE_THRESHOLD_NOT_MET':
            return '관망' + score_text
        if row.get('reason') == 'SYMBOL_OUTSIDE_DAILY_TRAINING':
            return '학습 종목 외 · 주문 보류'
        if row.get('reason') == 'MINUTE_EXIT_TIME_UNAVAILABLE':
            return '장마감 청산 시간 부족 · 주문 보류'
        return '5분봉 입력 확인 필요 · 주문 보류'

    def _model_buy_display(self, model, item, *, compact=False):
        """Display each model's own unit without treating a score as a probability."""
        checks = getattr(self, 'external_model_checks', {})
        if model not in checks or not checks[model].isChecked():
            return '꺼짐'
        if model in MINUTE_RESEARCH_IDS:
            return self._minute_buy_display(model, item)
        if model in QUOTE_SCORE_MODELS:
            display = self._model_score_display(model, item)
            return display[2].replace('대기', '관망') if compact else display[0]
        return self._preopen_score_display(model, item)

    def _model_sell_display(self, model, item):
        """Only report verified holding-lot exit conditions, never infer SELL from a BUY score."""
        key = f'{item.instrument.market.value}:{item.instrument.exchange}:{item.instrument.symbol}'
        group = getattr(self.portfolio_panel, '_exit_targets', {}).get(key)
        if not group:
            return '—'
        if group.get('error') or (group.get('lots') and not group.get('reconciled')):
            return '매도 확인 필요'
        lots = group.get('lots') if group.get('reconciled') else (group,)
        owned = [lot for lot in lots if (lot.get('strategy_id') or lot.get('model_id')) == model
                 and lot.get('quantity', 1) > 0]
        if not owned:
            return '—'
        snapshot = self.snapshots.get(item.id)
        fresh = (snapshot is not None and item.id in self.fresh_ids and item.id not in self.errors
                 and 0 <= (self.engine.clock() - snapshot.fetched_at).total_seconds() <= MAX_QUOTE_AGE)
        for lot in owned:
            if lot.get('error'):
                return '매도 확인 필요'
            if lot.get('sellable_quantity', 1) <= 0:
                continue
            if timed_exit_due(lot, item.instrument.market, self.engine.clock()):
                return '기간 매도 조건'
            if fresh:
                price = snapshot.quote.price
                upper, lower = lot.get('take_profit_price'), lot.get('stop_loss_price')
                if (isinstance(upper, Decimal) and price >= upper
                        or isinstance(lower, Decimal) and price <= lower):
                    return '가격 매도 조건'
        return '매도 대기' if any(lot.get('sellable_quantity', 1) > 0 for lot in owned) else '매도가능수량 없음'

    def _watch_values(self, item):
        name, price, checked_at = super()._watch_values(item)
        return (name, *(self._model_buy_display(model, item) for model in WATCH_MODEL_IDS),
                price, checked_at)

    def _update_model_score_row(self, key):
        item = self._items_by_id.get(key)
        row = self._watch_rows.get(key)
        if item is None or row is None:
            return
        view = self.watch_tables[item.instrument.market]
        for column, model in enumerate(WATCH_MODEL_IDS, 1):
            cell = view.item(row, column)
            if cell is None:
                continue
            if model in QUOTE_SCORE_MODELS:
                _, tip, _ = self._model_score_display(model, item)
            else:
                tip = (f'{PROTOTYPE_TITLES[model]} · 장전 동결 매수 후보 점수\n'
                       f'{PROTOTYPE_METHODS[model]}\n{PROTOTYPE_TARGETS[model]} · 매도 여부는 보유분 조건으로 별도 확인\n'
                       '이전 장/마지막 판단 표시는 화면 기록이며 현재 주문 허가가 아닙니다.')
            value = self._model_buy_display(model, item)
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
        self._update_model_detail()
        self._update_latest_model_summary()

    def _refresh_visible_model_display(self, *_):
        """Recheck time-dependent labels without repainting hidden watch rows."""
        if not hasattr(self, 'model_score_summary'):
            return
        current = self.workspace_tabs.currentWidget()
        if current is self.watch_page:
            self._refresh_model_scores()
        elif (current is self.signal_connection_page
              and self.signal_connection_page.currentWidget() is self.mark14_panel):
            self._update_model_detail()
            self._update_latest_model_summary()

    def _update_model_detail(self, *_):
        panel = getattr(self, 'mark14_panel', None)
        if panel is None:
            return
        item = self.selected_item()
        if item is None:
            panel.selected_result.setText('선택 종목 —')
            return
        model = panel.model_selector.currentData()
        if model not in ALL_MODEL_IDS:
            panel.selected_result.setText('모델 선택 전')
            return
        panel.selected_result.setText(
            f'{self._model_name(model)} · {item.name or item.instrument.symbol} · '
            f'매수: {self._model_buy_display(model, item, compact=True)} · '
            f'매도: {self._model_sell_display(model, item)}')
        if model in MINUTE_TRANSFER_IDS:
            row = self._minute_model_diagnostic(model, item)
            panel.selected_result.setToolTip(
                '완료 일봉만 학습·실제 완료 5분봉 추론. 점수는 분봉 적중확률이나 수익률이 아닙니다.\n'
                + (f"최근 5분봉 {row.get('minute_bar_label', '—')} · 신호봉 {row.get('signal_bar_label', '—')}"
                   + (f"\n보류 이유: {row['error']}" if row.get('error') else '')
                   if row is not None else '해당 종목의 현재 분봉 판단은 아직 없습니다.'))
        elif model in MINUTE_HEDGE_IDS:
            panel.selected_result.setToolTip('인버스 ETF 두 다리의 동시 체결·비율 검증 전까지 주문 신호를 내지 않습니다.')
        else:
            panel.selected_result.setToolTip(PROTOTYPE_NOTICES.get(model, ''))

    def _set_latest_model_summary(self, text, *, full_text=None):
        self.latest_model_summary.setText(text)
        self.latest_model_summary.setToolTip(full_text or
            '최근 조회 기준. 장중 확률 평균에는 기존 장벽 모델만 포함하며 다른 모델 점수는 합산하지 않습니다.')
        # The header has a fixed height. Keep every model reachable by its own
        # horizontal scroll instead of making the chart disappear below it.
        self.latest_model_summary.adjustSize()

    def _update_latest_model_summary(self):
        """Show one atomic barrier snapshot plus every active model's own verdict."""
        if not hasattr(self, 'latest_model_summary'):
            return
        models = tuple(self._chosen_external_models())
        # Only the original three have the legacy probability average. The
        # ten daily-proxy scores are separate, optional display diagnostics.
        barrier_models = tuple(model for model in models
                               if model in BARRIER_MODELS and model not in INTRADAY_MODEL_IDS)
        completed = self._last_complete_model_summary
        if completed is not None and (completed[0] != models
                                      or completed[1] not in getattr(self, '_items_by_id', {})):
            completed = self._last_complete_model_summary = None
        item = getattr(self, '_items_by_id', {}).get(self._last_queried_watch_id)
        snapshot = self.snapshots.get(self._last_queried_watch_id)
        if item is None or snapshot is None:
            self._last_complete_model_summary = None
            self._set_latest_model_summary('최근 조회 종목 — · 모델 추정확률 —')
            return
        stamp = snapshot.fetched_at.astimezone().strftime('%m/%d %H:%M:%S')
        quote = (f'{item.name or item.instrument.symbol} ({item.instrument.symbol}) · {stamp}'
                 f' · 당시가 {snapshot.quote.price} {item.instrument.currency}')
        if not models:
            self._last_complete_model_summary = None
            self._set_latest_model_summary('최근 조회 ' + quote + ' · AI 모델 연결 없음')
            return
        minute_models = tuple(model for model in models if model in MINUTE_TRANSFER_IDS)
        minute_rows = {model: self._minute_model_diagnostic(model, item) for model in minute_models}
        minute_ready = sum(row is not None and row.get('reason') == 'MINUTE_BUY_CANDIDATE'
                           for row in minute_rows.values())
        minute_observed = sum(row is not None for row in minute_rows.values())
        minute_brief = (f' · 5분봉 매수 후보 {minute_ready}/{len(minute_models)}'
                        if minute_models and minute_observed == len(minute_models) else
                        f' · 5분봉 판단 대기 {len(minute_models) - minute_observed}개'
                        if minute_models else '')
        hedge_brief = (' · 인버스 연구 주문 보류'
                       if any(model in MINUTE_HEDGE_IDS for model in models) else '')
        scores = [self._current_model_score(model, item) for model in barrier_models]
        if any(score is None for score in scores):
            if completed is not None and not minute_models:
                self._set_latest_model_summary(completed[2])
                return
            pending = (' · '.join(f'{self._model_name(model)} —' for model in models)
                       if len(models) <= 3 else f'모델 판단 대기 {len(models)}개')
            self._set_latest_model_summary('최근 조회 ' + quote + ' · 모델 추정확률 —'
                                           + minute_brief + hedge_brief + ' · ' + pending)
            return
        probability_by_model = dict(zip(barrier_models, scores))
        proxy_scores = {model: self._current_model_score(model, item)
                        for model in models if model in (*INTRADAY_MODEL_IDS, *TARGET_HORIZON_MODEL_IDS)}
        proxy_pending = any(score is None for score in proxy_scores.values())
        details = []
        for index, model in enumerate(models):
            if model in probability_by_model:
                score = probability_by_model[model]
                verdict = '매수' if score['reason'] == 'PREDICTED_DAILY_BARRIER_SUCCESS' else '관망'
                buy = f"{verdict}({score['probability']:.1%})"
            elif model in INTRADAY_MODEL_IDS:
                score = proxy_scores[model]
                verdict = ('매수' if score is not None
                           and score['reason'] == 'PREDICTED_DAILY_BARRIER_SUCCESS' else '관망')
                buy = (f"{verdict}({score['probability']:.1%} 대리점수)"
                       if score is not None else '대리점수 —')
            elif model in TARGET_HORIZON_MODEL_IDS:
                score = proxy_scores[model]
                verdict = ('연구 후보' if score is not None
                           and score['reason'] == 'TARGET_HORIZON_CANDIDATE' else '관망')
                buy = (f"{verdict}({score['probability']:.1%} 시가 대리점수)"
                       if score is not None else '대리점수 —')
            elif model in MINUTE_RESEARCH_IDS:
                buy = self._minute_buy_display(model, item)
            else:
                buy = self._preopen_score_display(model, item)
            sell = self._model_sell_display(model, item)
            priority = (0 if '매수' in buy or '매도 조건' in sell else
                        1 if buy not in {'—', '대리점수 —'} else 2)
            details.append((priority, index, f'{self._model_name(model)} {buy} / {sell}'))
        average = (' · 장중 확률 평균 '
                   + f"{sum((score['probability'] for score in scores), Decimal(0)) / Decimal(len(scores)):.1%}"
                   if scores else '')
        text = ('최근 조회 ' if proxy_pending else '최근 완료 조회 ') + quote + average
        text += minute_brief + hedge_brief
        if proxy_pending:
            text += ' · 대리점수 조회 중'
        top = sorted(details)[:3]
        text += ' · ' + '  |  '.join(value for _, _, value in top)
        if len(details) > len(top):
            text += f' · 외 {len(details) - len(top)}개 모델'
        full_text = quote + average + '\n' + '\n'.join(value for _, _, value in details)
        if not proxy_pending and minute_observed == len(minute_models):
            self._last_complete_model_summary = (models, item.id, text)
        self._set_latest_model_summary(text, full_text=full_text)

    def _update_selected_model_scores(self):
        if not hasattr(self, 'model_score_summary'):
            return
        item = self.selected_item()
        if item is None:
            self.model_score_summary.setText('현재가 —')
            self.price_change_summary.setText('전일 대비 —')
            return
        snapshot = self.snapshots.get(item.id)
        if snapshot is None or item.id in self.errors:
            self.model_score_summary.setText('현재가 —')
            self.model_score_summary.setToolTip(str(self.errors.get(item.id) or '시세 조회 전'))
        else:
            self.model_score_summary.setText(f'현재가 {snapshot.quote.price} {item.instrument.currency}')
            self.model_score_summary.setToolTip(
                f'{item.instrument.symbol} · 조회 {snapshot.fetched_at.astimezone():%m/%d %H:%M:%S}'
                + (' · 이전 조회값' if item.id not in self.fresh_ids else ''))
        change = self._previous_close_change(item, snapshot)
        if change is None:
            self.price_change_summary.setText('전일 대비 —')
            self.price_change_summary.setToolTip('전 거래일의 완료 일봉·신선한 시세를 모두 확인해야 표시합니다.')
            color = '#e9f5ff'
        else:
            rate, previous_day, previous_close = change
            self.price_change_summary.setText(f'전일 대비 {rate:+.2f}%')
            self.price_change_summary.setToolTip(
                f'{previous_day:%Y-%m-%d} 완료 일봉 종가 {previous_close} {item.instrument.currency} 기준')
            color = '#78e6c7' if rate > 0 else '#ed7892' if rate < 0 else '#e9f5ff'
        self.price_change_summary.setStyleSheet(
            f'QLabel#priceChangeSummary {{ background: #193148; color: {color}; '
            'border: 1px solid #31536d; border-radius: 6px; padding: 4px 8px; font-weight: 600; }')

    def _previous_close_change(self, item, snapshot):
        """Use only the preceding exchange session's completed daily bar."""
        if snapshot is None or item.id in self.errors or item.id not in self.fresh_ids:
            return None
        from dockdack.history import market_time
        from dockdack.market_schedule import session_on
        market = item.instrument.market
        quote_day = market_time(market, snapshot.fetched_at).date()
        previous_day = next((quote_day - timedelta(days=offset) for offset in range(1, 16)
                             if session_on(market, quote_day - timedelta(days=offset)) is not None), None)
        if previous_day is None:
            return None
        bar = next((bar for bar in reversed(snapshot.history.bars) if bar.day == previous_day), None)
        if bar is None or not bar.close.is_finite() or bar.close <= 0:
            return None
        price = snapshot.quote.price
        if not price.is_finite() or price <= 0:
            return None
        return ((price / bar.close - Decimal(1)) * Decimal(100), previous_day, bar.close)

    def select_item(self, *_):
        super().select_item(*_)
        self._update_selected_model_scores()
        self._update_model_detail()

    def _update_watch_row(self, key):
        super()._update_watch_row(key)
        self._update_model_score_row(key)
        self._update_selected_model_scores()

    def reload_tables(self, *, items=None, rules=None):
        super().reload_tables(items=items, rules=rules)
        self._sync_watch_model_columns()
        valid_ids = set(getattr(self, '_items_by_id', {}))
        self._last_model_scores = {key: score for key, score in self._last_model_scores.items()
                                   if key[1] in valid_ids}
        self._last_preopen_scores = {key: score for key, score in self._last_preopen_scores.items()
                                     if key[1] in valid_ids}
        if self._last_queried_watch_id not in valid_ids:
            self._last_queried_watch_id = None
        self._refresh_model_scores()

    def _move_legacy_settings(self):
        """Move advanced controls and logs without changing saved connections or policy."""
        self.advanced_mode_panel = QWidget()
        options = QVBoxLayout(self.advanced_mode_panel)
        self.advanced_mode_note = QLabel(
            '일반 AI 자동매매는 AI 추론 모델에서 모델을 선택하세요.\n'
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
        self.advanced_sources_panel = QWidget()
        sources_layout = QVBoxLayout(self.advanced_sources_panel)
        sources_note = QLabel('사용자 추가 신호기 · 출처 ID와 JSON 경로')
        sources_layout.addWidget(sources_note)
        for widget in (self.additional_sources, self.source_status):
            self.external_grid.removeWidget(widget)
            sources_layout.addWidget(widget)
        self.tabs.insertTab(1, self.advanced_sources_panel, '직접 신호기')
        self.workspace_tabs.removeTab(self.workspace_tabs.indexOf(self.operations_panel))
        self.tabs.addTab(self.operations_panel, '서버·감시 로그')
        self.tabs.currentChanged.connect(lambda _: self._reload_activity(visible_force=True))
        self.tabs.setCurrentWidget(self.advanced_mode_panel)
        index = self.workspace_tabs.indexOf(self.tabs)
        self.workspace_tabs.setTabText(index, '고급설정')
        self.workspace_tabs.setTabVisible(index, True)
        # Compatibility handle for saved/pre-existing callbacks. The ordinary
        # navigation is the permanently visible top-level tab.
        self.advanced_settings_button = QPushButton('고급설정', self)
        self.advanced_settings_button.setCheckable(True)
        self.advanced_settings_button.setAutoDefault(False)
        self.advanced_settings_button.setObjectName('linkButton')
        self.advanced_settings_button.setToolTip('고급설정 탭에서 수동 규칙·구형 모델·직접 신호기·서버 로그를 확인합니다.')
        self.advanced_settings_button.toggled.connect(self._show_advanced_settings)
        self.advanced_settings_button.hide()
        self.external_mode.toggled.connect(self._update_connection)

    def _workspace_changed(self, *_):
        previous = getattr(self, '_last_workspace_page', None)
        current = self.workspace_tabs.currentWidget()
        super()._workspace_changed()
        if not hasattr(self, '_last_workspace_page'):
            return  # Base constructor has not finished wiring this desktop.
        self._last_workspace_page = current
        if current is self.watch_page and previous is not current:
            self._refresh_model_scores()
        elif current is self.signal_connection_page and previous is not current:
            self._refresh_visible_model_display()
        if current is self.portfolio_panel and previous is not current:
            self.refresh_portfolio()  # PortfolioCache keeps the per-market 60 s minimum.

    def _workspace_tab_clicked(self, index):
        # QTabBar emits tabBarClicked before currentChanged. A click on another
        # tab is handled by _workspace_changed; only re-clicks refresh here.
        if (index == self.workspace_tabs.currentIndex()
                and self.workspace_tabs.widget(index) is self.portfolio_panel):
            self.refresh_portfolio()

    def _activity_page(self):
        # Present the nested log page to the base activity loader so its normal
        # current-category filtering and periodic throttle still apply.
        advanced = getattr(self, 'tabs', None)
        workspace = getattr(self, 'workspace_tabs', None)
        ai_page = getattr(self, 'signal_connection_page', None)
        if (ai_page is not None and workspace is not None
                and workspace.currentWidget() is ai_page
                and ai_page.currentWidget() is getattr(self, 'model_performance_panel', None)):
            return self.model_performance_panel
        if (advanced is not None and workspace is not None
                and workspace.currentWidget() is advanced
                and advanced.currentWidget() is getattr(self, 'operations_panel', None)):
            return self.operations_panel
        return super()._activity_page()

    def _show_advanced_settings(self, visible):
        if not visible and self.workspace_tabs.currentWidget() is self.tabs:
            self.workspace_tabs.setCurrentWidget(self.signal_connection_page)
        if visible:
            self.workspace_tabs.setCurrentWidget(self.tabs)

    def open_connection_settings(self):
        self.workspace_tabs.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self.external_panel)
        if self.monitoring or self.worker:
            self.message.setText('현재 적용 설정입니다. 연결을 변경하려면 감시·주문을 중지하세요.')

    def _chosen_trigger(self):
        return self.model_trigger.currentData() if hasattr(self, 'model_trigger') else 'none'

    def _chosen_external_models(self):
        return tuple(model for model, check in getattr(self, 'external_model_checks', {}).items()
                     if check.isChecked())

    def _set_all_models_selected(self, selected):
        # A bulk selection is one configuration change, not one worker/feed
        # rebuild per model. It never starts monitoring or arms orders.
        if (selected_mode(self.service) is not TradingMode.DEMO
                or not self.external_panel.isEnabled()
                or self.monitoring or self.worker is not None or self.pending_auto_arm):
            return
        checks = self.external_model_checks
        if all(check.isChecked() == selected for check in checks.values()):
            return
        for check in checks.values():
            blocked = check.blockSignals(True)
            try:
                check.setChecked(selected)
            finally:
                check.blockSignals(blocked)
        self.mark14_panel._update_selection_count()
        self._model_trigger_changed()
        self._queue_save_preferences()

    def _sync_watch_model_columns(self, *_):
        """Keep fixed model data indices; collapse only unchecked display columns."""
        checks = getattr(self, 'external_model_checks', None)
        if checks is None:
            return
        selected = {model for model, check in checks.items() if check.isChecked()}
        expected = 3 + len(WATCH_MODEL_IDS)
        for view in self.watch_tables.values():
            if view.columnCount() != expected:
                continue
            for column, model in enumerate(WATCH_MODEL_IDS, 1):
                view.setColumnHidden(column, model not in selected)
            view.setColumnHidden(expected - 2, False)  # Current price.
            view.setColumnHidden(expected - 1, False)  # Last lookup.

    def _prototype_output_path(self, model):
        return self.store.path.parent / 'exchange' / 'external-models' / model / 'signals.json'

    def _close_external_feeds(self):
        for feed in self._prototype_feeds.values():
            feed.close()
        self._prototype_feeds = {}
        self._mark14_preopen_ready_sessions.clear()
        self._mark14_preopen_attempts.clear()
        self._last_model_scores.clear()
        if hasattr(self, 'mark14_panel'):
            self.mark14_panel.model_results.clear()
            self.mark14_panel._show_selected_model()

    def _preopen_checkpoint(self, progress):
        """Freeze each selected model against one verified top-100 batch."""
        feeds = {model: self._prototype_feeds[model] for model in PREOPEN_MODEL_IDS
                 if model in self._prototype_feeds}
        if not feeds or self.engine._stop.is_set():
            return
        from dockdack.history import market_time
        from dockdack.market_schedule import session_on
        from dockdack.mark1_4_preopen import collect_preopen_candidates
        now = self.engine.clock()
        for market in (Market.DOMESTIC, Market.US):
            try:
                session = session_on(market, market_time(market, now).date())
            except ValueError as exc:
                for model in feeds:
                    progress(('mark14_preopen', {'model_id': model, 'market': market.value,
                        'state': 'unavailable', 'reason': str(exc), 'candidates': []}))
                continue
            if session is None or not session.opened - timedelta(minutes=10) <= now < session.opened:
                continue
            pending = []
            for model, feed in feeds.items():
                key = (model, market.value, session.opened)
                if key in self._mark14_preopen_ready_sessions:
                    if feed.client.is_alive:
                        continue
                    # A restarted child lost its frozen plan. Prepare anew.
                    self._mark14_preopen_ready_sessions.discard(key)
                previous = self._mark14_preopen_attempts.get(key)
                if previous is None or (now - previous).total_seconds() >= 30:
                    pending.append((model, feed, key))
            if not pending:
                continue
            gathered = collect_preopen_candidates(self.store, self.engine, market)
            if not gathered.ok:
                for model, _, key in pending:
                    self._mark14_preopen_attempts[key] = now
                    progress(('mark14_preopen', {'model_id': model, 'market': market.value,
                        'state': 'unavailable', 'session_open': session.opened.isoformat(),
                        'reason': gathered.reason, 'candidates': []}))
                continue
            for model, feed, key in pending:
                if self.engine._stop.is_set() or self.engine.clock() >= session.opened:
                    break
                self._mark14_preopen_attempts[key] = now
                try:
                    result = feed.prepare_preopen(market.value, gathered.candidates,
                                                  now=self.engine.clock(), session=session)
                except Exception as exc:
                    result = {'market': market.value, 'state': 'unavailable',
                              'session_open': session.opened.isoformat(),
                              'reason': f'{type(exc).__name__}: {exc}', 'candidates': []}
                if result.get('state') == 'prepared':
                    self._mark14_preopen_ready_sessions.add(key)
                if model != MARK14_TRIGGER:
                    result = {**result, 'model_id': model}
                progress(('mark14_preopen', result))

    def _frozen_open_watch_ids(self, market, now):
        """Protect only prepared DEMO selections until five minutes after open."""
        if selected_mode(self.service) is not TradingMode.DEMO:
            return ()
        from dockdack.lstm30_adapter import timestamp
        from dockdack.history import market_time
        from dockdack.market_schedule import session_on
        try:
            session = session_on(market, market_time(market, now).date())
            if session is None or not session.opened <= now < session.opened + timedelta(minutes=5):
                return ()
            protected = []
            for model in PREOPEN_MODEL_IDS:
                feed = self._prototype_feeds.get(model)
                if feed is None or not getattr(feed.client, 'is_alive', False):
                    continue
                plan = feed.plans.get(market.value)
                if not isinstance(plan, dict) or plan.get('state') != 'prepared' or plan.get('scored_count') != 100:
                    continue
                opened = timestamp(plan['session_open'])
                if opened != session.opened:
                    continue
                selected = tuple(row['watch_id'] for row in plan['candidates']
                                 if row.get('selected') is True)
                if (plan.get('selected_count') != len(selected) or len(selected) > 10
                        or len(set(selected)) != len(selected)
                        or any(not key.startswith(market.value + ':') for key in selected)):
                    continue
                protected.extend(selected)
            return tuple(dict.fromkeys(protected))
        except (KeyError, TypeError, ValueError, AttributeError):
            return ()

    def _poll_priority_watch_ids(self):
        """Visit frozen opening candidates first; holdings exits still precede buys."""
        now = self.engine.clock()
        return tuple(dict.fromkeys(key for market in Market
                                   for key in self._frozen_open_watch_ids(market, now)))

    @property
    def builtin_confirmation_notice(self):
        selected = self._chosen_external_models()
        legacy = sum(model in (MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER)
                     for model in selected)
        preopen = sum(model in PREOPEN_MODEL_IDS for model in selected)
        proxy = sum(model in INTRADAY_MODEL_IDS for model in selected)
        target_horizon = sum(model in TARGET_HORIZON_MODEL_IDS for model in selected)
        notices = []
        if legacy:
            notices.append(f'장중 확률 모델 {legacy}개: 각 모델의 50% 초과 신호만 모의 매수 후보로 사용하고 '
                           '실제 매수분에 모델별 익절·손절 기준을 적용합니다. 추정확률은 수익 보장이 아닙니다.')
            notices.extend(PROTOTYPE_NOTICES[model] for model in selected
                           if model in (MARK1_TRIGGER, MARK11_TRIGGER, MARK12_TRIGGER))
        if preopen:
            notices.append(f'장전 점수 모델 {preopen}개: 점수는 확률이 아니며 모델마다 후보·매도 시점이 다릅니다. '
                           '1회 매수 비중은 모델 선택 화면 설정과 주문 상한을 적용합니다. 모의 전용입니다.')
        if proxy:
            notices.append(f'일봉 대리 모델 {proxy}개: 완료 일봉과 현재가로 일봉 전체 대리사건을 판단합니다. '
                           '장중 진입 후 경로·수익성 미검증이며 모의 전용입니다.')
        if target_horizon:
            notices.append(f'목표가·보유기간 연구 모델 {target_horizon}개: 완료 일봉과 현재가로 다음 날 시가 기준 사건을 '
                           '대리 평가합니다. 손절 없이 목표가 또는 체결일 포함 H번째 거래 세션 마감 5분 전부터 매도를 '
                           '시도합니다. 장중 경로·목표가 체결·수익성이 미검증인 모의 전용 신호입니다.')
        return '\n'.join(notices)

    def _model_buy_sizing_confirmation(self):
        selected = self._chosen_external_models()
        if not selected:
            return (f'1회 매수: 시장별 계좌 평가금액의 {self.buy_percent.value():g}% '
                    '· 정수 주식 수 내림\n')
        rows = [f'{PROTOTYPE_TITLES[model]} {self.mark14_panel.model_buy_percents[model].value():g}%'
                for model in selected if model not in MINUTE_HEDGE_IDS]
        research_count = sum(model in MINUTE_HEDGE_IDS for model in selected)
        research_note = (f'인버스 연구 {research_count}종은 주문 불가·매수 비중 없음.\n'
                         if research_count else '')
        if not rows:
            return research_note + f'기타 신호 기본 {self.buy_percent.value():g}% · 정수 주식 수 내림\n'
        return ('모델별 1회 매수 상한 (시장별 계좌 평가금액 기준):\n'
                + '\n'.join(' · '.join(rows[index:index + 4])
                            for index in range(0, len(rows), 4))
                + '\n' + research_note
                + f'기타 신호 기본 {self.buy_percent.value():g}% · 정수 주식 수 내림\n')

    def _progress(self, data):
        if len(data) == 2 and data[0] == 'mark14_preopen':
            self.mark14_panel.show_preopen(data[1])
            self._refresh_model_scores()
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
        # Rebuilding dependent labels can ask Qt to reveal a focused widget at
        # the end of the long card list. Keep the user's current card in view.
        choice_scroll = getattr(getattr(self, 'mark14_panel', None), 'choices_scroll', None)
        if choice_scroll is not None:
            offset = choice_scroll.verticalScrollBar().value()
            QTimer.singleShot(0, lambda area=choice_scroll, value=offset:
                              area.verticalScrollBar().setValue(value))
        choice = self._chosen_trigger()
        enabled_models = self._chosen_external_models()
        if enabled_models and selected_mode(self.service) is not TradingMode.DEMO:
            for check in self.external_model_checks.values():
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
            self._sync_watch_model_columns()
            self.engine.disarm()
            self.message.setText('prototype 외부 신호기는 모의투자에서만 사용할 수 있습니다. 실전 주문에 연결하지 않았습니다.')
            return
        # Programmatic changes cannot swap a running worker's active strategy.
        if self.monitoring or self.worker is not None or self.pending_auto_arm:
            self.stop_monitoring()
        self.engine.disarm()
        self._sync_watch_model_columns()
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
        self.model_notice.setText('\n\n'.join(PROTOTYPE_NOTICES[model]
                                               for model in enabled_models if model in BARRIER_MODELS))
        self.model_notice.hide()
        self._sync_model_mode_controls()
        self._update_connection()
        self._refresh_model_scores()

    def _sync_model_mode_controls(self):
        if not hasattr(self, 'model_trigger'):
            return
        if self._chosen_external_models():
            self.environment_selector.buttons[TradingMode.REAL].setEnabled(False)
            self.environment_selector.buttons[TradingMode.REAL].setToolTip('prototype 외부 신호기는 모의투자 전용입니다. 모든 모델 연결을 끄면 환경 전환할 수 있습니다.')
        else:
            self.environment_selector.apply(selected_mode(self.service), pending=self._pending_environment is not None)
        if hasattr(self, 'advanced_mode_panel'):
            # These widgets used to inherit the external panel's editing lock.
            self.advanced_mode_panel.setEnabled(self.external_panel.isEnabled())
            self.advanced_sources_panel.setEnabled(self.external_panel.isEnabled())
            self.external_mode.setEnabled(not (self._chosen_external_models()
                                               or self._chosen_trigger() != 'none'
                                               or self.random_demo.isChecked()))

    def request_environment(self, mode):
        if TradingMode(mode) is TradingMode.REAL and self._chosen_external_models():
            self.engine.disarm()
            self.message.setText('prototype 외부 신호기는 모의투자 전용입니다. 실전 전환 전에 모든 모델 연결을 끄세요.')
            return
        self._preferences_timer.stop()
        self._save_preferences()
        return super().request_environment(mode)

    def _sync_environment(self):
        super()._sync_environment()
        self._sync_model_mode_controls()

    def _external_config_key(self):
        return (*super()._external_config_key(), self._chosen_trigger(), self._chosen_external_models())

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
            self._preferences_timer.stop()
            self._last_model_scores.clear()
            self._last_preopen_scores.clear()
            self._last_queried_watch_id = None
            self.external_quantity.setValue(999999999)
            self._apply_execution_preferences()
            self._restore_preferences(after_switch=True)
            self._refresh_model_scores()

    def update_controls(self):
        super().update_controls()
        # The sizing input has moved out of the externally disabled settings
        # panel, so preserve the same edit lock while a sweep/monitor runs.
        if hasattr(self, 'buy_percent') and hasattr(self, 'external_panel'):
            editing = self.external_panel.isEnabled()
            self.buy_percent.setEnabled(editing)
            # Model checkboxes are outside the old external panel too. A click
            # during a sweep would otherwise close live feeds from the GUI
            # thread while the worker is still consuming them.
            for check in getattr(self, 'external_model_checks', {}).values():
                check.setEnabled(editing and selected_mode(self.service) is TradingMode.DEMO)
            if hasattr(self, 'mark14_panel'):
                can_select = editing and selected_mode(self.service) is TradingMode.DEMO
                self.mark14_panel.select_all_button.setEnabled(can_select)
                self.mark14_panel.clear_all_button.setEnabled(can_select)
                self.mark14_panel.bulk_buy_percent.setEnabled(can_select)
                self.mark14_panel.apply_bulk_buy_percent_button.setEnabled(can_select)
                for field in self.mark14_panel.model_buy_percents.values():
                    field.setEnabled(editing and selected_mode(self.service) is TradingMode.DEMO)
        self._sync_model_mode_controls()

    def _update_connection(self):
        super()._update_connection()
        if hasattr(self, 'model_status'):
            if hasattr(self, 'ai_model_count'):
                count = len(self._chosen_external_models())
                self.ai_model_count.setText(str(count))
                self.ai_connection_card.setAccessibleName(f'AI 모델 {count}개 선택')
            producer = self.test_producer
            choice = self._chosen_trigger()
            selected_models = self._chosen_external_models()
            def selected_status(model):
                fallback = ('인버스 헤지 연구 선택됨 · 주문 연결 없음' if model in MINUTE_HEDGE_IDS
                            else '외부 연결 선택됨 · 첫 조회 시 별도 프로세스 시작')
                return getattr(self._prototype_feeds.get(model), 'status',
                               f'{PROTOTYPE_TITLES[model]} · {fallback}')
            text = '\n'.join(
                selected_status(model) for model in selected_models)
            if not text:
                text = ((producer.status if isinstance(producer, DesktopModelBridge) else '내장 LSTM · 첫 장중 조회 시 모델 확인')
                        if choice == 'lstm30' else '외부 AI 연결 꺼짐 · 직접 연결한 외부 JSON은 별도 사용')
            failed = tuple(model for model in self._chosen_external_models()
                           if any(term in str(getattr(self._prototype_feeds.get(model), 'status', '')).split('\n', 1)[0]
                                  for term in ('실패', '불가', '오류')))
            self.ai_connection_warning.setVisible(bool(failed))
            self.ai_connection_card.setAccessibleName(
                f'AI 모델 {len(self._chosen_external_models())}개 선택'
                + (f' · {len(failed)}개 신호 점검' if failed else ''))
            self.model_status.setToolTip(text)
            if hasattr(self, 'ai_connection_card'):
                research_count = sum(model in MINUTE_HEDGE_IDS for model in selected_models)
                minute_count = sum(model in MINUTE_TRANSFER_IDS for model in selected_models)
                scope = (f' · 분봉 매매 후보 {minute_count}개' if minute_count else '')
                scope += (f' · 인버스 연구 전용 {research_count}개(주문 불가)' if research_count else '')
                self.ai_connection_card.setToolTip(
                    f'선택한 모델 {len(selected_models)}개{scope}. 선택 수는 감시·주문 상태가 아닙니다.\n' + text)
            if self._chosen_external_models():
                text = '\n'.join(selected_status(model).splitlines()[0] for model in selected_models)
                self.connection_summary.setText(f'AI 모델 {len(self._chosen_external_models())}개 선택')
            elif self.external_mode.isChecked():
                self.connection_summary.setText('AI 모델 연결 없음')
            self.model_status.setText(text)
            builtin = producer.builtin if isinstance(producer, ExternalFeedGroup) else producer
            if isinstance(builtin, DesktopModelBridge) and builtin._load_error:
                warning = '내장 LSTM 실행 불가 · AI 추론 모델에서 원인 확인 · 다른 연결은 별도 운영'
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
            from dockdack.signals.mark1_4_external import Mark14ExternalFeed
            from dockdack.mark1_trigger import SharedPrototypeAccounts
            factory = self._external_feed_factory or ExternalPrototypeFeed
            shared = SharedPrototypeAccounts(self)
            minute_feed = None
            policy = result[2]
            try:
                for model in self._chosen_external_models():
                    source, path = PROTOTYPE_SOURCES[model], self._prototype_output_path(model)
                    if source in seen_sources or path.resolve() in seen_paths:
                        raise ValueError('외부 AI 신호 출처/파일이 다른 연결과 중복됩니다.')
                    extra = ExternalPolicy(source, policy.max_quantity, policy.max_krw, policy.max_usd)
                    bundle = {MARK1_TRIGGER: self._mark1_bundle,
                              MARK11_TRIGGER: self._mark11_bundle,
                              MARK12_TRIGGER: self._mark12_bundle,
                              MARK14_TRIGGER: self._mark14_bundle}.get(model)
                    kwargs = {'bundle_root': bundle}
                    if self._external_feed_factory is None:
                        kwargs['account_snapshots'] = shared
                    if self._external_feed_factory is None and model in MINUTE_TRANSFER_IDS:
                        from dockdack.signals.daily_proxy_minute import DailyProxyMinuteFeed, SharedDomesticMinuteFeed
                        if minute_feed is None:
                            minute_feed = SharedDomesticMinuteFeed(self)
                        feed = DailyProxyMinuteFeed(
                            self, model, extra, path, minute_feed=minute_feed,
                            account_snapshots=shared)
                    elif self._external_feed_factory is None and model in MINUTE_HEDGE_IDS:
                        from dockdack.signals.daily_proxy_minute import MinuteResearchHoldFeed
                        feed = MinuteResearchHoldFeed(self, model, extra, path)
                    elif self._external_feed_factory is None and model in PREOPEN_MODEL_IDS:
                        from dockdack.signals.preopen_series import PreopenExperimentalFeed
                        feed_type = Mark14ExternalFeed if model == MARK14_TRIGGER else PreopenExperimentalFeed
                        feed = feed_type(self, model, extra, path, **kwargs)
                    else:
                        feed_type = factory
                        feed = feed_type(self, model, extra, path, **kwargs)
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
        self._preferences_timer.stop()
        self._save_preferences()
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


def desktop_model_choices(trigger, no_model, external_models, *, trading_mode=TradingMode.DEMO):
    models = list(external_models)
    if trigger is None and not no_model and not models and trading_mode is TradingMode.DEMO:
        # A fresh DEMO desktop shows every model. Selection never arms orders;
        # persisted explicit OFF choices are restored by the window afterward.
        return 'none', list(PROTOTYPE_NOTICES)
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
    parser.add_argument('--mark14-bundle', type=Path, help='mark1.4 prototype 저장 모델 폴더')
    parser.add_argument('--external-model', action='append', choices=tuple(PROTOTYPE_NOTICES), default=[],
                        help='외부 AI 신호기. 복수 모델을 함께 쓰려면 각각 지정 (자동주문은 OFF)')
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
        trigger, models = desktop_model_choices(args.trigger, args.no_model, args.external_model,
                                                trading_mode=selected_mode(service))
        window = V00Window(service=service, store=store, builtin=not args.no_model,
                           trigger=trigger, mark1_bundle=args.mark1_bundle,
                           mark11_bundle=args.mark11_bundle, mark12_bundle=args.mark12_bundle,
                           mark14_bundle=args.mark14_bundle,
                           external_models=models, session_lock=lock,
                           restore_model_choices=not (args.no_model or args.trigger is not None or args.external_model))
        if legacy_notice:
            window.message.setText(legacy_notice)
        window.show()
        window.refresh_portfolio()
        return app.exec()
    finally:
        lock.release()


if __name__ == '__main__':
    raise SystemExit(main())
