"""Account-scoped desktop preferences, separated from rendering and feeds."""
import sqlite3
from math import isfinite
from PySide6.QtCore import QTimer
from dockdack.environment_store import selected_mode
from dockdack.models import TradingMode
from dockdack.minute_model_catalog import MINUTE_HEDGE_IDS, MINUTE_TRANSFER_IDS


class DesktopPreferencesMixin:
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
            'bulk_buy_percent': self.mark14_panel.bulk_buy_percent.value(),
            'symbol_exposure_percent': self.symbol_exposure_percent.value(),
            'model_exposure_percent': self.model_exposure_percent.value(),
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

    def _commit_buy_percent_edits(self):
        # Keyboard tracking is off: a typed value is not yet value() until
        # focus/Enter commits it. Closing or switching must save that edit too.
        for field in (self.buy_percent, self.mark14_panel.bulk_buy_percent,
                      *(field for model, field in self.mark14_panel.model_buy_percents.items()
                        if model not in MINUTE_HEDGE_IDS)):
            field.interpretText()

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
        previous_target = panel.bulk_buy_percent.value()
        blocked = panel.bulk_buy_percent.blockSignals(True)
        try:
            panel.bulk_buy_percent.interpretText()
        finally:
            panel.bulk_buy_percent.blockSignals(blocked)
        value = panel.bulk_buy_percent.value()
        fields = [field for model, field in panel.model_buy_percents.items()
                  if model not in MINUTE_HEDGE_IDS]
        if all(field.value() == value for field in fields):
            if value != previous_target:
                self._queue_save_preferences()
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
                       self.external_usd, self.buy_percent, self.mark14_panel.bulk_buy_percent):
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
        from dockdack.ui.v00_app import PROTOTYPE_NOTICES, PREOPEN_MODELS
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
                    self.mark14_panel.bulk_buy_percent,
                    self.symbol_exposure_percent, self.model_exposure_percent,
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
            self.symbol_exposure_percent.setValue(0)
            self.model_exposure_percent.setValue(0)
            restore_number('symbol_exposure_percent', self.symbol_exposure_percent)
            restore_number('model_exposure_percent', self.model_exposure_percent)
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
            # Older settings stored the individual values but not the bulk
            # input. Recover a shared percentage without applying it again.
            order_percents = {field.value()
                              for model, field in self.mark14_panel.model_buy_percents.items()
                              if model not in MINUTE_HEDGE_IDS}
            self.mark14_panel.bulk_buy_percent.setValue(
                next(iter(order_percents)) if len(order_percents) == 1 else 1)
            restore_number('bulk_buy_percent', self.mark14_panel.bulk_buy_percent)
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
