"""Worker-side local ledger/log collection; deliberately contains no Qt objects.

One collector belongs to one store/environment and one serialized worker. The
GUI only applies its finished payload. FIFO and daily totals share one complete
ledger calculation; the 500-row presentation cap never truncates accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dockdack.trade_journal import daily_trade_journal
from dockdack.model_performance import model_realized_performance


MAX_VISIBLE_ROWS = 500


@dataclass(frozen=True)
class LedgerSnapshot:
    revision: Any
    ledger: tuple[dict, ...]
    journal: dict
    model_performance: dict | None = None

    @property
    def performance(self):
        return self.journal["performance"]


class LedgerCollector:
    """Read/compute only when the durable ledger revision actually changes.

    `collect` may run in a background worker and performs no broker requests or
    writes. A store without revision support retains the equality-check fallback
    for simple callers/tests. Consumers must not mutate returned dictionaries.
    """

    def __init__(self, store):
        self.store = store
        self._snapshot = None
        self._revision = object()

    def collect(self, *, force=False):
        revision_reader = getattr(self.store, "ledger_revision", None)
        revision = revision_reader() if callable(revision_reader) else None
        if not force and revision is not None and revision == self._revision:
            return None
        ledger = self.store.order_history(limit=None)
        if self._snapshot is not None and ledger == self._snapshot.ledger:
            self._revision = revision
            return self._snapshot if force else None
        ledger = tuple(dict(row) for row in ledger)
        journal = daily_trade_journal(ledger)
        models = model_realized_performance(ledger, mode=getattr(self.store, 'mode', 'demo'),
                                           performance=journal['performance'])
        snapshot = LedgerSnapshot(revision, ledger, journal, models)
        self._snapshot, self._revision = snapshot, revision
        return snapshot


def collect_event_logs(store, categories, *, previous_heads=None, force=False):
    """Read bounded changed log categories, safe to call outside the Qt thread."""
    heads = store.event_heads()
    previous_heads = previous_heads or {}
    events = {}
    for category in tuple(dict.fromkeys(categories)):
        if category not in {"system", "monitor", "signal", "order"}:
            raise ValueError("지원하지 않는 로그 분류입니다.")
        if force or previous_heads.get(category) != heads[category]:
            events[category] = store.events(limit=MAX_VISIBLE_ROWS, category=category)
    return {"heads": heads, "events": events}
