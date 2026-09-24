"""Local database jobs with immutable inputs, no window or broker dependency.

Only their Qt completion signals may update widgets. Store identity accompanies
each result so a replaced environment cannot receive another account's data.
"""
from dockdack.activity_snapshot import collect_event_logs
from dockdack.gui import Worker


class WorkspaceReadWorker(Worker):
    """One serialized metadata/cache read and optional model lookback upgrade."""
    def __init__(self, store, *, minimum_days=0):
        self.store, self.minimum_days = store, minimum_days
        super().__init__(self.collect)

    def collect(self):
        if self.minimum_days:
            with self.store.connection() as db:
                db.execute('UPDATE watchlist SET days=? WHERE days<?',
                           (self.minimum_days, self.minimum_days))
        items = self.store.items()
        return self.store, items, self.store.rules(limit=500), self.store.cached_snapshots(items)


class ActivityReadWorker(Worker):
    """Compute journal/accounting and event log snapshots outside Qt callbacks."""
    def __init__(self, store, collector, *, categories, previous_heads, ledger_needed, log_reader=None):
        self.store, self.collector = store, collector
        self.categories, self.previous_heads = tuple(categories), dict(previous_heads)
        self.ledger_needed = ledger_needed
        self.log_reader = collect_event_logs if log_reader is None else log_reader
        super().__init__(self.collect)

    def collect(self):
        logs = self.log_reader(self.store, self.categories, previous_heads=self.previous_heads)
        snapshot = self.collector.collect() if self.ledger_needed else None
        return self.store, logs, snapshot
