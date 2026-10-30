"""Bounded, thread-safe session timings without credentials or order payloads."""
from collections import Counter, deque
from threading import Lock


class PipelineMetrics:
    def __init__(self, capacity=300):
        self._samples = deque(maxlen=capacity)
        self._counts = Counter()
        self._lock = Lock()

    def record(self, reason, seconds):
        with self._lock:
            self._samples.append(max(0.0, float(seconds)))
            self._counts[str(reason)] += 1

    def snapshot(self):
        with self._lock:
            ordered = sorted(self._samples)
            return {'samples': len(ordered), 'counts': dict(self._counts),
                    'p95_seconds': ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)] if ordered else 0.0,
                    'max_seconds': ordered[-1] if ordered else 0.0}

    def summary(self):
        stats = self.snapshot()
        return f"처리 {stats['samples']}회 · 최근 처리시간 P95 {stats['p95_seconds']:.2f}초 · 최대 {stats['max_seconds']:.2f}초"
