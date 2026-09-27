"""Adapters from sealed research bundles to a common pre-open scoring wire."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dockdack.mark1_8_bundle import load_bundle


class Mark18PreopenPredictor:
    """Preserve both Mark1.8's rank decision and its learned equity fraction."""

    def __init__(self, bundle_root: str | Path, market: str):
        self.bundle = load_bundle(Path(bundle_root), market=market)
        self.metadata = self.bundle.metadata
        self._universe = {(row["symbol"], row["exchange"])
                          for row in self.metadata["selected_symbols"]}

    def score_many(self, windows, symbols) -> list[dict]:
        values = np.asarray(windows, dtype=np.float32)
        if (values.ndim != 3 or values.shape[1:] != (30, 5) or
                len(values) != len(symbols) or not 1 <= len(values) <= 100 or
                not np.isfinite(values).all()):
            raise ValueError("Mark1.8 requires 1–100 complete 30×5 windows")
        pairs = tuple(symbols)
        if (len(set(pairs)) != len(pairs) or
                any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                    not all(isinstance(part, str) and part for part in pair)
                    for pair in pairs)):
            raise ValueError("Mark1.8 requires distinct symbol/exchange pairs")
        scores, fractions = self.bundle.predict(values)
        threshold = float(self.bundle.threshold)
        if (scores.shape != (len(values),) or fractions.shape != (len(values),) or
                not np.isfinite(scores).all() or not np.isfinite(fractions).all() or
                np.any(fractions < 0) or np.any(fractions > .1 + 1e-7)):
            raise ValueError("Mark1.8 produced an invalid rank score or equity fraction")
        return [{
            "symbol": symbol, "exchange": exchange,
            "score": float(score),
            "equity_fraction": float(fraction),
            "frozen_numeric_score_threshold": threshold,
            "score_metric": self.metadata["score_metric"],
            "score_unit": self.metadata["score_unit"],
            "above_frozen_threshold": bool(score > threshold and fraction > 0),
            "in_training_universe": (symbol, exchange) in self._universe,
            "out_of_training_universe": (symbol, exchange) not in self._universe,
        } for (symbol, exchange), score, fraction in zip(pairs, scores, fractions)]
