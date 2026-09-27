"""Offline Mark1.3 timing and cost tests with disposable raw SQLite data."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dockdack.mark1_3_research import (chronological_splits, fixed_open_close_returns,
                                       load_raw_daily_candidates, run_experiment)
from test_clean_daily_dataset import catalog_item, create_source, price_rows, sessions


class Mark13ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.calendar = sessions(220)
        self.start = self.calendar[0]
        self.train_end = self.calendar[105]
        self.validation_end = self.calendar[160]
        self.test_end = self.calendar[-1]

    def _rows(self):
        rows = price_rows(self.calendar, count=len(self.calendar))
        for index, row in enumerate(rows):
            opened = 100 + 3 * math.sin(index / 8)
            closed = opened * (1 + .007 * math.sin(index / 3))
            row['open'] = f'{opened:.8f}'
            row['close'] = f'{closed:.8f}'
            row['high'] = f'{max(opened, closed) * 1.01:.8f}'
            row['low'] = f'{min(opened, closed) * .99:.8f}'
        return rows

    def _source(self, rows, name):
        path = self.folder / name
        create_source(path, rows, [catalog_item()], 'domestic')
        return path

    def _load(self, path):
        return load_raw_daily_candidates(path, 'domestic', start=self.start,
                                         train_end=self.train_end, test_end=self.test_end,
                                         max_symbols=1, session_dates=self.calendar)

    def test_target_prices_change_proxy_but_not_preopen_features(self):
        original = self._rows()
        changed = deepcopy(original)
        opened = float(changed[200]['open']) * 1.1
        closed = opened * .98
        changed[200]['open'] = f'{opened:.8f}'
        changed[200]['close'] = f'{closed:.8f}'
        changed[200]['high'] = f'{opened * 1.01:.8f}'
        changed[200]['low'] = f'{closed * .99:.8f}'
        base = self._load(self._source(original, 'base.sqlite3'))
        altered = self._load(self._source(changed, 'altered.sqlite3'))
        date = self.calendar[200]
        before = np.flatnonzero(base.target_dates == date)
        after = np.flatnonzero(altered.target_dates == date)
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        np.testing.assert_array_equal(base.features[before], altered.features[after])
        self.assertNotEqual(base.entry_open[before[0]], altered.entry_open[after[0]])
        self.assertNotEqual(base.exit_close[before[0]], altered.exit_close[after[0]])

    def test_missing_target_remains_preopen_candidate_and_source_is_unchanged(self):
        rows = self._rows()
        rows.pop(205)
        source = self._source(rows, 'missing.sqlite3')
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        samples = self._load(source)
        missing = np.flatnonzero(samples.target_dates == self.calendar[205])
        self.assertEqual(len(missing), 1)
        self.assertFalse(samples.observed[missing[0]])
        self.assertTrue(np.isfinite(samples.features[missing[0]]).all())
        self.assertGreaterEqual(samples.source['unobserved_or_untradable_target'], 1)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_chronological_smoke_reports_frequency_cost_and_no_deployment(self):
        source = self._source(self._rows(), 'smoke.sqlite3')
        samples = self._load(source)
        splits = chronological_splits(samples, train_end=self.train_end,
                                      validation_end=self.validation_end, test_end=self.test_end,
                                      sessions=tuple(self.calendar))
        self.assertLess(samples.target_ordinals[splits['train']].max() + 30,
                        samples.target_ordinals[splits['validation']].min())
        self.assertLess(samples.target_ordinals[splits['validation']].max() + 30,
                        samples.target_ordinals[splits['test']].min())
        gross, net = fixed_open_close_returns(np.asarray([100.]), np.asarray([101.]), cost_bps=20)
        self.assertAlmostEqual(gross[0], .01)
        self.assertLess(net[0], gross[0])
        report, checkpoint = run_experiment(samples, splits, epochs=2, train_cap=100, cost_bps=20)
        self.assertTrue(report['research_only'])
        self.assertFalse(report['deployment_allowed'])
        self.assertFalse(checkpoint['deployment_allowed'])
        self.assertEqual(report['entry'], 't+1 OPEN proxy')
        self.assertIn('t+1 CLOSE', report['exit'])
        self.assertEqual(report['results']['test']['no_trade']['signals'], 0)
        self.assertEqual(report['results']['test']['no_trade']['equal_weight_session_compound_observed_only'], 0)
        self.assertEqual(report['results']['test']['always']['coverage'], 1)
        self.assertIn('mean_net_return_observed', report['results']['test']['neural_positive'])
        self.assertGreater(report['results']['test']['always']['signals_per_session'], 0)
        self.assertGreaterEqual(report['neural_threshold_net_return'], 0)
        provenance = report['score_diagnostics']['threshold_provenance']
        self.assertEqual(provenance['actual_threshold_net_return'],
                         report['neural_threshold_net_return'])
        self.assertEqual(provenance['zero_floor_active'],
                         provenance['unfloored_validation_quantile_net_return'] < 0)
        for name in ('train', 'validation', 'test'):
            distribution = report['score_diagnostics']['distribution_by_split'][name]
            self.assertEqual(distribution['count'], report['splits'][name]['candidates'])
            self.assertLessEqual(distribution['quantiles']['min'],
                                 distribution['quantiles']['median'])
            self.assertLessEqual(distribution['quantiles']['median'],
                                 distribution['quantiles']['max'])
        sweep = report['diagnostic_only_unfloored_validation_coverage_sweep']
        self.assertEqual([item['target_validation_coverage'] for item in sweep],
                         [.01, .05, .10, .25, .50])
        self.assertTrue(all(item['test']['eligible_candidates'] ==
                            report['splits']['test']['candidates'] for item in sweep))
        self.assertEqual(set(report['test_cost_grid_bps']), {'0', '10', '20', '40', '80'})
        self.assertGreater(report['test_cost_grid_bps']['0']['always']['mean_net_return_observed'],
                           report['test_cost_grid_bps']['40']['always']['mean_net_return_observed'])


if __name__ == '__main__':
    unittest.main()
