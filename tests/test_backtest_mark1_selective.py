"""Synthetic-only frozen selective backtest and provenance regression tests."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    from examples import backtest_mark1_selective as runner
    from dockdack.mark1_data import barrier_outcomes


@unittest.skipUnless(AVAILABLE, "requires optional NumPy/Torch")
class SelectiveBacktestTests(unittest.TestCase):
    @staticmethod
    def calibration():
        return {"method": "platt_monotone", "slope": 1., "bias": 0., "fit_samples": 100, "weighted": False}

    @staticmethod
    def dataset():
        count = 4
        histories = np.full((count, 30, 5), 100., dtype=np.float32)
        histories[..., 4] = 100000
        ohlc = np.array([[100., 102., 99.5, 101.], [100., 100.5, 98., 99.],
                         [100., 102., 98., 100.], [100., 100.5, 99.5, 100.]])
        return SimpleNamespace(bars=histories.reshape(-1, 5), starts=np.arange(count) * 30,
                               target_ohlc=ohlc, target_dates=np.arange(20138, 20142),
                               symbol_ids=np.array([0, 0, 1, 1]),
                               manifest={"symbols": [{"symbol_id": 0, "symbol": "A", "exchange": "KRX"},
                                                     {"symbol_id": 1, "symbol": "B", "exchange": "KRX"}]})

    @contextlib.contextmanager
    def training_fixture(self):
        with tempfile.TemporaryDirectory(prefix='selective-provenance-') as directory:
            root, repo = Path(directory) / 'training', Path(directory) / 'repo'
            root.mkdir()
            code = {}
            for name in runner.CODE_FILES:
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                code[name] = runner.file_hash(path)
            originals = {}
            for market in ('domestic', 'us'):
                path = repo / f'models/mark1/{market}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(market.encode())
                originals[market] = runner.file_hash(path)
            complete = {}
            for market in ('domestic', 'us'):
                folder = root / market
                folder.mkdir()
                source = {'market': market, 'target': runner.PROTOCOL['target'], 'version': 2,
                          'purge_sessions': 30, 'start': '2010-01-01', 'seed': 42,
                          'max_train_samples': 200000, 'max_eval_samples': 60000,
                          'database_path': str(root / f'{market}.sqlite3'), 'database_sha256': 'a' * 64}
                ranking = [{'architecture': 'cat_binary6', 'minimum_precision_lower': .6}]
                summary = {'market': market, 'completed': True, 'selected': 'cat_binary6',
                           'ranking': ranking, 'source': source, 'ensembles': {},
                           'research_only': True, 'deployment_allowed': False, 'research_qualified': False}
                for fold in runner.FOLDS:
                    ensemble = {'calibration': {'success': self.calibration(), 'stop': None},
                                'policy_selection': {'chosen_policy': {'threshold': .65, 'stop_probability_cap': 1.},
                                                     'research_only': True, 'deployment_allowed': False},
                                'member_sha256': {}}
                    for seed in runner.SEEDS:
                        trial = folder / fold / f'cat_binary6-{seed}'
                        trial.mkdir(parents=True)
                        artifact = runner.model_path(trial, 'cat_binary6')
                        artifact.write_bytes(f'{market}-{fold}-{seed}'.encode())
                        digest = runner.file_hash(artifact)
                        request = {'model_name': 'cat_binary6', 'seed': seed,
                                   'wrapper_sha256': code['dockdack/mark1_selective_models.py'],
                                   'class_names': list(runner.CLASS_NAMES), 'train_shape': [5, len(runner.FEATURE_NAMES)]}
                        runner.save_json(trial / 'request.json', request)
                        runner.save_json(trial / 'metadata.json', {'request': request, 'model_sha256': digest})
                        runner.save_json(artifact.with_name(artifact.name + '.json'), {
                            'model_name': 'cat_binary6', 'model_sha256': digest, 'owner': runner.OWNER,
                            'schema_version': runner.SCHEMA_VERSION, 'feature_count': len(runner.FEATURE_NAMES),
                            'class_names': list(runner.CLASS_NAMES)})
                        ensemble['member_sha256'][str(seed)] = digest
                    summary['ensembles'][fold] = ensemble
                    runner.save_json(folder / fold / 'ensemble.json', ensemble)
                runner.save_json(folder / 'source.json', source)
                runner.save_json(folder / 'summary.json', summary)
                runner.save_json(folder / 'selection_locked.json', {
                    'market': market, 'selected': 'cat_binary6', 'source': source,
                    'ranking': ranking, 'selection_rule': runner.PROTOCOL['selection']})
                complete[market] = summary
            runner.save_json(root / 'summary.json', complete)
            runner.save_json(root / 'protocol.json', {'protocol': runner.PROTOCOL, 'code_sha256': code})
            runner.save_json(root / 'protected_checkpoints.json', {
                str(repo / f'models/mark1/{key}.pt'): value for key, value in originals.items()})
            with patch.object(runner, 'ROOT', repo), patch.dict(runner.deep.ORIGINAL_CHECKPOINT_SHA256, originals):
                yield root, repo, complete

    def test_complete_training_contract_both_markets_verified(self):
        with self.training_fixture() as (root, _, complete):
            summary, source, hashes = runner.verify_training_artifacts(root, 'domestic')
            self.assertEqual(summary, complete['domestic'])
            self.assertEqual(source, summary['source'])
            self.assertGreater(len(hashes), 50)
            self.assertTrue(all(runner.file_hash(path) == digest for path, digest in hashes.items()))

    def test_other_market_incomplete_fails_before_any_evaluation_data(self):
        with self.training_fixture() as (root, _, complete):
            complete['us']['completed'] = False
            runner.save_json(root / 'summary.json', complete)
            with patch.object(runner.deep, 'load_frozen_cache') as cache:
                with self.assertRaisesRegex(ValueError, 'Both markets'):
                    runner.run_market('domestic', SimpleNamespace(training_run=root))
                cache.assert_not_called()

    def test_other_market_selected_model_tamper_rejected(self):
        with self.training_fixture() as (root, _, _):
            artifact = runner.model_path(root / 'us/walk_2022/cat_binary6-43', 'cat_binary6')
            artifact.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                runner.verify_training_artifacts(root, 'domestic')

    def test_frozen_code_tamper_rejected(self):
        with self.training_fixture() as (root, repo, _):
            (repo / runner.CODE_FILES[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'code changed'):
                runner.verify_training_artifacts(root, 'domestic')

    def test_selected_policy_and_source_lock_mismatch_rejected(self):
        with self.training_fixture() as (root, _, _):
            path = root / 'domestic/selection_locked.json'
            lock = runner.read_json(path)
            lock['selected'] = 'cat_binary8'
            runner.save_json(path, lock)
            with self.assertRaisesRegex(ValueError, 'selection mismatch'):
                runner.verify_training_artifacts(root, 'domestic')

    def test_native_metadata_wrong_seed_rejected(self):
        with self.training_fixture() as (root, _, _):
            path = root / 'domestic/walk_2024/cat_binary6-42/request.json'
            request = runner.read_json(path)
            request['seed'] = 7
            runner.save_json(path, request)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                runner.verify_training_artifacts(root, 'domestic')

    def test_previous_comparators_bound_to_recorded_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(deep_training_run=root / 'deep', deep_backtest_run=root / 'old-evaluation',
                                   baseline_dir=root / 'baseline')
            baseline = args.baseline_dir / 'domestic.pt'
            baseline.parent.mkdir()
            baseline.write_bytes(b'original baseline')
            hashes = {str(baseline.resolve()): runner.file_hash(baseline)}
            for seed in runner.SEEDS:
                path = args.deep_training_run / 'domestic/walk_2024' / f'inception-{seed}/model.pt'
                path.parent.mkdir(parents=True)
                path.write_bytes(str(seed).encode())
                hashes[str(path.resolve())] = runner.file_hash(path)
            contract = {'database_sha256': 'a' * 64}
            prior = {'selected': 'inception', 'ensemble_calibration': self.calibration()}
            old_path = args.deep_backtest_run / 'domestic/summary.json'
            old_path.parent.mkdir(parents=True)
            runner.save_json(old_path, {'source': contract, 'completed': True, 'selected_architecture': 'inception',
                                       'ensemble_calibration': self.calibration(), 'artifact_sha256': hashes})
            with patch.object(runner.deep, 'verify_training_artifacts', return_value=(prior, contract, {})), \
                    patch.dict(runner.deep.ORIGINAL_CHECKPOINT_SHA256, {'domestic': runner.file_hash(baseline)}):
                actual, verified = runner.verify_comparators(args, 'domestic', contract)
                self.assertEqual(actual, prior)
                self.assertEqual(len(verified), 5)
                path.write_bytes(b'changed previous deep checkpoint')
                with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                    runner.verify_comparators(args, 'domestic', contract)

    def test_previous_comparators_reject_different_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(deep_training_run=root / 'deep', deep_backtest_run=root / 'old')
            folder = args.deep_backtest_run / 'domestic'
            folder.mkdir(parents=True)
            runner.save_json(folder / 'summary.json', {'source': {'database_sha256': 'a' * 64}})
            with patch.object(runner.deep, 'verify_training_artifacts', return_value=({}, {'database_sha256': 'a' * 64}, {})):
                with self.assertRaisesRegex(ValueError, 'source/selection'):
                    runner.verify_comparators(args, 'domestic', {'database_sha256': 'b' * 64})

    def test_incomplete_training_missing_files_fails_without_cache(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.deep, 'load_frozen_cache') as cache:
            with self.assertRaises(FileNotFoundError):
                runner.run_market('domestic', SimpleNamespace(training_run=Path(temporary)))
            cache.assert_not_called()

    def test_selected_candidates_preserve_real_probabilities(self):
        data, indices = self.dataset(), np.arange(4)
        probabilities, selected = np.array([.91, .8, .5, .7]), np.array([False, True, False, True])
        candidates = runner.selected_candidates(data, indices, probabilities, selected)
        self.assertEqual([row['probability'] for row in candidates], [.8, .7])
        self.assertEqual([row['date'] for row in candidates], data.target_dates[[1, 3]].tolist())
        self.assertEqual([row['liquidity_shares'] for row in candidates], [100000., 100000.])
        np.testing.assert_array_equal(probabilities, [.91, .8, .5, .7])

    def test_invalid_masks_or_probabilities_rejected(self):
        data, indices = self.dataset(), np.arange(4)
        for probability, mask in (([.9, .8, .5, .7], [True, False, True, False]),
                                  ([.9, .8, .6, .7], [1, 0, 1, 0]),
                                  ([np.nan, .8, .6, .7], [False] * 4),
                                  ([1.1, .8, .6, .7], [False] * 4)):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                runner.selected_candidates(data, indices, probability, mask)

    def test_logit_average_frozen_calibration_and_chunked_features(self):
        with self.training_fixture() as (root, _, complete):
            data, indices = self.dataset(), np.arange(4)
            args = SimpleNamespace(training_run=root, batch_size=2)
            calls = []

            def fake_features(incoming, picked, batch_size):
                calls.append(picked.copy())
                return np.repeat(picked[:, None], len(runner.FEATURE_NAMES), axis=1).astype(np.float32)

            def fake_load(name, path):
                return int(path.parent.name.rsplit('-', 1)[1])

            def fake_predict(model, name, values):
                return {'success_logits': values[:, 0].astype(np.float64) + (model - 43), 'stop_logits': None}

            with patch.object(runner, 'feature_array', side_effect=fake_features), \
                    patch.object(runner, 'load_model', side_effect=fake_load), \
                    patch.object(runner, 'predict_raw', side_effect=fake_predict):
                p, stop, mask, hashes = runner.infer_selective(data, indices, 'domestic', args, complete['domestic'])
            expected = 1 / (1 + np.exp(-np.arange(4)))
            np.testing.assert_allclose(p, expected)
            np.testing.assert_array_equal(mask, expected > .65)
            self.assertIsNone(stop)
            self.assertEqual(len(hashes), 3)
            self.assertEqual(len(calls), 2)
            np.testing.assert_array_equal(np.concatenate(calls), indices)

    def test_native_invalid_predictions_fail_closed(self):
        with self.training_fixture() as (root, _, complete):
            args = SimpleNamespace(training_run=root, batch_size=2)
            with patch.object(runner, 'load_model', return_value=object()), \
                    patch.object(runner, 'predict_raw', return_value={'success_logits': np.array([np.nan, .1]), 'stop_logits': None}):
                with self.assertRaisesRegex(ValueError, 'raw prediction'):
                    runner.infer_selective(self.dataset(), np.arange(4), 'domestic', args, complete['domestic'])

    def test_calendar_stability_zero_signals_and_concentration(self):
        dates = np.array(['2025-03-01', '2025-03-02', '2025-04-01', '2026-01-02'], dtype='datetime64[D]').astype(int)
        result = runner.signal_stability([1, 0, 1, 0], [.01, -.009, .01, 0], dates,
                                          [1, 1, 2, 3], np.array([True, True, True, False]))
        self.assertEqual(result['signal_count'], 3)
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertAlmostEqual(result['largest_symbol_share'], 2 / 3)
        self.assertEqual(result['largest_symbol_id'], 1)
        self.assertEqual(result['yearly'][-1]['period'], '2026')
        self.assertIsNone(result['yearly'][-1]['precision'])
        self.assertIsNone(result['yearly'][-1]['net_mean_return'])
        self.assertEqual([item['period'] for item in result['quarterly']], ['2025Q1', '2025Q2', '2026Q1'])
        self.assertAlmostEqual(result['quarterly'][1]['net_mean_return'], .008)

    def test_full_market_produces_all32_ledgers_and_true_brier(self):
        data, indices = self.dataset(), np.arange(4)
        target = data.target_ohlc
        outcomes = barrier_outcomes(target[:, 1], target[:, 2], target[:, 3], target[:, 0])
        old_p = {'baseline': np.array([.8, .5, .7, .5]), 'deep': np.array([.5, .9, .8, .5])}
        new_p, selected = np.array([.9, .6, .8, .5]), np.array([True, False, False, False])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.sqlite3'
            source.write_bytes(b'synthetic source never queried')
            contract = {'database_path': str(source), 'database_sha256': runner.file_hash(source)}
            training = {'selected': 'cat_binary6', 'research_qualified': False, 'ensembles': {'walk_2024': {
                'calibration': {'success': self.calibration(), 'stop': None},
                'policy_selection': {'chosen_policy': {'threshold': .85, 'stop_probability_cap': 1.}}}}}
            args = SimpleNamespace(training_run=root / 'training', deep_training_run=root / 'deep',
                output_dir=root / 'output', baseline_dir=root / 'baseline', cache_dir=root / 'cache',
                device='cpu', batch_size=2, initial_krw=10000000., initial_usd=10000.,
                max_positions=20, position_fraction=.05, volume_fraction=.001)
            prices = {(int(data.symbol_ids[i]), int(data.target_dates[i])): tuple(target[i]) for i in indices}
            zero = (0, int(data.target_dates[-1]) + 1)
            prices[zero] = (100., 200., 1., 100.)
            actual_simulate, calls = runner.simulate_portfolio, []

            def simulate(candidates, panel, sessions, **kwargs):
                self.assertNotIn(zero, panel)
                calls.append(kwargs)
                return actual_simulate(candidates, panel, sessions, **kwargs)

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner, 'verify_training_artifacts', return_value=(training, contract, {})))
                stack.enter_context(patch.object(runner, 'verify_comparators', return_value=({}, {})))
                stack.enter_context(patch.object(runner.deep, 'load_frozen_cache', return_value=data))
                stack.enter_context(patch.object(runner.deep, 'read_sessions', return_value=np.arange(20100, 20150)))
                stack.enter_context(patch.object(runner.deep, 'common_test_indices', return_value=indices))
                stack.enter_context(patch.object(runner, 'load_price_panel', return_value=(prices, data.target_dates.tolist(), {'zero_volume_keys': [zero]})))
                stack.enter_context(patch.object(runner.deep, 'infer_common', return_value=(old_p, outcomes, data.target_dates, {})))
                stack.enter_context(patch.object(runner, 'infer_selective', return_value=(new_p, None, selected, {})))
                stack.enter_context(patch.object(runner, 'simulate_portfolio', side_effect=simulate))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                result = runner.run_market('domestic', args)
            self.assertTrue(result['completed'])
            self.assertFalse(result['deployment_allowed'])
            self.assertEqual(set(result['models']), set(runner.COMPARATORS))
            self.assertEqual(len(calls), 32)
            self.assertEqual(len(result['cost_sensitivity']), 32)
            self.assertEqual(result['models']['selective']['raw_signals'], 1)
            self.assertEqual(result['models']['unfiltered']['raw_signals'], 3)
            self.assertEqual(result['models']['selective']['classification']['overall']['brier'],
                             result['models']['unfiltered']['classification']['overall']['brier'])
            self.assertAlmostEqual(result['models']['selective']['classification']['overall']['brier'],
                                   np.square(new_p - outcomes['success']).mean())
            for name in runner.COMPARATORS:
                for mode in ('carry', 'eod'):
                    ledger = runner.read_json(args.output_dir / 'domestic' / f'{name}-{mode}.json')
                    self.assertTrue(ledger['diagnostic_only'])
                    self.assertFalse(ledger['deployment_allowed'])
                    self.assertTrue(all('symbol' in trade for trade in ledger['trades']))
                self.assertIn('quarterly', result['models'][name]['stability'])
            with np.load(args.output_dir / 'domestic/predictions.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['sample_indices'], indices)
                np.testing.assert_array_equal(saved['selective_probabilities'], saved['unfiltered_probabilities'])
                np.testing.assert_array_equal(saved['selective_selected'], selected)
            self.assertEqual(runner.file_hash(source), contract['database_sha256'])

    def test_main_refuses_overwrite_before_reading_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, 'verify_training_artifacts') as verify:
            with self.assertRaises(FileExistsError):
                runner.main(['--output-dir', temporary])
            verify.assert_not_called()

    def test_main_rejects_partial_training_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'new'
            with patch.object(runner, 'verify_training_artifacts', side_effect=ValueError('not complete')):
                with self.assertRaises(ValueError):
                    runner.main(['--output-dir', str(target)])
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
