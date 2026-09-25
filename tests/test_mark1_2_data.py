"""MK1.2 data regressions: synthetic arrays and disposable SQLite files only."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from dockdack.mark1_data import Mark1Dataset, TARGET
from dockdack.mark1_deep_data import class_targets
from dockdack.mark1_deep_models import features_from_history, success_logit
from dockdack.mark1_2_data import (BuildAugmentedBank, FACTORS, RECEIPT_PATH,
                                  _quarantine, load_dataset, read_sessions)
from dockdack.research_artifacts import cache_key, sha256_file


def fixture_dataset(market="us"):
    prices = 95. + np.arange(100, dtype=np.float64) / 10
    bars = np.column_stack((prices, prices + 2, prices - 2, prices + .2,
                            np.full(100, 10000.))).astype(np.float32)
    targets = np.asarray([[100., 101., 99.2, 100.2],
                          [100., 100.9, 99.1, 100.],
                          [100., 101., 99.1, 100.],
                          [100., 100.9, 99.2, 100.],
                          [100., 102., 98., 100.],
                          [100., 101., 99.2, 100.]], dtype=np.float64)
    return Mark1Dataset(bars, np.arange(6, dtype=np.int64),
                        np.arange(18000, 18006, dtype=np.int64),
                        np.asarray([10, 20, 30, 40, 50, 10], np.int64), targets,
                        {"train": np.asarray([0, 1], np.int64),
                         "tune": np.asarray([2], np.int64),
                         "calibration": np.asarray([3], np.int64),
                         "selection": np.asarray([4], np.int64),
                         "test": np.asarray([5], np.int64)},
                        {"market": market, "selected_symbols": 5,
                         "symbols": [{"symbol_id": i, "symbol": symbol}
                                     for i, symbol in zip([10, 20, 30, 40, 50],
                                                          ["AAPL", "FCEL", "BNED", "BBSI", "SONY"])]})


class QueryModel(nn.Module):
    def __init__(self, fail=False, nonfinite=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(.5)
        self.fail, self.nonfinite = fail, nonfinite
        self.observed = []

    def forward(self, features):
        self.observed.append((features.dtype, features.device.type, self.training,
                              torch.is_grad_enabled()))
        if self.fail:
            raise RuntimeError("fixture model failure")
        query = features[:, -1, 0] * self.weight
        zero = torch.zeros_like(query)
        if self.nonfinite:
            query = query * float("nan")
        return torch.stack((query, zero, zero, zero), dim=1)


class Mark12BankTests(unittest.TestCase):
    def setUp(self):
        self.dataset = fixture_dataset()
        self.splits = {"train": np.arange(6, dtype=np.int64),
                       "empty": np.asarray([], dtype=np.int64)}

    def bank(self, dataset=None, **kwargs):
        return BuildAugmentedBank(dataset or self.dataset, self.splits, "cpu", **kwargs)

    def test_exact_barriers_stop_first_and_all_factor_labels(self):
        bank = self.bank()
        part = bank.parts["train"]
        np.testing.assert_array_equal(part["classes"][:, 0], [0, 1, 2, 3, 2, 0])
        np.testing.assert_array_equal(part["classes"].numpy(), class_targets(self.dataset.target_ohlc, FACTORS))
        np.testing.assert_array_equal(part["outcomes"]["success"], [True, False, False, False, False, True])
        self.assertEqual(part["outcomes"]["gross_return"][2], -.009)
        self.assertEqual(part["classes"].dtype, torch.int64)
        self.assertEqual(bank.bars.dtype, torch.float32)
        self.assertEqual(bank.factors.device, bank.bars.device)
        np.testing.assert_array_equal(part["symbols"], self.dataset.symbol_ids)

    def test_label_arithmetic_retains_float64_before_feature_cast(self):
        targets = self.dataset.target_ohlc.copy()
        targets[0, 1] = 101 - 1e-7  # FP32 rounds up to a take touch; float64 must not.
        self.assertEqual(np.float32(targets[0, 1]), np.float32(101))
        bank = self.bank(replace(self.dataset, target_ohlc=targets))
        self.assertEqual(bank.parts["train"]["classes"][0, 0].item(), 3)
        self.assertFalse(bank.parts["train"]["outcomes"]["success"][0])

    def test_price_augmentation_changes_query_only_and_preserves_inputs(self):
        original = copy.deepcopy(self.dataset)
        bank = self.bank()
        indices = torch.arange(6)
        actual = bank.features(bank.parts["train"], indices)
        augmented = bank.features(bank.parts["train"], indices, torch.ones(6, dtype=torch.long))
        self.assertEqual(tuple(actual.shape), (6, 31, 18))
        torch.testing.assert_close(actual[:, :30], augmented[:, :30], rtol=0, atol=0)
        mask = torch.ones(18, dtype=torch.bool)
        mask[[0, 11]] = False
        torch.testing.assert_close(actual[:, -1, mask], augmented[:, -1, mask], rtol=0, atol=0)
        self.assertFalse(torch.equal(actual[:, -1, 0], augmented[:, -1, 0]))
        for name in ("bars", "starts", "target_dates", "symbol_ids", "target_ohlc"):
            np.testing.assert_array_equal(getattr(self.dataset, name), getattr(original, name))
        self.assertEqual(self.dataset.manifest, original.manifest)
        self.assertEqual(bank.bars.numel(), self.dataset.bars.size)

    def test_target_high_low_close_never_enter_features(self):
        changed = self.dataset.target_ohlc.copy()
        changed[:, 1], changed[:, 2], changed[:, 3] = 110, 90, 103
        banks = [self.bank(), self.bank(replace(self.dataset, target_ohlc=changed))]
        torch.testing.assert_close(banks[0].features(banks[0].parts["train"], torch.arange(6)),
                                   banks[1].features(banks[1].parts["train"], torch.arange(6)), rtol=0, atol=0)
        self.assertFalse(torch.equal(banks[0].parts["train"]["classes"], banks[1].parts["train"]["classes"]))

    def test_batched_logits_are_fp32_factor_specific_and_restore_states(self):
        bank, model = self.bank(), QueryModel()
        model.train()
        model.dropout.eval()
        part = bank.parts["train"]
        index = torch.arange(6)
        expected = success_logit(model(bank.features(part, index, torch.full_like(index, 2)))).detach().numpy()
        model.observed.clear()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = bank.logits(model, "train", batch_size=2, device="cpu", factor_index=2)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(model.observed, [(torch.float32, "cpu", False, False)] * 3)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertFalse(np.array_equal(actual, bank.logits(model, "train")))

    def test_logits_failure_restores_mode_and_empty_split_is_supported(self):
        bank, model = self.bank(), QueryModel(fail=True)
        model.train()
        model.dropout.eval()
        with self.assertRaisesRegex(RuntimeError, "fixture model"):
            bank.logits(model, "train", device="cpu")
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertEqual(bank.logits(model, "empty").shape, (0,))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            bank.logits(QueryModel(nonfinite=True), "train")

    def test_invalid_inputs_are_rejected_without_silent_clipping(self):
        for factors in ((.99, 1.), (1., 1.), (1., float("nan")), (1., -1),
                        (1., 1e100), (1., 1. + 1e-10), (1., 1e38)):
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                self.bank(factors=factors)
        for indices in ([0, 0], [-1], [6], [1., 2.], [2, 1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                BuildAugmentedBank(self.dataset, {"train": indices}, "cpu")
        for starts in (np.asarray([-1] * 6), np.asarray([100] * 6)):
            with self.assertRaises(ValueError):
                self.bank(replace(self.dataset, starts=starts))
        broken = self.dataset.bars.copy()
        broken[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.bank(replace(self.dataset, bars=broken))
        bank = self.bank()
        for value in (-1, 5, True):
            with self.assertRaises(ValueError):
                bank.logits(QueryModel(), "train", factor_index=value)
        with self.assertRaises(ValueError):
            bank.logits(QueryModel().double(), "train")
        with self.assertRaises((IndexError, RuntimeError)):
            bank.features(bank.parts["train"], torch.tensor([-1]))
        with self.assertRaisesRegex(ValueError, "more than one split"):
            BuildAugmentedBank(self.dataset, {"train": [0], "test": [0]}, "cpu")

    def test_cuda_resident_memory_guard_runs_before_allocating(self):
        with patch("dockdack.mark1_2_data.torch.cuda.mem_get_info", return_value=(2, 2)):
            with self.assertRaisesRegex(MemoryError, "half"):
                BuildAugmentedBank(self.dataset, self.splits, "cuda")

    def test_quarantine_keeps_raw_bars_and_remaps_every_split_without_mutation(self):
        original = copy.deepcopy(self.dataset)
        result, contract = _quarantine(self.dataset, "us")
        self.assertIs(result.bars, self.dataset.bars)
        np.testing.assert_array_equal(result.starts, [0, 5])
        np.testing.assert_array_equal(result.target_dates, self.dataset.target_dates[[0, 5]])
        np.testing.assert_array_equal(result.target_ohlc, self.dataset.target_ohlc[[0, 5]])
        np.testing.assert_array_equal(result.splits["train"], [0])
        np.testing.assert_array_equal(result.splits["test"], [1])
        for split in ("tune", "calibration", "selection"):
            self.assertEqual(len(result.splits[split]), 0)
        self.assertEqual(contract["quarantine_policy"]["symbols"], ["FCEL", "BNED", "BBSI", "SONY"])
        self.assertEqual(sum(row["excluded_samples"] for row in contract["quarantine"]), 4)
        self.assertEqual(self.dataset.manifest, original.manifest)
        self.assertEqual(result.manifest["selected_symbols"], 1)
        for key in original.splits:
            np.testing.assert_array_equal(self.dataset.splits[key], original.splits[key])
        domestic, record = _quarantine(fixture_dataset("domestic"), "domestic")
        self.assertEqual(len(domestic.starts), 6)
        self.assertEqual(record["quarantine"], [])


class Mark12FrozenLoadTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "data/kiwoom_daily/clean-20260916-v1/us_daily_clean.sqlite3"
        self.database.parent.mkdir(parents=True)
        with sqlite3.connect(self.database) as db:
            db.executescript("""CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
                CREATE TABLE daily_bars(dummy); CREATE TABLE training_samples(dummy);
                CREATE TABLE sessions(session_date TEXT,ordinal INTEGER); CREATE TABLE instruments(dummy);""")
            db.executemany("INSERT INTO metadata VALUES (?,?)", [("market", '"us"'), ("build_status", '"complete"')])
            db.executemany("INSERT INTO sessions VALUES (?,?)", [("2024-01-02", 1), ("2024-01-03", 2)])
        db.close()
        self.source = {"version": 2, "market": "us", "target": TARGET, "seed": 42,
                       "database_path": "C:/Users/user/Desktop/dockdack-data-collection/data/kiwoom_daily/clean-20260916-v1/us_daily_clean.sqlite3",
                       "database_sha256": sha256_file(self.database)}
        self.source_path = self.root / "outputs/mark1/selective-20260916/us/source.json"
        self.source_path.parent.mkdir(parents=True)
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")
        self.cache = self.root / f"outputs/mark1/cache/us-{cache_key(self.source)}.npz"
        self.cache.parent.mkdir(parents=True)
        dataset = fixture_dataset()
        np.savez(self.cache, **{name: getattr(dataset, name) for name in
                              ("bars", "starts", "target_dates", "symbol_ids", "target_ohlc")},
                 **{f"split_{name}": value for name, value in dataset.splits.items()},
                 manifest=json.dumps(dataset.manifest), cache_config=json.dumps(self.source))
        self.record = {"read_only_database_and_cache_checks": {"us": {
            "database_sha256": self.source["database_sha256"], "cache_sha256": sha256_file(self.cache),
            "cache_key": cache_key(self.source), "source_contract_unchanged": True,
            "read_before_after_verified": True}}}
        self.record_path = self.root / RECEIPT_PATH
        self.record_path.parent.mkdir(parents=True)
        self.record_path.write_text(json.dumps(self.record), encoding="utf-8")

    def test_load_uses_explicit_receipt_preserves_logical_source_and_all_input_bytes(self):
        paths = (self.database, self.cache, self.source_path, self.record_path)
        before = {path: sha256_file(path) for path in paths}
        files_before = set(self.root.rglob("*"))
        dataset, source, receipt = load_dataset("us", self.root)
        self.assertEqual(source, self.source)
        self.assertEqual(len(dataset.starts), 2)
        self.assertEqual(receipt["cache_key"], cache_key(self.source))
        self.assertEqual(Path(receipt["source"]["physical_path"]), self.database)
        self.assertTrue(receipt["read_only"])
        self.assertEqual(receipt["experiment"]["raw_bar_count"], 100)
        self.assertEqual(before, {path: sha256_file(path) for path in paths})
        self.assertEqual(files_before, set(self.root.rglob("*")))

    def test_changed_cache_and_changed_database_are_rejected(self):
        original = self.cache.read_bytes()
        with self.cache.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "receipt"):
            load_dataset("us", self.root)
        self.cache.write_bytes(original)
        with self.database.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            load_dataset("us", self.root)

    def test_wrong_or_missing_receipt_and_source_market_fail_closed(self):
        for field, value in (("cache_key", "wrong"), ("database_sha256", "0" * 64),
                             ("source_contract_unchanged", False), ("read_before_after_verified", False)):
            changed = copy.deepcopy(self.record)
            changed["read_only_database_and_cache_checks"]["us"][field] = value
            self.record_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_dataset("us", self.root)
        self.record_path.write_text(json.dumps(self.record), encoding="utf-8")
        self.source_path.write_text(json.dumps({**self.source, "market": "domestic"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_dataset("us", self.root)
        with self.assertRaises(ValueError):
            load_dataset("other", self.root)

    def test_read_sessions_readonly_missing_database_and_wal_rejected(self):
        before = sha256_file(self.database)
        dates = read_sessions(self.database)
        np.testing.assert_array_equal(dates, np.asarray(["2024-01-02", "2024-01-03"], dtype="datetime64[D]").astype(np.int64))
        self.assertEqual(sha256_file(self.database), before)
        missing = self.root / "missing.sqlite3"
        with self.assertRaises(FileNotFoundError):
            read_sessions(missing)
        self.assertFalse(missing.exists())
        Path(str(self.database) + "-wal").write_bytes(b"pending changes")
        with self.assertRaisesRegex(ValueError, "WAL"):
            read_sessions(self.database)


if __name__ == "__main__":
    unittest.main()
