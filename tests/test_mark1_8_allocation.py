from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dockdack.mark1_4_evolution import EvolutionSamples
from dockdack.mark1_8_allocation import AllocationNet, fit_allocation, simulate_allocations
from dockdack.mark1_8_bundle import load_bundle, write_bundle_manifest
from dockdack.research_artifacts import write_new_json


ROOT = Path(__file__).resolve().parents[1]


def _samples():
    symbols = np.tile(np.arange(11), 2)
    count = len(symbols)
    windows = np.ones((count, 30, 5), dtype=np.float32)
    windows[:, :, 4] = 1_000
    return EvolutionSamples(
        windows=windows,
        target_dates=np.repeat(np.array(["2022-01-03", "2022-01-04"]), 11),
        target_ordinals=np.repeat(np.array([1, 2]), 11),
        symbol_ids=symbols,
        entry_open=np.full(count, 100.0),
        exit_close=np.full(count, 101.0),
        source={"market": "domestic"},
    )


class AllocationTests(unittest.TestCase):
    def test_exact_simulator_caps_names_shares_and_cash(self):
        samples = _samples()
        scores = np.tile(np.arange(11, dtype=np.float32) / 11, 2)
        sizes = np.full(22, .1, dtype=np.float32)
        result = simulate_allocations(samples, np.arange(22), scores, sizes,
                                      threshold=0, cost_bps=20,
                                      initial_equity=10_000)
        self.assertFalse(result["incomplete_data"])
        self.assertEqual(result["signals"], 20)
        self.assertEqual(result["executed_trades"], 20)
        self.assertEqual(result["daily"][0]["selected_symbol_ids"],
                         list(range(10, 0, -1)))
        self.assertEqual(max(result["daily"][0]["shares"]), 9)
        self.assertGreater(result["final_equity"], 10_000)
        self.assertEqual(len(result["daily"]), 2)

    def test_selected_missing_target_invalidates_exact_path(self):
        samples = _samples()
        samples.exit_close[10] = np.nan
        result = simulate_allocations(samples, np.arange(22),
                                      np.tile(np.arange(11) / 11, 2),
                                      np.full(22, .1), threshold=0,
                                      initial_equity=10_000)
        self.assertTrue(result["incomplete_data"])
        self.assertIsNone(result["compound_net_return"])
        self.assertEqual(result["unresolved_selected_outcomes"], 1)
        self.assertIsNone(result["daily"][0]["return"])
        self.assertIsNone(result["daily"][1]["return"])

    def test_size_guard_and_cuda_refusal(self):
        samples = _samples()
        with self.assertRaisesRegex(ValueError, "allocation"):
            simulate_allocations(samples, np.arange(22), np.ones(22),
                                 np.full(22, .11), threshold=0)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                fit_allocation(samples, np.arange(22), seed=41)

    def test_bundle_checks_market_hash_and_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            report = {"version": "Mark1.8", "market": "us", "research_only": True,
                      "deployment_allowed": False, "calibration": {"winner": {
                          "seed": 41, "threshold": .4}}}
            write_new_json(tmp_path / "report.json", report)
            for seed in (41, 42, 43):
                with (tmp_path / f"seed{seed}-model.pt").open("xb") as stream:
                    torch.save({"state_dict": AllocationNet().state_dict(), "seed": seed,
                                "research_only": True, "deployment_allowed": False}, stream)
                with (tmp_path / f"seed{seed}-scores.npz").open("xb") as stream:
                    np.savez_compressed(stream, scores=np.array([.5]), sizes=np.array([.05]))
            write_bundle_manifest(tmp_path)
            bundle = load_bundle(tmp_path, market="us")
            score, size = bundle.predict(_samples().windows[:2])
            self.assertEqual(len(score), len(size))
            self.assertEqual(len(size), 2)
            self.assertTrue(np.all((score >= 0) & (score <= 1)))
            self.assertTrue(np.all((size >= 0) & (size <= .1)))
            self.assertLessEqual(len(bundle.select(
                _samples().windows[:2], np.array(["AAA", "BBB"]))), 2)
            with self.assertRaisesRegex(ValueError, "unique"):
                bundle.select(_samples().windows[:2], np.array(["AAA", "AAA"]))
            with self.assertRaisesRegex(ValueError, "contract"):
                load_bundle(tmp_path, market="domestic")
            with (tmp_path / "seed41-scores.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "Changed"):
                load_bundle(tmp_path, market="us")

    def test_tracked_bundle_reproduces_saved_score_and_size(self):
        root = ROOT / "models" / "mark1_8"
        for market in ("domestic", "us"):
            with self.subTest(market=market):
                bundle = load_bundle(root, market=market)
                self.assertTrue(bundle.metadata["bundle_manifest_sha256"])
                self.assertTrue(bundle.metadata["cash_outperformed_selected_in_2021"])
                with np.load(root / f"{market}-audit.npz", allow_pickle=False) as audit:
                    scores, sizes = bundle.predict(audit["windows"])
                    self.assertTrue(np.allclose(scores, audit["scores"], atol=1e-6))
                    self.assertTrue(np.allclose(sizes, audit["sizes"], atol=1e-6))

    def test_tracked_bundle_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "mark1_8"
            shutil.copytree(ROOT / "models" / "mark1_8", copied)
            with (copied / "domestic-audit.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "policy"):
                load_bundle(copied, market="domestic")
