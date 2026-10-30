from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from dockdack.mark1_special_inference import MarkSpecialPredictor


ROOT = Path(__file__).resolve().parents[1]


class MarkSpecialInferenceTests(unittest.TestCase):
    def test_audited_bundle_reproduces_saved_scores(self):
        for version, folder in (("1.9", "mark1_9"), ("1.10", "mark1_10")):
            for market in ("domestic", "us"):
                with self.subTest(version=version, market=market):
                    root = ROOT / "models" / folder
                    predictor = MarkSpecialPredictor(root, market, version)
                    self.assertTrue(predictor.metadata["research_only"])
                    self.assertFalse(predictor.metadata["deployment_allowed"])
                    with np.load(root / f"{market}-audit.npz", allow_pickle=False) as audit:
                        windows, scores = audit["windows"], audit["scores"]
                    exchange = "KRX" if market == "domestic" else "ND"
                    identities = [(f"T{index:03}", exchange) for index in range(len(windows))]
                    rows = predictor.score_many(windows, identities)
                    self.assertTrue(np.allclose([row["score"] for row in rows], scores,
                                                atol=3e-6, rtol=1e-6))
                    self.assertTrue(all(not row["in_frozen_universe"] for row in rows))
                    if version == "1.9":
                        self.assertTrue(all(row["predicted_sigma_percent"] > 0 for row in rows))
                        self.assertTrue(np.allclose(
                            [row["predicted_mean_net_percent"] - row["predicted_sigma_percent"]
                             for row in rows],
                            [row["score"] for row in rows]))
                    else:
                        self.assertTrue(all("predicted_sigma_percent" not in row for row in rows))

    def test_market_mismatch_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ROOT / "models" / "mark1_9"
            destination = Path(directory) / "mark1_9"
            shutil.copytree(source, destination)
            with self.assertRaisesRegex(ValueError, "Expected"):
                MarkSpecialPredictor(destination, "other", "1.9")
            with (destination / "domestic-audit.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "policy"):
                MarkSpecialPredictor(destination, "domestic", "1.9")
