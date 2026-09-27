from pathlib import Path
import shutil

import numpy as np
import pytest

from dockdack.mark1_special_inference import MarkSpecialPredictor


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version,folder", [("1.9", "mark1_9"),
                                            ("1.10", "mark1_10")])
@pytest.mark.parametrize("market", ["domestic", "us"])
def test_audited_bundle_reproduces_saved_scores(version, folder, market):
    root = ROOT / "models" / folder
    predictor = MarkSpecialPredictor(root, market, version)
    assert predictor.metadata["research_only"]
    assert not predictor.metadata["deployment_allowed"]
    with np.load(root / f"{market}-audit.npz", allow_pickle=False) as audit:
        windows, scores = audit["windows"], audit["scores"]
    exchange = "KRX" if market == "domestic" else "ND"
    identities = [(f"T{index:03}", exchange) for index in range(len(windows))]
    rows = predictor.score_many(windows, identities)
    assert np.allclose([row["score"] for row in rows], scores,
                       atol=3e-6, rtol=1e-6)
    assert all(not row["in_frozen_universe"] for row in rows)
    if version == "1.9":
        assert all(row["predicted_sigma_percent"] > 0 for row in rows)
        assert np.allclose([row["predicted_mean_net_percent"] -
                            row["predicted_sigma_percent"] for row in rows],
                           [row["score"] for row in rows])
    else:
        assert all("predicted_sigma_percent" not in row for row in rows)


def test_market_mismatch_and_tamper_fail_closed(tmp_path):
    source = ROOT / "models" / "mark1_9"
    destination = tmp_path / "mark1_9"
    shutil.copytree(source, destination)
    with pytest.raises(ValueError, match="Expected"):
        MarkSpecialPredictor(destination, "other", "1.9")
    with (destination / "domestic-audit.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="policy"):
        MarkSpecialPredictor(destination, "domestic", "1.9")
