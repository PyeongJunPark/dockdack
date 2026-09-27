import numpy as np
import pytest
import torch
from pathlib import Path
import shutil

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


def test_exact_simulator_caps_names_shares_and_cash():
    samples = _samples()
    scores = np.tile(np.arange(11, dtype=np.float32) / 11, 2)
    sizes = np.full(22, .1, dtype=np.float32)
    result = simulate_allocations(samples, np.arange(22), scores, sizes,
                                  threshold=0, cost_bps=20,
                                  initial_equity=10_000)
    assert not result["incomplete_data"]
    assert result["signals"] == 20
    assert result["executed_trades"] == 20
    assert result["daily"][0]["selected_symbol_ids"] == list(range(10, 0, -1))
    assert max(result["daily"][0]["shares"]) == 9
    assert result["final_equity"] > 10_000
    assert len(result["daily"]) == 2


def test_selected_missing_target_invalidates_exact_path():
    samples = _samples()
    samples.exit_close[10] = np.nan
    result = simulate_allocations(samples, np.arange(22),
                                  np.tile(np.arange(11) / 11, 2),
                                  np.full(22, .1), threshold=0,
                                  initial_equity=10_000)
    assert result["incomplete_data"]
    assert result["compound_net_return"] is None
    assert result["unresolved_selected_outcomes"] == 1
    assert result["daily"][0]["return"] is None
    assert result["daily"][1]["return"] is None


def test_size_guard_and_cuda_refusal(monkeypatch):
    samples = _samples()
    with pytest.raises(ValueError, match="allocation"):
        simulate_allocations(samples, np.arange(22), np.ones(22),
                             np.full(22, .11), threshold=0)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        fit_allocation(samples, np.arange(22), seed=41)


def test_bundle_checks_market_hash_and_frozen_threshold(tmp_path):
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
    assert len(score) == len(size) == 2
    assert np.all((score >= 0) & (score <= 1))
    assert np.all((size >= 0) & (size <= .1))
    assert len(bundle.select(_samples().windows[:2], np.array(["AAA", "BBB"]))) <= 2
    with pytest.raises(ValueError, match="unique"):
        bundle.select(_samples().windows[:2], np.array(["AAA", "AAA"]))
    with pytest.raises(ValueError, match="contract"):
        load_bundle(tmp_path, market="domestic")
    with (tmp_path / "seed41-scores.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Changed"):
        load_bundle(tmp_path, market="us")


@pytest.mark.parametrize("market", ["domestic", "us"])
def test_tracked_bundle_reproduces_saved_score_and_size(market):
    root = ROOT / "models" / "mark1_8"
    bundle = load_bundle(root, market=market)
    assert bundle.metadata["bundle_manifest_sha256"]
    assert bundle.metadata["cash_outperformed_selected_in_2021"]
    with np.load(root / f"{market}-audit.npz", allow_pickle=False) as audit:
        scores, sizes = bundle.predict(audit["windows"])
        assert np.allclose(scores, audit["scores"], atol=1e-6)
        assert np.allclose(sizes, audit["sizes"], atol=1e-6)


def test_tracked_bundle_rejects_tampering(tmp_path):
    copied = tmp_path / "mark1_8"
    shutil.copytree(ROOT / "models" / "mark1_8", copied)
    with (copied / "domestic-audit.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="policy"):
        load_bundle(copied, market="domestic")
