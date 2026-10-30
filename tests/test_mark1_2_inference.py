"""Synthetic offline fixtures only: no actual training, accounts, or databases."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dockdack.mark1_2_inference import (
    FLAGS, MARKETS, SEEDS, SEMANTICS, Predictor, checked_file, load_weights,
    read_json, sha256_file,
)
from dockdack.mark1_2_models import MODEL_NAMES, build_model
from dockdack.mark1_deep_models import CLASS_NAMES, FEATURE_NAMES, TARGET, features_from_history, success_logit
from dockdack.mark1_metrics import calibrated_probability
from examples.export_mark1_2 import ROOT, TRAINING_CODE_FILES, export_bundle, inspect_training_run, synthetic_cases


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def make_run(root):
    """Tiny untrained models exercise the exact trainer serialization contract."""
    config = dict(input_size=18, sequence_length=31, width=4, dropout=.2)
    calibration = dict(method="platt_monotone", weighted=False, fit_samples=24, slope=1.1, bias=-.2)
    recipe = dict(version="mark1-2-neural-price-augmentation-v1", target=TARGET,
                  take_profit_pct=1., stop_loss_pct=.9, threshold=.5, both_touch="stop_first_failure",
                  ensemble_seeds=list(SEEDS), price_factors=[1., .99, .995, 1.005, 1.01],
                  architectures=list(MODEL_NAMES), model_config=config, selection="fixture frozen selection",
                  deployment_allowed=False, intraday_path_verified=False)
    code = {name: sha256_file(ROOT / name) for name in TRAINING_CODE_FILES}
    write_json(root / "protocol.json", dict(phase="train", markets=list(MARKETS), protocol=recipe, code_sha256=code))
    write_json(root / "status.json", dict(status="completed", phase="train", training_only=True))
    summaries = {}
    for market, architecture in zip(MARKETS, ("rnn", "cnn")):
        source, receipt = {"synthetic_test_fixture": True}, {"not_an_actual_dataset": True}
        ensemble = {fold: dict(calibration=calibration, qualification={"qualified": False})
                    for fold in ("walk_2022", "walk_2024")}
        ranking = [dict(architecture=architecture)]
        summary = dict(completed=True, pilot=False, market=market, selected=architecture, ranking=ranking,
                       protocol=recipe, source=source, data_receipt=receipt, ensemble=ensemble,
                       research_qualified=False, deployment_allowed=False, intraday_path_verified=False)
        summaries[market] = summary
        write_json(root / market / "summary.json", summary)
        write_json(root / market / "selection_locked.json", dict(selected=architecture, ranking=ranking,
                   rule=recipe["selection"], source=source, code_sha256=code))
        for fold, item in ensemble.items():
            write_json(root / market / fold / "ensemble.json", item)
        for seed in SEEDS:
            context = dict(market=market, fold="walk_2024", source=source, data_receipt=receipt, code_sha256=code)
            contract = dict(context=context, architecture=architecture, seed=seed, pilot=False, protocol=recipe)
            folder = root / market / "walk_2024" / f"{architecture}-{seed}"
            write_json(folder / "contract.json", contract)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                model = build_model(architecture, **config)
            checkpoint = dict(state_dict=model.state_dict(), architecture=architecture, model_config=config,
                feature_names=list(FEATURE_NAMES), class_names=list(CLASS_NAMES), target=TARGET,
                calibration=calibration, threshold=.5, take_profit_pct=1., stop_loss_pct=.9, seed=seed,
                context=context, contract=contract, **FLAGS)
            torch.save(checkpoint, folder / "model.pt")
            write_json(folder / "history.json", {"fixture": True})
            np.savez(folder / "predictions.npz", probabilities=np.zeros(3, dtype=np.float64))
            artifacts = {name: sha256_file(folder / name) for name in ("model.pt", "predictions.npz", "history.json")}
            write_json(folder / "model.sha256.json", {"sha256": artifacts["model.pt"]})
            write_json(folder / "result.json", dict(completed=True, contract=contract, architecture=architecture,
                       seed=seed, calibration=calibration, artifact_sha256=artifacts))
    write_json(root / "summary.json", summaries)


class UnsafeFixture:
    pass


class TestMark12Inference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.base = tempfile.TemporaryDirectory(prefix="dockdack-mark12-inference-fixture-")
        cls.base_run = Path(cls.base.name) / "run"
        cls.base_bundle = Path(cls.base.name) / "bundle"
        make_run(cls.base_run)
        cls.export = export_bundle(cls.base_run, cls.base_bundle)

    @classmethod
    def tearDownClass(cls):
        cls.base.cleanup()
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dockdack-mark12-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.bundle = self.root / "bundle"

    def copied_run(self):
        shutil.copytree(self.base_run, self.run)
        return self.run

    def copied_bundle(self):
        shutil.copytree(self.base_bundle, self.bundle)
        return self.bundle

    def change_json(self, path, callback):
        value = read_json(path)
        callback(value)
        write_json(path, value)

    def reseal_manifest(self, callback):
        path = self.bundle / "manifest.json"
        self.change_json(path, callback)
        (self.bundle / "manifest.sha256").write_text(sha256_file(path), encoding="ascii")

    def trial(self):
        return self.run / "domestic/walk_2024/rnn-42"

    def reseal_trial_model(self, callback):
        folder = self.trial()
        checkpoint = load_weights(folder / "model.pt")
        callback(checkpoint)
        torch.save(checkpoint, folder / "model.pt")
        digest = sha256_file(folder / "model.pt")
        self.change_json(folder / "result.json", lambda value: value["artifact_sha256"].update({"model.pt": digest}))
        write_json(folder / "model.sha256.json", {"sha256": digest})

    def test_export_verifies_twenty_four_cases_and_exact_bundle_files(self):
        validation = self.export["validation"]
        self.assertEqual(validation["cases"], 24)
        self.assertEqual(validation["cases_per_market"], {market: 12 for market in MARKETS})
        self.assertEqual(validation["maximum_absolute_difference"], {market: 0. for market in MARKETS})
        files = {path.relative_to(self.base_bundle).as_posix() for path in self.base_bundle.rglob("*") if path.is_file()}
        self.assertEqual(files, {"manifest.json", "manifest.sha256", "export-validation.json"}
                         | {f"{market}/seed{seed}.pt" for market in MARKETS for seed in SEEDS})
        manifest = read_json(self.base_bundle / "manifest.json")
        self.assertEqual(manifest["semantics"]["ensemble_accumulation_dtype"], "float32")
        self.assertEqual(manifest["semantics"]["platt_dtype"], "float64")
        self.assertEqual(manifest["source_run"]["source_files_sha256"], inspect_training_run(self.base_run)["source_files_sha256"])

    def test_both_market_predictions_match_direct_fp32_ensemble(self):
        source = inspect_training_run(self.base_run)
        history, entries = synthetic_cases()
        before_history, before_entries = history.copy(), entries.copy()
        with torch.inference_mode():
            features = features_from_history(torch.from_numpy(history), torch.from_numpy(entries), validate=True)
            for market in MARKETS:
                item = source["markets"][market]
                raw = []
                for member in item["members"]:
                    model = build_model(item["selected"], **item["model_config"]).eval()
                    model.load_state_dict(member["checkpoint"]["state_dict"])
                    raw.append(success_logit(model(features)).numpy())
                average = np.mean(raw, axis=0)
                self.assertEqual(average.dtype, np.float32)
                expected = calibrated_probability(average, item["calibration"])
                predictor = Predictor(self.base_bundle, market)
                actual = predictor.predict_proba(history, entries)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float64)
                self.assertTrue(np.all((actual >= 0) & (actual <= 1)))
        np.testing.assert_array_equal(history, before_history)
        np.testing.assert_array_equal(entries, before_entries)

    def test_single_history_scalar_query_and_metadata_are_isolated(self):
        history, entries = synthetic_cases()
        predictor = Predictor(self.base_bundle, "domestic")
        result = predictor.predict_proba(history[0], float(entries[0]))
        self.assertEqual(result.shape, (1,))
        np.testing.assert_allclose(result, predictor.predict_proba(history[:1], entries[:1]), atol=1e-12)
        predictor.metadata["semantics"]["threshold"] = 0
        self.assertEqual(predictor.metadata["semantics"]["threshold"], .5)
        self.assertFalse(predictor.metadata["deployment_allowed"])

    def test_bundle_is_portable_without_source_directory_access(self):
        self.copied_bundle()
        history, prices = synthetic_cases()
        predictor = Predictor(self.bundle, "us")
        self.assertEqual(predictor.predict_proba(history, prices).shape, (12,))
        self.assertNotIn(str(self.base_run), (self.bundle / "manifest.json").read_text())

    def test_export_does_not_modify_source(self):
        self.copied_run()
        before = {p.relative_to(self.run): sha256_file(p) for p in self.run.rglob("*") if p.is_file()}
        export_bundle(self.run, self.bundle)
        after = {p.relative_to(self.run): sha256_file(p) for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(list(self.root.glob(".mark1-2-building-*")))

    def test_existing_destination_is_never_modified(self):
        self.bundle.mkdir()
        (self.bundle / "keep.txt").write_text("old model")
        with self.assertRaisesRegex(ValueError, "Destination exists"):
            export_bundle(self.root / "missing-run", self.bundle)
        self.assertEqual((self.bundle / "keep.txt").read_text(), "old model")

    def test_failed_export_discards_only_its_staging_directory(self):
        keep = self.root / "keep.txt"
        keep.write_text("preserve")
        with patch("examples.export_mark1_2.Predictor", side_effect=ValueError("fixture failure")):
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                export_bundle(self.base_run, self.bundle)
        self.assertFalse(self.bundle.exists())
        self.assertEqual(keep.read_text(), "preserve")
        self.assertFalse(list(self.root.glob(".mark1-2-building-*")))

    def test_pilot_or_running_sources_are_rejected(self):
        self.copied_run()
        self.change_json(self.run / "protocol.json", lambda x: x.update(phase="pilot"))
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)
        shutil.copyfile(self.base_run / "protocol.json", self.run / "protocol.json")
        self.change_json(self.run / "status.json", lambda x: x.update(status="running"))
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)

    def test_missing_market_and_unlocked_selection_are_rejected(self):
        self.copied_run()
        self.change_json(self.run / "summary.json", lambda x: x.pop("us"))
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)
        shutil.copyfile(self.base_run / "summary.json", self.run / "summary.json")
        self.change_json(self.run / "domestic/selection_locked.json", lambda x: x.update(selected="lstm"))
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)

    def test_source_code_change_is_rejected(self):
        self.copied_run()
        self.change_json(self.run / "protocol.json", lambda x: x["code_sha256"].update({TRAINING_CODE_FILES[0]: "0" * 64}))
        with self.assertRaisesRegex(ValueError, "checksum"):
            inspect_training_run(self.run)

    def test_changed_prediction_and_checkpoint_seal_are_rejected(self):
        self.copied_run()
        folder = self.trial()
        (folder / "predictions.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)
        shutil.copyfile(self.base_run / "domestic/walk_2024/rnn-42/predictions.npz", folder / "predictions.npz")
        self.change_json(folder / "model.sha256.json", lambda x: x.update(sha256="0" * 64))
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)

    def test_empty_artifact_map_is_rejected(self):
        self.copied_run()
        self.change_json(self.trial() / "result.json", lambda x: x.update(artifact_sha256={}))
        with self.assertRaises(ValueError):
            inspect_training_run(self.run)

    def test_checkpoint_identity_and_nonfinite_state_are_rejected(self):
        self.copied_run()
        self.reseal_trial_model(lambda x: x.update(seed=44))
        with self.assertRaisesRegex(ValueError, "identity"):
            inspect_training_run(self.run)
        shutil.copytree(self.base_run / "domestic/walk_2024/rnn-42", self.trial(), dirs_exist_ok=True)
        def corrupt_state(payload):
            next(iter(payload["state_dict"].values())).flatten()[0] = float("nan")
        self.reseal_trial_model(corrupt_state)
        with self.assertRaisesRegex(ValueError, "finite"):
            inspect_training_run(self.run)

    def test_unsafe_pickle_is_rejected_without_allowlisting(self):
        path = self.root / "unsafe.pt"
        torch.save({"state_dict": UnsafeFixture()}, path)
        with self.assertRaises(Exception) as result:
            load_weights(path)
        self.assertIn("Weights only load failed", str(result.exception))

    def test_manifest_checksum_and_extra_files_are_rejected(self):
        self.copied_bundle()
        (self.bundle / "manifest.sha256").write_text("0" * 64)
        with self.assertRaises(ValueError):
            Predictor(self.bundle, "domestic")
        shutil.copyfile(self.base_bundle / "manifest.sha256", self.bundle / "manifest.sha256")
        (self.bundle / "unlisted.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            Predictor(self.bundle, "domestic")

    def test_other_market_checkpoint_is_also_hash_verified(self):
        self.copied_bundle()
        (self.bundle / "us/seed44.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Predictor(self.bundle, "domestic")

    def test_runtime_code_and_research_flags_are_checked(self):
        self.copied_bundle()
        self.reseal_manifest(lambda x: x["runtime_code_sha256"].update(inference="0" * 64))
        with self.assertRaisesRegex(ValueError, "contract"):
            Predictor(self.bundle, "domestic")
        shutil.copyfile(self.base_bundle / "manifest.json", self.bundle / "manifest.json")
        self.reseal_manifest(lambda x: x["risk_flags"].update(deployment_allowed=True))
        with self.assertRaisesRegex(ValueError, "flags"):
            Predictor(self.bundle, "domestic")

    def test_numeric_values_cannot_impersonate_boolean_flags(self):
        self.copied_bundle()
        self.reseal_manifest(lambda x: x["risk_flags"].update(research_only=1))
        with self.assertRaisesRegex(ValueError, "booleans"):
            Predictor(self.bundle, "domestic")

    def test_source_provenance_cannot_be_omitted_or_mixed(self):
        self.copied_bundle()
        self.reseal_manifest(lambda x: x["source_run"].update(protocol_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "provenance"):
            Predictor(self.bundle, "domestic")
        shutil.copyfile(self.base_bundle / "manifest.json", self.bundle / "manifest.json")
        self.reseal_manifest(lambda x: x["markets"]["domestic"]["members"][0].update(source_checkpoint_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "provenance"):
            Predictor(self.bundle, "domestic")

    def test_nonmonotone_calibration_is_rejected(self):
        self.copied_bundle()
        self.reseal_manifest(lambda x: x["markets"]["domestic"]["calibration"].update(slope=-1))
        with self.assertRaises(ValueError):
            Predictor(self.bundle, "domestic")

    def test_path_traversal_is_rejected(self):
        self.copied_bundle()
        self.reseal_manifest(lambda x: x["markets"]["domestic"]["members"][0].update(path="../outside.pt"))
        with self.assertRaises(ValueError):
            Predictor(self.bundle, "domestic")
        for value in ("../outside.pt", "C:/outside.pt", "a\\b", "a//b", "/absolute"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                checked_file(self.bundle, value, "0" * 64)

    def test_symlink_artifact_is_rejected_when_supported(self):
        link = self.root / "link"
        try:
            link.symlink_to(self.base_bundle, target_is_directory=True)
        except OSError:
            self.skipTest("OS does not permit unprivileged symlinks")
        with self.assertRaisesRegex(ValueError, "Linked"):
            Predictor(link, "domestic")

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_json(path)

    def test_only_cpu_and_known_markets_are_accepted(self):
        for market, device in (("domestic", "cuda"), ("unknown", "cpu"), ("us", "cpu:0")):
            with self.subTest(market=market, device=device), self.assertRaises(ValueError):
                Predictor(self.base_bundle, market, device=device)

    def test_invalid_inputs_fail_closed(self):
        history, prices = synthetic_cases()
        predictor = Predictor(self.base_bundle, "domestic")
        cases = [(history[:, :29], prices), (np.zeros((1, 31, 5)), 1.),
                 (history[:0], prices[:0]), (history, prices[:-1]), (history, -1.),
                 (history, np.nan), (history, True), (history.astype(object), prices),
                 (history, 1e300), (history, 1e-300)]
        bad = history.copy()
        bad[0, 0, 1] = .1
        cases.append((bad, prices))
        bad = history.copy()
        bad[0, 0, 4] = -1
        cases.append((bad, prices))
        for index, (bars, entries) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                predictor.predict_proba(bars, entries)

    def test_outer_autocast_does_not_change_cpu_fp32_contract(self):
        history, prices = synthetic_cases()
        predictor = Predictor(self.base_bundle, "us")
        expected = predictor.predict_proba(history, prices)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = predictor.predict_proba(history.astype(np.float64), prices.astype(np.float64))
        np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
