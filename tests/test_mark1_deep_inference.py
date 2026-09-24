"""Safe CPU ensemble loading and strictly non-deploying prediction contract."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_ML = (importlib.util.find_spec("torch") is not None
          and importlib.util.find_spec("numpy") is not None)
if HAS_ML:
    import numpy as np
    import torch
    from dockdack.mark1_deep_data import FOLDS
    from dockdack.mark1_deep_inference import DeepPredictor
    from dockdack.mark1_deep_models import CLASS_NAMES, FEATURE_NAMES, MODEL_NAMES, TARGET, build_model, success_logit
    from dockdack.mark1_metrics import calibrated_probability
    from examples.train_mark1_deep import PROTOCOL


@unittest.skipUnless(HAS_ML, "requires optional ml dependencies")
class DeepInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mark1-deep-inference-")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.folder = self.make_fixture(self.root / "domestic")

    @staticmethod
    def bars():
        close = np.linspace(99, 100, 30)
        return np.stack((close - .1, close + 1, close - 1, close,
                         np.linspace(1000, 1500, 30)), axis=-1)

    @staticmethod
    def save_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_fixture(self, folder, *, selected="mlp_deep", market="domestic", qualified=False):
        folder.mkdir(parents=True, exist_ok=True)
        config = dict(input_size=18, sequence_length=31, width=4, dropout=.2)
        protocol = copy.deepcopy(PROTOCOL)
        protocol["model_config"] = config
        source = dict(version=2, market=market, target=TARGET, database_sha256="a" * 64,
                      database_path="deliberately-not-opened.sqlite3")
        calibration = dict(method="platt_monotone", slope=1., bias=0., fit_samples=100)
        splits = {name: dict(samples=100, first="2023-02-15", last="2023-12-31", symbols=10)
                  for name in ("train", "tune", "calibration", "selection")}
        for seed in (42, 43, 44):
            path = folder / "walk_2024" / f"{selected}-{seed}" / "model.pt"
            path.parent.mkdir(parents=True)
            model = build_model(selected, **config)
            state = {name: torch.zeros_like(value) for name, value in model.state_dict().items()}
            state["head.classifier.3.bias"] = torch.tensor([math.log(3), 0., 0., 0.])
            checkpoint = dict(state_dict=state, architecture=selected, model_config=config,
                              feature_names=FEATURE_NAMES, class_names=CLASS_NAMES, target=TARGET,
                              context=dict(market=market, fold="walk_2024", source=source, splits=splits),
                              seed=seed, best_epoch=1, research_only=True, intraday_path_verified=False,
                              calibration=calibration, threshold=.5, take_profit_pct=1., stop_loss_pct=.9,
                              protocol=protocol)
            torch.save(checkpoint, path)
        summary = dict(market=market, selected=selected, protocol=protocol,
                       ensemble_calibration=calibration, research_qualified=qualified,
                       ensemble_qualification=dict(qualified=qualified),
                       fold_qualification={name: dict(qualified=qualified) for name in FOLDS},
                       seed_results=[dict(seed=seed, architecture=selected) for seed in (42, 43, 44)])
        self.save_json(folder / "summary.json", summary)
        self.save_json(folder / "source.json", source)
        return folder

    def edit_json(self, filename, mutation):
        path = self.folder / filename
        value = json.loads(path.read_text("utf-8"))
        mutation(value)
        self.save_json(path, value)

    def edit_checkpoint(self, mutation, *, seed=42):
        path = self.folder / "walk_2024" / f"mlp_deep-{seed}" / "model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        mutation(checkpoint)
        torch.save(checkpoint, path)

    def test_exact_half_holds_and_every_result_denies_deployment(self):
        predictor = DeepPredictor(self.folder)
        result = predictor.predict(self.bars(), 100.)
        self.assertEqual(result["probability_success"], .5)
        self.assertFalse(result["predicts_success"])
        self.assertEqual(result["buy_threshold"], .5)
        self.assertEqual(result["market"], "domestic")
        self.assertEqual(result["target"], TARGET)
        self.assertEqual(result["model_name"], "mlp_deep")
        self.assertTrue(result["research_only"])
        self.assertFalse(result["research_qualified"])
        self.assertFalse(result["deployment_allowed"])
        self.assertFalse(result["intraday_path_verified"])
        self.assertEqual(result["inference_precision"], "float32_cpu")
        self.assertTrue(all(not model.training for model in predictor.models))

    def test_strictly_above_half_can_predict_success_but_not_deploy(self):
        for seed in (42, 43, 44):
            self.edit_checkpoint(lambda value: value["state_dict"]["head.classifier.3.bias"].add_(
                torch.tensor([.001, 0., 0., 0.])), seed=seed)
        result = DeepPredictor(self.folder).predict(self.bars(), 100.)
        self.assertGreater(result["probability_success"], .5)
        self.assertTrue(result["predicts_success"])
        self.assertFalse(result["deployment_allowed"])

    def test_qualified_ensemble_still_cannot_deploy(self):
        folder = self.make_fixture(self.root / "qualified", qualified=True)
        result = DeepPredictor(folder).predict(self.bars(), "100")
        self.assertTrue(result["research_qualified"])
        self.assertFalse(result["deployment_allowed"])

    def test_ensemble_averages_raw_success_logits_then_calibrates(self):
        outputs = ([4., 1., -1., 0.], [-2., 0., 0., 0.], [.1, 2., 3., -4.])
        for seed, values in zip((42, 43, 44), outputs):
            self.edit_checkpoint(lambda checkpoint, values=values: checkpoint["state_dict"].__setitem__(
                "head.classifier.3.bias", torch.tensor(values)), seed=seed)
        self.edit_json("summary.json", lambda value: value["ensemble_calibration"].update(slope=1.7, bias=.2))
        logits = [success_logit(torch.tensor([values])).item() for values in outputs]
        expected = calibrated_probability([math.fsum(logits) / 3],
                                          dict(method="platt_monotone", slope=1.7, bias=.2))[0]
        actual = DeepPredictor(self.folder).predict(self.bars(), 100.)
        self.assertEqual(actual["probability_success"], expected)

    def test_every_architecture_loads_safely_and_us_metadata_is_retained(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                folder = self.make_fixture(self.root / name, selected=name, market="us")
                result = DeepPredictor(folder).predict(self.bars(), 100.)
                self.assertEqual(result["model_name"], name)
                self.assertEqual(result["market"], "us")
                self.assertEqual(result["probability_success"], .5)

    def test_inference_does_not_mutate_inputs_files_or_cpu_rng(self):
        before = {str(path.relative_to(self.folder)): path.read_bytes()
                  for path in self.folder.rglob("*") if path.is_file()}
        bars = self.bars()
        original = bars.copy()
        rng = torch.get_rng_state().clone()
        predictor = DeepPredictor(self.folder)
        predictor.predict(bars, 100.)
        np.testing.assert_array_equal(bars, original)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        after = {str(path.relative_to(self.folder)): path.read_bytes()
                 for path in self.folder.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_safe_weights_only_loading_is_always_requested(self):
        original = torch.load
        calls = []

        def observed_load(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        with patch.object(torch, "load", side_effect=observed_load):
            DeepPredictor(self.folder)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(value.get("weights_only") is True and value.get("map_location") == "cpu" for value in calls))

    def test_missing_or_malformed_summary_rejected(self):
        with self.assertRaises(ValueError):
            DeepPredictor(self.root / "missing")
        self.save_json(self.folder / "summary.json", [])
        with self.assertRaises(ValueError):
            DeepPredictor(self.folder)

    def test_invalid_market_architecture_or_qualification_rejected(self):
        original = (self.folder / "summary.json").read_text("utf-8")
        for key, invalid in (("market", "forex"), ("selected", "../resnet18"),
                             ("selected", []), ("research_qualified", 1)):
            (self.folder / "summary.json").write_text(original, encoding="utf-8")
            self.edit_json("summary.json", lambda value: value.__setitem__(key, invalid))
            with self.subTest(key=key, value=invalid):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_invalid_source_provenance_rejected(self):
        original = (self.folder / "source.json").read_text("utf-8")
        for key, invalid in (("market", "us"), ("target", "different"),
                             ("version", True), ("database_sha256", "invalid")):
            (self.folder / "source.json").write_text(original, encoding="utf-8")
            self.edit_json("source.json", lambda value: value.__setitem__(key, invalid))
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_invalid_protocol_or_configuration_rejected(self):
        original = (self.folder / "summary.json").read_text("utf-8")
        changes = (("threshold", .4), ("take", .02), ("stop", .01),
                   ("threshold_rule", "greater_or_equal"), ("both_touch", "take_first"),
                   ("ensemble_seeds", [42, 43, 45]), ("ensemble", "mean_probabilities"),
                   ("version", "unknown"))
        for key, invalid in changes:
            (self.folder / "summary.json").write_text(original, encoding="utf-8")
            self.edit_json("summary.json", lambda value: value["protocol"].__setitem__(key, invalid))
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)
        for key, invalid in (("input_size", 9), ("sequence_length", 30), ("width", 0),
                             ("width", 50000), ("width", True), ("dropout", True),
                             ("dropout", 1), ("dropout", float("nan"))):
            (self.folder / "summary.json").write_text(original, encoding="utf-8")
            self.edit_json("summary.json", lambda value: value["protocol"]["model_config"].__setitem__(key, invalid))
            with self.subTest(key=key, value=invalid):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_invalid_ensemble_calibration_rejected(self):
        original = (self.folder / "summary.json").read_text("utf-8")
        for key, invalid in (("method", "identity"), ("slope", -1.), ("slope", True),
                             ("bias", float("nan")), ("fit_samples", 0), ("fit_samples", 99)):
            (self.folder / "summary.json").write_text(original, encoding="utf-8")
            self.edit_json("summary.json", lambda value: value["ensemble_calibration"].__setitem__(key, invalid))
            with self.subTest(key=key, value=invalid):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_seed_summary_or_qualification_disagreement_rejected(self):
        self.edit_json("summary.json", lambda value: value["seed_results"][1].update(seed=42))
        with self.assertRaises(ValueError):
            DeepPredictor(self.folder)
        self.edit_json("summary.json", lambda value: value["seed_results"][1].update(seed=43))
        self.edit_json("summary.json", lambda value: value.update(research_qualified=True))
        with self.assertRaises(ValueError):
            DeepPredictor(self.folder)

    def test_bad_checkpoint_contracts_rejected(self):
        path = self.folder / "walk_2024" / "mlp_deep-42" / "model.pt"
        original = path.read_bytes()
        changes = (("architecture", "resnet18"), ("seed", 43), ("target", "other"),
                   ("feature_names", tuple(reversed(FEATURE_NAMES))),
                   ("class_names", ("both_touch", "stop_only", "take_only", "neither")),
                   ("threshold", .4), ("take_profit_pct", 2), ("stop_loss_pct", 1),
                   ("research_only", False), ("intraday_path_verified", True), ("best_epoch", 0))
        for key, invalid in changes:
            path.write_bytes(original)
            self.edit_checkpoint(lambda value: value.__setitem__(key, invalid))
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_checkpoint_market_fold_source_and_membership_mismatch_rejected(self):
        path = self.folder / "walk_2024" / "mlp_deep-42" / "model.pt"
        original = path.read_bytes()
        changes = (lambda value: value["context"].update(market="us"),
                   lambda value: value["context"].update(fold="walk_2022"),
                   lambda value: value["context"]["source"].update(database_sha256="b" * 64),
                   lambda value: value["context"]["splits"]["selection"].update(samples=99),
                   lambda value: value["context"]["splits"]["train"].update(samples=0),
                   lambda value: value["calibration"].update(fit_samples=99),
                   lambda value: value["protocol"].update(loss="changed"))
        for index, change in enumerate(changes):
            path.write_bytes(original)
            self.edit_checkpoint(change)
            with self.subTest(change=index):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_nonfinite_or_shape_mismatched_weights_rejected(self):
        path = self.folder / "walk_2024" / "mlp_deep-42" / "model.pt"
        original = path.read_bytes()
        changes = (lambda state: state["head.classifier.3.bias"].fill_(float("nan")),
                   lambda state: state["head.classifier.3.bias"].fill_(float("inf")),
                   lambda state: state.__setitem__("head.classifier.3.bias", torch.zeros(3)),
                   lambda state: state.__setitem__("head.classifier.3.bias", torch.zeros(4, dtype=torch.long)),
                   lambda state: state.pop("head.classifier.3.bias"))
        for index, change in enumerate(changes):
            path.write_bytes(original)
            self.edit_checkpoint(lambda value: change(value["state_dict"]))
            with self.subTest(change=index):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder)

    def test_missing_seed_checkpoint_rejected(self):
        (self.folder / "walk_2024" / "mlp_deep-44" / "model.pt").unlink()
        with self.assertRaises(ValueError):
            DeepPredictor(self.folder)

    def test_bad_candidate_price_or_history_rejected(self):
        predictor = DeepPredictor(self.folder)
        for candidate in (True, np.bool_(True), 0, -1, "nan", "inf", "1e500", "0.00000000000000000000000000000000000000000000001"):
            with self.subTest(price=candidate):
                with self.assertRaises(ValueError):
                    predictor.predict(self.bars(), candidate)
        malformed = (self.bars()[:29], np.zeros((31, 5)), np.ones((30, 5), dtype=bool),
                     np.ones((30, 5), dtype=complex), [["bad"]] * 30)
        for bars in malformed:
            with self.assertRaises(ValueError):
                predictor.predict(bars, 100.)
        for channel, value in ((0, float("nan")), (1, 0), (2, 1000), (3, -1), (4, -1)):
            bars = self.bars()
            bars[0, channel] = value
            with self.assertRaises(ValueError):
                predictor.predict(bars, 100.)

    def test_nonfinite_runtime_logit_rejected(self):
        predictor = DeepPredictor(self.folder)
        with torch.no_grad():
            predictor.models[0].head.classifier[-1].bias.fill_(float("nan"))
        with self.assertRaises(ValueError):
            predictor.predict(self.bars(), 100.)

    def test_unsupported_device_rejected_without_gpu_use(self):
        for device in ("meta", "nonsense", False):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    DeepPredictor(self.folder, device=device)


if __name__ == "__main__":
    unittest.main()
