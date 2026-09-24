"""Fast, synthetic CPU-only backend checks; no data DB or GPU required."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CAT = importlib.util.find_spec("catboost") is not None
HAS_LGB = importlib.util.find_spec("lightgbm") is not None
if HAS_NUMPY:
    import numpy as np
    from dockdack.mark1_selective_models import (
        MODEL_NAMES, fit_model, joint_logits, load_model, model_path, predict_raw, save_model,
    )


@unittest.skipUnless(HAS_NUMPY, "optional numpy required")
class SelectiveMathTests(unittest.TestCase):
    def test_names_and_extensions(self):
        self.assertEqual(MODEL_NAMES, ("cat_binary6", "cat_binary8", "cat_joint6", "lgbm_binary"))
        self.assertEqual(model_path("example", "cat_binary6").suffix, ".cbm")
        self.assertEqual(model_path("example", "lgbm_binary").suffix, ".txt")
        with self.assertRaises(ValueError):
            model_path("example", "unknown")

    def test_joint_matches_probability_sums(self):
        probabilities = np.array([[.2, .3, .1, .4], [.7, .1, .15, .05]])
        result = joint_logits(np.log(probabilities))
        np.testing.assert_allclose(1 / (1 + np.exp(-result["success_logits"])), probabilities[:, 0])
        np.testing.assert_allclose(1 / (1 + np.exp(-result["stop_logits"])), probabilities[:, 1:3].sum(1))

    def test_joint_stable_extreme_and_translation_invariant(self):
        raw = np.array([[1000, -1000, 0, 1], [-1000, 1000, 999, 998]], dtype=float)
        result, shifted = joint_logits(raw), joint_logits(raw + 100000)
        for key in result:
            self.assertTrue(np.isfinite(result[key]).all())
            np.testing.assert_allclose(result[key], shifted[key], atol=1e-10)

    def test_joint_invalid_inputs(self):
        for values in (np.zeros((2, 3)), np.zeros(4), [[0, 0, 0, np.inf]], [[0, 0, 0, np.nan]]):
            with self.assertRaises(ValueError):
                joint_logits(values)

    def test_invalid_request_fails_before_dependency_or_disk(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "not-created"
            x = np.ones((8, 3), np.float32)
            y = np.arange(8, dtype=np.int32) % 4
            for option in ({"max_iterations": 0}, {"early_stopping": True}, {"threads": -1},
                           {"seed": -1}, {"task_type": "cuda"}):
                with self.assertRaises(ValueError):
                    fit_model("cat_binary6", x, y, x, y, folder, **option)
            with self.assertRaises(ValueError):
                fit_model("cat_binary6", x, y.astype(float), x, y, folder)
            with self.assertRaises(ValueError):
                fit_model("cat_binary6", x, y, x[:, :2], y, folder)
            self.assertFalse(folder.exists())

    def test_nonfinite_and_invalid_classes_rejected(self):
        x = np.ones((8, 3), np.float32)
        y = np.arange(8, dtype=np.int32) % 4
        bad = x.copy()
        bad[0, 0] = np.nan
        for train_x, train_y in ((bad, y), (x, y + 1), (x, np.zeros(8, dtype=np.int32))):
            with self.assertRaises(ValueError):
                fit_model("cat_binary6", train_x, train_y, x, y, "never-created")
        with self.assertRaises(ValueError):
            fit_model("cat_joint6", x, y % 2, x, y, "never-created")

    def test_prediction_explicit_cpu_and_empty(self):
        class Fake:
            classes_ = np.arange(2)
            n_features_in_ = 2

            def predict(self, matrix, **kwargs):
                self.kwargs = kwargs
                return np.arange(len(matrix), dtype=float)

        model = Fake()
        values = predict_raw(model, "cat_binary6", np.zeros((3, 2)))
        self.assertEqual(model.kwargs["task_type"], "CPU")
        self.assertIsNone(values["stop_logits"])
        self.assertEqual(values["success_logits"].dtype, np.float64)
        self.assertEqual(predict_raw(model, "cat_binary6", np.zeros((0, 2)))["success_logits"].shape, (0,))
        with self.assertRaises(ValueError):
            predict_raw(model, "cat_binary6", np.zeros((3, 4)))

    def test_bad_joint_model_class_order_rejected(self):
        class Fake:
            classes_ = np.array([1, 0, 2, 3])
            n_features_in_ = 2
        with self.assertRaises(ValueError):
            predict_raw(Fake(), "cat_joint6", np.zeros((2, 2)))


@unittest.skipUnless(HAS_NUMPY and HAS_CAT, "optional CatBoost required")
class CatBoostSyntheticTests(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(14)
        x = rng.normal(size=(800, 6)).astype(np.float32)
        y = ((x[:, 0] > 0).astype(np.int32) + 2 * (x[:, 1] > 0)).astype(np.int32)
        return x[:600], y[:600], x[600:], y[600:]

    def test_all_cat_models_cpu_artifact_roundtrip(self):
        for name in MODEL_NAMES[:3]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                arrays = self.data()
                before = [array.copy() for array in arrays]
                model, metadata = fit_model(name, *arrays, temporary, task_type="CPU",
                                            max_iterations=10, early_stopping=3, threads=2)
                actual = predict_raw(model, name, arrays[2])
                restored = predict_raw(load_model(name, model_path(temporary, name)), name, arrays[2])
                np.testing.assert_allclose(actual["success_logits"], restored["success_logits"], atol=0, rtol=0)
                if name == "cat_joint6":
                    np.testing.assert_allclose(actual["stop_logits"], restored["stop_logits"], atol=0, rtol=0)
                self.assertTrue(1 <= metadata["best_iteration"] <= metadata["trained_iterations"] <= 10)
                self.assertEqual(metadata["effective_task_type"], "CPU")
                self.assertEqual(len(metadata["model_sha256"]), 64)
                self.assertFalse(metadata["deployment_allowed"])
                for old, new in zip(before, arrays):
                    np.testing.assert_array_equal(old, new)

    def test_exact_repeat_reuses_and_different_config_or_data_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays = self.data()
            kwargs = dict(task_type="CPU", max_iterations=10, early_stopping=3, threads=2)
            first, metadata = fit_model("cat_binary6", *arrays, temporary, **kwargs)
            with patch("catboost.CatBoostClassifier.fit", side_effect=AssertionError("must not retrain")):
                second, reused = fit_model("cat_binary6", *arrays, temporary, **kwargs)
            self.assertTrue(reused["reused"])
            self.assertEqual(metadata["model_sha256"], reused["model_sha256"])
            with self.assertRaisesRegex(ValueError, "incompatible cached"):
                fit_model("cat_binary6", *arrays, temporary, seed=43, **kwargs)
            altered = list(arrays)
            altered[0] = altered[0].copy()
            altered[0][0, 0] += .1
            with self.assertRaisesRegex(ValueError, "incompatible cached"):
                fit_model("cat_binary6", *altered, temporary, **kwargs)

    def test_ownership_checksum_name_and_no_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arrays = self.data()
            kwargs = dict(task_type="CPU", max_iterations=10, early_stopping=3, threads=2)
            unowned = folder / "unowned"
            unowned.mkdir()
            (unowned / "user.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ownership"):
                fit_model("cat_binary6", *arrays, unowned, **kwargs)
            self.assertEqual((unowned / "user.txt").read_text(), "preserve")
            model, _ = fit_model("cat_binary6", *arrays, folder / "owned", **kwargs)
            path = model_path(folder / "owned", "cat_binary6")
            with self.assertRaises(FileExistsError):
                save_model(model, "cat_binary6", path)
            with self.assertRaisesRegex(ValueError, "contract/checksum"):
                load_model("cat_binary8", path)
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "contract/checksum"):
                load_model("cat_binary6", path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                fit_model("cat_binary6", *arrays, folder / "owned", **kwargs)

    def test_owned_partial_fit_can_restart_without_deleting_user_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays = self.data()
            kwargs = dict(task_type="CPU", max_iterations=10, early_stopping=3, threads=2)
            with patch("catboost.CatBoostClassifier.fit", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    fit_model("cat_binary6", *arrays, temporary, **kwargs)
            marker = Path(temporary) / "keep-note.txt"
            marker.write_text("preserve", encoding="utf-8")
            _, result = fit_model("cat_binary6", *arrays, temporary, **kwargs)
            self.assertFalse(result["reused"])
            self.assertEqual(marker.read_text(), "preserve")


@unittest.skipUnless(HAS_NUMPY and HAS_LGB, "optional LightGBM required")
class LightGBMSyntheticTests(unittest.TestCase):
    def test_cpu_roundtrip_reuse_and_fixed_params(self):
        rng = np.random.default_rng(25)
        x = rng.normal(size=(1400, 5)).astype(np.float32)
        y = ((x[:, 0] > 0).astype(np.int32) + 2 * (x[:, 1] > 0)).astype(np.int32)
        with tempfile.TemporaryDirectory() as temporary:
            args = ("lgbm_binary", x[:1000], y[:1000], x[1000:], y[1000:], temporary)
            model, metadata = fit_model(*args, task_type="GPU", max_iterations=10, early_stopping=3, threads=2)
            self.assertEqual(metadata["effective_task_type"], "CPU")
            self.assertTrue(metadata["params"]["deterministic"])
            self.assertEqual(metadata["params"]["metric"], "average_precision")
            self.assertEqual(metadata["params"]["min_data_in_leaf"], 200)
            actual = predict_raw(model, "lgbm_binary", x[1000:])
            restored = predict_raw(load_model("lgbm_binary", model_path(temporary, "lgbm_binary")), "lgbm_binary", x[1000:])
            np.testing.assert_allclose(actual["success_logits"], restored["success_logits"], rtol=0, atol=0)
            self.assertIsNone(actual["stop_logits"])
            with patch("lightgbm.train", side_effect=AssertionError("must not retrain")):
                _, again = fit_model(*args, task_type="GPU", max_iterations=10, early_stopping=3, threads=2)
            self.assertTrue(again["reused"])


if __name__ == "__main__":
    unittest.main()
