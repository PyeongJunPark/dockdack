"""Offline trainer contracts: no real accounts, database, or fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from examples.train_mark1_2 import (
    FACTORS, FOLDS, MODEL_NAMES, PROTOCOL, augmented_loss, binary_loss,
    epoch_generator, load_checkpoint, lock_json, rank_architectures, save_checkpoint,
    save_resume, load_resume, run_lock, verify_completed_trial,
)
from dockdack.mark1_deep_models import success_logit


class Mark12TrainingTests(unittest.TestCase):
    def test_loss_matches_declared_actual_to_synthetic_weights(self):
        actual = torch.tensor([[.2, .1, -.3, .4], [-.4, .7, .2, -.1]], requires_grad=True)
        augmented = torch.tensor([[.8, -.1, .2, -.6], [.3, -.2, .6, .1]], requires_grad=True)
        labels, synthetic = torch.tensor([0, 2]), torch.tensor([2, 0])
        expected = (F.binary_cross_entropy_with_logits(success_logit(actual), (labels == 0).float())
                    + .25 * F.cross_entropy(actual, labels)
                    + .5 * F.binary_cross_entropy_with_logits(success_logit(augmented), (synthetic == 0).float())
                    + .125 * F.cross_entropy(augmented, synthetic))
        loss = augmented_loss(actual, augmented, labels, synthetic)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for value in (actual.grad, augmented.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(float(value.abs().sum()), 0)

    def test_sampler_is_architecture_independent_and_epoch_dependent(self):
        first = torch.randperm(100, generator=epoch_generator(42, 1, "cpu"))
        torch.rand(1234)
        second = torch.randperm(100, generator=epoch_generator(42, 1, "cpu"))
        third = torch.randperm(100, generator=epoch_generator(42, 2, "cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_stable_binary_loss_at_large_logits(self):
        self.assertEqual(binary_loss(np.array([1000., -1000.]), np.array([1., 0.])), 0.)

    def test_contract_canonicalization_reuses_tuples_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "protocol.json"
            lock_json(path, {"factors": FACTORS})
            original = path.read_bytes()
            lock_json(path, {"factors": list(FACTORS)})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                lock_json(path, {"factors": [1.]})
            self.assertEqual(path.read_bytes(), original)

    def test_checkpoint_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, {"weight": torch.tensor([1., 2.]), "version": "test"})
            torch.testing.assert_close(load_checkpoint(path)["weight"], torch.tensor([1., 2.]))
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_checkpoint(path)

    def test_incomplete_publication_keeps_previous_resume_generation(self):
        import examples.train_mark1_2 as trainer
        with tempfile.TemporaryDirectory() as folder:
            save_resume(folder, {"history": [1], "weight": torch.tensor([1.])})
            original_writer = trainer.atomic_json
            def interrupted(path, value):
                if Path(path).name == "resume.json":
                    raise OSError("simulated interruption before pointer publication")
                return original_writer(path, value)
            with patch.object(trainer, "atomic_json", side_effect=interrupted):
                with self.assertRaises(OSError):
                    save_resume(folder, {"history": [1, 2], "weight": torch.tensor([2.])})
            self.assertEqual(load_resume(folder, "cpu")["history"], [1])
            save_resume(folder, {"history": [1, 2], "weight": torch.tensor([2.])})
            self.assertEqual(load_resume(folder, "cpu")["history"], [1, 2])

    def test_run_lock_rejects_second_writer_then_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            with run_lock(folder):
                with self.assertRaises(OSError):
                    with run_lock(folder):
                        self.fail("second writer acquired the run")
            with run_lock(folder):
                pass

    def test_empty_missing_or_traversing_artifact_sets_are_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            for hashes in ({}, {"../escape": "a" * 64}, {"model.pt": "a" * 64}):
                with self.assertRaisesRegex(ValueError, "artifact set"):
                    verify_completed_trial(folder, {"completed": True, "contract": {}, "artifact_sha256": hashes}, {})

    def test_protocol_excludes_test_year_and_never_authorizes_deployment(self):
        self.assertEqual(tuple(FACTORS), (1., .99, .995, 1.005, 1.01))
        self.assertEqual(PROTOCOL["loss_actual_to_augmented_weight"], "2:1")
        self.assertFalse(PROTOCOL["deployment_allowed"])
        self.assertFalse(PROTOCOL["intraday_path_verified"])
        self.assertEqual(PROTOCOL["threshold"], .5)
        self.assertTrue(all(fold["selection_year"] < 2025 for fold in FOLDS.values()))

    def results(self):
        return {fold: {name: {"brier_skill": i * .01,
            "qualification": {"qualified": False},
            "selection": {"block_bootstrap": {"reason": "insufficient", "precision_lower": None,
                                                "net_mean_lower": None}}}
            for i, name in enumerate(MODEL_NAMES)} for fold in FOLDS}

    def test_no_signal_evidence_uses_explicit_diagnostic_fallback(self):
        results = self.results()
        ranking = rank_architectures(results)
        self.assertEqual(ranking[0]["architecture"], MODEL_NAMES[-1])
        self.assertFalse(any(row["qualified_both_folds"] for row in ranking))
        self.assertFalse(any(row["supported_both_folds"] for row in ranking))

    def test_supported_both_folds_required_not_one_lucky_fold(self):
        results = self.results()
        supported = {"reason": None, "precision_lower": .60, "net_mean_lower": .0001}
        for fold in FOLDS:
            results[fold][MODEL_NAMES[0]]["selection"]["block_bootstrap"] = copy.deepcopy(supported)
        results[next(iter(FOLDS))][MODEL_NAMES[-1]]["selection"]["block_bootstrap"] = {
            **supported, "precision_lower": .99}
        ranking = rank_architectures(results)
        self.assertEqual(ranking[0]["architecture"], MODEL_NAMES[0])
        self.assertTrue(ranking[0]["supported_both_folds"])

    def test_qualified_candidate_precedes_higher_precision_but_failed_candidate(self):
        results = self.results()
        for fold in FOLDS:
            for name, precision in ((MODEL_NAMES[0], .60), (MODEL_NAMES[1], .61)):
                results[fold][name]["selection"]["block_bootstrap"] = {
                    "reason": None, "precision_lower": precision, "net_mean_lower": .0001}
            results[fold][MODEL_NAMES[0]]["qualification"]["qualified"] = True
        self.assertEqual(rank_architectures(results)[0]["architecture"], MODEL_NAMES[0])


if __name__ == "__main__":
    unittest.main()
