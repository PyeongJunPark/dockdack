"""Offline common-architecture tests; optional isolated CUDA shutdown smoke."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from dockdack import mark1_deep_models as frozen
    from dockdack.mark1_2_models import (
        CLASS_NAMES, FEATURE_NAMES, MODEL_NAMES, build_model, features_from_history,
        parameter_count, success_logit, validate_features,
    )


CUDA_SMOKE_SCRIPT = r'''
import io, json
import torch
from dockdack.mark1_2_models import MODEL_NAMES, build_model, success_logit
if not torch.cuda.is_available():
    raise RuntimeError("CUDA explicitly requested but unavailable")
torch.set_num_threads(2)
torch.manual_seed(123)
features = torch.randn(3,31,18,device="cuda")
before = torch.backends.cudnn.enabled
checks = []
for name in MODEL_NAMES:
    model = build_model(name,width=8,dropout=.2).to("cuda").train()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits = model(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            success_logit(logits.float()),torch.tensor([0.,1.,0.],device="cuda"))
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()):
        raise RuntimeError(name + " invalid gradients")
    if torch.backends.cudnn.enabled != before:
        raise RuntimeError(name + " leaked backend setting")
    if name in ("rnn","lstm","gru") and logits.dtype != torch.float32:
        raise RuntimeError(name + " recurrent AMP guard failed")
    model.eval()
    with torch.no_grad():
        expected = model(features).cpu()
    buffer = io.BytesIO()
    torch.save(model.state_dict(),buffer)
    buffer.seek(0)
    clone = build_model(name,width=8,dropout=.2).to("cuda").eval()
    clone.load_state_dict(torch.load(buffer,map_location="cuda",weights_only=True))
    with torch.no_grad():
        actual = clone(features).cpu()
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    checks.append({"name":name,"shape":list(actual.shape),"train_dtype":str(logits.dtype),"finite_gradients":True,"reload_equal":True})
    del model, clone
torch.cuda.synchronize()
print(json.dumps({"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"checks":checks,"cudnn_restored":torch.backends.cudnn.enabled==before}))
'''


@unittest.skipUnless(HAS_TORCH, "requires optional ml dependencies")
class Mark12ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(314)
        self.features = torch.randn(3, 31, 18)

    def test_common_exports_reuse_frozen_feature_semantics(self):
        self.assertEqual(MODEL_NAMES, ("mlp", "rnn", "lstm", "gru", "cnn", "resnet18", "resnet34"))
        self.assertIs(features_from_history, frozen.features_from_history)
        self.assertIs(success_logit, frozen.success_logit)
        self.assertIs(validate_features, frozen.validate_features)
        self.assertEqual(FEATURE_NAMES, frozen.FEATURE_NAMES)
        self.assertEqual(CLASS_NAMES, ("take_only", "stop_only", "both_touch", "neither"))

    def test_every_architecture_returns_finite_four_logits(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name, width=8).eval()
                output = model(self.features)
                self.assertEqual(output.shape, (3, 4))
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreater(parameter_count(model), 0)

    def test_same_four_class_and_binary_loss_have_finite_gradients(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name, width=8).train()
                features = self.features.clone().requires_grad_()
                output = model(features)
                classes = torch.tensor([0, 2, 3])
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(success_logit(output), (classes == 0).float())
                        + .25 * torch.nn.functional.cross_entropy(output, classes))
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad[:, -1].abs().sum().item(), 0)
                self.assertGreater(features.grad[:, :-1].abs().sum().item(), 0)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_query_changes_never_enter_history_encoder(self):
        modified = self.features.clone()
        modified[:, -1] += 2
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name, width=8).eval()
                encoded = []
                def capture(module, args, output):
                    encoded.append((output[0] if isinstance(output, tuple) else output).detach().clone())
                handle = model.history_encoder.register_forward_hook(capture)
                first, second = model(self.features), model(modified)
                handle.remove()
                torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)
                self.assertFalse(torch.equal(first, second))

    def test_recurrence_receives_only_thirty_history_tokens(self):
        for name in ("rnn", "lstm", "gru"):
            model = build_model(name, width=8).eval()
            seen = []
            handle = model.history_encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
            model(self.features)
            handle.remove()
            self.assertEqual(seen, [torch.Size((3, 30, 18))])
            self.assertEqual(model.history_encoder.num_layers, 2)
            self.assertEqual(model.history_encoder.hidden_size, 16)
            self.assertFalse(model.history_encoder.bidirectional)

    def test_save_weights_reload_is_exact(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name, width=8).eval()
                expected = model(self.features)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                clone = build_model(name, width=8).eval()
                clone.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
                torch.testing.assert_close(clone(self.features), expected, rtol=0, atol=0)

    def test_mlp_resnets_delegate_to_existing_architecture(self):
        for name in ("mlp", "resnet18", "resnet34"):
            with self.subTest(name=name):
                torch.manual_seed(51)
                model = build_model(name, width=8)
                torch.manual_seed(51)
                original = frozen.build_model("mlp_deep" if name == "mlp" else name, width=8)
                self.assertEqual(type(model), type(original))
                self.assertEqual(parameter_count(model), parameter_count(original))
                for key in original.state_dict():
                    torch.testing.assert_close(model.state_dict()[key], original.state_dict()[key], rtol=0, atol=0)

    def test_input_not_mutated(self):
        before = self.features.clone()
        for name in MODEL_NAMES:
            build_model(name, width=8).eval()(self.features)
            torch.testing.assert_close(self.features, before, rtol=0, atol=0)

    def test_invalid_parameters_are_rejected_for_every_architecture(self):
        for name in MODEL_NAMES:
            for options in ({"input_size": 9}, {"input_size": True}, {"sequence_length": 30},
                            {"sequence_length": 31.0}, {"width": 3}, {"width": True}, {"width": 8.5},
                            {"dropout": True}, {"dropout": -0.1}, {"dropout": 1.0}, {"dropout": float("nan")}):
                with self.subTest(name=name, options=options):
                    with self.assertRaises(ValueError):
                        build_model(name, **options)
        for name in (None, "transformer", "mlp_deep", ""):
            with self.assertRaises(ValueError):
                build_model(name)

    def test_invalid_shape_and_dtype_are_rejected(self):
        for name in MODEL_NAMES:
            model = build_model(name, width=8)
            for features in (None, self.features[:, :30], self.features[:, :, :9], self.features[:0], self.features.long()):
                with self.subTest(name=name, shape=getattr(features, "shape", None)):
                    with self.assertRaises(ValueError):
                        model(features)

    def test_recurrent_cpu_never_changes_cudnn_and_disables_outer_amp(self):
        for name in ("rnn", "lstm", "gru"):
            model = build_model(name, width=8).train()
            with patch("torch.backends.cudnn.flags", side_effect=AssertionError("CPU must not touch cuDNN")):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(self.features)
            self.assertEqual(output.dtype, torch.float32)
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_recurrent_requires_float32_parameters_with_clear_error(self):
        for name in ("rnn", "lstm", "gru"):
            with self.assertRaisesRegex(ValueError, "float32"):
                build_model(name, width=8).half()(self.features)

    def test_explicit_finite_input_validation(self):
        invalid = self.features.clone()
        invalid[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            validate_features(invalid)

    @unittest.skipUnless(os.environ.get("DOCKDACK_MARK12_CUDA_SMOKE") == "1", "opt-in isolated CUDA process")
    def test_cuda_forward_backward_reload_and_normal_process_exit(self):
        process = subprocess.run([sys.executable, "-X", "utf8", "-c", CUDA_SMOKE_SCRIPT],
                                 cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, msg=process.stdout + process.stderr)
        report = json.loads(process.stdout.strip().splitlines()[-1])
        self.assertEqual(tuple(item["name"] for item in report["checks"]), MODEL_NAMES)
        self.assertTrue(report["cudnn_restored"])
        self.assertTrue(all(item["finite_gradients"] and item["reload_equal"] for item in report["checks"]))


if __name__ == "__main__":
    unittest.main()
