"""Both ordinary-desktop model choices remain offline until manual activation."""
from contextlib import redirect_stdout
import io
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from examples import run_desktop_gui as launcher


class Mark11DesktopLauncherTests(unittest.TestCase):
    def test_gui_import_does_not_require_numpy_or_torch(self):
        code = ('import sys; sys.modules["numpy"] = None; sys.modules["torch"] = None; '
                'import dockdack.v00_app; '
                'assert "dockdack.mark1_intraday_models" not in sys.modules; '
                'assert "dockdack.mark1_intraday_extra_models" not in sys.modules; '
                'assert "dockdack.mark1_target_horizon_models" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], cwd=launcher.ROOT,
                                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_external_cli_never_implicitly_adds_legacy_lstm(self):
        from dockdack.v00_app import PROTOTYPE_NOTICES, desktop_model_choices
        self.assertEqual(desktop_model_choices(None, False, ['mark1-prototype']), ('none', ['mark1-prototype']))
        self.assertEqual(desktop_model_choices('lstm30', False, ['mark1-prototype']), ('lstm30', ['mark1-prototype']))
        # A fresh DEMO desktop selects every model but never arms orders.
        self.assertEqual(desktop_model_choices(None, False, []),
                         ('none', list(PROTOTYPE_NOTICES)))

    def test_explicit_mark11_paths_pass_through_unchanged(self):
        args = ['--trigger', 'mark1-2-prototype', '--mark12-bundle=neural-model', '--mark11-bundle=half-model',
                '--mark1-bundle=old-model', '--store=paper.sqlite3', '--env-file=fake.env']
        self.assertEqual(launcher.desktop_arguments(args), args)

    def test_check_loads_all_models_for_both_markets_without_starting_gui(self):
        from dockdack.signals.preopen_series import PREOPEN_MODELS
        from dockdack.signals.mark1_target_horizon_trigger import MODEL_IDS as TARGET_HORIZON_MODEL_IDS
        with patch('dockdack.mark1_prototype_inference.PrototypePredictor') as old, \
                patch('dockdack.mark1_1_prototype_inference.Mark11PrototypePredictor') as new, \
                patch('dockdack.signals.mark1_2_trigger.Mark12PrototypePredictor') as neural, \
                patch('dockdack.mark1_4_inference.Mark14Predictor') as mark14, \
                patch('dockdack.mark1_target_horizon_inference.MarkTargetHorizonPredictor') as target_horizon, \
                patch('dockdack.signals.preopen_series.load_preopen_predictor') as preopen, \
                patch('dockdack.v00_app.main', side_effect=AssertionError('Do not start the GUI')), \
                patch('requests.sessions.Session.request', side_effect=AssertionError('No network')), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main(['--check']), 0)
        self.assertEqual([call.args for call in old.call_args_list],
                         [(launcher.ROOT / 'models/mark1_prototype', market) for market in ('domestic', 'us')])
        self.assertEqual([call.args for call in new.call_args_list],
                         [(launcher.ROOT / 'models/mark1_1_prototype', market) for market in ('domestic', 'us')])
        self.assertEqual([call.args for call in neural.call_args_list],
                         [(launcher.ROOT / 'models/mark1_2_prototype', market) for market in ('domestic', 'us')])
        self.assertEqual([call.args for call in mark14.call_args_list],
                         [(launcher.ROOT / 'models/mark1_4', market) for market in ('domestic', 'us')])
        self.assertEqual([call.args for call in preopen.call_args_list],
                         [(model_id, launcher.ROOT / spec.bundle_directory, market)
                          for market in ('domestic', 'us') for model_id, spec in PREOPEN_MODELS.items()])
        self.assertEqual([call.args for call in target_horizon.call_args_list],
                         [(launcher.ROOT / 'models/mark1_target_horizon_v1', market, model_id)
                          for market in ('domestic', 'us') for model_id in TARGET_HORIZON_MODEL_IDS])
        self.assertIn('mark1.0-mark1.28', output.getvalue())
        self.assertIn('no monitoring, orders or network', output.getvalue())

    def test_help_names_all_choices_and_separate_bundle_arguments(self):
        from dockdack.v00_app import main
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as raised:
            main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        for option in ('mark1-prototype', 'mark1-1-prototype', 'mark1-2-prototype',
                       '--mark1-bundle', '--mark11-bundle', '--mark12-bundle'):
            self.assertIn(option, output.getvalue())


if __name__ == '__main__':
    unittest.main()
