"""Both ordinary-desktop model choices remain offline until manual activation."""
from contextlib import redirect_stdout
import io
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from examples import run_desktop_gui as launcher


class Mark11DesktopLauncherTests(unittest.TestCase):
    def test_external_cli_never_implicitly_adds_legacy_lstm(self):
        from dockdack.v00_app import desktop_model_choices
        self.assertEqual(desktop_model_choices(None, False, ['mark1-prototype']), ('none', ['mark1-prototype']))
        self.assertEqual(desktop_model_choices('lstm30', False, ['mark1-prototype']), ('lstm30', ['mark1-prototype']))
        self.assertEqual(desktop_model_choices(None, False, []), ('none', ['mark1-prototype', 'mark1-1-prototype']))

    def test_explicit_mark11_paths_pass_through_unchanged(self):
        args = ['--trigger', 'mark1-1-prototype', '--mark11-bundle=half-model',
                '--mark1-bundle=old-model', '--store=paper.sqlite3', '--env-file=fake.env']
        self.assertEqual(launcher.desktop_arguments(args), args)

    def test_check_loads_both_models_for_both_markets_without_starting_gui(self):
        with patch('dockdack.mark1_prototype_inference.PrototypePredictor') as old, \
                patch('dockdack.mark1_1_prototype_inference.Mark11PrototypePredictor') as new, \
                patch('dockdack.v00_app.main', side_effect=AssertionError('Do not start the GUI')), \
                patch('requests.sessions.Session.request', side_effect=AssertionError('No network')), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launcher.main(['--check']), 0)
        self.assertEqual([call.args for call in old.call_args_list],
                         [(launcher.ROOT / 'models/mark1_prototype', market) for market in ('domestic', 'us')])
        self.assertEqual([call.args for call in new.call_args_list],
                         [(launcher.ROOT / 'models/mark1_1_prototype', market) for market in ('domestic', 'us')])
        self.assertIn('mark1.1 prototype', output.getvalue())
        self.assertIn('no monitoring, orders or network', output.getvalue())

    def test_help_names_both_choices_and_separate_bundle_arguments(self):
        from dockdack.v00_app import main
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as raised:
            main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        for option in ('mark1-prototype', 'mark1-1-prototype', '--mark1-bundle', '--mark11-bundle'):
            self.assertIn(option, output.getvalue())


if __name__ == '__main__':
    unittest.main()
