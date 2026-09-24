"""Exercise the VBS wrapper -> GUI startup path with disposable ledgers only."""
from contextlib import ExitStack, closing, redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None
if HAS_QT:
    from PySide6.QtWidgets import QApplication, QMessageBox
    from dockdack import v00_app
from dockdack.models import TradingMode
from dockdack.persistence.environment_store import scoped_store_path
from dockdack.watchlist import WatchStore
from examples import run_desktop_gui as launcher


@unittest.skipUnless(HAS_QT, 'Install GUI extra')
class DesktopLauncherScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dockdack'
        self.root.mkdir()
        self.source = self.root / '.dockdack/lstm30-demo/watchlist.sqlite3'
        self.legacy = WatchStore(self.source)
        self.legacy.event('', 'legacy startup regression marker')
        self.before = self.source.read_bytes()
        self.service = SimpleNamespace(mode=TradingMode.DEMO, storage_scope='a' * 64)
        self.target = scoped_store_path(self.service, base_folder=self.source.parent)
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        for name, value in (
            ('dockdack.runtime_paths.app_home', self.root),
            ('dockdack.v00_app.TradingService', self.service),
            ('dotenv.load_dotenv', None),
            ('examples.run_desktop_gui.configure_local_dependencies', None),
            ('dockdack.v00_app.set_windows_app_id', None),
            ('dockdack.v00_app.apply_branding', None),
        ):
            self.context.enter_context(patch(name, return_value=value))
        self.context.enter_context(patch('requests.sessions.Session.request', side_effect=AssertionError('No network')))
        self.application = Mock()
        self.application.exec.return_value = 0
        self.context.enter_context(patch('dockdack.v00_app.QApplication.instance', return_value=self.application))
        self.window = self.context.enter_context(patch('dockdack.v00_app.V00Window'))
        self.question = self.context.enter_context(patch('dockdack.v00_app.QMessageBox.question',
                                                        return_value=QMessageBox.StandardButton.Cancel))
        self.warning = self.context.enter_context(patch('dockdack.v00_app.QMessageBox.warning'))

    def launch(self, *extra):
        return launcher.main(['--no-model', '--external-model', 'mark1-prototype',
                              '--external-model', 'mark1-1-prototype', *extra])

    def assert_source_preserved(self):
        self.assertEqual(self.source.read_bytes(), self.before)
        with closing(sqlite3.connect(self.source.as_uri() + '?mode=ro', uri=True)) as db:
            self.assertEqual(db.execute('SELECT scope FROM app_environment').fetchone(), ('demo',))

    def test_cancel_asks_about_legacy_source_and_creates_no_destination(self):
        self.assertEqual(self.launch(), 0)
        self.question.assert_called_once()
        self.assertEqual(self.question.call_args.args[-1], QMessageBox.StandardButton.Cancel)
        self.assertIn(str(self.source), self.question.call_args.args[2])
        self.window.assert_not_called()
        self.warning.assert_not_called()
        self.assertFalse(self.target.exists())
        self.assert_source_preserved()

    def test_confirmed_migration_copies_then_opens_current_scope(self):
        self.question.return_value = QMessageBox.StandardButton.Yes
        self.assertEqual(self.launch(), 0)
        self.question.assert_called_once()
        self.warning.assert_not_called()
        store = self.window.call_args.kwargs['store']
        self.assertEqual(store.path, self.target)
        self.assertEqual(store.storage_scope, self.service.storage_scope)
        self.assertTrue(any(row['message'] == 'legacy startup regression marker' for row in store.events()))
        self.assertFalse(self.window.call_args.kwargs['builtin'])
        self.assertEqual(self.window.call_args.kwargs['external_models'], ['mark1-prototype', 'mark1-1-prototype'])
        self.assert_source_preserved()

    def test_no_opens_empty_scoped_ledger_without_erasing_source(self):
        self.question.return_value = QMessageBox.StandardButton.No
        self.assertEqual(self.launch(), 0)
        self.question.assert_called_once()
        store = self.window.call_args.kwargs['store']
        self.assertEqual(store.path, self.target)
        self.assertFalse(any(row['message'] == 'legacy startup regression marker' for row in store.events()))
        self.assert_source_preserved()

    def test_existing_scoped_destination_reused_without_reimport(self):
        current = WatchStore(self.target, storage_scope=self.service.storage_scope)
        current.event('', 'already scoped marker')
        self.assertEqual(self.launch(), 0)
        self.question.assert_not_called()
        self.warning.assert_not_called()
        messages = [row['message'] for row in self.window.call_args.kwargs['store'].events()]
        self.assertIn('already scoped marker', messages)
        self.assertNotIn('legacy startup regression marker', messages)
        self.assert_source_preserved()

    def test_explicit_legacy_path_as_store_still_fails_without_rebinding(self):
        self.assertEqual(self.launch('--store', str(self.source)), 1)
        self.question.assert_not_called()
        self.window.assert_not_called()
        self.assertIn('API 인증 범위가 다릅니다', self.warning.call_args.args[2])
        self.assertFalse(self.target.exists())
        self.assert_source_preserved()

    def test_explicit_other_account_remains_blocked(self):
        other = WatchStore(self.root / 'other.sqlite3', storage_scope='b' * 64)
        before = other.path.read_bytes()
        self.assertEqual(self.launch('--store', str(other.path)), 1)
        self.question.assert_not_called()
        self.window.assert_not_called()
        self.assertIn('API 인증 범위가 다릅니다', self.warning.call_args.args[2])
        self.assertEqual(other.path.read_bytes(), before)
        self.assert_source_preserved()

    def test_sibling_source_requires_confirmation_too(self):
        sibling_root = self.root.parent / 'dockdack-mark_1'
        sibling_root.mkdir()
        with patch('dockdack.runtime_paths.app_home', return_value=sibling_root):
            self.assertEqual(self.launch(), 0)
        self.question.assert_called_once()
        self.assertIn(str(self.source), self.question.call_args.args[2])
        self.assertFalse(self.target.exists())
        self.assert_source_preserved()

    def test_nonexistent_explicit_legacy_source_does_not_open_empty_ledger(self):
        self.assertEqual(self.launch('--legacy-store', str(self.root / 'missing.sqlite3')), 1)
        self.question.assert_not_called()
        self.window.assert_not_called()
        self.assertIn('기존 장부 파일이 없습니다', self.warning.call_args.args[2])
        self.assertFalse(self.target.exists())

    def test_source_and_explicit_destination_options_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.launch('--store=some.sqlite3', '--legacy-store=old.sqlite3')
        self.assertEqual(error.exception.code, 2)
        self.window.assert_not_called()


if __name__ == '__main__':
    unittest.main()
