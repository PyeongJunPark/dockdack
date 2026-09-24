import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples import run_desktop_gui as launcher


class DesktopLauncherTests(unittest.TestCase):
    def test_sibling_ledger_is_only_a_migration_source_not_an_explicit_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dockdack-mark_1"
            sibling = Path(folder) / "dockdack"
            root.mkdir()
            ledger = sibling / ".dockdack/lstm30-demo/watchlist.sqlite3"
            ledger.parent.mkdir(parents=True)
            ledger.touch()
            config = sibling / ".env"
            config.touch()
            args = launcher.desktop_arguments(["--trigger", "mark1-prototype"], root)
            self.assertEqual(args, ["--trigger", "mark1-prototype", "--legacy-store", str(ledger.resolve()),
                                    "--env-file", str(config.resolve())])
            self.assertEqual(list(root.iterdir()), [])

    def test_explicit_paths_and_strategy_kept(self):
        args = ["--store=custom.sqlite3", "--env-file", "custom.env", "--no-model"]
        self.assertEqual(launcher.desktop_arguments(args), args)

    def test_local_store_takes_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dockdack-mark_1"
            local = root / ".dockdack/watchlist.sqlite3"
            sibling = root.parent / "dockdack/.dockdack/lstm30-demo/watchlist.sqlite3"
            for path in (local, sibling):
                path.parent.mkdir(parents=True)
                path.touch()
            self.assertEqual(launcher.desktop_arguments([], root), ["--legacy-store", str(local.resolve())])

    def test_explicit_legacy_source_is_not_replaced(self):
        args = ["--legacy-store=old.sqlite3", "--env-file=custom.env", "--no-model"]
        self.assertEqual(launcher.desktop_arguments(args), args)

    def test_configured_data_home_selects_both_config_and_legacy_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger = root / '.dockdack/watchlist.sqlite3'
            ledger.parent.mkdir()
            ledger.touch()
            (root / '.env').touch()
            with patch.dict('os.environ', {'DOCKDACK_HOME': str(root)}):
                args = launcher.desktop_arguments([])
            self.assertEqual(args, ['--legacy-store', str(ledger.resolve()),
                                    '--env-file', str((root / '.env').resolve())])

    def test_no_cp313_paths_added_to_other_python(self):
        before = list(sys.path)
        with patch.object(launcher.sys, "version_info", (3, 14, 0)):
            launcher.configure_local_dependencies()
        self.assertEqual(sys.path, before)

    def test_normal_vbs_not_separate_prototype_window(self):
        source = (launcher.ROOT / "DockDack.vbs").read_text(encoding="utf-8")
        self.assertIn("-m examples.run_desktop_gui --no-model", source)
        self.assertIn("--external-model mark1-prototype --external-model mark1-1-prototype", source)
        self.assertNotIn("run_mark1_prototype_gui", source)


if __name__ == "__main__":
    unittest.main()
