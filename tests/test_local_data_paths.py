"""Dataset default selection never depends on a removed sibling checkout."""
import argparse
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dockdack.local_data_paths import CLEAN_DAILY_RELATIVE, default_clean_database_dir


class LocalDataPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "dockdack-mark_1"
        self.local = self.root / CLEAN_DAILY_RELATIVE
        self.canonical = self.base / "dockdack" / CLEAN_DAILY_RELATIVE
        self.legacy = self.base / "dockdack-data-collection" / CLEAN_DAILY_RELATIVE

    def test_existing_local_dataset_wins_over_both_siblings(self):
        for path in (self.local, self.canonical, self.legacy):
            path.mkdir(parents=True)
        self.assertEqual(default_clean_database_dir(self.root), self.local)

    def test_canonical_sibling_wins_over_legacy_data_checkout(self):
        self.canonical.mkdir(parents=True)
        self.legacy.mkdir(parents=True)
        self.assertEqual(default_clean_database_dir(self.root), self.canonical)

    def test_legacy_data_checkout_remains_compatible_when_needed(self):
        self.legacy.mkdir(parents=True)
        self.assertEqual(default_clean_database_dir(self.root), self.legacy)

    def test_no_dataset_returns_local_default_without_creating_anything(self):
        self.assertEqual(default_clean_database_dir(self.root), self.local)
        self.assertFalse(self.root.exists())
        self.assertEqual(list(self.base.iterdir()), [])

    def test_consolidated_checkout_works_with_no_other_directory(self):
        self.canonical.mkdir(parents=True)
        self.assertEqual(default_clean_database_dir(self.base / "dockdack"), self.canonical)
        self.assertFalse(self.legacy.exists())

    def test_file_in_place_of_directory_is_not_a_dataset(self):
        self.local.parent.mkdir(parents=True)
        self.local.touch()
        self.legacy.mkdir(parents=True)
        self.assertEqual(default_clean_database_dir(self.root), self.legacy)

    def test_preview_default_and_explicit_cli_path_use_one_shared_selection(self):
        self.local.mkdir(parents=True)
        explicit = self.base / "explicit-input"
        for name in ("preview_mark1_dual_gui", "preview_mark1_normal_gui", "preview_mark1_prototype_gui"):
            module = importlib.import_module("examples." + name)
            for argv, expected in (([], self.local), (["--database-dir", str(explicit)], explicit)):
                with self.subTest(module=name, explicit=bool(argv)):
                    parsed = []
                    original = argparse.ArgumentParser.parse_args
                    class ParsedOnly(Exception):
                        pass
                    def parse_only(parser, values):
                        parsed.append(original(parser, values))
                        raise ParsedOnly
                    with patch.object(module, "ROOT", self.root), patch.object(argparse.ArgumentParser, "parse_args", parse_only):
                        with self.assertRaises(ParsedOnly):
                            module.main(argv)
                    self.assertEqual(parsed[0].database_dir, expected)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Install the research dependencies")
    def test_us_audit_default_and_explicit_file_are_preserved(self):
        module = importlib.import_module("examples.audit_mark1_us_data")
        self.local.mkdir(parents=True)
        explicit = self.base / "another-us.sqlite3"
        for argv, expected in (([], self.local / "us_daily_clean.sqlite3"), (["--clean", str(explicit)], explicit)):
            parsed = []
            original = argparse.ArgumentParser.parse_args
            class ParsedOnly(Exception):
                pass
            def parse_only(parser, values):
                parsed.append(original(parser, values))
                raise ParsedOnly
            with patch.object(module, "ROOT", self.root), patch.object(argparse.ArgumentParser, "parse_args", parse_only):
                with self.assertRaises(ParsedOnly):
                    module.main(argv)
            self.assertEqual(parsed[0].clean, expected)


if __name__ == "__main__":
    unittest.main()
