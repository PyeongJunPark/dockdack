import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.run_mark1_prototype_gui import config_file, parser_for_cli


class PrototypeLauncherTests(unittest.TestCase):
    def test_defaults_cannot_arm_or_start_monitoring(self):
        args = parser_for_cli().parse_args([])
        self.assertEqual(args.bundle.name, "mark1_prototype")
        self.assertEqual(args.runtime_dir.name, "mark1-prototype")
        self.assertFalse(args.top_domestic100)
        self.assertFalse(args.top_us100)
        self.assertFalse(hasattr(args, "arm"))
        self.assertFalse(hasattr(args, "start"))

    def test_unknown_arming_option_rejected(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser_for_cli().parse_args(["--arm"])

    def test_explicit_missing_config_rejected(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            config_file(Path(folder) / "missing.env")

    def test_config_selection_does_not_read_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with patch("examples.run_mark1_prototype_gui.ROOT", path), patch.object(Path, "is_file", return_value=True):
                self.assertEqual(config_file(), path / ".env")


if __name__ == "__main__":
    unittest.main()
