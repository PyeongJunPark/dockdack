"""One-shot continuation safeguards; no training or account calls."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from examples.finish_mark1_2 import completed_report, owned_child, publish_report, wait_for_process
from dockdack.research_artifacts import sha256_file
from examples import finish_mark1_2 as finish


class FinishMark12Tests(unittest.TestCase):
    def test_workspace_parent_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "outputs"
            parent.mkdir()
            self.assertEqual(owned_child(parent / "new-run", parent), parent / "new-run")
            for target in (parent, parent / ".." / "other"):
                with self.assertRaises(ValueError):
                    owned_child(target, parent)

    def test_report_publication_preserves_different_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.md"
            source.write_text("# Verified example\n", encoding="utf-8")
            destination = Path(temporary) / "report" / "REPORT.md"
            digest = publish_report(source, destination)
            self.assertEqual(publish_report(source, destination), digest)
            source.write_text("# Different example\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                publish_report(source, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "# Verified example\n")

    @unittest.skipUnless(sys.platform == "win32", "Windows process handle")
    def test_wait_confirms_the_bound_process_exit_code(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.3); raise SystemExit(7)"],
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            self.assertEqual(wait_for_process(process.pid), 7)
        finally:
            process.wait(timeout=5)

    def test_only_sealed_completed_research_report_can_be_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            report = folder / "REPORT.md"
            report.write_text("# Result", encoding="utf-8")
            receipt = {"completed": True, "markets": ["domestic", "us"],
                       "deployment_allowed": False, "orders_started": False,
                       "output_sha256": {"REPORT.md": sha256_file(report)}}
            def save():
                (folder / "completed.json").write_text(json.dumps(receipt), encoding="utf-8")
            save()
            self.assertEqual(completed_report(folder), report)
            receipt["orders_started"] = True
            save()
            with self.assertRaises(ValueError):
                completed_report(folder)
            receipt["orders_started"] = False
            save()
            report.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                completed_report(folder)

    def test_pipeline_continues_once_only_after_successful_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, state, backtest = (root / "outputs/mark1" / name for name in ("train", "pipeline", "backtest"))
            bundle, published = root / "models/new", root / "reports/new/REPORT.md"
            training.mkdir(parents=True)
            (training / "status.json").write_text('{"status":"completed"}', encoding="utf-8")
            calls = []
            def command(arguments, **kwargs):
                module = arguments[5]
                calls.append(module)
                if module == "examples.export_mark1_2":
                    bundle.mkdir(parents=True)
                else:
                    self.assertEqual(module, "examples.backtest_mark1_2")
                    backtest.mkdir(parents=True)
                    report = backtest / "REPORT.md"
                    report.write_text("# Synthetic finished result", encoding="utf-8")
                    receipt = {"completed": True, "markets": ["domestic", "us"],
                               "deployment_allowed": False, "orders_started": False,
                               "output_sha256": {"REPORT.md": sha256_file(report)}}
                    (backtest / "completed.json").write_text(json.dumps(receipt), encoding="utf-8")
                return SimpleNamespace(returncode=0)
            with patch.object(finish, "ROOT", root), patch.object(finish, "wait_for_process", return_value=0), \
                    patch.object(finish.subprocess, "run", side_effect=command):
                finish.main(["--training-pid", "123", "--training-run", str(training),
                             "--output-dir", str(state), "--bundle", str(bundle),
                             "--backtest-dir", str(backtest), "--report", str(published)])
            self.assertEqual(calls, ["examples.export_mark1_2", "examples.backtest_mark1_2"])
            status = json.loads((state / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["report_sha256"], sha256_file(published))

    def test_training_failure_does_not_run_export_or_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "outputs/mark1/train"
            state = root / "outputs/mark1/pipeline"
            with patch.object(finish, "ROOT", root), patch.object(finish, "wait_for_process", return_value=7), \
                    patch.object(finish.subprocess, "run") as commands:
                with self.assertRaisesRegex(RuntimeError, "exit=7"):
                    finish.main(["--training-pid", "123", "--training-run", str(training),
                                 "--output-dir", str(state), "--bundle", str(root / "models/new"),
                                 "--backtest-dir", str(root / "outputs/mark1/backtest"),
                                 "--report", str(root / "reports/new/REPORT.md")])
                commands.assert_not_called()
            self.assertEqual(json.loads((state / "status.json").read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
