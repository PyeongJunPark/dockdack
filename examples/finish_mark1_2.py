"""One-shot continuation of an already-started Mark1.2 training job.

Wait for that exact local process, then export and backtest. This does not
restart training, alter trading settings, send orders, or schedule recurrence.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

from dockdack.research_artifacts import read_json, sha256_file
from examples.train_mark1_2 import atomic_json, run_lock

ROOT = Path(__file__).resolve().parents[1]


def wait_for_process(pid):
    """Hold a Windows process handle, so a later PID reuse cannot fool us."""
    if sys.platform != "win32" or type(pid) is not int or pid <= 0:
        raise ValueError("A positive local Windows training PID is required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Cannot bind the original training process")
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 30_000)
            if result == 0:
                break
            if result != 258:
                raise OSError(ctypes.get_last_error(), "Training process wait failed")
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise OSError(ctypes.get_last_error(), "Cannot confirm training exit code")
        return exit_code.value
    finally:
        kernel.CloseHandle(handle)


def owned_child(path, parent):
    original = Path(path).absolute()
    resolved, parent = original.resolve(), Path(parent).resolve()
    if resolved == parent or not resolved.is_relative_to(parent):
        raise ValueError("Output must be a dedicated workspace child")
    if any(item.is_symlink() for item in (original, *original.parents) if item != parent):
        raise ValueError("Linked output paths are forbidden")
    return resolved


def publish_report(source, destination):
    """Copy only the finished Markdown result; never rewrite an existing report."""
    source = Path(source)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Finished backtest report is missing")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or sha256_file(source) != sha256_file(destination):
            raise FileExistsError("Preserving an existing different report")
    else:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    return sha256_file(destination)


def completed_report(folder):
    """Do not publish a partial or changed evaluation report."""
    folder = Path(folder)
    receipt = read_json(folder / "completed.json")
    if (receipt.get("completed") is not True
            or receipt.get("markets") != ["domestic", "us"]
            or receipt.get("deployment_allowed") is not False
            or receipt.get("orders_started") is not False):
        raise ValueError("Both research-only backtests must be complete")
    source = folder / "REPORT.md"
    expected = receipt.get("output_sha256", {}).get("REPORT.md")
    if not isinstance(expected, str) or sha256_file(source) != expected:
        raise ValueError("Completed report hash mismatch")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/mark1_2_prototype")
    parser.add_argument("--backtest-dir", type=Path, default=ROOT / "outputs/mark1/mark1-2-backtest-20260924-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/mark1-2-20260924/REPORT.md")
    args = parser.parse_args(argv)
    args.training_run = owned_child(args.training_run, ROOT / "outputs/mark1")
    args.output_dir = owned_child(args.output_dir, ROOT / "outputs/mark1")
    args.bundle = owned_child(args.bundle, ROOT / "models")
    args.backtest_dir = owned_child(args.backtest_dir, ROOT / "outputs/mark1")
    args.report = owned_child(args.report, ROOT / "reports")
    if len({args.training_run, args.output_dir, args.backtest_dir}) != 3:
        parser.error("Training, continuation and backtest directories must differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.output_dir):
        record = dict(training_run=str(args.training_run), training_pid=args.training_pid,
                      bundle=str(args.bundle), backtest_dir=str(args.backtest_dir), report=str(args.report),
                      started_at_utc=datetime.now(timezone.utc).isoformat(), no_accounts_or_orders=True)
        status_path = args.output_dir / "status.json"
        def status(phase, **extra):
            atomic_json(status_path, {**record, "status": phase, **extra})
        def command(module, arguments, log_name):
            with (args.output_dir / log_name).open("w", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, "-u", "-X", "utf8", "-m", module, *arguments],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            if process.returncode:
                raise RuntimeError(f"{module} exited {process.returncode}; see {log_name}")
        try:
            if args.bundle.exists() or args.backtest_dir.exists():
                raise FileExistsError("Preserving existing bundle/backtest; choose unused output paths")
            status("waiting_for_training")
            code = wait_for_process(args.training_pid)
            if code != 0 or read_json(args.training_run / "status.json").get("status") != "completed":
                raise RuntimeError(f"Training did not finish successfully (exit={code})")
            status("exporting")
            command("examples.export_mark1_2", ["--training-run", str(args.training_run),
                    "--output-dir", str(args.bundle)], "export.log")
            status("backtesting")
            command("examples.backtest_mark1_2", ["--training-run", str(args.training_run),
                    "--bundle", str(args.bundle), "--output-dir", str(args.backtest_dir)], "backtest.log")
            report_hash = publish_report(completed_report(args.backtest_dir), args.report)
            status("completed", report_sha256=report_hash, finished_at_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            status("failed", error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
