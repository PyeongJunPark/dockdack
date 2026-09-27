"""Sealed model artifacts must survive Windows and Unix Git checkouts byte-for-byte."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUNDLES = (
    "mark1_4", "mark1_series", "mark1_8", "mark1_9", "mark1_10",
    "mark1_horizons",
)
FROZEN_SCORERS = (
    "dockdack/mark1_4_evolution.py",
    "dockdack/mark1_4_followup_models.py",
    "dockdack/mark1_4_sparse.py",
    "dockdack/mark1_series_models.py",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(shutil.which("git"), "Git checkout behavior requires Git")
class SealedCheckoutBytesTests(unittest.TestCase):
    def test_sealed_models_in_both_checkout_modes(self):
        bundle_paths = [f"models/{name}" for name in BUNDLES]
        tracked = subprocess.run(
            ["git", "ls-files", "--", *bundle_paths, *FROZEN_SCORERS],
            cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.splitlines()
        # Binary checkpoints are not affected by text conversion. Cover the
        # JSON and scorer bytes whose SHA-256 values are checked at runtime.
        sealed_paths = [path for path in tracked if path.endswith((".json", ".py"))]
        self.assertGreater(len(sealed_paths), 15)

        for bundle in BUNDLES:
            base = ROOT / "models" / bundle
            self.assertEqual(
                _digest(base / "manifest.json"),
                (base / "manifest.sha256").read_text(encoding="ascii").strip(),
                bundle,
            )
            manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
            code_hashes = (manifest.get("research_code_sha256")
                           or manifest.get("scoring_code_sha256") or {})
            for name, expected in code_hashes.items():
                self.assertEqual(_digest(ROOT / "dockdack" / name), expected, name)

        for autocrlf in ("false", "true"):
            with self.subTest(core_autocrlf=autocrlf), tempfile.TemporaryDirectory(
                prefix="dockdack-sealed-checkout-"
            ) as directory:
                target = Path(directory)
                subprocess.run(
                    ["git", "-c", f"core.autocrlf={autocrlf}", "checkout-index",
                     f"--prefix={target.as_posix()}/", "--", *sealed_paths],
                    cwd=ROOT, capture_output=True, text=True, check=True,
                )
                for relative in sealed_paths:
                    self.assertEqual(
                        (target / relative).read_bytes(), (ROOT / relative).read_bytes(),
                        f"{relative} changed with core.autocrlf={autocrlf}",
                    )


if __name__ == "__main__":
    unittest.main()
