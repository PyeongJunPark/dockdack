"""Compatibility entry point for a one-model external GUI preview.

The implementation is shared with the simultaneous external-model proof. It
uses a temporary ledger, fake flat account and read-only historical databases;
the obsolete built-in predictor injection path is no longer used.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from dockdack.local_data_paths import default_clean_database_dir
from examples.run_desktop_gui import ROOT


def main(argv=None):
    from examples.preview_mark1_dual_gui import main as external_preview
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'models/mark1_prototype')
    parser.add_argument('--database-dir', type=Path,
                        default=default_clean_database_dir(ROOT))
    parser.add_argument('--day', type=date.fromisoformat, default=date(2024, 7, 15))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/mark1/normal-external-gui-preview.png')
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != '.png':
        parser.error('--output must be a PNG file')
    return external_preview(['--model', 'mark1-prototype', '--mark1-bundle', str(args.bundle),
                             '--database-dir', str(args.database_dir), '--day', args.day.isoformat(),
                             '--output-prefix', str(args.output.with_suffix(''))])


if __name__ == '__main__':
    raise SystemExit(main())
