"""Offline training for separately sealed MK1.18–MK1.22 daily-proxy models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dockdack.mark1_intraday_extra_models import (
    ARCHITECTURES, EFFECTIVE_LOOKBACK, VARIANTS, build_model, feature_matrix,
)
from examples.train_mark1_intraday import ROOT, _write_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models" / "mark1_intraday_extra")
    args = parser.parse_args(argv)
    manifest = _write_bundle(
        args.workspace.resolve(), args.output.resolve(),
        variants=VARIANTS, architectures=ARCHITECTURES,
        lookbacks=EFFECTIVE_LOOKBACK, feature_fn=feature_matrix,
        model_builder=build_model,
        feature_code="dockdack/mark1_intraday_extra_models.py",
        owner="dockdack.mark1_intraday_extra",
    )
    print(json.dumps({"completed": True, "output": str(args.output.resolve()),
                      "variants": list(manifest["variants"]),
                      "markets": list(manifest["markets"]),
                      "intraday_path_verified": False,
                      "profitability_validated": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
