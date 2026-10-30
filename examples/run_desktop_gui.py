"""Launch the ordinary unified GUI, OFF; no strategy-specific window or account calls."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure_local_dependencies():
    """Reuse this worktree's optional cp313 packages without changing a venv."""
    if sys.version_info[:2] == (3, 13):
        for name in ("selective-deps", "prototype-gui-deps"):
            path = ROOT / "outputs" / "mark1" / name
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))


def desktop_arguments(argv, root=None):
    """Find a legacy source, never mistake it for an account-bound destination."""
    if root is None:
        from dockdack.runtime_paths import app_home
        root = app_home()
    args = list(argv)
    has = lambda option: any(value == option or value.startswith(option + "=") for value in args)
    sibling = root.parent / "dockdack"
    if not has("--store") and not has("--legacy-store"):
        candidates = (root / ".dockdack/lstm30-demo/watchlist.sqlite3", root / ".dockdack/watchlist.sqlite3",
                      sibling / ".dockdack/lstm30-demo/watchlist.sqlite3", sibling / ".dockdack/watchlist.sqlite3")
        path = next((path for path in candidates if path.is_file()), None)
        if path is not None:
            # The GUI must ask about ownership before copying into a scoped
            # destination. --store deliberately bypasses that migration flow.
            args.extend(("--legacy-store", str(path.resolve())))
    if not has("--env-file"):
        path = next((path for path in (root / ".env", sibling / ".env") if path.is_file()), None)
        if path is not None:
            args.extend(("--env-file", str(path.resolve())))
    return args


def main(argv=None):
    configure_local_dependencies()
    args = list(sys.argv[1:] if argv is None else argv)
    from dockdack.v00_app import main as desktop_main
    if args == ["--check"]:
        # Explicitly offline: verify model loading, not credentials or an account.
        from dockdack.mark1_prototype_inference import PrototypePredictor
        from dockdack.mark1_1_prototype_inference import Mark11PrototypePredictor
        from dockdack.signals.mark1_2_trigger import Mark12PrototypePredictor
        from dockdack.mark1_4_inference import Mark14Predictor
        from dockdack.mark1_intraday_inference import MarkIntradayPredictor
        from dockdack.mark1_intraday_models import VARIANTS as INTRADAY_VARIANTS
        from dockdack.mark1_intraday_extra_models import VARIANTS as EXTRA_VARIANTS
        from dockdack.mark1_target_horizon_inference import MarkTargetHorizonPredictor
        from dockdack.signals.mark1_target_horizon_trigger import MODEL_IDS as TARGET_HORIZON_MODEL_IDS
        from dockdack.signals.preopen_series import PREOPEN_MODELS, load_preopen_predictor
        for market in ("domestic", "us"):
            PrototypePredictor(ROOT / "models/mark1_prototype", market)
            Mark11PrototypePredictor(ROOT / "models/mark1_1_prototype", market)
            Mark12PrototypePredictor(ROOT / "models/mark1_2_prototype", market)
            Mark14Predictor(ROOT / "models/mark1_4", market)
            for model_id, spec in PREOPEN_MODELS.items():
                load_preopen_predictor(model_id, ROOT / spec.bundle_directory, market)
            for variant in INTRADAY_VARIANTS:
                MarkIntradayPredictor(ROOT / "models/mark1_intraday", market, variant)
            for variant in EXTRA_VARIANTS:
                MarkIntradayPredictor(ROOT / "models/mark1_intraday_extra", market, variant)
            for model_id in TARGET_HORIZON_MODEL_IDS:
                MarkTargetHorizonPredictor(ROOT / "models/mark1_target_horizon_v1", market, model_id)
        print("Unified GUI + saved KR/US mark1.0-mark1.28 model bundles: OK; no monitoring, orders or network")
        return 0
    return desktop_main(desktop_arguments(args))


if __name__ == "__main__":
    raise SystemExit(main())
