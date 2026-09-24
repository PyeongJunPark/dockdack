"""Launch mark1 prototype with saved models; monitoring OFF, orders prohibited."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure_local_dependencies():
    """Reuse isolated local cp313 packages, without modifying either broker venv."""
    if sys.version_info[:2] == (3, 13):
        for name in ("selective-deps", "prototype-gui-deps"):
            path = ROOT / "outputs" / "mark1" / name
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/mark1_prototype")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".dockdack/mark1-prototype")
    parser.add_argument("--env-file", type=Path, help="Optional local API configuration; never copied")
    parser.add_argument("--symbol", action="append", help="MARKET:EXCHANGE:SYMBOL, repeatable")
    parser.add_argument("--top-domestic100", action="store_true", help="Select KR TOP100 when monitoring is manually started")
    parser.add_argument("--top-us100", action="store_true", help="Select US TOP100 when monitoring is manually started")
    parser.add_argument("--check", action="store_true", help="Offline bundle/import check; no window, account or network")
    return parser


def config_file(explicit=None):
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise ValueError("The specified local configuration file does not exist")
        return path
    # This worktree deliberately has no secret file. Read the user's existing
    # checkout configuration only in this process, never copy or print it.
    for path in (ROOT / ".env", ROOT.parent / "dockdack" / ".env"):
        if path.is_file():
            return path
    return None


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    configure_local_dependencies()
    from PySide6.QtWidgets import QApplication, QMessageBox
    from dockdack.gui_service import Instrument, TradingService
    from dockdack.lstm30_gui import DEFAULT_ITEMS
    from dockdack.mark1_prototype_gui import Mark1PrototypeWatchlistDialog
    from dockdack.mark1_prototype_inference import PrototypePredictor
    from dockdack.models import Market, TradingMode
    from dockdack.watchlist import WatchItem

    items = None
    if args.symbol:
        try:
            items = [WatchItem(Instrument(Market(market), symbol, exchange), days=31)
                     for market, exchange, symbol in (value.split(":") for value in args.symbol)]
            if any(not item.instrument.symbol.strip() or not item.instrument.exchange.strip() for item in items):
                raise ValueError("Empty symbol or exchange")
        except (ValueError, TypeError) as exc:
            parser.error(f"Invalid --symbol: {exc}")
    ranked = ({Market.DOMESTIC} if args.top_domestic100 else set()) | ({Market.US} if args.top_us100 else set())
    markets = {item.instrument.market.value for item in (items or DEFAULT_ITEMS)} | {market.value for market in ranked}
    bundle = args.bundle.resolve()
    if args.check:
        predictors = {market: PrototypePredictor(bundle, market) for market in sorted(markets)}
        summary_keys = ("model_name", "version", "bundle_manifest_sha256", "feature_count", "seeds",
                        "buy_threshold", "take_profit_pct", "stop_loss_pct", "research_only",
                        "research_qualified", "deployment_allowed", "known_data_quality_issues")
        print(json.dumps({"title": "mark1 prototype", "bundle": str(bundle), "models": {
            market: {key: predictor.metadata[key] for key in summary_keys} for market, predictor in predictors.items()},
            "monitoring": False, "orders_permitted": False, "network_used": False}, ensure_ascii=False, indent=2))
        return 0

    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        path = config_file(args.env_file)
        if path is not None:
            from dotenv import load_dotenv
            load_dotenv(path, override=False)
        predictors = {market: PrototypePredictor(bundle, market) for market in sorted(markets)}
        window = Mark1PrototypeWatchlistDialog(
            TradingService(mode=TradingMode.DEMO), runtime_dir=args.runtime_dir.resolve(),
            predictors=predictors, items=items, ranked_markets=ranked,
            checkpoint_paths={market: bundle / "manifest.json" for market in markets},
        )
    except Exception as exc:
        # Keep arbitrary broker/config exception details out of the visible log.
        QMessageBox.critical(None, "mark1 prototype 시작 실패",
                             "모델·실행 환경·전용 실행 폴더를 확인하세요. 주문은 시작되지 않았습니다.\n"
                             f"오류 종류: {type(exc).__name__}\n"
                             "문서: docs/MARK1_PROTOTYPE.md")
        return 1
    previous = {}
    try:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous[sig] = signal.signal(sig, lambda *_: window.stop_monitoring())
        window.workspace_tabs.setCurrentIndex(window.workspace_tabs.count() - 1)
        window.resize(1500, 950)
        from dockdack.lstm30_adapter import atomic_json
        from dockdack.watchlist import utc_now
        atomic_json(window.runtime_dir / "prototype-startup.json", {
            "record_type": "startup_snapshot_not_live_status", "created_at": utc_now().isoformat(),
            "pid": os.getpid(), "title": window.windowTitle(), "gui_base_commit": "b4b925c",
            "monitoring": window.monitoring, "orders_permitted": window.engine.orders_enabled,
            "mode": window.service.mode.value, "source_id": window.source_id,
            "bundle_manifest_sha256": {market: predictor.metadata["bundle_manifest_sha256"]
                                       for market, predictor in predictors.items()},
        })
        window.show()
        # Deliberately no start_session / arm / close-liquidation invocation.
        return app.exec()
    finally:
        window._close_when_idle = True
        window._pending_environment = None
        window.stop_monitoring()
        window.pool.waitForDone()
        window.inspection_pool.waitForDone()
        window.activity_pool.waitForDone()
        app.processEvents()
        window.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
