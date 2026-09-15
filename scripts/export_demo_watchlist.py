"""Read-only: import Korean/US turnover TOP100 and export N daily bars. No orders."""

import argparse
from pathlib import Path
from urllib.parse import urlsplit

from dockdack import KiwoomBroker, KiwoomConfig, Market, TradingMode
from dockdack.autotrade import AutoTrader
from dockdack.gui_service import TradingService
from dockdack.http import RequestsTransport
from dockdack.signal_bridge import export_charts
from dockdack.watchlist import default_store


class ReadOnlyTransport(RequestsTransport):
    def request(self, method, url, **kwargs):
        target = urlsplit(url)
        allowed = {"/oauth2/token", "/api/dostk/rkinfo", "/api/us/rkinfo",
                   "/api/dostk/chart", "/api/us/chart", "/api/dostk/stkinfo",
                   "/api/us/stkinfo", "/api/us/mrkcond", "/api/dostk/mrkcond"}
        if target.scheme != "https" or target.netloc != "mockapi.kiwoom.com" or target.path not in allowed:
            raise ValueError("Read-only demo transport: this endpoint is not allowed")
        return super().request(method, url, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="N for newly added stocks; existing N is preserved")
    parser.add_argument("--output", type=Path, default=Path(".dockdack/exchange/charts.json"))
    args = parser.parse_args()
    if not 1 <= args.days <= 1000:
        parser.error("--days must be 1..1000")
    service = TradingService(lambda market: KiwoomBroker(
        KiwoomConfig.from_env(TradingMode.DEMO, market=market), transport=ReadOnlyTransport()))
    store = default_store()
    rankings = service.top_turnover(Market.DOMESTIC, 100) + service.top_turnover(Market.US, 100)
    store.add_ranked(rankings, days=args.days)
    print("Imported domestic=100 us=100; existing watchlist preserved", flush=True)
    engine, errors = AutoTrader(service, store), {}
    items = store.items()
    for index, item in enumerate(items, 1):
        try:
            engine.snapshot(item)
        except Exception as exc:
            errors[item.id] = str(exc)
            print(f"ERROR {item.id}: {exc}", flush=True)
        if index % 10 == 0 or index == len(items):
            print(f"Charts {index}/{len(items)}; errors={len(errors)}", flush=True)
    payload = export_charts(store, args.output, errors=errors)
    complete = sum(r.get("complete", False) for r in payload["stocks"])
    print(f"Exported {args.output.resolve()}; complete={complete}/{len(items)}; orders_enabled=False", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
