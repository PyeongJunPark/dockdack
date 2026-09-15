"""Copy existing saved charts to the durable history cache; no network or orders."""

from dockdack.history_cache import HistoryCache
from dockdack.watchlist import default_store, utc_now


def main():
    store = default_store()
    cache = HistoryCache(store, None, utc_now)
    count = bars = 0
    for item in store.items():
        saved = cache.load(item)
        if saved:
            history, fetched_at = saved
            cache.save(item, history, fetched_at)
            count += 1
            bars += len(history.bars)
    print(f"Saved history cache: stocks={count}, bars={bars}; network_requests=0; orders=0")


if __name__ == "__main__":
    main()
