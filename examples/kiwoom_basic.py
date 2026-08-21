"""Read-only Kiwoom examples. Set the documented environment variables first."""

from dockdack import KiwoomBroker, Market, USExchange


def main() -> None:
    broker = KiwoomBroker.from_env()

    samsung = broker.quote_domestic("005930")
    apple = broker.quote_us("AAPL", exchange=USExchange.NASDAQ)
    print(f"{samsung.name}({samsung.symbol}): {samsung.price} {samsung.currency}")
    print(f"{apple.name}({apple.symbol}): {apple.price} {apple.currency}")

    matches = broker.search_stocks("삼성", market=Market.DOMESTIC, limit=10)
    print("검색 결과:", [(stock.symbol, stock.name) for stock in matches])


if __name__ == "__main__":
    main()
