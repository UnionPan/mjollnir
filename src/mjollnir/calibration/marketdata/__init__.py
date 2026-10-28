"""Data fetching and cleaning utilities.

The data types load eagerly; each fetcher loads with its first use, so this
package imports without the [data] extra (yfinance, requests) that the fetchers
need.
"""

from mjollnir._lazy import lazy_exports

from .data_provider import DataProvider, MarketData, OptionChain, OptionQuote

__getattr__, __dir__, _LAZY = lazy_exports(__name__, globals(), {
    "YFinanceFetcher": ".yfinance_fetcher",
    "fetch_spy_data": ".yfinance_fetcher",
    "CoinGeckoFetcher": ".coingecko_fetcher",
    "BinanceFetcher": ".binance_fetcher",
    "GeminiFetcher": ".gemini_fetcher",
    "KrakenFetcher": ".kraken_fetcher",
    "CryptoFetcher": ".crypto_fetcher",
    "download_bitcoin": ".crypto_fetcher",
    "download_crypto_basket": ".crypto_fetcher",
    "SyntheticOptionChainGenerator": ".synthetic_crypto",
    "RegimeVolatilityProfile": ".synthetic_crypto",
    "quick_generate_option_chains": ".synthetic_crypto",
    # synthetic_equity / synthetic_merton_equity live in the low-level
    # mjollnir.synthetic_data layer; re-exported here for the historical import path.
    "SyntheticEquityOptionChainGenerator": "mjollnir.synthetic_data",
    "HestonVolatilityProfile": "mjollnir.synthetic_data",
    "get_default_moneyness_by_maturity": "mjollnir.synthetic_data",
    "SyntheticMertonOptionChainGenerator": "mjollnir.synthetic_data",
    "MertonVolatilityProfile": "mjollnir.synthetic_data",
})

__all__ = ["DataProvider", "MarketData", "OptionChain", "OptionQuote", *_LAZY]
