"""Data fetching and cleaning utilities."""

from .data_provider import DataProvider, MarketData, OptionChain, OptionQuote
from .yfinance_fetcher import YFinanceFetcher, fetch_spy_data
from .coingecko_fetcher import CoinGeckoFetcher
from .binance_fetcher import BinanceFetcher
from .kraken_fetcher import KrakenFetcher
from .crypto_fetcher import CryptoFetcher, download_bitcoin, download_crypto_basket
from .synthetic_crypto import (
    SyntheticOptionChainGenerator,
    RegimeVolatilityProfile,
    quick_generate_option_chains,
)
# synthetic_equity / synthetic_merton_equity now live in the low-level
# mjollnir.synthetic layer; re-exported here for the historical import path.
from mjollnir.synthetic import (
    SyntheticEquityOptionChainGenerator,
    HestonVolatilityProfile,
    get_default_moneyness_by_maturity,
    SyntheticMertonOptionChainGenerator,
    MertonVolatilityProfile,
)

__all__ = [
    'DataProvider',
    'MarketData',
    'OptionChain',
    'OptionQuote',
    'YFinanceFetcher',
    'fetch_spy_data',
    'CoinGeckoFetcher',
    'BinanceFetcher',
    'KrakenFetcher',
    'CryptoFetcher',
    'download_bitcoin',
    'download_crypto_basket',
    # Synthetic option generation
    'SyntheticOptionChainGenerator',
    'RegimeVolatilityProfile',
    'quick_generate_option_chains',
    'SyntheticEquityOptionChainGenerator',
    'HestonVolatilityProfile',
    'get_default_moneyness_by_maturity',
    'SyntheticMertonOptionChainGenerator',
    'MertonVolatilityProfile',
]
