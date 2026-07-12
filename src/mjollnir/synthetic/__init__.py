"""
mjollnir.synthetic — shared low-level substrate for option-chain data.

Top-level in the namespace, low in the dependency DAG: depends only on
``mjollnir.pricer``, and is imported by *both* ``mjollnir.simulations`` (to build
the option chains an RL env prices and hedges) and ``mjollnir.calibration`` (for
synthetic evaluation data). It holds the shared option-chain value types and the
counterfactual chain generators. Keeping it here — inside the versioned substrate —
is what keeps env dynamics reproducible: the chains the agent sees are frozen with
the package version, not produced by a loosely-versioned outside tool.
"""

from .data_provider import DataProvider, MarketData, OptionChain, OptionQuote
from .synthetic_equity import (
    SyntheticEquityOptionChainGenerator,
    HestonVolatilityProfile,
    get_default_moneyness_by_maturity,
)
from .synthetic_merton_equity import (
    SyntheticMertonOptionChainGenerator,
    MertonVolatilityProfile,
)

__all__ = [
    # shared value types
    "OptionChain",
    "OptionQuote",
    "MarketData",
    "DataProvider",
    # counterfactual generators
    "SyntheticEquityOptionChainGenerator",
    "HestonVolatilityProfile",
    "get_default_moneyness_by_maturity",
    "SyntheticMertonOptionChainGenerator",
    "MertonVolatilityProfile",
]
