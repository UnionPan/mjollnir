"""
Derivatives module - comprehensive derivative contracts

author: Yunian Pan
email: yp1170@nyu.edu
"""

# Base classes
from .base import Derivative, PathIndependentDerivative, PathDependentDerivative

# Vanilla derivatives
from .vanilla import (
    EuropeanCall,
    EuropeanPut,
    DigitalCall,
    DigitalPut,
    Straddle,
    Strangle,
    ButterflySpread,
)

# American derivatives
from .american import (
    AmericanOption,
    AmericanCall,
    AmericanPut,
    PerpetualAmericanOption,
    BermudanOption,
)

# Path-dependent derivatives
from .path_dependent import (
    AsianOption,
    BarrierOption,
    LookbackOption,
    CliquetOption,
)

# Multi-asset derivatives
from .multi_asset import (
    BasketOption,
    SpreadOption,
    RainbowOption,
    ExchangeOption,
    QuantoOption,
)

# Interest rate derivatives
from .rates import (
    Caplet,
    Floorlet,
    InterestRateSwap,
    Swaption,
    YieldCurveOption,
    BondOption,
)

__all__ = [
    "AmericanCall",
    # American
    "AmericanOption",
    "AmericanPut",
    # Path-dependent
    "AsianOption",
    "BarrierOption",
    # Multi-asset
    "BasketOption",
    "BermudanOption",
    "BondOption",
    "ButterflySpread",
    # Rates
    "Caplet",
    "CliquetOption",
    # Base
    "Derivative",
    "DigitalCall",
    "DigitalPut",
    # Vanilla
    "EuropeanCall",
    "EuropeanPut",
    "ExchangeOption",
    "Floorlet",
    "InterestRateSwap",
    "LookbackOption",
    "PathDependentDerivative",
    "PathIndependentDerivative",
    "PerpetualAmericanOption",
    "QuantoOption",
    "RainbowOption",
    "SpreadOption",
    "Straddle",
    "Strangle",
    "Swaption",
    "YieldCurveOption",
]
