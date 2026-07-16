"""Shared value objects for the hedging environments.

Extracted from ``heston_env`` so that consumers (rollout code, caches,
liability specs) can import the parameter and liability types without
importing a gym environment. ``heston_env`` re-exports both names, so
existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HestonParams", "Liability"]


@dataclass
class HestonParams:
    """
    Heston model parameters.

    Default parameters calibrated for realistic volatility smile visible even
    in short-dated options (7-30 days):
    - Fast mean reversion (kappa=6.0)
    - High vol-of-vol (xi=1.0) - creates visible smile in short maturities
    - Strong negative correlation (rho=-0.8) - creates equity-style left skew

    These parameters satisfy Feller condition: 2κθ = 1.08 > ξ² = 1.0
    and create typical equity/crypto volatility patterns:
    - OTM puts: Higher IV (crash protection premium)
    - ATM: Peak IV
    - OTM calls: Lower IV (smile/smirk pattern)

    Note: 30% vol represents volatile assets (crypto, meme stocks).
    For standard equities, use v_0=theta=0.04 (20% vol), xi=0.5, kappa=4.0.
    """
    S_0: float = 1.0        # Initial stock price
    v_0: float = 0.09       # Initial variance (σ₀ = 30%)
    mu: float = 0.0         # Drift (risk-neutral: mu=0)
    kappa: float = 6.0      # Mean reversion speed (fast)
    theta: float = 0.09     # Long-run variance (σ_LR = 30%)
    xi: float = 1.0         # Vol-of-vol (high for visible short-dated smile)
    rho: float = -0.8       # Correlation (strong negative = equity-style skew)


@dataclass
class Liability:
    """
    Liability to be hedged.

    For hedging task, agent is short this liability and must hedge it
    using underlying + options from the grid.
    """
    option_type: str        # 'call' or 'put'
    strike: float          # Strike price (can be relative to S_0)
    maturity_days: int     # Days to maturity
    quantity: float = -1.0  # Negative = short position

    # MTM tracking
    initial_price: float = None
    current_price: float = None
    payoff: float = None
