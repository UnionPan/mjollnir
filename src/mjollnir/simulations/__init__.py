"""
Simulation Environments for Reinforcement Learning

Provides gym-compatible environments for:
- Heston stochastic volatility model with multi-dimensional trading
- Option hedging and trading on fixed grid
- Portfolio management under partial observability (POMDP)

Features:
- Professional processes.Heston simulation (Milstein/Euler)
- Multi-dimensional action space (underlying + options)
- Fixed grid representation (moneyness × TTM)
- Synthetic equity option chains with realistic pricing
- T=246 steps, dt=1/252 (~1 trading year)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from mjollnir._lazy import lazy_exports

# The environments, their cache and the renderer need the [envs] extra
# (gymnasium, zarr, matplotlib); each loads with its first use.
__getattr__, __dir__, _LAZY = lazy_exports(__name__, globals(), {
    "HestonEnv": ".heston_env",
    "HestonParams": ".heston_env",
    "Liability": ".heston_env",
    "make_heston_env": ".heston_env",
    "MertonEnv": ".merton_env",
    "MertonParams": ".merton_env",
    "RoughBergomiEnv": ".rough_bergomi_env",
    "RoughBergomiParams": ".rough_bergomi_env",
    "CachedHestonEnv": ".heston_cache",
    "build_heston_cache": ".heston_cache",
    "HestonEnvRenderer": ".renderer",
})


def make_heston_trading_env(**kwargs):
    """Convenience wrapper: Heston trading env without options."""
    from .heston_env import HestonEnv

    return HestonEnv(task='trading', include_options=False, **kwargs)


def make_heston_hedging_env(**kwargs):
    """Convenience wrapper: Heston hedging env with options."""
    from .heston_env import HestonEnv

    return HestonEnv(task='hedging', include_options=True, **kwargs)


def make_heston_no_options_env(**kwargs):
    """Convenience wrapper: Heston env without options."""
    from .heston_env import HestonEnv

    return HestonEnv(include_options=False, **kwargs)

__all__ = [*_LAZY, 'make_heston_hedging_env', 'make_heston_no_options_env', 'make_heston_trading_env']
