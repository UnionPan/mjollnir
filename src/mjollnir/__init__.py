"""
mjollnir — a stable, reproducible, differentiable market-simulation & calibration
substrate for RL research (deep hedging) and backtesting.

Two consumption modes:

* ``mjollnir.jax``          — the JAX simulation kernel (Heston QE step + differentiable
                              Fourier/COS pricer + runtime config). The frozen public API
                              that RL environments pin against.
* ``mjollnir.calibration``  — calibration objects and CLIs that source model parameters
                              from market data (physical- and risk-neutral-measure).

Extracted from an internal research monorepo.
"""

__version__ = "0.1.0"
