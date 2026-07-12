"""
mjollnir — counterfactual market simulation and backtesting: differentiable,
reproducible market models enabling RL research, with market-model calibration
pipelines.

Two consumption modes:

* ``mjollnir.jax``          — the JAX simulation kernel (Heston QE step + differentiable
                              Fourier/COS pricer + runtime config). The frozen public API
                              that RL environments pin against.
* ``mjollnir.calibration``  — calibration objects and CLIs that source model parameters
                              from market data (physical- and risk-neutral-measure).
"""

__version__ = "0.1.0"
