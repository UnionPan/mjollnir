"""Option books and classical hedgers for counterfactual hedging backtests.

A hedging experiment fixes a *book* from one calibration: a short European
option, the premium received for it, and every hedger's assumptions. The book
is then run through a *world*: the same calibration, or a counterfactual
:class:`~mjollnir.scenarios.Scenario` applied to it. Keeping the book fixed
while the world changes is what makes the comparison counterfactual: the
trader sold and set up the hedge before learning which world arrived.

Hedgers are :class:`~mjollnir.backtest.ParametricStrategy` values for
:func:`mjollnir.backtest.run_backtest`: module-level functions plus numeric
parameters, so a new strike, vol or calibration reuses the compiled backtest.
Positions hedge a *short* option, so a hedger holds the option's delta.

Rates are zero throughout: ``run_backtest`` pays no interest on cash, so books
are priced at ``r = 0`` for consistency.

Example::

    from mjollnir.hedging import EuropeanOption, compare_hedgers
    from mjollnir.scenarios import vol_spike

    call = EuropeanOption(strike=100.0, maturity=21 / 252)
    calm = compare_hedgers(book, call, n_paths=4096, seed=7)
    shocked = compare_hedgers(book, call, n_paths=4096, seed=7, scenario=vol_spike(2.0))
    print(calm["bs_delta"].pnl.std(), shocked["bs_delta"].pnl.std())
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from mjollnir.backtest import BacktestResult, ParametricStrategy, run_backtest
from mjollnir.jax import fourier_price, heston_cos_delta
from mjollnir.params import ParamSet
from mjollnir.pricer._jax_iv import implied_vol_batch_np
from mjollnir.scenarios import Scenario

__all__ = [
    "EuropeanOption",
    "Strategy",
    "bs_delta",
    "compare_hedgers",
    "heston_delta",
    "heston_price",
    "implied_vol",
    "no_hedge",
    "standard_hedgers",
]

Strategy = Callable[[Any, Any, Any, Any], Any]

# Widest COS window for hedge deltas, as a multiple of max(theta, v0): covers worlds whose
# variance rises up to this factor (e.g. a 2x vol spike) at the full truncation width.
_DELTA_WINDOW_VARIANCE_MULTIPLE = 4.0


@dataclass(frozen=True)
class EuropeanOption:
    """A European vanilla option, owed by the book at maturity."""

    strike: float
    maturity: float  # years
    is_call: bool = True

    def payoff(self, spot):
        """Payoff at maturity; jax- and numpy-traceable."""
        intrinsic = spot - self.strike if self.is_call else self.strike - spot
        return jnp.maximum(intrinsic, 0.0)

    def steps(self, dt: float) -> int:
        """Number of ``dt`` steps to maturity; ``maturity`` must be a multiple of ``dt``."""
        n = round(self.maturity / dt)
        if n < 1 or not math.isclose(n * dt, self.maturity, rel_tol=1e-9):
            raise ValueError(f"maturity {self.maturity} is not a positive multiple of dt={dt}")
        return n


def heston_price(option: EuropeanOption, params: ParamSet, spot: float,
                 variance: float | None = None) -> float:
    """Heston COS price of ``option`` under ``params`` (``v0`` unless ``variance`` is given)."""
    p = params.params
    v = p["v0"] if variance is None else variance
    return float(fourier_price(spot, option.strike, option.maturity, 0.0, 0.0, v,
                               p["kappa"], p["theta"], p["sigma_v"], p["rho"], option.is_call))


def implied_vol(option: EuropeanOption, price: float, spot: float) -> float:
    """Black-Scholes implied vol of ``price``; raises if the price is not invertible."""
    iv = implied_vol_batch_np(
        S=spot, K=np.array([option.strike]), T=option.maturity, r=0.0, q=0.0,
        market_prices=np.array([price]), is_call=np.array([option.is_call]),
    )[0]
    if not np.isfinite(iv):
        raise ValueError(f"price {price} of {option} at spot {spot} has no implied vol")
    return float(iv)


def _option_params(option: EuropeanOption, dt: float) -> dict[str, Any]:
    return {"strike": option.strike, "maturity": option.maturity,
            "is_call": option.is_call, "dt": dt}


def _no_hedge(p, spot, variance, position, step):
    return jnp.zeros_like(spot)


def _bs_delta(p, spot, variance, position, step):
    tau = p["maturity"] - step * p["dt"]
    sigma = p["sigma"]
    d1 = (jnp.log(spot / p["strike"]) + 0.5 * sigma**2 * tau) / (sigma * jnp.sqrt(tau))
    delta = norm.cdf(d1)
    return jnp.where(p["is_call"], delta, delta - 1.0)


def _heston_delta_scalar(spot, variance, tau, strike, kappa, theta, sigma_v, rho, is_call, v_top):
    return heston_cos_delta(spot, variance, tau, strike, 0.0, 0.0, kappa, theta, sigma_v, rho,
                            is_call, v_top)


_heston_spot_delta = jnp.vectorize(_heston_delta_scalar)


def _heston_delta(p, spot, variance, position, step):
    tau = p["maturity"] - step * p["dt"]
    return _heston_spot_delta(spot, variance, tau, p["strike"], p["kappa"], p["theta"],
                              p["sigma_v"], p["rho"], p["is_call"], p["v_top"])


def no_hedge() -> ParametricStrategy:
    """Hold nothing: the book's P&L is the premium minus the payoff."""
    return ParametricStrategy(_no_hedge, ())


def bs_delta(option: EuropeanOption, sigma: float, dt: float) -> ParametricStrategy:
    """Black-Scholes delta at a fixed volatility ``sigma``."""
    return ParametricStrategy(_bs_delta, {**_option_params(option, dt), "sigma": float(sigma)})


def heston_delta(option: EuropeanOption, params: ParamSet, dt: float) -> ParametricStrategy:
    """Heston delta ``dC/dS`` from ``params``' dynamics at the path's current variance.

    The hedger observes the simulated variance, so this is the model-consistent
    benchmark for a desk that tracks volatility, not a filter-based hedger. Deltas come
    from :func:`mjollnir.jax.heston_cos_delta`, which shares the characteristic-function
    work across paths.
    """
    dynamics = {k: float(params.params[k]) for k in ("kappa", "theta", "sigma_v", "rho")}
    v_top = _DELTA_WINDOW_VARIANCE_MULTIPLE * max(dynamics["theta"], float(params.params["v0"]))
    return ParametricStrategy(_heston_delta, {**_option_params(option, dt), **dynamics,
                                              "v_top": v_top})


def standard_hedgers(option: EuropeanOption, book: ParamSet, *, dt: float,
                     spot0: float = 100.0) -> dict[str, Strategy]:
    """No hedge, Black-Scholes delta at the premium's implied vol, and Heston delta."""
    sigma = implied_vol(option, heston_price(option, book, spot0), spot0)
    return {
        "no_hedge": no_hedge(),
        "bs_delta": bs_delta(option, sigma, dt),
        "heston_delta": heston_delta(option, book, dt),
    }


def compare_hedgers(
    book: ParamSet,
    option: EuropeanOption,
    *,
    hedgers: dict[str, Strategy] | None = None,
    scenario: Scenario | None = None,
    n_paths: int = 4096,
    dt: float = 1.0 / 252,
    spot0: float = 100.0,
    mu: float | None = None,
    cost_bps: float = 0.0,
    seed: int = 0,
) -> dict[str, BacktestResult]:
    """Run a short-option book through one world for several hedgers.

    The premium (Heston price under ``book``) and the hedgers come from the
    book; ``scenario``, if given, changes only the simulated world. Every
    hedger sees the same paths (common random numbers), so differences between
    results are differences between strategies.

    Args:
        book: Heston ParamSet the book was priced and hedged with.
        option: the option sold.
        hedgers: name -> strategy; defaults to :func:`standard_hedgers`.
        scenario: counterfactual applied to the simulated world only.
        mu: simulation drift; defaults to the book's ``mu`` (0 if absent).
    """
    premium = heston_price(option, book, spot0)
    if hedgers is None:
        hedgers = standard_hedgers(option, book, dt=dt, spot0=spot0)
    drift = float(book.params.get("mu", 0.0)) if mu is None else mu
    return {
        name: run_backtest(
            book, strategy, n_paths=n_paths, horizon_steps=option.steps(dt), dt=dt,
            mu=drift, spot0=spot0, initial_cash=premium, cost_bps=cost_bps,
            scenario=scenario, liability=option.payoff, seed=seed,
        )
        for name, strategy in hedgers.items()
    }
