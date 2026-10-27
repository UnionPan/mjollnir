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

Every world in :mod:`mjollnir.worlds` has a model price and a model hedger
(:func:`model_price`, :func:`model_delta`): exact for GBM, Heston, Merton and
regime switching; for GARCH and rough Bergomi the price is Monte Carlo in the
model's own world and the hedge a labelled approximation (Black-Scholes delta
at the model's forecast or current variance). Rates and dividends are carried
consistently: the book is priced and hedged at ``rate``/``dividend``, cash
earns ``rate`` in the backtest, held shares earn ``dividend``, and the
simulated spot drifts at ``rate - dividend`` unless told otherwise.

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

from mjollnir import model_pricing
from mjollnir.backtest import BacktestResult, ParametricStrategy, run_backtest, run_backtests
from mjollnir.jax import fourier_price, heston_cos_delta
from mjollnir.params import ParamSet
from mjollnir.pricer._jax_iv import implied_vol_batch_np
from mjollnir.scenarios import Scenario
from mjollnir.worlds import market_params

__all__ = [
    "MODEL_HEDGERS",
    "EuropeanOption",
    "Strategy",
    "bs_delta",
    "compare_hedgers",
    "heston_delta",
    "heston_price",
    "implied_vol",
    "model_delta",
    "model_price",
    "no_hedge",
    "standard_hedgers",
]

MONTE_CARLO_PATHS = 32768  # premium of a model without a fast pricer: ~0.3% standard error

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
                 variance: float | None = None, rate: float = 0.0, dividend: float = 0.0) -> float:
    """Heston COS price of ``option`` under ``params`` (``v0`` unless ``variance`` is given)."""
    p = params.params
    v = p["v0"] if variance is None else variance
    return float(fourier_price(spot, option.strike, option.maturity, rate, dividend, v,
                               p["kappa"], p["theta"], p["sigma_v"], p["rho"], option.is_call))


def implied_vol(option: EuropeanOption, price: float, spot: float,
                rate: float = 0.0, dividend: float = 0.0) -> float:
    """Black-Scholes implied vol of ``price``; raises if the price is not invertible."""
    iv = implied_vol_batch_np(
        S=spot, K=np.array([option.strike]), T=option.maturity, r=rate, q=dividend,
        market_prices=np.array([price]), is_call=np.array([option.is_call]),
    )[0]
    if not np.isfinite(iv):
        raise ValueError(f"price {price} of {option} at spot {spot} has no implied vol")
    return float(iv)


def _monte_carlo_price(option, params, spot, rate, dividend, dt, seed):
    steps = option.steps(dt)
    result = run_backtest(params, _hold_nothing, n_paths=MONTE_CARLO_PATHS, horizon_steps=steps, dt=dt,
                          mu=rate - dividend, spot0=spot, seed=seed)
    payoff = np.asarray(option.payoff(result.spots[:, -1]))
    return float(np.exp(-rate * option.maturity) * payoff.mean())


def model_price(option: EuropeanOption, params: ParamSet, spot: float, *, rate: float = 0.0,
                dividend: float = 0.0, dt: float = 1.0 / 252, seed: int = 0) -> float:
    """Price of ``option`` under the risk-neutral version of ``params``' model."""
    p, T, K, call = params.params, option.maturity, option.strike, option.is_call
    if params.model == "gbm":
        return float(model_pricing.bs_price(spot, K, T, rate, dividend, p["sigma"], call))
    if params.model == "heston":
        return heston_price(option, params, spot, rate=rate, dividend=dividend)
    if params.model == "merton":
        return float(model_pricing.merton_price(spot, K, T, rate, dividend, p["sigma"], p["lambda_j"],
                                                p["mu_j"], p["sigma_j"], call))
    if params.model == "regime_switching_gbm":
        m = market_params(params.model, p)
        table = model_pricing.regime_table(m["sigmas"], m["transition"], option.steps(dt), dt, rate, dividend)
        put = K * model_pricing.regime_put_price(table, 0, int(m["regime0"]), float(np.log(spot / K)))
        return put + (spot * np.exp(-dividend * T) - K * np.exp(-rate * T) if call else 0.0)
    return _monte_carlo_price(option, params, spot, rate, dividend, dt, seed)


def _option_params(option: EuropeanOption, dt: float, rate: float = 0.0, dividend: float = 0.0) -> dict[str, Any]:
    return {"strike": option.strike, "maturity": option.maturity, "is_call": option.is_call, "dt": dt,
            "rate": rate, "dividend": dividend}


def _tau(p, step):
    return p["maturity"] - step * p["dt"]


def _hold_nothing(spot, state, position, step):
    return jnp.zeros_like(spot)


def _no_hedge(p, spot, state, position, step):
    return jnp.zeros_like(spot)


def _bs_delta(p, spot, state, position, step):
    return model_pricing.bs_delta(spot, p["strike"], _tau(p, step), p["rate"], p["dividend"], p["sigma"],
                                  p["is_call"])


def _heston_delta_scalar(spot, variance, tau, strike, rate, dividend, kappa, theta, sigma_v, rho, is_call,
                         v_top):
    return heston_cos_delta(spot, variance, tau, strike, rate, dividend, kappa, theta, sigma_v, rho,
                            is_call, v_top)


_heston_spot_delta = jnp.vectorize(_heston_delta_scalar)


def _heston_delta(p, spot, variance, position, step):
    return _heston_spot_delta(spot, variance, _tau(p, step), p["strike"], p["rate"], p["dividend"],
                              p["kappa"], p["theta"], p["sigma_v"], p["rho"], p["is_call"], p["v_top"])


def _merton_delta(p, spot, state, position, step):
    return model_pricing.merton_delta(spot, p["strike"], _tau(p, step), p["rate"], p["dividend"],
                                      p["sigma"], p["lambda_j"], p["mu_j"], p["sigma_j"], p["is_call"])


def _regime_delta(p, spot, regime, position, step):
    tau = _tau(p, step)
    put = model_pricing.regime_put_delta(p["table"], step, regime, jnp.log(spot / p["strike"]), tau,
                                         p["dividend"])
    return jnp.where(p["is_call"], put + jnp.exp(-p["dividend"] * tau), put)


def _garch_delta(p, spot, h, position, step):
    """Black-Scholes delta at the GARCH forecast of the average variance to expiry."""
    remaining = jnp.maximum(p["steps"] - step, 1)
    phi = p["persistence"]
    decay = (1.0 - phi**remaining) / ((1.0 - phi) * remaining)
    average = p["long_run"] + (h - p["long_run"]) * decay
    sigma = jnp.sqrt(jnp.maximum(average, 1e-12) / p["dt"])
    return model_pricing.bs_delta(spot, p["strike"], _tau(p, step), p["rate"], p["dividend"], sigma,
                                  p["is_call"])


def _rough_delta(p, spot, variance, position, step):
    """Black-Scholes delta at today's instantaneous variance."""
    sigma = jnp.sqrt(jnp.maximum(variance, 1e-12))
    return model_pricing.bs_delta(spot, p["strike"], _tau(p, step), p["rate"], p["dividend"], sigma,
                                  p["is_call"])


def no_hedge() -> ParametricStrategy:
    """Hold nothing: the book's P&L is the premium minus the payoff."""
    return ParametricStrategy(_no_hedge, ())


def bs_delta(option: EuropeanOption, sigma: float, dt: float, rate: float = 0.0,
             dividend: float = 0.0) -> ParametricStrategy:
    """Black-Scholes delta at a fixed volatility ``sigma``."""
    return ParametricStrategy(_bs_delta, {**_option_params(option, dt, rate, dividend), "sigma": float(sigma)})


def heston_delta(option: EuropeanOption, params: ParamSet, dt: float, rate: float = 0.0,
                 dividend: float = 0.0) -> ParametricStrategy:
    """Heston delta ``dC/dS`` from ``params``' dynamics at the path's current variance.

    The hedger observes the simulated variance, so this is the model-consistent
    benchmark for a desk that tracks volatility, not a filter-based hedger. Deltas come
    from :func:`mjollnir.jax.heston_cos_delta`, which shares the characteristic-function
    work across paths.
    """
    dynamics = {k: float(params.params[k]) for k in ("kappa", "theta", "sigma_v", "rho")}
    v_top = _DELTA_WINDOW_VARIANCE_MULTIPLE * max(dynamics["theta"], float(params.params["v0"]))
    return ParametricStrategy(_heston_delta, {**_option_params(option, dt, rate, dividend), **dynamics,
                                              "v_top": v_top})


# model -> (hedger name, description); the description says what the hedger knows
MODEL_HEDGERS = {
    "gbm": ("gbm_delta", "Black-Scholes delta at the model's volatility"),
    "heston": ("heston_delta", "Heston delta at the current variance"),
    "merton": ("merton_delta", "Merton delta: the jump-diffusion's own price sensitivity"),
    "regime_switching_gbm": ("regime_delta", "Regime-switching delta, knowing today's regime"),
    "garch": ("garch_delta", "Approximation: Black-Scholes delta at the GARCH variance forecast to expiry"),
    "rough_bergomi": ("rough_delta", "Approximation: Black-Scholes delta at today's instantaneous variance"),
}


def model_delta(option: EuropeanOption, params: ParamSet, dt: float, rate: float = 0.0,
                dividend: float = 0.0) -> ParametricStrategy:
    """The hedge a desk that believes ``params``' model would run."""
    p, base = params.params, _option_params(option, dt, rate, dividend)
    if params.model == "gbm":
        return bs_delta(option, p["sigma"], dt, rate, dividend)
    if params.model == "heston":
        return heston_delta(option, params, dt, rate, dividend)
    if params.model == "merton":
        return ParametricStrategy(_merton_delta, {**base, **{k: float(p[k]) for k in
                                                             ("sigma", "lambda_j", "mu_j", "sigma_j")}})
    if params.model == "regime_switching_gbm":
        m = market_params(params.model, p)
        table = model_pricing.regime_table(m["sigmas"], m["transition"], option.steps(dt), dt, rate, dividend)
        return ParametricStrategy(_regime_delta, {**base, "table": table})
    if params.model == "garch":
        persistence = float(p["alpha"] + p["beta"])
        return ParametricStrategy(_garch_delta, {**base, "persistence": persistence,
                                                 "long_run": float(p["omega"]) / (1.0 - persistence),
                                                 "steps": option.steps(dt)})
    if params.model == "rough_bergomi":
        return ParametricStrategy(_rough_delta, base)
    raise ValueError(f"no model hedger for {params.model!r}")


def standard_hedgers(option: EuropeanOption, book: ParamSet, *, dt: float, spot0: float = 100.0,
                     rate: float = 0.0, dividend: float = 0.0,
                     premium: float | None = None) -> dict[str, Strategy]:
    """No hedge, Black-Scholes delta at the premium's implied vol, and the model's own delta."""
    if premium is None:
        premium = model_price(option, book, spot0, rate=rate, dividend=dividend, dt=dt)
    sigma = implied_vol(option, premium, spot0, rate, dividend)
    name, _ = MODEL_HEDGERS[book.model]
    return {
        "no_hedge": no_hedge(),
        "bs_delta": bs_delta(option, sigma, dt, rate, dividend),
        name: model_delta(option, book, dt, rate, dividend),
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
    rate: float = 0.0,
    dividend: float = 0.0,
    premium: float | None = None,
    cost_bps: float = 0.0,
    seed: int = 0,
) -> dict[str, BacktestResult]:
    """Run a short-option book through one world for several hedgers.

    The premium (the model price under ``book`` unless given, e.g. a market
    price) and the hedgers come from the book; ``scenario``, if given, changes
    only the simulated world. Every hedger sees the same paths (common random
    numbers), so differences between results are differences between strategies.

    Args:
        book: ParamSet the book was priced and hedged with (any model with a world).
        option: the option sold.
        hedgers: name -> strategy; defaults to :func:`standard_hedgers`.
        scenario: counterfactual applied to the simulated world only.
        mu: spot drift of the simulation; defaults to ``rate - dividend``.
        rate: the book's risk-free rate (continuous).
        dividend: the underlying's dividend yield (continuous).
        premium: the price received; defaults to :func:`model_price`.
    """
    if premium is None:
        premium = model_price(option, book, spot0, rate=rate, dividend=dividend, dt=dt)
    if hedgers is None:
        hedgers = standard_hedgers(option, book, dt=dt, spot0=spot0, rate=rate, dividend=dividend,
                                   premium=premium)
    drift = rate - dividend if mu is None else mu
    common = {"n_paths": n_paths, "horizon_steps": option.steps(dt), "dt": dt, "mu": drift, "rate": rate,
              "dividend": dividend, "spot0": spot0, "initial_cash": premium, "cost_bps": cost_bps,
              "scenario": scenario, "liability": option.payoff, "seed": seed}
    if all(isinstance(h, ParametricStrategy) for h in hedgers.values()):
        # one program, one simulation of the world for all of them
        return run_backtests(book, hedgers, **common)
    return {name: run_backtest(book, strategy, **common) for name, strategy in hedgers.items()}
