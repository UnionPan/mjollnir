"""Simulated worlds: each model's spot dynamics as one JAX step.

A *world* is how a market evolves under a calibrated model, step by step,
together with the state a hedger observes. Every model provides two pure
functions, so one backtest engine (:mod:`mjollnir.backtest`) runs them all:

``init(params, key, horizon, dt) -> (state, path, key)``
    The observed state at time 0 and the episode's precomputed inputs
    (``path``; rough Bergomi's variance path, otherwise an empty array).
``step(spot, state, t, params, path, dt, mu, key) -> (spot, state, key)``
    One step of length ``dt`` with spot drift ``mu`` (r - q plus any excess).

``state`` is what the model's hedger sees: the variance for stochastic
volatility (Heston, rough Bergomi), the conditional per-step variance for
GARCH, the regime for regime switching, and sigma^2 for the constant-
volatility models. ``params`` is a dict of numbers that enter the compiled
program traced, so one program serves every calibration of a model.

Parameter names (ParamSet ``model``):

- ``gbm``: sigma
- ``heston``: v0, kappa, theta, sigma_v, rho
- ``merton``: sigma, lambda_j, mu_j, sigma_j
- ``regime_switching_gbm``: sigma_<k> and per-step transition probabilities
  p_<i><j> for regimes k, i, j = 0..n-1, and regime0 (the regime today)
- ``garch``: omega, alpha, beta (per-step units) and h0 (today's per-step
  conditional variance)
- ``rough_bergomi``: xi0, eta, rho, H

Each world's mean spot grows exactly at ``mu``: jumps, regimes and GARCH
variance are compensated, so a world simulated at ``mu = r - q`` is risk
neutral in the drift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mjollnir.processes._jax_qe import qe_heston_step

__all__ = ["MODELS", "World", "market_params", "world_for"]


class World(NamedTuple):
    init: Callable[..., Any]
    step: Callable[..., Any]


_EMPTY = jnp.zeros((0,))


def _no_path(params, key, horizon, dt):
    return _EMPTY, key


# ── GBM ─────────────────────────────────────────────────────────────────────

def _gbm_init(params, key, horizon, dt):
    return params["sigma"] ** 2, _EMPTY, key


def _gbm_step(spot, state, t, params, path, dt, mu, key):
    key, k = jax.random.split(key)
    sigma = params["sigma"]
    z = jax.random.normal(k, jnp.shape(spot))
    return spot * jnp.exp((mu - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * z), state, key


# ── Heston ──────────────────────────────────────────────────────────────────

def _heston_init(params, key, horizon, dt):
    return params["v0"], _EMPTY, key


def _heston_step(spot, state, t, params, path, dt, mu, key):
    spot, variance, key = qe_heston_step(spot, state, dt, mu, params["kappa"], params["theta"],
                                         params["sigma_v"], params["rho"], key=key)
    return spot, variance, key


# ── Merton ──────────────────────────────────────────────────────────────────

def _merton_init(params, key, horizon, dt):
    return params["sigma"] ** 2, _EMPTY, key


def _merton_step(spot, state, t, params, path, dt, mu, key):
    key, kz, kn, kj = jax.random.split(key, 4)
    sigma, lam, mu_j, sigma_j = params["sigma"], params["lambda_j"], params["mu_j"], params["sigma_j"]
    compensator = jnp.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    n_jumps = jax.random.poisson(kn, lam * dt, jnp.shape(spot)).astype(spot.dtype)
    jump = n_jumps * mu_j + jnp.sqrt(n_jumps) * sigma_j * jax.random.normal(kj, jnp.shape(spot))
    z = jax.random.normal(kz, jnp.shape(spot))
    log_step = (mu - 0.5 * sigma**2 - lam * compensator) * dt + sigma * jnp.sqrt(dt) * z + jump
    return spot * jnp.exp(log_step), state, key


# ── Regime-switching GBM ────────────────────────────────────────────────────

def _regime_init(params, key, horizon, dt):
    return params["regime0"], _EMPTY, key


def _regime_step(spot, state, t, params, path, dt, mu, key):
    key, kz, kr = jax.random.split(key, 3)
    sigmas, transition = params["sigmas"], params["transition"]
    regime = state.astype(jnp.int32)
    sigma = sigmas[regime]
    z = jax.random.normal(kz, jnp.shape(spot))
    spot = spot * jnp.exp((mu - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * z)
    regime = jax.random.categorical(kr, jnp.log(jnp.maximum(transition[regime], 1e-300)))
    return spot, regime.astype(state.dtype), key


# ── GARCH(1,1) ──────────────────────────────────────────────────────────────

def _garch_init(params, key, horizon, dt):
    return params["h0"], _EMPTY, key


def _garch_step(spot, state, t, params, path, dt, mu, key):
    """One GARCH step; ``state`` is this step's per-step conditional variance.
    The -h/2 term makes E[S_{t+1} / S_t] = exp(mu dt) exactly."""
    key, k = jax.random.split(key)
    eps = jnp.sqrt(state) * jax.random.normal(k, jnp.shape(spot))
    spot = spot * jnp.exp(mu * dt - 0.5 * state + eps)
    state = jnp.maximum(params["omega"] + params["alpha"] * eps**2 + params["beta"] * state, 1e-12)
    return spot, state, key


# ── Rough Bergomi ───────────────────────────────────────────────────────────

def hybrid_scheme_weights(H, horizon: int, dt):
    """Hybrid-scheme (kappa = 1) weights for the Riemann-Liouville fBm
    sqrt(2H) int_0^t (t - s)^(H - 1/2) dW_s on a grid of ``horizon`` steps.

    Returns the Toeplitz matrix applied to the Brownian increments for every
    step but the most recent (evaluated at the optimal points b_k of
    Bennedsen, Lunde and Pakkanen 2017), and the covariance of the most
    recent step's exact kernel integral with its own increment.
    """
    alpha = H - 0.5
    k = jnp.arange(2, horizon + 1, dtype=jnp.float64)
    b = ((k ** (alpha + 1.0) - (k - 1.0) ** (alpha + 1.0)) / (alpha + 1.0)) ** (1.0 / alpha)
    g = (b * dt) ** alpha  # weight of the increment k steps back, k >= 2
    lags = jnp.arange(horizon)[:, None] - jnp.arange(horizon)[None, :] + 1  # lag of increment j at step i
    padded = jnp.concatenate([jnp.zeros(2), g])
    toeplitz = jnp.where(lags >= 2, padded[jnp.clip(lags, 0, horizon)], 0.0)
    var_w, var_z = dt, dt ** (2.0 * alpha + 1.0) / (2.0 * alpha + 1.0)
    cov = dt ** (alpha + 1.0) / (alpha + 1.0)
    return toeplitz, var_w, var_z, cov


def _rough_init(params, key, horizon, dt):
    """Variance path v_0..v_{horizon-1} and the standardized increments driving it."""
    key, kw, kz = jax.random.split(key, 3)
    H, eta, xi0 = params["H"], params["eta"], params["xi0"]
    toeplitz, var_w, var_z, cov = hybrid_scheme_weights(H, horizon, dt)
    z_w = jax.random.normal(kw, (horizon,))
    z_perp = jax.random.normal(kz, (horizon,))
    dw = jnp.sqrt(var_w) * z_w
    beta = cov / var_w
    exact = beta * dw + jnp.sqrt(jnp.maximum(var_z - beta**2 * var_w, 0.0)) * z_perp  # last-step integral
    # W^H at the END of step i (time (i + 1) dt): the exact last-step term plus the rest
    w_h = jnp.sqrt(2.0 * H) * (exact + toeplitz @ dw)
    t_end = (jnp.arange(horizon) + 1.0) * dt
    v_end = xi0 * jnp.exp(eta * w_h - 0.5 * eta**2 * t_end ** (2.0 * H))
    variance = jnp.concatenate([jnp.reshape(xi0, (1,)), v_end[:-1]])  # v at the START of each step
    return xi0, jnp.stack([variance, z_w]), key


def _rough_step(spot, state, t, params, path, dt, mu, key):
    key, k = jax.random.split(key)
    variance, z_w = path[0, t], path[1, t]
    rho = params["rho"]
    z = rho * z_w + jnp.sqrt(1.0 - rho**2) * jax.random.normal(k, jnp.shape(spot))
    spot = spot * jnp.exp((mu - 0.5 * variance) * dt + jnp.sqrt(variance * dt) * z)
    next_variance = path[0, jnp.minimum(t + 1, path.shape[1] - 1)]
    return spot, next_variance.astype(state.dtype), key


# ── registry ────────────────────────────────────────────────────────────────

MODELS: dict[str, World] = {
    "gbm": World(_gbm_init, _gbm_step),
    "heston": World(_heston_init, _heston_step),
    "merton": World(_merton_init, _merton_step),
    "regime_switching_gbm": World(_regime_init, _regime_step),
    "garch": World(_garch_init, _garch_step),
    "rough_bergomi": World(_rough_init, _rough_step),
}

_REQUIRED = {
    "gbm": ("sigma",),
    "heston": ("v0", "kappa", "theta", "sigma_v", "rho"),
    "merton": ("sigma", "lambda_j", "mu_j", "sigma_j"),
    "garch": ("omega", "alpha", "beta", "h0"),
    "rough_bergomi": ("xi0", "eta", "rho", "H"),
}


def world_for(model: str) -> World:
    if model not in MODELS:
        raise ValueError(f"no simulated world for model {model!r}; known: {sorted(MODELS)}")
    return MODELS[model]


def regime_count(params) -> int:
    return sum(1 for k in params if k.startswith("sigma_") and k[6:].isdigit())


def market_params(model: str, params) -> dict[str, Any]:
    """A ParamSet's parameters as the arrays a world reads (validated)."""
    params = dict(params)
    if model == "regime_switching_gbm":
        n = regime_count(params)
        if n < 1:
            raise ValueError("regime_switching_gbm needs sigma_0..sigma_<n-1>")
        transition = np.array([[params[f"p_{i}{j}"] for j in range(n)] for i in range(n)], dtype=float)
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("every row of the regime transition matrix must sum to 1")
        regime0 = round(float(params.get("regime0", 0)))
        if not 0 <= regime0 < n:
            raise ValueError(f"regime0 must be in 0..{n - 1}")
        return {"sigmas": jnp.asarray([params[f"sigma_{k}"] for k in range(n)], jnp.float64),
                "transition": jnp.asarray(transition, jnp.float64),
                "regime0": jnp.asarray(float(regime0), jnp.float64)}
    missing = set(_REQUIRED[model]) - params.keys()
    if missing:
        raise ValueError(f"{model} ParamSet missing {sorted(missing)}")
    return {k: jnp.asarray(params[k], jnp.float64) for k in _REQUIRED[model]}
