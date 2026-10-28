"""Prices and deltas of European options under each world's model.

What a hedger needs from a model: the premium of the option it sold and the
delta that hedges it, both under the model's risk-neutral version (the drift
replaced by r - q, the volatility dynamics kept as calibrated).

- Black-Scholes: closed form (``bs_price``, ``bs_delta``).
- Merton: the Poisson mixture of Black-Scholes prices and deltas (Merton
  1976), with enough terms for frequent small jumps.
- Regime-switching GBM: COS on the chain's characteristic function
  ``E[exp(iu log(S_T/S_t)) | regime i] = [exp(tau (L + diag psi(u))) 1]_i``,
  with L the chain's generator and psi_j the j-th regime's Black-Scholes
  exponent; tabulated over (step, regime, log-moneyness) once per hedger, so
  a hedge is a table lookup. Puts by COS, calls by parity.

Heston prices and deltas live in :mod:`mjollnir.jax`; GARCH and rough
Bergomi have no fast pricer and are priced by Monte Carlo in their own world
(:func:`mjollnir.hedging.model_price`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


__all__ = ["RegimeTable", "bs_delta", "bs_gamma", "bs_price", "bs_theta", "bs_vega", "merton_delta", "merton_price",
           "regime_table"]

MERTON_TERMS = 128   # covers lambda * tau up to ~90 jumps (frequent-small-jump fits)
REGIME_COS_TERMS = 512
REGIME_GRID = 481    # log-moneyness points of a regime table (interpolation error ~2e-6 in price)
REGIME_X_HALF_WIDTH = 0.8


def _d1(spot, strike, tau, rate, dividend, sigma):
    return (jnp.log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * tau) / (sigma * jnp.sqrt(tau))


def bs_price(spot, strike, tau, rate, dividend, sigma, is_call):
    """Black-Scholes price with continuous rate and dividend yield."""
    d1 = _d1(spot, strike, tau, rate, dividend, sigma)
    d2 = d1 - sigma * jnp.sqrt(tau)
    call = spot * jnp.exp(-dividend * tau) * norm.cdf(d1) - strike * jnp.exp(-rate * tau) * norm.cdf(d2)
    put = call - spot * jnp.exp(-dividend * tau) + strike * jnp.exp(-rate * tau)
    return jnp.where(is_call, call, put)


def bs_delta(spot, strike, tau, rate, dividend, sigma, is_call):
    """Black-Scholes delta dV/dS with continuous rate and dividend yield."""
    call = jnp.exp(-dividend * tau) * norm.cdf(_d1(spot, strike, tau, rate, dividend, sigma))
    return jnp.where(is_call, call, call - jnp.exp(-dividend * tau))


def bs_gamma(spot, strike, tau, rate, dividend, sigma):
    """Black-Scholes gamma d2V/dS2 (the same for calls and puts)."""
    d1 = _d1(spot, strike, tau, rate, dividend, sigma)
    return jnp.exp(-dividend * tau) * norm.pdf(d1) / (spot * sigma * jnp.sqrt(tau))


def bs_vega(spot, strike, tau, rate, dividend, sigma):
    """Black-Scholes vega dV/dsigma, per unit of volatility (the same for calls and puts)."""
    d1 = _d1(spot, strike, tau, rate, dividend, sigma)
    return spot * jnp.exp(-dividend * tau) * norm.pdf(d1) * jnp.sqrt(tau)


def bs_theta(spot, strike, tau, rate, dividend, sigma, is_call):
    """Black-Scholes theta dV/dt, per year of calendar time (the negative of dV/dtau)."""
    d1 = _d1(spot, strike, tau, rate, dividend, sigma)
    d2 = d1 - sigma * jnp.sqrt(tau)
    decay = -spot * jnp.exp(-dividend * tau) * norm.pdf(d1) * sigma / (2.0 * jnp.sqrt(tau))
    call = decay + dividend * spot * jnp.exp(-dividend * tau) * norm.cdf(d1) - rate * strike * jnp.exp(-rate * tau) * norm.cdf(d2)
    put = decay - dividend * spot * jnp.exp(-dividend * tau) * norm.cdf(-d1) + rate * strike * jnp.exp(-rate * tau) * norm.cdf(-d2)
    return jnp.where(is_call, call, put)


def _merton_terms(tau, rate, sigma, lambda_j, mu_j, sigma_j):
    """Poisson weights, per-term vols and per-term rates of the Merton mixture."""
    m = jnp.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam_prime = lambda_j * (1.0 + m)
    k = jnp.arange(MERTON_TERMS, dtype=jnp.float64)
    log_w = -lam_prime * tau + k * jnp.log(jnp.maximum(lam_prime * tau, 1e-300)) - jax.lax.lgamma(k + 1.0)
    sigma_k = jnp.sqrt(sigma**2 + k * sigma_j**2 / tau)
    rate_k = rate - lambda_j * m + k * jnp.log1p(m) / tau
    return jnp.exp(log_w), sigma_k, rate_k


def merton_price(spot, strike, tau, rate, dividend, sigma, lambda_j, mu_j, sigma_j, is_call):
    # weights at lambda' = lambda (1 + m) already carry exp((rate_k - rate) tau), so each
    # term is a plain Black-Scholes price at its own rate (Merton 1976, eq. 19)
    w, sigma_k, rate_k = _merton_terms(tau, rate, sigma, lambda_j, mu_j, sigma_j)
    return jnp.sum(w * bs_price(spot, strike, tau, rate_k, dividend, sigma_k, is_call))


def merton_delta(spot, strike, tau, rate, dividend, sigma, lambda_j, mu_j, sigma_j, is_call):
    w, sigma_k, rate_k = _merton_terms(tau, rate, sigma, lambda_j, mu_j, sigma_j)
    return jnp.sum(w * bs_delta(spot, strike, tau, rate_k, dividend, sigma_k, is_call))


class RegimeTable(NamedTuple):
    """Regime-switching deltas (and prices) on a grid, for table-lookup hedging.

    ``delta[t, i, g]`` is the put delta per unit strike at step ``t`` (time to
    expiry ``(horizon - t) dt``), regime ``i`` and log-moneyness ``x[g] =
    log(S/K)``; ``price`` likewise (per unit strike).
    """

    x: jax.Array
    delta: jax.Array
    price: jax.Array


def _generator(transition, dt):
    """Generator of a per-step transition matrix: (P - I) / dt, exact to O(dt)."""
    return (transition - jnp.eye(transition.shape[0])) / dt


def _psi_put(k, a, b):
    """COS coefficients of the put payoff (1 - e^y)^+ on [a, 0], per unit strike (numpy)."""
    w = k * np.pi / (b - a)
    chi = (np.cos(w * (0.0 - a)) + w * np.sin(w * (0.0 - a)) - (1.0 + 0.0 * w) * np.exp(a)) / (1.0 + w**2)
    chi = np.where(k == 0, 1.0 - np.exp(a), chi)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(k == 0, -a, (b - a) / (k * np.pi) * (np.sin(w * (0.0 - a)) - np.sin(0.0)))
    return 2.0 / (b - a) * (psi - chi)


def regime_table(sigmas, transition, horizon: int, dt, rate, dividend) -> RegimeTable:
    """Put prices and deltas over (step, regime, log-moneyness) for ``horizon`` steps.

    A precomputation (numpy, outside any compiled program): for every maturity
    and COS term, ``exp(tau M(u)) 1`` comes from one batched eigendecomposition
    of ``M(u) = L + diag(psi(u))``. Cached on its inputs: a book's premium and
    its hedger use the same table.
    """
    return _regime_table(tuple(np.asarray(sigmas, dtype=float).ravel().tolist()),
                         tuple(np.asarray(transition, dtype=float).ravel().tolist()),
                         int(horizon), float(dt), float(rate), float(dividend))


@lru_cache(maxsize=64)
def _regime_table(sigmas, transition, horizon, dt, rate, dividend) -> RegimeTable:
    sigmas = np.asarray(sigmas, dtype=float)
    transition = np.asarray(transition, dtype=float).reshape(sigmas.size, sigmas.size)
    n = sigmas.size
    generator = (transition - np.eye(n)) / dt
    x = np.linspace(-REGIME_X_HALF_WIDTH, REGIME_X_HALF_WIDTH, REGIME_GRID)
    k = np.arange(REGIME_COS_TERMS, dtype=float)
    taus = (horizon - np.arange(horizon)) * dt                         # (H,)
    sigma_max = sigmas.max()
    centre = (rate - dividend - 0.5 * sigma_max**2) * taus
    half = 10.0 * sigma_max * np.sqrt(taus) + 0.1
    a = x[0] + centre - half                                           # (H,)
    b = x[-1] + centre + half
    u = k[None, :] * np.pi / (b - a)[:, None]                          # (H, N)

    psi = 1j * u[..., None] * (rate - dividend - 0.5 * sigmas**2) - 0.5 * sigmas**2 * u[..., None] ** 2  # (H, N, n)
    m = generator[None, None, :, :] + psi[..., None] * np.eye(n)       # (H, N, n, n)
    eigvals, eigvecs = np.linalg.eig(m)
    weights = np.linalg.solve(eigvecs, np.ones((*m.shape[:2], n, 1), dtype=complex))[..., 0]  # V^-1 1
    phi = np.einsum("hnij,hnj->hni", eigvecs, np.exp(taus[:, None, None] * eigvals) * weights)  # (H, N, n)

    coeff = _psi_put(k[None, :], a[:, None], b[:, None]) * np.where(k == 0, 0.5, 1.0)[None, :]  # (H, N)
    phase = np.exp(1j * u[:, None, :] * (x[None, :, None] - a[:, None, None]))                  # (H, G, N)
    disc = np.exp(-rate * taus)[:, None, None]
    price = disc * np.real(np.einsum("hgn,hn,hni->hig", phase, coeff, phi))                     # (H, n, G)
    dprice = disc * np.real(np.einsum("hgn,hn,hni->hig", phase, 1j * u * coeff, phi))
    # per unit strike, V = K p(x) with x = log(S/K), so dV/dS = p'(x) / exp(x)
    return RegimeTable(x=jnp.asarray(x), delta=jnp.asarray(dprice / np.exp(x)[None, None, :]),
                       price=jnp.asarray(price))


def regime_put_delta(table: RegimeTable, step, regime, log_moneyness, tau, dividend):
    """Put delta from the table; beyond the grid, the deep in/out-of-the-money limits."""
    row = table.delta[step, regime.astype(jnp.int32)]
    inside = jnp.interp(log_moneyness, table.x, row)
    deep_itm_put = -jnp.exp(-dividend * tau)
    return jnp.where(log_moneyness < table.x[0], deep_itm_put,
                     jnp.where(log_moneyness > table.x[-1], 0.0, inside))


def regime_put_price(table: RegimeTable, step: int, regime: int, log_moneyness: float) -> float:
    return float(np.interp(log_moneyness, np.asarray(table.x), np.asarray(table.price[step, regime])))
