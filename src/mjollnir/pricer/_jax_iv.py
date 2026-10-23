"""
Vectorized Black-Scholes implied volatility inversion via JAX.

Replaces per-option root-finding loops with a single JIT-compiled batch
operation over all options simultaneously.

Two solvers are provided:

- ``implied_vol_bisection`` (the default behind ``implied_vol_batch_np``):
  bisection on ``[sigma_min, sigma_max]``. A Black-Scholes price is strictly
  increasing in sigma, so bisection always converges, needs no vega, and
  reports NaN for any price outside ``(BS(sigma_min), BS(sigma_max))``.
- ``implied_vol_batch``: the original fixed-iteration damped Newton. It never
  reports failure: unattainable prices come back clipped to the bounds, and
  low-vega quotes (short-dated wings) can stall at the initial guess. It is
  kept bit-identical because the synthetic option-chain generator's IVs feed
  environment observations (see docs/guide/stability.md).

author: Yunian Pan
email: yp1170@nyu.edu
"""

import jax
import jax.numpy as jnp
import numpy as np

from ..processes._jax_utils import to_numpy
from ._pricer_runtime import guard_pricer


# ============================================================================
# Vectorized BS price + vega (all options at once)
# ============================================================================
def _bs_price_and_vega(S, K, T, r, q, sigma, is_call):
    """
    Vectorized Black-Scholes price and vega.

    All inputs are arrays of shape (n,). Returns (prices, vegas) of shape (n,).
    """
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Normal CDF and PDF via JAX
    nd1 = jax.scipy.stats.norm.cdf(d1)
    nd2 = jax.scipy.stats.norm.cdf(d2)
    npd1 = jax.scipy.stats.norm.pdf(d1)

    # Call and put prices
    call_price = S * jnp.exp(-q * T) * nd1 - K * jnp.exp(-r * T) * nd2
    put_price = K * jnp.exp(-r * T) * (1.0 - nd2) - S * jnp.exp(-q * T) * (1.0 - nd1)

    prices = jnp.where(is_call, call_price, put_price)
    vega = S * jnp.exp(-q * T) * npd1 * sqrt_T

    return prices, vega


# ============================================================================
# Vectorized Newton-Raphson IV solver
# ============================================================================
@guard_pricer
@jax.jit
def implied_vol_batch(
    S: jnp.ndarray,
    K: jnp.ndarray,
    T: jnp.ndarray,
    r: jnp.ndarray,
    q: jnp.ndarray,
    market_prices: jnp.ndarray,
    is_call: jnp.ndarray,
    sigma_init: jnp.ndarray,
    sigma_min: float = 0.005,
    sigma_max: float = 5.0,
) -> jnp.ndarray:
    """
    Batch implied volatility via vectorized Newton-Raphson.

    All N options are solved simultaneously in each iteration.
    Fixed iteration count (no early exit) for JIT compatibility.

    Args:
        S: Spot prices, shape (n,)
        K: Strikes, shape (n,)
        T: Maturities, shape (n,)
        r: Risk-free rates, shape (n,)
        q: Dividend yields, shape (n,)
        market_prices: Market option prices, shape (n,)
        is_call: Boolean call/put flags, shape (n,)
        sigma_init: Initial IV guesses, shape (n,)
        sigma_min: Floor for IV
        sigma_max: Ceiling for IV

    Returns:
        Implied volatilities, shape (n,)
    """
    sigma = jnp.clip(sigma_init, sigma_min, sigma_max)

    def newton_step(sigma, _):
        prices, vega = _bs_price_and_vega(S, K, T, r, q, sigma, is_call)
        diff = prices - market_prices

        # Newton update with vega floor to avoid division by zero
        vega_safe = jnp.maximum(vega, 1e-10)
        update = diff / vega_safe

        # Damped Newton: clip step size
        update = jnp.clip(update, -0.5, 0.5)

        sigma_new = sigma - update
        sigma_new = jnp.clip(sigma_new, sigma_min, sigma_max)

        return sigma_new, None

    # Run 20 fixed iterations via lax.scan (no Python loop)
    sigma_final, _ = jax.lax.scan(newton_step, sigma, jnp.arange(20))

    return sigma_final


# ============================================================================
# Vectorized bisection IV solver (reports failure as NaN)
# ============================================================================
_BISECTION_STEPS = 100  # halves [sigma_min, sigma_max] far below float64 resolution


def _bs_price(S, K, T, r, q, sigma, is_call):
    """
    Vectorized Black-Scholes price, accurate in the far wings.

    Puts use ``N(-d)`` rather than ``1 - N(d)``, which cancels to zero for
    deep out-of-the-money puts. ``_bs_price_and_vega`` keeps the original
    form because the legacy Newton solver must stay bit-identical.
    """
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    cdf = jax.scipy.stats.norm.cdf
    call_price = S * jnp.exp(-q * T) * cdf(d1) - K * jnp.exp(-r * T) * cdf(d2)
    put_price = K * jnp.exp(-r * T) * cdf(-d2) - S * jnp.exp(-q * T) * cdf(-d1)
    return jnp.where(is_call, call_price, put_price)


@guard_pricer
@jax.jit
def implied_vol_bisection(
    S: jnp.ndarray,
    K: jnp.ndarray,
    T: jnp.ndarray,
    r: jnp.ndarray,
    q: jnp.ndarray,
    market_prices: jnp.ndarray,
    is_call: jnp.ndarray,
    sigma_min: float = 0.005,
    sigma_max: float = 5.0,
    price_tolerance: float = 0.0,
) -> jnp.ndarray:
    """
    Batch implied volatility via vectorized bisection.

    Args:
        S, K, T, r, q: Spot, strike, maturity, rate, dividend yield, shape (n,)
        market_prices: Option prices, shape (n,)
        is_call: Boolean call/put flags, shape (n,)
        sigma_min: Lower end of the search range
        sigma_max: Upper end of the search range
        price_tolerance: Absolute pricing error of ``market_prices``. A price
            within this distance of ``BS(sigma_min)`` (for a deep in-the-money
            option, essentially its intrinsic value) carries no volatility
            information and returns NaN.

    Returns:
        Implied volatilities, shape (n,). NaN where the price is not strictly
        inside ``(BS(sigma_min) + price_tolerance, BS(sigma_max))``: below
        intrinsic, above the no-arbitrage upper bound, a zero wing price, a time
        value inside the pricing error, an IV outside the search range, a
        non-finite input, or an expired option.
    """
    lo = jnp.full_like(market_prices, sigma_min)
    hi = jnp.full_like(market_prices, sigma_max)
    price_lo = _bs_price(S, K, T, r, q, lo, is_call)
    price_hi = _bs_price(S, K, T, r, q, hi, is_call)
    attainable = (market_prices > price_lo + price_tolerance) & (market_prices < price_hi)

    def bisect(bounds, _):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        price_mid = _bs_price(S, K, T, r, q, mid, is_call)
        below = price_mid < market_prices
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(bisect, (lo, hi), None, length=_BISECTION_STEPS)
    return jnp.where(attainable, 0.5 * (lo + hi), jnp.nan)


# ============================================================================
# Convenience: numpy in/out wrapper
# ============================================================================
def implied_vol_batch_np(
    S, K, T, r, q, market_prices, is_call,
    sigma_init=None,
    sigma_min=0.005, sigma_max=5.0,
    on_failure="nan",
    price_tolerance=0.0,
):
    """
    Batch IV inversion with numpy arrays in and out.

    Args:
        S: float or array — spot price(s)
        K: array — strikes
        T: float or array — maturities
        r: float or array — risk-free rate
        q: float or array — dividend yield
        market_prices: array — option prices
        is_call: bool array or list — True for calls
        sigma_init: array — Newton initial guesses, ``on_failure="clip"`` only
            (default: 0.2)
        sigma_min: lower end of the IV search range
        sigma_max: upper end of the IV search range
        on_failure: ``"nan"`` (default) solves by bisection and returns NaN
            wherever the price cannot be inverted. ``"clip"`` runs the legacy
            fixed-iteration Newton (``implied_vol_batch``), which always
            returns a value in ``[sigma_min, sigma_max]`` even when it is
            wrong; use it only to reproduce historical outputs.
        price_tolerance: absolute error of ``market_prices`` (``on_failure="nan"``
            only); time value below it is treated as uninformative (NaN). Pass
            the pricer's accuracy when inverting model prices.

    Returns:
        numpy array of implied volatilities
    """
    if on_failure not in ("nan", "clip"):
        raise ValueError(f"on_failure must be 'nan' or 'clip', got {on_failure!r}")
    if on_failure == "clip" and price_tolerance:
        raise ValueError("price_tolerance needs on_failure='nan': clip mode never reports failure")
    n = len(K)

    S_j = jnp.broadcast_to(jnp.float64(S), (n,)) if np.isscalar(S) else jnp.array(S, dtype=jnp.float64)
    K_j = jnp.array(K, dtype=jnp.float64)
    T_j = jnp.broadcast_to(jnp.float64(T), (n,)) if np.isscalar(T) else jnp.array(T, dtype=jnp.float64)
    r_j = jnp.broadcast_to(jnp.float64(r), (n,)) if np.isscalar(r) else jnp.array(r, dtype=jnp.float64)
    q_j = jnp.broadcast_to(jnp.float64(q), (n,)) if np.isscalar(q) else jnp.array(q, dtype=jnp.float64)
    prices_j = jnp.array(market_prices, dtype=jnp.float64)
    is_call_j = jnp.array(is_call, dtype=bool) if not isinstance(is_call, jnp.ndarray) else is_call

    if on_failure == "nan":
        result = implied_vol_bisection(
            S_j, K_j, T_j, r_j, q_j, prices_j, is_call_j, sigma_min, sigma_max, price_tolerance,
        )
        return to_numpy(result)

    if sigma_init is None:
        sigma_init_j = jnp.full(n, 0.2)
    else:
        sigma_init_j = jnp.array(sigma_init, dtype=jnp.float64)

    result = implied_vol_batch(
        S_j, K_j, T_j, r_j, q_j, prices_j, is_call_j,
        sigma_init_j, sigma_min, sigma_max,
    )

    return to_numpy(result)
