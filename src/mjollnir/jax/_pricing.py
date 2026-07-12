"""Differentiable Heston COS pricing for the public ``mjollnir.jax`` facade.

``mjollnir.pricer._jax_fourier_pricer.jax_cos_price_heston`` is a numpy-orchestrated
convenience wrapper: it computes the COS truncation range with ``np.log``/``np.sqrt``
and casts the result to ``float``, so it cannot be traced by ``jax.grad``/``jax.jit``.
The RL layer historically worked around this by composing the private internals
(``jax_heston_cf`` + ``_jax_cos_price_multi``) itself.

This module is that composition, done once, correctly, in pure ``jax.numpy`` — the
truncation range is traced too, so the price is differentiable with respect to *all*
market inputs (``S0``, ``v0``, and every Heston parameter) and safe inside
``jit``/``vmap``/``lax.scan``. For concrete (non-traced) inputs it reproduces
``jax_cos_price_heston`` exactly (asserted by ``tests/test_parity_golden.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from mjollnir.pricer._jax_fourier_pricer import (
    HestonCFParams,
    _jax_cos_price_multi,
    _jax_cos_price_single,
    jax_heston_cf,
)

__all__ = ["fourier_price", "fourier_price_batch"]


def _traced_truncation_range(log_S0, drift, sigma_approx, T, L):
    """COS truncation range ``[a, b]`` in pure jnp (jit/vmap-traceable).

    The bounds are wrapped in ``stop_gradient``: they only choose the integration
    grid, and the COS price is independent of them up to truncation error, so
    differentiating *through* them adds no signal — it only re-introduces the
    well-known ``jnp.where`` NaN-cotangent problem in the COS coefficient
    formulas. This mirrors the deep-hedging convention of a concrete grid with a
    fully differentiated characteristic function, while remaining traceable so
    the pricer works on batched/vmapped inputs.
    """
    c1 = log_S0 + drift * T
    c2 = jnp.maximum(sigma_approx**2 * T, 1e-12)
    half = L * jnp.sqrt(c2)
    return jax.lax.stop_gradient(c1 - half), jax.lax.stop_gradient(c1 + half)


def _effective_variance(v0, kappa, theta, T):
    """Mean of the integrated CIR variance over ``[0, T]`` divided by ``T``.

    Sizing the COS truncation grid from ``sqrt(v0)`` alone under-covers the
    log-return distribution whenever variance mean-reverts away from ``v0``
    (found by the hypothesis property suite: low ``v0``, high ``theta``, long
    ``T`` broke put-call parity by ~1e-1). The mean integrated variance
    ``theta + (v0 - theta) * (1 - exp(-kappa T)) / (kappa T)`` accounts for
    the reversion; it equals ``v0`` exactly when ``v0 == theta``, so golden
    values at that point are unchanged.
    """
    kT = kappa * T
    w = (1.0 - jnp.exp(-kT)) / jnp.maximum(kT, 1e-12)
    return theta + (v0 - theta) * w


def _cover_strikes(a, b, strikes, margin: float = 0.25):
    """Expand the COS window so every strike's payoff kink lies inside it.

    If ``log(K)`` falls outside ``[a, b]`` the payoff cosine-coefficients are
    computed over a clipped domain and the returned "price" is unbounded junk
    (found by the property suite: deep-OTM strike beyond the grid). The clamp
    is inactive for any strike already covered, so at-the-money golden values
    are unchanged. ``stop_gradient`` for the same reason as the range itself.
    """
    log_k = jnp.log(strikes)
    lo = jax.lax.stop_gradient(jnp.minimum(a, jnp.min(log_k) - margin))
    hi = jax.lax.stop_gradient(jnp.maximum(b, jnp.max(log_k) + margin))
    return lo, hi


def fourier_price(
    S0, K, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    is_call: bool = True, N: int = 256, L: float = 12.0,
):
    """Differentiable Heston COS price for a single vanilla option.

    Mirrors ``jax_cos_price_heston`` (same math, same defaults) but stays fully
    inside JAX, so ``jax.grad``/``jax.jit``/``jax.vmap`` work with respect to any
    of the market arguments, and sizes the truncation grid from the
    mean-reversion-aware effective variance rather than ``v0`` alone.
    ``N`` and ``L`` must be static Python values.
    """
    mu = r - q
    log_S0 = jnp.log(S0)
    v_eff = _effective_variance(v0, kappa, theta, T)
    sigma_approx = jnp.sqrt(v_eff)
    a, b = _traced_truncation_range(log_S0, mu - 0.5 * v_eff, sigma_approx, T, L)
    a, b = _cover_strikes(a, b, K)
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    cf_vals = jax_heston_cf(params, u, log_S0, v0, T)
    return _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call)


def fourier_price_batch(
    S0, strikes, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    is_call, N: int = 256, L: float = 12.0,
):
    """Differentiable Heston COS prices across a strike vector.

    ``strikes`` and ``is_call`` are arrays of shape ``(M,)``; returns ``(M,)``.
    """
    mu = r - q
    log_S0 = jnp.log(S0)
    v_eff = _effective_variance(v0, kappa, theta, T)
    sigma_approx = jnp.sqrt(v_eff)
    a, b = _traced_truncation_range(log_S0, mu - 0.5 * v_eff, sigma_approx, T, L)
    a, b = _cover_strikes(a, b, jnp.asarray(strikes))
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    cf_vals = jax_heston_cf(params, u, log_S0, v0, T)
    return _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
    )
