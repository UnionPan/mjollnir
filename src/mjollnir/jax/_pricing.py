"""Differentiable Heston COS pricing for the public ``mjollnir.jax`` facade.

``mjollnir.pricer._jax_fourier_pricer.jax_cos_price_heston`` is a numpy-orchestrated
convenience wrapper: it takes ``np.log`` of the spot and casts the result to ``float``,
so it cannot be traced by ``jax.grad``/``jax.jit``. Both share the COS window,
:func:`~mjollnir.pricer._jax_fourier_pricer.heston_cos_range`.
The RL layer historically worked around this by composing the private internals
(``jax_heston_cf`` + ``_jax_cos_price_multi``) itself.

This module is that composition, done once, correctly, in pure ``jax.numpy`` — the
truncation range is traced too, so the price is differentiable with respect to *all*
market inputs (``S0``, ``v0``, and every Heston parameter) and safe inside
``jit``/``vmap``/``lax.scan``. For concrete (non-traced) inputs it reproduces
``jax_cos_price_heston`` exactly (asserted by ``tests/test_parity_golden.py``).
"""

from __future__ import annotations

import jax.numpy as jnp

from mjollnir.pricer._jax_fourier_pricer import (
    HESTON_COS_TERMS,
    HestonCFParams,
    _jax_cos_price_multi,
    _jax_cos_price_single,
    heston_cos_range,
    jax_heston_cf,
)

__all__ = ["fourier_price", "fourier_price_batch"]


def fourier_price(
    S0, K, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    is_call: bool = True, N: int = HESTON_COS_TERMS, L: float = 12.0,
):
    """Differentiable Heston COS price for a single vanilla option.

    Same math and defaults as ``jax_cos_price_heston`` (the window from
    :func:`heston_cos_range`), but fully inside JAX, so ``jax.grad``/
    ``jax.jit``/``jax.vmap`` work with respect to any of the market arguments.
    ``N`` and ``L`` must be static Python values.
    """
    mu = r - q
    log_S0 = jnp.log(S0)
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    a, b = heston_cos_range(params, log_S0, v0, T, L, strikes=jnp.reshape(K, (1,)))
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    cf_vals = jax_heston_cf(params, u, log_S0, v0, T)
    return _jax_cos_price_single(cf_vals, K, T, r, a, b, is_call, forward_pv=S0 * jnp.exp(-q * T))


def fourier_price_batch(
    S0, strikes, T, r, q,
    v0, kappa, theta, sigma_v, rho,
    is_call, N: int = HESTON_COS_TERMS, L: float = 12.0,
):
    """Differentiable Heston COS prices across a strike vector.

    ``strikes`` and ``is_call`` are arrays of shape ``(M,)``; returns ``(M,)``.
    """
    mu = r - q
    log_S0 = jnp.log(S0)
    params = HestonCFParams(v0=v0, kappa=kappa, theta=theta,
                            sigma_v=sigma_v, rho=rho, mu=mu)
    a, b = heston_cos_range(params, log_S0, v0, T, L, strikes=jnp.asarray(strikes))
    k = jnp.arange(N)
    u = k * jnp.pi / (b - a)
    cf_vals = jax_heston_cf(params, u, log_S0, v0, T)
    return _jax_cos_price_multi(
        cf_vals, jnp.asarray(strikes), T, r, a, b, jnp.asarray(is_call),
        forward_pv=S0 * jnp.exp(-q * T),
    )
