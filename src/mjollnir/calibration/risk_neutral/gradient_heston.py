"""Gradient-based risk-neutral Heston calibration.

Where the classic :class:`HestonCalibrator` drives scipy least-squares with
finite-difference gradients, this calibrator differentiates the library's own
COS pricer exactly: the loss is built from
:func:`mjollnir.jax.fourier_price_batch` and optimized with ``jax.grad`` —
the "third row" of the differentiation table (∂price/∂params) made into a
pipeline. Exact gradients mean no step-size tuning, no 2x5 extra pricings
per iteration, and jit-compiled end-to-end fitting.

Parameters are optimized in unconstrained space (softplus for the positive
ones, tanh for rho) with cosine-decayed Adam and multi-start on ``(v0, rho)``
— the same recipe that closed the batched-Merton likelihood gap.

The result is returned as a Q-measure :class:`~mjollnir.params.ParamSet`, so
a calibration plugs directly into the artifact store, the scenario library,
and the simulation kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mjollnir.jax._pricing import fourier_price_batch
from mjollnir.params import ParamSet

__all__ = ["GradientHestonResult", "fit_heston_surface"]

_N_STEPS = 3000
_SCHEDULE = optax.cosine_decay_schedule(init_value=5e-2, decay_steps=_N_STEPS)


def _softplus(x):
    return jnp.logaddexp(x, 0.0)


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def _constrain(u):
    """Unconstrained R^5 -> (v0, kappa, theta, sigma_v, rho)."""
    return (
        _softplus(u[0]),          # v0 > 0
        _softplus(u[1]),          # kappa > 0
        _softplus(u[2]),          # theta > 0
        _softplus(u[3]),          # sigma_v > 0
        0.999 * jnp.tanh(u[4]),   # rho in (-1, 1)
    )


@dataclass(frozen=True)
class GradientHestonResult:
    param_set: ParamSet
    rmse: float           # root-mean-square price error at the optimum
    n_quotes: int
    converged: bool       # all params finite and strictly inside bounds


def _make_loss(surfaces, S0, r, q):
    """Weighted MSE over the whole surface; jit/grad-safe.

    ``surfaces`` is a tuple of ``(T, strikes, is_call, mids, weights)`` per
    maturity — strikes vary per slice, so slices are summed in a Python loop
    (unrolled by jit; maturity counts are small).
    """

    def loss(u):
        v0, kappa, theta, sigma_v, rho = _constrain(u)
        total = 0.0
        total_w = 0.0
        for T, strikes, is_call, mids, w in surfaces:
            model = fourier_price_batch(
                S0, strikes, T, r, q, v0, kappa, theta, sigma_v, rho, is_call,
            )
            total = total + jnp.sum(w * (model - mids) ** 2)
            total_w = total_w + jnp.sum(w)
        return total / total_w

    return loss


def fit_heston_surface(
    quotes,
    S0: float,
    r: float,
    q: float = 0.0,
    *,
    weights=None,
    asset: str | None = None,
    source: str = "gradient COS calibration",
) -> GradientHestonResult:
    """Calibrate Heston to an option surface with exact gradients.

    Args:
        quotes: iterable of ``(strike, maturity_years, is_call, mid_price)``
            tuples (e.g. built from an ``OptionChain``).
        S0: spot price.
        r: risk-free rate (continuously compounded).
        q: dividend yield.
        weights: optional per-quote weights (default: equal).
        asset: recorded on the resulting ParamSet.
        source: provenance string for the ParamSet.

    Returns:
        :class:`GradientHestonResult` with a Q-measure ParamSet.
    """
    quotes = list(quotes)
    if not quotes:
        raise ValueError("no quotes supplied")
    w_all = np.ones(len(quotes)) if weights is None else np.asarray(weights, float)

    # group by maturity so each slice shares one CF evaluation
    by_T: dict[float, list[int]] = {}
    for i, (_, T, _, _) in enumerate(quotes):
        by_T.setdefault(round(float(T), 10), []).append(i)
    surfaces = []
    for T, idx in sorted(by_T.items()):
        surfaces.append((
            T,
            jnp.asarray([quotes[i][0] for i in idx], jnp.float64),
            jnp.asarray([bool(quotes[i][2]) for i in idx]),
            jnp.asarray([quotes[i][3] for i in idx], jnp.float64),
            jnp.asarray([w_all[i] for i in idx], jnp.float64),
        ))

    loss = _make_loss(tuple(surfaces), S0, r, q)
    optimizer = optax.adam(learning_rate=_SCHEDULE)

    @jax.jit
    def fit_from(u0):
        state = optimizer.init(u0)

        def step(carry, _):
            u, st = carry
            val, g = jax.value_and_grad(loss)(u)
            updates, st = optimizer.update(g, st)
            return (optax.apply_updates(u, updates), st), val

        (u, _), _ = jax.lax.scan(step, (u0, state), None, length=_N_STEPS)
        return u, loss(u)

    # multi-start over (v0, rho); kappa/theta/sigma_v share sane inits
    starts = []
    for v0_init in (0.02, 0.09):
        for rho_init in (-0.7, -0.2):
            starts.append(jnp.asarray([
                _inv_softplus(v0_init),
                _inv_softplus(2.0),
                _inv_softplus(0.05),
                _inv_softplus(0.4),
                float(np.arctanh(rho_init / 0.999)),
            ]))

    best_u, best_val = None, np.inf
    for u0 in starts:
        u, val = fit_from(u0)
        if float(val) < best_val:
            best_u, best_val = u, float(val)

    v0, kappa, theta, sigma_v, rho = (float(x) for x in _constrain(best_u))
    params = {"v0": v0, "kappa": kappa, "theta": theta,
              "sigma_v": sigma_v, "rho": rho}
    converged = all(np.isfinite(list(params.values()))) and sigma_v > 1e-4
    ps = ParamSet.create("heston", "Q", params, asset=asset, source=source)
    return GradientHestonResult(
        param_set=ps,
        rmse=float(np.sqrt(best_val)),
        n_quotes=len(quotes),
        converged=bool(converged),
    )
