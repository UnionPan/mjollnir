"""Gradient-based risk-neutral Heston calibration.

Where the classic :class:`HestonCalibrator` drives scipy with finite-difference
gradients, this calibrator differentiates the library's own COS pricer exactly
and solves the least-squares problem with Levenberg-Marquardt: a forward-mode
Jacobian through :func:`mjollnir.jax.fourier_price_batch`, a damped
Gauss-Newton step, a fixed number of iterations.

The cost of a fit is fixed by construction, which matters wherever a stranger
can ask for one (a public API):

- Quotes are padded to a size bucket and every market input is a traced
  argument, so one compiled program serves every chain of that size; nothing
  recompiles per request.
- Multi-start runs in parallel under ``vmap``, each start for a fixed number of
  iterations.

Parameters are fitted in unconstrained coordinates (softplus for the positive
ones, tanh for rho). The Gauss-Newton covariance at the optimum, mapped through
those transforms, gives standard errors: a chain that cannot separate kappa
from theta says so with a large error rather than a confident number.

The result is returned as a Q-measure :class:`~mjollnir.params.ParamSet`, so a
calibration plugs directly into the artifact store, the scenario library, and
the simulation kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from mjollnir.params import ParamSet
from mjollnir.pricer._jax_fourier_pricer import (
    HestonCFParams,
    _cos_cf_weights,
    _cos_payoff_coefficients,
    jax_heston_cf,
)
from mjollnir.pricer._jax_iv import implied_vol_batch_np

__all__ = ["BUCKETS", "MATURITY_BUCKETS", "MAX_QUOTES", "GradientHestonResult", "fit_heston_surface"]

PARAM_NAMES = ("v0", "kappa", "theta", "sigma_v", "rho")
BUCKETS = (16, 32, 64, 128, 256, 512)
MAX_QUOTES = BUCKETS[-1]
MATURITY_BUCKETS = (4, 8, 16, 32, 64)
_STRIKE_MARGIN = 0.25  # log-strike margin of each expiry's COS window
_COS_L = 12.0
_MAX_ITERATIONS = 40  # hard cap: the worst-case cost of a fit
_STALL_ITERATIONS = 3  # stop after this many steps in a row without real progress
_PROGRESS = 1e-9  # relative cost reduction that counts as progress
_COS_TERMS = 256


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


def _constrain_vec(u):
    return jnp.stack(_constrain(u))


@dataclass(frozen=True)
class GradientHestonResult:
    param_set: ParamSet
    rmse: float           # root of the weighted mean-square price error at the optimum
    n_quotes: int
    converged: bool       # all params finite and the fit moved off its start
    std_errors: dict[str, float] = field(default_factory=dict)  # Gauss-Newton, per parameter
    iv_rmse: float | None = None  # model vs market implied vol, over quotes both invert


def _windows(S0, r, q, expiries, lo_logk, hi_logk, window_vol):
    """A fixed COS window per expiry, sized from the market, covering its strikes.

    Fixing the window makes the payoff coefficients independent of the model
    parameters, so they are computed once per fit; ``window_vol`` is generous
    (well above the chain's implied vols) and ``_COS_L`` wide, so a window holds
    every density the fit can wander through.
    """
    centre = np.log(S0) + (r - q - 0.5 * window_vol**2) * expiries
    half = _COS_L * window_vol * np.sqrt(expiries)
    return (np.minimum(centre - half, lo_logk - _STRIKE_MARGIN),
            np.maximum(centre + half, hi_logk + _STRIKE_MARGIN))


def _model_prices(u, S0, coefficients, discounts, parity, quote_expiry, expiries, a, b, r, q):
    """COS price of every quote: the characteristic function once per expiry (a
    chain has a handful), then one dot product per quote with its precomputed
    put coefficients; calls add their parity term (bounded put payoffs keep
    wide windows accurate)."""
    v0, kappa, theta, sigma_v, rho = _constrain(u)
    cf_params = HestonCFParams(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho, mu=r - q)
    k = jnp.arange(_COS_TERMS)

    def expiry_weights(T, a_e, b_e):
        cf = jax_heston_cf(cf_params, k * jnp.pi / (b_e - a_e), jnp.log(S0), v0, T)
        return _cos_cf_weights(cf, a_e, b_e)

    weights = jax.vmap(expiry_weights)(expiries, a, b)          # (expiries, N)
    return discounts * jnp.sum(weights[quote_expiry] * coefficients, axis=-1) + parity


def _put_coefficients(strikes, quote_expiry, a, b):
    def one(K, e):
        return _cos_payoff_coefficients(K[None], a[e], b[e], jnp.zeros(1, dtype=bool), _COS_TERMS)[0]

    return jax.jit(jax.vmap(one))(strikes, quote_expiry)


@cache
def _fitter(quote_bucket: int, expiry_bucket: int):
    """The compiled multi-start Levenberg-Marquardt fit for one pair of size buckets."""
    del quote_bucket, expiry_bucket  # the cache key: shapes, hence one compile, per pair

    def fit(starts, S0, market, mids, sqrt_w, r, q):
        def residuals(u):
            return sqrt_w * (_model_prices(u, S0, *market, r, q) - mids)

        def cost(u):
            res = residuals(u)
            return 0.5 * jnp.dot(res, res)

        def solve(u0):
            def step(carry):
                u, lam, c, it, stalled = carry
                res = residuals(u)
                jac = jax.jacfwd(residuals)(u)
                hess = jac.T @ jac
                damped = hess + lam * jnp.diag(jnp.diag(hess) + 1e-12)
                u_new = u - jnp.linalg.solve(damped, jac.T @ res)
                c_new = cost(u_new)
                better = jnp.isfinite(c_new) & (c_new < c)
                progress = better & (c - c_new > _PROGRESS * jnp.maximum(c, 1e-300))
                return (jnp.where(better, u_new, u),
                        jnp.clip(jnp.where(better, lam / 3.0, lam * 4.0), 1e-9, 1e9),
                        jnp.where(better, c_new, c),
                        it + 1,
                        jnp.where(progress, 0, stalled + 1))

            def running(carry):
                *_, it, stalled = carry
                return (it < _MAX_ITERATIONS) & (stalled < _STALL_ITERATIONS)

            start = (u0, jnp.asarray(1e-2), cost(u0), jnp.asarray(0), jnp.asarray(0))
            u, _, c, _, _ = jax.lax.while_loop(running, step, start)
            return u, c

        us, costs = jax.vmap(solve)(starts)
        best = jnp.argmin(jnp.where(jnp.isfinite(costs), costs, jnp.inf))
        u = us[best]
        jac = jax.jacfwd(residuals)(u)
        return u, costs[best], jac.T @ jac, jax.jacfwd(_constrain_vec)(u)

    return jax.jit(fit)


def _atm_variance(quotes, S0, r, q) -> float:
    """Implied variance of the quote nearest the money (a start for v0 and theta)."""
    k, t, c, m = min(quotes, key=lambda x: (abs(np.log(x[0] / S0)), x[1]))
    iv = implied_vol_batch_np(np.array([S0]), np.array([k]), np.array([t]), np.array([r]),
                              np.array([q]), np.array([m]), np.array([c]))[0]
    return float(iv**2) if np.isfinite(iv) else 0.04


def fit_heston_surface(
    quotes,
    S0: float,
    r: float,
    q: float = 0.0,
    *,
    weights=None,
    asset: str | None = None,
    source: str = "gradient COS calibration",
    buckets: tuple[int, int] | None = None,
) -> GradientHestonResult:
    """Calibrate Heston to an option surface with exact gradients.

    Args:
        quotes: iterable of ``(strike, maturity_years, is_call, mid_price)``
            tuples (e.g. built from an ``OptionChain``); at most ``MAX_QUOTES``.
        S0: spot price.
        r: risk-free rate (continuously compounded).
        q: dividend yield.
        weights: optional per-quote weights on squared price errors (default:
            equal); ``1 / vega**2`` makes the fit approximately one in implied vol.
        asset: recorded on the resulting ParamSet.
        source: provenance string for the ParamSet.
        buckets: fixed ``(quotes, expiries)`` padding, from ``BUCKETS`` and
            ``MATURITY_BUCKETS``; a service that pins them compiles one program
            and warms it once. Default: the smallest buckets that fit.

    Returns:
        :class:`GradientHestonResult` with a Q-measure ParamSet.
    """
    quotes = list(quotes)
    if not quotes:
        raise ValueError("no quotes supplied")
    if len(quotes) > MAX_QUOTES:
        raise ValueError(f"at most {MAX_QUOTES} quotes per fit, got {len(quotes)}")
    n = len(quotes)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, non-negative, one per quote")

    arr = np.array([(k, t, float(c), m) for k, t, c, m in quotes], float)
    expiries, quote_expiry = np.unique(np.round(arr[:, 1], 10), return_inverse=True)
    if len(expiries) > MATURITY_BUCKETS[-1]:
        raise ValueError(f"at most {MATURITY_BUCKETS[-1]} expiries per fit, got {len(expiries)}")
    log_k = np.log(arr[:, 0])
    lo_logk = np.array([log_k[quote_expiry == e].min() for e in range(len(expiries))])
    hi_logk = np.array([log_k[quote_expiry == e].max() for e in range(len(expiries))])
    spot, rate, div = (np.full(n, x) for x in (S0, r, q))
    calls = arr[:, 2] > 0.5
    iv_market = implied_vol_batch_np(spot, arr[:, 0], arr[:, 1], rate, div, arr[:, 3], calls)
    finite_iv = iv_market[np.isfinite(iv_market)]
    window_vol = float(np.clip(2.0 * (finite_iv.max() if finite_iv.size else 0.5), 0.3, 4.0))
    if buckets is None:
        quote_bucket = next(b for b in BUCKETS if b >= n)
        expiry_bucket = next(b for b in MATURITY_BUCKETS if b >= len(expiries))
    else:
        quote_bucket, expiry_bucket = buckets
        if quote_bucket not in BUCKETS or expiry_bucket not in MATURITY_BUCKETS:
            raise ValueError(f"buckets must come from {BUCKETS} and {MATURITY_BUCKETS}")
        if n > quote_bucket or len(expiries) > expiry_bucket:
            raise ValueError(f"{n} quotes over {len(expiries)} expiries do not fit buckets {buckets}")
    # pads repeat the first quote (zero weight) and the first expiry: finite and inert
    pad = quote_bucket - n
    arr = np.vstack([arr, np.repeat(arr[:1], pad, axis=0)])
    quote_expiry = np.concatenate([quote_expiry, np.full(pad, quote_expiry[0])])
    extra = expiry_bucket - len(expiries)
    expiries, lo_logk, hi_logk = (np.concatenate([x, np.full(extra, x[0])]) for x in (expiries, lo_logk, hi_logk))
    sqrt_w = np.sqrt(np.concatenate([w / max(w.mean(), 1e-300), np.zeros(pad)]))
    a, b = _windows(S0, r, q, expiries, lo_logk, hi_logk, window_vol)
    coefficients = _put_coefficients(jnp.asarray(arr[:, 0]), jnp.asarray(quote_expiry),
                                     jnp.asarray(a), jnp.asarray(b))
    # call = put + S0 exp(-qT) - K exp(-rT)
    parity = np.where(arr[:, 2] > 0.5, S0 * np.exp(-q * arr[:, 1]) - arr[:, 0] * np.exp(-r * arr[:, 1]), 0.0)
    market = (coefficients, jnp.asarray(np.exp(-r * arr[:, 1])), jnp.asarray(parity), jnp.asarray(quote_expiry),
              jnp.asarray(expiries), jnp.asarray(a), jnp.asarray(b))

    var0 = _atm_variance(quotes, S0, r, q)
    starts = jnp.asarray([
        [_inv_softplus(scale * var0), _inv_softplus(2.0), _inv_softplus(var0),
         _inv_softplus(0.5), float(np.arctanh(rho0 / 0.999))]
        for scale in (0.5, 1.0, 2.0) for rho0 in (-0.7, -0.2)
    ])

    u, best_cost, hess, dparams = _fitter(quote_bucket, expiry_bucket)(
        starts, float(S0), market, jnp.asarray(arr[:, 3]), jnp.asarray(sqrt_w), float(r), float(q),
    )
    values = np.asarray(_constrain_vec(u), float)
    params = dict(zip(PARAM_NAMES, (float(x) for x in values), strict=True))
    rmse = float(np.sqrt(2.0 * float(best_cost) / n))

    # Gauss-Newton covariance: s^2 (J'J)^-1 in u, then the delta method into params.
    dof = max(n - len(PARAM_NAMES), 1)
    s2 = 2.0 * float(best_cost) / dof
    try:
        cov_u = s2 * np.linalg.inv(np.asarray(hess, float))
        g = np.asarray(dparams, float)
        std = np.sqrt(np.clip(np.diag(g @ cov_u @ g.T), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(PARAM_NAMES), np.inf)
    std_errors = dict(zip(PARAM_NAMES, (float(x) for x in std), strict=True))

    model = np.asarray(_model_prices(u, float(S0), *market, float(r), float(q)))[:n]
    iv_model = implied_vol_batch_np(spot, arr[:n, 0], arr[:n, 1], rate, div, model, calls)
    both = np.isfinite(iv_model) & np.isfinite(iv_market)
    iv_rmse = float(np.sqrt(np.mean((iv_model[both] - iv_market[both]) ** 2))) if both.any() else None

    converged = bool(np.all(np.isfinite(values)) and params["sigma_v"] > 1e-4 and np.isfinite(best_cost))
    ps = ParamSet.create("heston", "Q", params, asset=asset, source=source)
    return GradientHestonResult(
        param_set=ps, rmse=rmse, n_quotes=n, converged=converged,
        std_errors=std_errors, iv_rmse=iv_rmse,
    )
