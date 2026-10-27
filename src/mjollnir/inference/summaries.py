"""Summary statistics of a daily return history, for amortized inference.

One implementation serves training (millions of simulated histories on a GPU)
and inference (a visitor's real history), so the network never sees a
mismatch. Histories have different lengths: each is a row padded to
``max_len`` with a mask, and every statistic is a masked average.

Returns are demeaned first: the drift is not identified by a few years of
closes (the drift policy reports it separately), and the posterior must not
lean on it. Each statistic targets something the models differ in:

- the level of volatility and its distribution over time (theta, regimes);
- how fast volatility clustering decays: autocorrelations of r^2 and |r|
  across lags (kappa, GARCH persistence);
- leverage: today's return against later volatility (rho);
- tails and outliers (Merton's jumps);
- the variogram of log|r| across lags (roughness H, vol-of-vol);
- the history's length, so the posterior widens on short ones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["FEATURE_NAMES", "pad_returns", "summary_features"]

ACF_SQ_LAGS = (1, 2, 5, 10, 21, 63)
ACF_ABS_LAGS = (1, 5, 21, 63)
VARIOGRAM_LAGS = (1, 2, 5, 10, 21)
DT = 1.0 / 252

FEATURE_NAMES = (
    ["log_vol", "skew", "log_kurtosis"]
    + [f"acf_sq_{k}" for k in ACF_SQ_LAGS]
    + [f"acf_abs_{k}" for k in ACF_ABS_LAGS]
    + ["leverage_1", "leverage_5", "leverage_21"]
    + ["log_rv21_sd", "log_rv5_sd", "rv21_q90_over_q10"]
    + ["tail_3sd", "tail_5sd", "max_abs_over_sd"]
    + [f"variogram_{k}" for k in VARIOGRAM_LAGS]
    + ["log_n"]
)


def pad_returns(histories: list[np.ndarray], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack histories of different lengths (most recent ``max_len`` kept) with masks."""
    out = np.zeros((len(histories), max_len))
    mask = np.zeros((len(histories), max_len))
    for i, r in enumerate(histories):
        r = np.asarray(r, dtype=float)[-max_len:]
        out[i, : r.size] = r
        mask[i, : r.size] = 1.0
    return out, mask


def _masked_mean(x, mask):
    return jnp.sum(x * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1.0)


def _lagged_corr(x, y, mask, lag):
    """Masked correlation of x_t with y_{t+lag}."""
    a, b = x[..., :-lag], y[..., lag:]
    m = mask[..., :-lag] * mask[..., lag:]
    ma, mb = _masked_mean(a, m), _masked_mean(b, m)
    da, db = a - ma[..., None], b - mb[..., None]
    cov = _masked_mean(da * db, m)
    va, vb = _masked_mean(da**2, m), _masked_mean(db**2, m)
    return cov / jnp.sqrt(jnp.maximum(va * vb, 1e-30))


def _rolling_mean(x, mask, window):
    """Trailing ``window``-day mean of x, valid where the whole window lies in the history."""
    c = jnp.cumsum(x * mask, axis=-1)
    n = jnp.cumsum(mask, axis=-1)
    shifted_c = jnp.concatenate([jnp.zeros((*c.shape[:-1], window)), c[..., :-window]], axis=-1)
    shifted_n = jnp.concatenate([jnp.zeros((*n.shape[:-1], window)), n[..., :-window]], axis=-1)
    full = (n - shifted_n) >= window
    return (c - shifted_c) / window, full.astype(x.dtype) * mask


def _masked_quantile_ratio(x, valid):
    """q90 / q10 of x over valid entries (sorting pushes invalid entries to the end)."""
    big = jnp.where(valid > 0, x, jnp.inf)
    s = jnp.sort(big, axis=-1)
    n = jnp.sum(valid, axis=-1)
    q10 = jnp.take_along_axis(s, jnp.floor(0.1 * (n - 1)).astype(jnp.int32)[..., None], axis=-1)[..., 0]
    q90 = jnp.take_along_axis(s, jnp.floor(0.9 * (n - 1)).astype(jnp.int32)[..., None], axis=-1)[..., 0]
    return jnp.log(jnp.maximum(q90, 1e-30) / jnp.maximum(q10, 1e-30))


@jax.jit
def summary_features(returns, mask):
    """(B, F) summaries of (B, T) daily log returns with (B, T) masks."""
    returns, mask = jnp.asarray(returns), jnp.asarray(mask)
    r = (returns - _masked_mean(returns, mask)[..., None]) * mask
    var = _masked_mean(r**2, mask)
    sd = jnp.sqrt(jnp.maximum(var, 1e-30))
    z = r / sd[..., None]
    sq, ab = r**2, jnp.abs(r)

    feats = [
        0.5 * jnp.log(jnp.maximum(var / DT, 1e-30)),
        _masked_mean(z**3, mask),
        jnp.log(jnp.maximum(_masked_mean(z**4, mask), 1e-6)),
    ]
    feats += [_lagged_corr(sq, sq, mask, k) for k in ACF_SQ_LAGS]
    feats += [_lagged_corr(ab, ab, mask, k) for k in ACF_ABS_LAGS]
    feats += [_lagged_corr(r, sq, mask, k) for k in (1,)]
    rv5, valid5 = _rolling_mean(sq, mask, 5)
    rv21, valid21 = _rolling_mean(sq, mask, 21)
    # leverage over a horizon: today's return against the realized variance of the next days
    for horizon, rv, valid in ((5, rv5, valid5), (21, rv21, valid21)):
        future = jnp.concatenate([rv[..., horizon:], jnp.zeros((*rv.shape[:-1], horizon))], axis=-1)
        fvalid = jnp.concatenate([valid[..., horizon:], jnp.zeros((*valid.shape[:-1], horizon))], axis=-1)
        feats.append(_lagged_corr(r * fvalid, future * fvalid, fvalid * mask, 1))
    log_rv21 = jnp.log(jnp.maximum(rv21, 1e-30))
    log_rv5 = jnp.log(jnp.maximum(rv5, 1e-30))
    feats += [
        jnp.sqrt(jnp.maximum(_masked_mean(log_rv21**2, valid21) - _masked_mean(log_rv21, valid21) ** 2, 0.0)),
        jnp.sqrt(jnp.maximum(_masked_mean(log_rv5**2, valid5) - _masked_mean(log_rv5, valid5) ** 2, 0.0)),
        _masked_quantile_ratio(rv21, valid21),
        _masked_mean((jnp.abs(z) > 3.0).astype(r.dtype), mask),
        _masked_mean((jnp.abs(z) > 5.0).astype(r.dtype), mask),
        jnp.log(jnp.max(jnp.abs(z) * mask, axis=-1)),
    ]
    # log|r| variogram; exact zeros (stale prices) are left out of the pairs
    nonzero = mask * (ab > 1e-12)
    log_abs = jnp.log(jnp.where(nonzero > 0, ab, 1.0))
    for k in VARIOGRAM_LAGS:
        m = nonzero[..., :-k] * nonzero[..., k:]
        d = log_abs[..., k:] - log_abs[..., :-k]
        feats.append(_masked_mean(d**2, m) - _masked_mean(d, m) ** 2)
    feats.append(jnp.log(jnp.maximum(jnp.sum(mask, axis=-1), 1.0)))
    return jnp.stack(feats, axis=-1)
