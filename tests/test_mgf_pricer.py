"""The MGF grid pricers against closed forms and a fine COS reference, short dates and wings included."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from mjollnir.jax import configure_runtime, fourier_price
from mjollnir.model_pricing import merton_price
from mjollnir.pricer._jax_mgf_pricer import (
    bs_price_grid_jax,
    heston_price_grid_jax,
    heston_price_slice_fast,
    merton_price_grid_jax,
)

configure_runtime()

S, R, Q = 100.0, 0.03, 0.01
STRIKES = np.linspace(50.0, 250.0, 81)
CALLS = np.ones_like(STRIKES, dtype=bool)


def black_scholes(strikes, T, sigma, is_call=True):
    forward, discount = S * np.exp((R - Q) * T), np.exp(-R * T)
    d1 = (np.log(forward / strikes) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    call = discount * (forward * norm.cdf(d1) - strikes * norm.cdf(d1 - sigma * np.sqrt(T)))
    return call if is_call else call - discount * (forward - strikes)


@pytest.mark.parametrize("T, sigma", [(1 / 252, 0.2), (0.05, 0.2), (0.5, 0.2), (2.0, 0.3), (5.0, 0.6), (1.0, 0.08)])
def test_black_scholes_from_one_day_to_five_years(T, sigma):
    got = np.asarray(bs_price_grid_jax(S, jnp.asarray(STRIKES), jnp.asarray(CALLS), T, R, Q, sigma))
    # a one-day call struck at 250 is worth 1e-300: the deep wing must not read a cancellation error
    np.testing.assert_allclose(got, black_scholes(STRIKES, T, sigma), rtol=0, atol=1e-10)


HESTON = [  # v0, theta, kappa, xi, rho, T
    (0.04, 0.04, 2.0, 0.5, -0.7, 0.5),
    (0.04, 0.06, 1.0, 1.0, -0.7, 2.0),
    (0.09, 0.04, 0.5, 1.5, 0.3, 3.0),
    (0.02, 0.04, 3.0, 0.4, -0.5, 1 / 252),
    (0.23, 0.064, 102.0, 4.8, -0.45, 0.03),  # a fitted short-dated smile: extreme mean reversion and vol of vol
    (0.01, 0.02, 0.3, 0.2, -0.9, 5.0),
]


@pytest.mark.parametrize("v0, theta, kappa, xi, rho, T", HESTON)
def test_heston_matches_a_fine_cos_reference(v0, theta, kappa, xi, rho, T):
    strikes = np.linspace(60.0, 160.0, 41)
    got = np.asarray(heston_price_grid_jax(S, jnp.asarray(strikes), jnp.ones(41, bool), T, R, Q, v0, theta, kappa,
                                           xi, rho))
    reference = np.array([float(fourier_price(S, k, T, R, Q, v0, kappa, theta, xi, rho, True, N=4096))
                          for k in strikes])
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-6)


def test_heston_put_call_parity_and_types():
    strikes = np.array([80.0, 100.0, 120.0])
    T, params = 0.5, (0.04, 0.04, 2.0, 0.5, -0.7)
    calls = heston_price_slice_fast(S, strikes, T, R, Q, *params, ["call"] * 3)
    puts = heston_price_slice_fast(S, strikes, T, R, Q, *params, ["put"] * 3)
    np.testing.assert_allclose(calls - puts, np.exp(-R * T) * (S * np.exp((R - Q) * T) - strikes), atol=1e-10)


def test_moment_explosion_falls_back_to_lewis_contour():
    # moments of order 1.25 explode within ten years at this vol of vol and correlation
    v0, theta, kappa, xi, rho, T = 0.09, 0.04, 0.5, 3.0, 0.9, 10.0
    strikes = np.array([60.0, 100.0, 160.0])
    got = np.asarray(heston_price_grid_jax(S, jnp.asarray(strikes), jnp.ones(3, bool), T, R, Q, v0, theta, kappa,
                                           xi, rho))
    forward, discount = S * np.exp((R - Q) * T), np.exp(-R * T)
    assert np.all(np.isfinite(got))
    assert np.all(got >= np.maximum(discount * (forward - strikes), 0.0) - 1e-9) and np.all(got <= S)


@pytest.mark.parametrize("T", [1 / 252, 0.25, 2.0])
def test_merton_matches_its_series(T):
    sigma, lam, mu_j, sigma_j = 0.15, 3.0, -0.08, 0.1
    got = np.asarray(merton_price_grid_jax(S, jnp.asarray(STRIKES), jnp.asarray(CALLS), T, R, Q, sigma, lam, mu_j,
                                           sigma_j))
    reference = np.array([float(merton_price(S, k, T, R, Q, sigma, lam, mu_j, sigma_j, True)) for k in STRIKES])
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-8)


def test_a_pure_jump_merton_prices():
    # No diffusion: the grid takes its scale from the jumps. The no-jump outcome is a point mass, whose
    # transform never decays, so the truncated inversion is good to 1e-4 here, not 1e-8.
    T, lam, mu_j, sigma_j = 0.5, 5.0, -0.05, 0.1
    strikes = np.array([90.0, 100.0, 110.0])
    got = np.asarray(merton_price_grid_jax(S, jnp.asarray(strikes), jnp.ones(3, bool), T, R, Q, 0.0, lam, mu_j,
                                           sigma_j))
    reference = np.array([float(merton_price(S, k, T, R, Q, 1e-12, lam, mu_j, sigma_j, True)) for k in strikes])
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-4)
