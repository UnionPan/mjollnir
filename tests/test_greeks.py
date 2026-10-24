"""Tests for the fast Heston COS delta."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.jax import configure_runtime, fourier_price, heston_cos_delta

configure_runtime()

P = {"kappa": 1.7, "theta": 0.0743, "sigma_v": 0.53, "rho": -0.7}
R, Q, K = 0.03, 0.01, 100.0
V_TOP = 4 * 0.15


def _fast(spots, variances, tau, is_call, **kw):
    f = jax.vmap(lambda s, v: heston_cos_delta(s, v, tau, K, R, Q, P["kappa"], P["theta"],
                                               P["sigma_v"], P["rho"], is_call, V_TOP, **kw))
    return np.asarray(f(spots, variances))


def _reference(spots, variances, tau, is_call):
    def price(s, v):
        return fourier_price(s, K, tau, R, Q, v, P["kappa"], P["theta"], P["sigma_v"], P["rho"],
                             is_call, N=512)
    return np.asarray(jax.vmap(jax.grad(price))(spots, variances))


@pytest.mark.parametrize("days", [1, 5, 21, 63])
@pytest.mark.parametrize("is_call", [True, False])
def test_matches_autodiff_of_the_cos_price(days, is_call):
    tau = days / 252
    rng = np.random.default_rng(days)
    spots = jnp.asarray(K * np.exp(rng.normal(0, 3 * np.sqrt(0.15 * tau), 512)))
    variances = jnp.asarray(rng.uniform(0.005, 0.6, 512))  # spans the whole window ladder
    err = np.abs(_fast(spots, variances, tau, is_call) - _reference(spots, variances, tau, is_call))
    assert err.max() < 2e-4


def test_deep_in_and_out_of_the_money_limits():
    tau = 21 / 252
    spots = jnp.array([20.0, 500.0])
    variances = jnp.full(2, 0.04)
    np.testing.assert_allclose(_fast(spots, variances, tau, True), [0.0, np.exp(-Q * tau)], atol=1e-12)
    np.testing.assert_allclose(_fast(spots, variances, tau, False), [-np.exp(-Q * tau), 0.0], atol=1e-12)


def test_put_call_parity():
    tau = 0.25
    spots, variances = jnp.linspace(80, 120, 9), jnp.full(9, 0.06)
    np.testing.assert_allclose(_fast(spots, variances, tau, True) - _fast(spots, variances, tau, False),
                               np.exp(-Q * tau), atol=1e-12)


def test_jittable_with_traced_time_to_maturity():
    f = jax.jit(lambda tau: heston_cos_delta(100.0, 0.05, tau, K, R, Q, P["kappa"], P["theta"],
                                             P["sigma_v"], P["rho"], True, V_TOP))
    assert 0.4 < float(f(30 / 252)) < 0.7
