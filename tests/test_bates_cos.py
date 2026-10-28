"""Bates COS prices against the independent MGF inversion, where a window from today's variance fails."""

import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.jax import configure_runtime
from mjollnir.pricer._jax_fourier_pricer import jax_cos_price_bates_multi
from mjollnir.pricer._jax_mgf_pricer import bates_price_grid_jax

configure_runtime()
S, R, Q = 100.0, 0.02, 0.0


@pytest.mark.parametrize("v0, T, lam, mu_j, sigma_j", [
    (1e-4, 0.25, 0.5, -0.15, 0.1),   # variance far below its mean: the window once came from sqrt(v0) alone
    (1e-4, 2.0, 1.0, -0.1, 0.1),
    (0.04, 0.05, 3.0, -0.08, 0.05),  # frequent jumps over a short date
    (0.09, 1.0, 0.2, -0.3, 0.2),     # rare large crashes
])
def test_bates_cos_matches_the_mgf_inversion(v0, T, lam, mu_j, sigma_j):
    kappa, theta, xi, rho = 2.0, 0.04, 0.4, -0.6
    strikes = np.linspace(60.0, 140.0, 17)
    calls = strikes >= S
    cos = jax_cos_price_bates_multi(S, strikes, T, R, Q, v0, kappa, theta, xi, rho, lam, mu_j, sigma_j, calls)
    mgf = np.asarray(bates_price_grid_jax(S, jnp.asarray(strikes), jnp.asarray(calls), T, R, Q, v0, theta, kappa, xi,
                                          rho, lam, mu_j, sigma_j))
    np.testing.assert_allclose(cos, mgf, rtol=0, atol=1e-6)
