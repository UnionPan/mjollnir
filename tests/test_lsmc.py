"""Tests for differentiable LSMC — financial laws, not just smoke."""

import jax
import jax.numpy as jnp
import pytest

from mjollnir.jax import configure_runtime, fourier_price
from mjollnir.jax._lsmc import lsmc_price, simulate_heston_paths

configure_runtime()

HESTON = dict(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
S0, V0, T, R = 100.0, 0.04, 0.5, 0.04
N_STEPS = 50
DT = T / N_STEPS


def _paths(n_paths=20_000, mu=R, seed=0):
    return simulate_heston_paths(jax.random.PRNGKey(seed), S0, V0, DT, mu,
                                 **HESTON, n_paths=n_paths, n_steps=N_STEPS)


class TestLSMCLaws:
    def test_american_put_dominates_european(self):
        """With r > 0, the Bermudan put carries a positive early-exercise
        premium over the European priced by the COS kernel."""
        K = 110.0   # ITM put: premium is material here
        spots, variances = _paths()
        bermudan = float(lsmc_price(spots, variances,
                                    lambda s: jnp.maximum(K - s, 0.0), DT, R))
        european = float(fourier_price(S0, K, T, R, 0.0, V0, **HESTON,
                                       is_call=False))
        assert bermudan > european - 0.05          # never (materially) below
        assert bermudan - european > 0.1           # and the premium is real

    def test_american_call_equals_european_no_dividends(self):
        """No dividends: early exercise of a call is never optimal, so the
        LSMC price must match the European within Monte Carlo error."""
        K = 100.0
        spots, variances = _paths()
        bermudan = float(lsmc_price(spots, variances,
                                    lambda s: jnp.maximum(s - K, 0.0), DT, R))
        european = float(fourier_price(S0, K, T, R, 0.0, V0, **HESTON,
                                       is_call=True))
        assert bermudan == pytest.approx(european, rel=0.03)

    def test_deterministic_per_key(self):
        spots, variances = _paths(n_paths=2_000, seed=7)
        f = lambda s: jnp.maximum(105.0 - s, 0.0)
        p1 = float(lsmc_price(spots, variances, f, DT, R))
        spots2, variances2 = _paths(n_paths=2_000, seed=7)
        p2 = float(lsmc_price(spots2, variances2, f, DT, R))
        assert p1 == p2

    def test_gradient_to_heston_params(self):
        """d(Bermudan price)/d(v0) exists, is finite and positive —
        early-exercise vega through the whole pipeline by autodiff."""
        K = 105.0

        def price(v0):
            spots, variances = simulate_heston_paths(
                jax.random.PRNGKey(3), S0, v0, DT, R, **HESTON,
                n_paths=4_000, n_steps=N_STEPS)
            return lsmc_price(spots, variances,
                              lambda s: jnp.maximum(K - s, 0.0), DT, R)

        g = jax.grad(price)(0.04)
        assert bool(jnp.isfinite(g)) and float(g) > 0.0

    def test_intrinsic_lower_bound(self):
        """Deep-ITM put: price at least intrinsic-ish (exercise now is a
        feasible policy)."""
        K = 130.0
        spots, variances = _paths(n_paths=5_000)
        p = float(lsmc_price(spots, variances,
                             lambda s: jnp.maximum(K - s, 0.0), DT, R))
        assert p > (K - S0) * 0.97
