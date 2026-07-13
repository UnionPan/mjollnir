"""Tests for gradient-based risk-neutral Heston calibration."""

import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.calibration.risk_neutral.gradient_heston import fit_heston_surface
from mjollnir.jax import configure_runtime, fourier_price_batch

configure_runtime()

TRUE = dict(v0=0.05, kappa=2.5, theta=0.06, sigma_v=0.45, rho=-0.65)
S0, R, Q = 100.0, 0.02, 0.0


def _surface(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    quotes = []
    for T in (0.1, 0.25, 0.5, 1.0):
        strikes = jnp.asarray(np.linspace(80, 120, 9))
        is_call = jnp.asarray(strikes >= S0)   # OTM convention
        mids = fourier_price_batch(S0, strikes, T, R, Q, **TRUE, is_call=is_call)
        mids = np.asarray(mids) * (1.0 + noise * rng.standard_normal(len(strikes)))
        quotes += [(float(k), T, bool(c), float(m))
                   for k, c, m in zip(strikes, is_call, mids, strict=True)]
    return quotes


class TestGradientHeston:
    def test_exact_recovery(self):
        """Noise-free surface: every parameter recovered to <0.5%."""
        res = fit_heston_surface(_surface(), S0, R, Q, asset="TEST")
        assert res.converged
        assert res.rmse < 1e-4
        for k, true in TRUE.items():
            assert res.param_set.params[k] == pytest.approx(true, rel=5e-3), k

    def test_noisy_surface_stays_sane(self):
        """0.5% multiplicative price noise: fit remains in the right basin."""
        res = fit_heston_surface(_surface(noise=0.005, seed=3), S0, R, Q)
        assert res.converged
        p = res.param_set.params
        assert p["v0"] == pytest.approx(TRUE["v0"], rel=0.25)
        assert p["rho"] == pytest.approx(TRUE["rho"], abs=0.15)
        assert res.rmse < 0.2

    def test_returns_q_measure_paramset(self):
        """The artifact contract: Q-measure, model heston, hash-stable."""
        res = fit_heston_surface(_surface(), S0, R, Q, asset="SPY")
        ps = res.param_set
        assert ps.measure == "Q" and ps.model == "heston" and ps.asset == "SPY"
        assert ps.content_hash()  # persistable artifact

    def test_empty_quotes_raise(self):
        with pytest.raises(ValueError, match="no quotes"):
            fit_heston_surface([], S0, R, Q)
