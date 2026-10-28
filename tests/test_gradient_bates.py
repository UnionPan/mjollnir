"""Gradient-based risk-neutral Bates calibration: Heston with lognormal jumps."""

import numpy as np
import pytest

from mjollnir.calibration import fit_bates_surface
from mjollnir.calibration.risk_neutral.gradient_heston import fit_heston_surface
from mjollnir.jax import configure_runtime
from mjollnir.pricer._jax_fourier_pricer import jax_cos_price_bates

configure_runtime()

S0, R, Q = 100.0, 0.02, 0.0
# calm diffusion, and crash risk that steepens the short-dated skew
TRUE = dict(v0=0.03, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.6, lambda_j=0.8, mu_j=-0.12, sigma_j=0.08)
EXPIRIES = (0.04, 0.1, 0.25, 0.5, 1.0)


def _surface():
    quotes = []
    for T in EXPIRIES:
        for k in np.linspace(80.0, 115.0, 8):
            is_call = bool(k >= S0)  # out of the money
            price = float(jax_cos_price_bates(S0, float(k), T, R, Q, TRUE["v0"], TRUE["kappa"], TRUE["theta"],
                                              TRUE["sigma_v"], TRUE["rho"], TRUE["lambda_j"], TRUE["mu_j"],
                                              TRUE["sigma_j"], is_call=is_call, N=1024))
            quotes.append((float(k), T, is_call, price))
    return quotes


@pytest.fixture(scope="module")
def surface():
    return _surface()


def test_a_bates_surface_is_fitted_and_the_paramset_is_bates(surface):
    res = fit_bates_surface(surface, S0, R, Q, asset="TEST")
    assert res.converged and res.param_set.model == "bates" and res.param_set.measure == "Q"
    assert set(res.param_set.params) == set(TRUE)
    assert res.iv_rmse < 1e-3  # a tenth of a vol point
    for name, true in TRUE.items():  # every parameter, jumps included, comes back
        assert res.param_set.params[name] == pytest.approx(true, rel=2e-2), name


def test_jumps_fit_a_crash_skew_heston_has_to_strain_for(surface):
    bates = fit_bates_surface(surface, S0, R, Q)
    heston = fit_heston_surface(surface, S0, R, Q)
    assert bates.iv_rmse < 0.5 * heston.iv_rmse
