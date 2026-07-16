"""Smoke + sanity tests for the cross-asset calibration package."""

import numpy as np

from mjollnir.calibration.cross_asset import (
    dcc_corr_path,
    fit_dcc,
    fit_factor_model,
)


def _panel(n_assets=8, n_obs=600, k_true=2, seed=0):
    """Synthetic return panel with a known 2-factor structure."""
    rng = np.random.default_rng(seed)
    factors = 0.01 * rng.standard_normal((n_obs, k_true))
    loadings = rng.uniform(0.5, 1.5, (n_assets, k_true))
    idio = 0.005 * rng.standard_normal((n_obs, n_assets))
    return factors @ loadings.T + idio, [f"A{i}" for i in range(n_assets)]


class TestFactorModel:
    def test_recovers_factor_dimension(self):
        returns, tickers = _panel()
        fm = fit_factor_model(returns, tickers)
        assert 1 <= fm.k <= 3          # MP-edge selection near the true k=2
        assert fm.loadings.shape == (8, fm.k)
        assert (fm.resid_var > 0).all()

    def test_covariance_decomposition(self):
        """The model's contract is Sigma ~= B Omega B' + D, with the factor
        part carrying most of the variance for a 2-factor panel."""
        returns, tickers = _panel()
        fm = fit_factor_model(returns, tickers, k=2)
        sample = np.cov(returns, rowvar=False)
        factor_part = fm.loadings @ fm.factor_cov @ fm.loadings.T
        model = factor_part + np.diag(fm.resid_var)
        # diagonal reproduced within 50%
        np.testing.assert_allclose(np.diag(model), np.diag(sample), rtol=0.5)
        # factors explain most of the trace
        assert np.trace(factor_part) > 0.6 * np.trace(sample)


class TestDCC:
    def test_fit_and_corr_path(self):
        rng = np.random.default_rng(1)
        T = 400
        # two factors with slowly varying correlation
        z = rng.standard_normal((T, 2))
        z[:, 1] = 0.5 * z[:, 0] + np.sqrt(1 - 0.25) * z[:, 1]
        res = fit_dcc(0.01 * z)
        assert 0.0 <= res.a < 1.0 and 0.0 <= res.b < 1.0
        assert res.a + res.b < 1.0     # stationarity
        path = dcc_corr_path(res, 0.01 * z)
        assert path.shape[0] == T
        off_diag = path[:, 0, 1]
        assert np.all(np.abs(off_diag) <= 1.0)
        assert abs(off_diag.mean() - 0.5) < 0.2   # near the true correlation
