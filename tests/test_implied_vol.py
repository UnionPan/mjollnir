"""Tests for the batched Black-Scholes implied-vol inversion.

The solver must recover IVs accurately where a price is invertible and report
NaN where it is not, instead of returning a clipped or unconverged sigma that
looks like a real answer.
"""

import dataclasses
from datetime import date

import numpy as np
import pytest
from scipy.stats import norm

from mjollnir.jax import configure_runtime
from mjollnir.pricer._jax_iv import implied_vol_batch_np
from mjollnir.surface import ImpliedVolSurface
from mjollnir.synthetic_data import (
    HestonVolatilityProfile,
    SyntheticEquityOptionChainGenerator,
)

configure_runtime()

S, R, Q = 100.0, 0.03, 0.01


def _bs(K, T, sigma, is_call):
    d1 = (np.log(S / K) + (R - Q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-Q * T) * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    put = K * np.exp(-R * T) * norm.cdf(-d2) - S * np.exp(-Q * T) * norm.cdf(-d1)
    return np.where(is_call, call, put)


def _chain():
    prof = HestonVolatilityProfile(kappa=4.0, theta=0.04, xi=0.4, rho=-0.6,
                                   v0=0.04, atm_iv=0.20)
    gen = SyntheticEquityOptionChainGenerator(random_seed=0)
    return gen.generate_single_chain(reference_date=date(2026, 1, 2),
                                     spot_price=S, vol_profile=prof)


class TestImpliedVolBatch:
    @pytest.mark.parametrize("T", [7 / 365, 30 / 365, 1.0])
    def test_recovers_known_vols(self, T):
        K = np.linspace(70, 140, 29)
        is_call = K >= S
        sigma = 0.15 + 0.25 * np.abs(np.log(K / S))  # a smile
        iv = implied_vol_batch_np(S, K, T, R, Q, _bs(K, T, sigma, is_call), is_call)
        np.testing.assert_allclose(iv, sigma, atol=1e-8)

    def test_price_outside_no_arbitrage_bounds_is_nan(self):
        K = np.array([90.0, 110.0, 90.0, 110.0])
        is_call = np.array([True, True, False, False])
        T = 0.5
        intrinsic = np.where(
            is_call,
            np.maximum(S * np.exp(-Q * T) - K * np.exp(-R * T), 0.0),
            np.maximum(K * np.exp(-R * T) - S * np.exp(-Q * T), 0.0),
        )
        upper = np.where(is_call, S * np.exp(-Q * T), K * np.exp(-R * T))
        below = intrinsic - 0.01
        above = upper + 0.01
        assert np.isnan(implied_vol_batch_np(S, K, T, R, Q, below, is_call)).all()
        assert np.isnan(implied_vol_batch_np(S, K, T, R, Q, above, is_call)).all()

    def test_zero_price_far_otm_is_nan(self):
        # A clipped-to-zero wing price carries no vol information.
        iv = implied_vol_batch_np(S, np.array([160.0, 50.0]), 30 / 365, R, Q,
                                  np.zeros(2), np.array([True, False]))
        assert np.isnan(iv).all()

    def test_vol_beyond_search_range_is_nan(self):
        K = np.array([100.0])
        price = _bs(K, 0.5, np.array([6.0]), np.array([True]))
        iv = implied_vol_batch_np(S, K, 0.5, R, Q, price, np.array([True]), sigma_max=5.0)
        assert np.isnan(iv).all()

    def test_clip_mode_keeps_raw_solver_output(self):
        K = np.array([100.0, 160.0])
        is_call = np.array([True, True])
        prices = np.array([_bs(K[:1], 0.5, np.array([0.2]), is_call[:1])[0], 0.0])
        clipped = implied_vol_batch_np(S, K, 0.5, R, Q, prices, is_call, on_failure="clip")
        assert np.isfinite(clipped).all()
        assert clipped[0] == pytest.approx(0.2, abs=1e-8)
        assert 0.005 <= clipped[1] <= 5.0

    def test_rejects_unknown_failure_mode(self):
        with pytest.raises(ValueError, match="on_failure"):
            implied_vol_batch_np(S, np.array([100.0]), 0.5, R, Q, np.array([5.0]),
                                 np.array([True]), on_failure="zero")


class TestSurfaceInversion:
    def test_from_chain_recovers_ivs_when_quotes_have_none(self):
        chain = _chain()
        given = ImpliedVolSurface.from_chain(chain)
        stripped = dataclasses.replace(chain, options=[
            dataclasses.replace(o, implied_volatility=None) for o in chain.options
        ])
        recovered = ImpliedVolSurface.from_chain(stripped)
        # Re-inverting the quoted mids reproduces the generator's IVs on every
        # quote whose mid is invertible.
        assert len(recovered) > 0.8 * len(given)
        lookup = {(k, t, c): iv for k, t, c, iv in zip(
            given.strikes, given.maturities, given.is_call, given.ivs, strict=True)}
        for k, t, c, iv in zip(recovered.strikes, recovered.maturities,
                               recovered.is_call, recovered.ivs, strict=True):
            assert iv == pytest.approx(lookup[(k, t, c)], abs=1e-6)
