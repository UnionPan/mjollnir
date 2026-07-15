"""Tests for ImpliedVolSurface and the backtest driver."""

from datetime import date

import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.backtest import run_backtest
from mjollnir.jax import configure_runtime
from mjollnir.params import ParamSet
from mjollnir.scenarios import vol_spike
from mjollnir.surface import ImpliedVolSurface
from mjollnir.synthetic_data import (
    HestonVolatilityProfile,
    SyntheticEquityOptionChainGenerator,
)

configure_runtime()

HESTON = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7}


def _chain(spot=100.0):
    prof = HestonVolatilityProfile(kappa=4.0, theta=0.04, xi=0.4, rho=-0.6,
                                   v0=0.04, atm_iv=0.20)
    gen = SyntheticEquityOptionChainGenerator(random_seed=0)
    return gen.generate_single_chain(reference_date=date(2026, 1, 2),
                                     spot_price=spot, vol_profile=prof)


def _params():
    return ParamSet.create("heston", "Q", HESTON, asset="TEST")


class TestImpliedVolSurface:
    def test_from_chain(self):
        surf = ImpliedVolSurface.from_chain(_chain())
        assert len(surf) > 20
        assert np.isfinite(surf.ivs).all() and (surf.ivs > 0).all()
        assert (surf.total_variance > 0).all()
        assert surf.asset == _chain().underlying

    def test_synthetic_chain_is_arbitrage_free(self):
        """Chains generated from a coherent Heston model must pass the checks
        the property suite enforces on the pricer."""
        report = ImpliedVolSurface.from_chain(_chain()).arbitrage_report(tol=1e-6)
        assert report.ok, report.summary()

    def test_report_catches_planted_butterfly(self):
        surf = ImpliedVolSurface.from_chain(_chain())
        T0 = surf.expiries()[0]
        idx = np.where(np.isclose(surf.maturities, T0) & surf.is_call)[0]
        mids = surf.mids.copy()
        mid_strike = idx[np.argsort(surf.strikes[idx])][len(idx) // 2]
        mids[mid_strike] *= 1.5   # inflate one interior call -> concavity
        from dataclasses import replace
        bad = replace(surf, mids=mids)
        report = bad.arbitrage_report()
        assert not report.ok and report.butterfly

    def test_atm_term_structure(self):
        ts_T, ts_iv = ImpliedVolSurface.from_chain(_chain()).atm_term_structure()
        assert len(ts_T) >= 3 and (np.diff(ts_T) > 0).all()
        assert np.allclose(ts_iv, 0.20, atol=0.06)   # near the ATM IV of the profile

    def test_calibration_bridge(self):
        """surface.to_quotes feeds fit_heston_surface end to end."""
        from mjollnir.calibration import fit_heston_surface

        surf = ImpliedVolSurface.from_chain(_chain())
        res = fit_heston_surface(surf.to_quotes(), surf.spot, surf.rate,
                                 surf.dividend_yield, asset=surf.asset)
        assert res.converged
        # the generator priced with v0 = theta = 0.04; recovery should be sane
        assert res.param_set.params["theta"] == pytest.approx(0.04, rel=0.5)


class TestBacktest:
    def test_deterministic_and_provenance(self):
        ps = _params()
        strat = lambda s, v, pos, i: jnp.asarray(1.0)   # constant long
        r1 = run_backtest(ps, strat, n_paths=16, horizon_steps=32, seed=5)
        r2 = run_backtest(ps, strat, n_paths=16, horizon_steps=32, seed=5)
        np.testing.assert_array_equal(r1.wealth, r2.wealth)
        assert r1.params_hash == ps.content_hash()
        assert r1.wealth.shape == (16, 33)

    def test_flat_strategy_flat_wealth(self):
        """Zero position, zero costs: wealth stays exactly at initial cash."""
        strat = lambda s, v, pos, i: jnp.asarray(0.0)
        res = run_backtest(_params(), strat, n_paths=8, horizon_steps=16,
                           initial_cash=100.0)
        np.testing.assert_allclose(res.wealth, 100.0)
        assert res.max_drawdown == 0.0

    def test_long_positive_drift_earns(self):
        strat = lambda s, v, pos, i: jnp.asarray(1.0)
        res = run_backtest(_params(), strat, n_paths=256, horizon_steps=126,
                           mu=0.08, seed=1)
        assert res.mean_terminal > 0.0
        assert res.sharpe > 0.0

    def test_costs_hurt(self):
        churn = lambda s, v, pos, i: jnp.where(i % 2 == 0, 1.0, -1.0)
        free = run_backtest(_params(), churn, n_paths=32, horizon_steps=64, seed=2)
        costly = run_backtest(_params(), churn, n_paths=32, horizon_steps=64,
                              seed=2, cost_bps=10.0)
        assert costly.mean_terminal < free.mean_terminal

    def test_scenario_is_applied_and_recorded(self):
        strat = lambda s, v, pos, i: jnp.asarray(1.0)
        base = run_backtest(_params(), strat, n_paths=64, horizon_steps=64, seed=3)
        crisis = run_backtest(_params(), strat, n_paths=64, horizon_steps=64,
                              seed=3, scenario=vol_spike(2.5))
        assert crisis.scenario and "vol_spike" in crisis.scenario
        assert crisis.params_hash != base.params_hash
        assert crisis.std_terminal > base.std_terminal   # more vol => wider outcomes

    def test_rejects_non_heston(self):
        ps = ParamSet.create("merton", "P", {"sigma": 0.2})
        with pytest.raises(ValueError, match="heston"):
            run_backtest(ps, lambda s, v, p, i: 0.0)
