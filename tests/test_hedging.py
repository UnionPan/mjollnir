"""Tests for option books, classical hedgers and liability-aware backtests."""

import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.backtest import run_backtest
from mjollnir.hedging import (
    EuropeanOption,
    bs_delta,
    compare_hedgers,
    heston_delta,
    heston_price,
    implied_vol,
    no_hedge,
)
from mjollnir.jax import configure_runtime, fourier_price
from mjollnir.params import ParamSet
from mjollnir.scenarios import vol_spike

configure_runtime()

DT = 1 / 252
CALL = EuropeanOption(strike=100.0, maturity=21 * DT, is_call=True)
PUT = EuropeanOption(strike=100.0, maturity=21 * DT, is_call=False)


def _heston(**overrides):
    params = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma_v": 0.5, "rho": -0.7, "mu": 0.0}
    return ParamSet.create("heston", "P", {**params, **overrides}, asset="TEST")


def _near_constant_vol():
    # Vol-of-vol ~0 with v0 = theta: Heston collapses to Black-Scholes at 20% vol.
    return _heston(sigma_v=1e-4, rho=0.0)


class TestPricing:
    def test_heston_price_matches_cos_kernel(self):
        ps = _heston()
        p = ps.params
        expected = float(fourier_price(100.0, 100.0, CALL.maturity, 0.0, 0.0, p["v0"],
                                       p["kappa"], p["theta"], p["sigma_v"], p["rho"], True))
        assert heston_price(CALL, ps, spot=100.0) == pytest.approx(expected, rel=1e-12)

    def test_put_call_parity(self):
        ps = _heston()
        assert heston_price(CALL, ps, 100.0) - heston_price(PUT, ps, 100.0) == pytest.approx(0.0, abs=1e-8)

    def test_implied_vol_of_constant_vol_price_is_that_vol(self):
        price = heston_price(CALL, _near_constant_vol(), 100.0)
        assert implied_vol(CALL, price, spot=100.0) == pytest.approx(0.2, abs=1e-4)


class TestLiabilityBacktest:
    def test_liability_pnl_is_wealth_minus_payoff(self):
        ps = _heston()
        res = run_backtest(ps, no_hedge(), n_paths=64, horizon_steps=CALL.steps(DT), dt=DT,
                           initial_cash=2.0, liability=CALL.payoff, seed=1)
        payoff = np.maximum(res.spots[:, -1] - CALL.strike, 0.0)
        np.testing.assert_allclose(res.pnl, res.wealth[:, -1] - payoff)
        assert res.mean_terminal == pytest.approx(res.pnl.mean())

    def test_without_liability_pnl_is_terminal_wealth(self):
        res = run_backtest(_heston(), no_hedge(), n_paths=8, horizon_steps=8, seed=0)
        np.testing.assert_allclose(res.pnl, res.wealth[:, -1])

    def test_costs_are_reported_and_charged(self):
        ps = _heston()
        hedger = bs_delta(CALL, sigma=0.2, dt=DT)
        kw = {"n_paths": 128, "horizon_steps": CALL.steps(DT), "dt": DT, "liability": CALL.payoff, "seed": 4}
        free = run_backtest(ps, hedger, **kw)
        costly = run_backtest(ps, hedger, cost_bps=10.0, **kw)
        assert (costly.costs >= 0).all() and costly.costs.mean() > 0
        np.testing.assert_allclose(free.pnl - costly.pnl, costly.costs, rtol=1e-9, atol=1e-12)
        assert free.costs.max() == 0.0

    def test_positions_record_the_strategy(self):
        res = run_backtest(_heston(), bs_delta(CALL, sigma=0.2, dt=DT), n_paths=16,
                           horizon_steps=CALL.steps(DT), dt=DT, liability=CALL.payoff, seed=2)
        assert res.positions.shape == (16, CALL.steps(DT))
        assert ((res.positions >= 0) & (res.positions <= 1)).all()  # long call deltas


class TestHedgers:
    @pytest.mark.parametrize("make", ["none", "bs", "heston"])
    def test_no_free_lunch_without_drift_or_costs(self, make):
        ps = _heston()
        premium = heston_price(CALL, ps, 100.0)
        hedger = {
            "none": no_hedge(),
            "bs": bs_delta(CALL, sigma=implied_vol(CALL, premium, 100.0), dt=DT),
            "heston": heston_delta(CALL, ps, dt=DT),
        }[make]
        res = run_backtest(ps, hedger, n_paths=4096, horizon_steps=CALL.steps(DT), dt=DT,
                           initial_cash=premium, liability=CALL.payoff, seed=11)
        standard_error = res.pnl.std() / np.sqrt(res.pnl.size)
        assert abs(res.pnl.mean()) < 5 * standard_error

    def test_delta_hedge_error_shrinks_like_sqrt_dt(self):
        ps = _near_constant_vol()
        stds = []
        for steps_per_day in (1, 4):
            dt = DT / steps_per_day
            res = run_backtest(ps, bs_delta(CALL, sigma=0.2, dt=dt), n_paths=4096,
                               horizon_steps=CALL.steps(dt), dt=dt,
                               initial_cash=heston_price(CALL, ps, 100.0),
                               liability=CALL.payoff, seed=5)
            stds.append(res.pnl.std())
        assert stds[0] / stds[1] == pytest.approx(2.0, rel=0.2)

    def test_hedging_beats_no_hedge(self):
        ps = _heston()
        results = compare_hedgers(ps, CALL, n_paths=2048, dt=DT, seed=3)
        assert results["bs_delta"].pnl.std() < 0.35 * results["no_hedge"].pnl.std()
        assert results["heston_delta"].pnl.std() < 0.35 * results["no_hedge"].pnl.std()

    def test_compare_hedgers_uses_common_random_numbers(self):
        results = compare_hedgers(_heston(), CALL, n_paths=64, dt=DT, seed=9)
        spots = [r.spots for r in results.values()]
        for other in spots[1:]:
            np.testing.assert_array_equal(spots[0], other)

    def test_counterfactual_keeps_the_book_and_changes_the_world(self):
        ps = _heston()
        calm = compare_hedgers(ps, CALL, n_paths=2048, dt=DT, seed=3)
        shocked = compare_hedgers(ps, CALL, n_paths=2048, dt=DT, seed=3, scenario=vol_spike(2.0))
        # Same premium and hedger, a more volatile world: the seller loses on
        # average, and the stale-vol BS hedge is noisier than in the calm world.
        assert shocked["no_hedge"].pnl.mean() < calm["no_hedge"].pnl.mean()
        assert shocked["bs_delta"].pnl.std() > calm["bs_delta"].pnl.std()
        assert shocked["bs_delta"].scenario == "vol_spike[2x]"

    def test_strategies_are_vectorizable(self):
        hedger = heston_delta(CALL, _heston(), dt=DT)
        out = hedger(jnp.array([90.0, 100.0, 110.0]), jnp.full(3, 0.04), jnp.zeros(3), 0)
        assert out.shape == (3,) and bool(jnp.all(jnp.diff(out) > 0))

    def test_parametric_strategies_reuse_the_compiled_backtest(self):
        from mjollnir.backtest import _simulate

        compare_hedgers(_heston(), CALL, n_paths=32, dt=DT, seed=0)
        compiled = _simulate._cache_size()
        other_put = EuropeanOption(strike=105.0, maturity=CALL.maturity, is_call=False)
        compare_hedgers(_heston(kappa=3.0, v0=0.09), other_put, n_paths=32, dt=DT, seed=1,
                        scenario=vol_spike(2.0), cost_bps=5.0)
        assert _simulate._cache_size() == compiled

