"""Every world's price, hedgers and accounting agree with each other."""

import numpy as np
import pytest

from mjollnir.backtest import run_backtest
from mjollnir.hedging import (
    MODEL_HEDGERS,
    EuropeanOption,
    compare_hedgers,
    model_price,
    standard_hedgers,
)
from mjollnir.jax import configure_runtime
from mjollnir.params import ParamSet

configure_runtime()
DT, RATE, DIVIDEND, SPOT = 1 / 252, 0.04, 0.015, 100.0

BOOKS = {
    "gbm": {"sigma": 0.3},
    "heston": {"v0": 0.06, "kappa": 2.0, "theta": 0.05, "sigma_v": 0.5, "rho": -0.7},
    "merton": {"sigma": 0.15, "lambda_j": 4.0, "mu_j": -0.08, "sigma_j": 0.1},
    "regime_switching_gbm": {"sigma_0": 0.5, "sigma_1": 0.18, "p_00": 0.96, "p_01": 0.04,
                             "p_10": 0.02, "p_11": 0.98, "regime0": 1},
    "garch": {"omega": 4e-6, "alpha": 0.08, "beta": 0.9, "h0": 3e-4},
    "rough_bergomi": {"xi0": 0.05, "eta": 1.5, "rho": -0.7, "H": 0.1},
}


def _book(model):
    return ParamSet.create(model, "Q", BOOKS[model], asset="TEST")


@pytest.mark.parametrize("model", BOOKS)
@pytest.mark.parametrize("is_call", [True, False])
def test_every_hedger_breaks_even_on_average_in_a_risk_neutral_world(model, is_call):
    """At mu = r - q and zero costs, discounted wealth is a martingale: the mean
    P&L of any self-financing hedger is zero, which checks the premium, the world,
    the rate and dividend accounting and the hedger plumbing at once."""
    option = EuropeanOption(strike=105.0 if is_call else 95.0, maturity=21 * DT, is_call=is_call)
    results = compare_hedgers(_book(model), option, n_paths=8192, dt=DT, spot0=SPOT,
                              rate=RATE, dividend=DIVIDEND, seed=5)
    assert set(results) == {"no_hedge", "bs_delta", MODEL_HEDGERS[model][0]}
    for name, result in results.items():
        error = result.pnl.std() / np.sqrt(result.pnl.size)
        discounted_mean = result.pnl.mean() * np.exp(-RATE * option.maturity)
        assert abs(discounted_mean) < 4.5 * error + 2e-3, (model, name, discounted_mean, error)
    # The model's own hedge removes most of the unhedged risk where delta can:
    # diffusion risk. Jumps are not delta-hedgeable (Merton's price delta is not
    # even the variance-minimizing hedge there), so with large ones it only helps.
    # GARCH and rough Bergomi hedge with an approximation (Black-Scholes delta at a
    # forecast or current variance), which leaves more behind.
    bar = {"merton": 0.9, "garch": 0.6, "rough_bergomi": 0.6}.get(model, 0.5)
    hedged, unhedged = results[MODEL_HEDGERS[model][0]].pnl.std(), results["no_hedge"].pnl.std()
    assert hedged < bar * unhedged, (model, hedged, unhedged)


@pytest.mark.parametrize("model", ["gbm", "heston", "merton", "regime_switching_gbm"])
def test_fast_prices_agree_with_monte_carlo_in_the_models_own_world(model):
    option = EuropeanOption(strike=100.0, maturity=63 * DT, is_call=True)
    fast = model_price(option, _book(model), SPOT, rate=RATE, dividend=DIVIDEND, dt=DT)
    run = run_backtest(_book(model), lambda s, v, p, i: 0.0 * s, n_paths=65536, horizon_steps=63, dt=DT,
                       mu=RATE - DIVIDEND, spot0=SPOT, seed=9)
    payoff = np.exp(-RATE * option.maturity) * np.asarray(option.payoff(run.spots[:, -1]))
    error = payoff.std() / np.sqrt(payoff.size)
    assert abs(fast - payoff.mean()) < 4 * error, (model, fast, payoff.mean(), error)


def test_gbm_delta_hedging_converges_as_rebalancing_quickens():
    """Complete market: in its own world the Black-Scholes hedge's error falls
    like 1/sqrt(rebalancing frequency)."""
    book = _book("gbm")
    spreads = []
    for dt in (1 / 64, 1 / 256, 1 / 1024):  # 16, 64, 256 rebalances over 3 months
        option = EuropeanOption(strike=100.0, maturity=0.25)
        hedgers = standard_hedgers(option, book, dt=dt, rate=RATE, dividend=DIVIDEND)
        result = compare_hedgers(book, option, hedgers={"gbm_delta": hedgers["gbm_delta"]}, n_paths=4096,
                                 dt=dt, rate=RATE, dividend=DIVIDEND, seed=1)
        spreads.append(result["gbm_delta"].pnl.std())
    assert spreads[1] == pytest.approx(spreads[0] / 2, rel=0.2)
    assert spreads[2] == pytest.approx(spreads[1] / 2, rel=0.2)


def test_scenarios_refuse_parameters_a_model_does_not_have():
    from mjollnir.scenarios import correlation_breakdown

    with pytest.raises(ValueError, match="does not have"):
        compare_hedgers(_book("gbm"), EuropeanOption(100.0, 21 * DT), scenario=correlation_breakdown(-0.9),
                        n_paths=64, dt=DT)


@pytest.mark.parametrize("model", BOOKS)
def test_a_vol_spike_widens_every_models_hedging_error(model):
    from mjollnir.scenarios import scenario_for

    book = _book(model)
    option = EuropeanOption(strike=100.0, maturity=21 * DT)
    scenario = scenario_for("vol_spike", 2.0, book)
    calm = compare_hedgers(book, option, n_paths=2048, dt=DT, rate=RATE, dividend=DIVIDEND, seed=3)
    shocked = compare_hedgers(book, option, n_paths=2048, dt=DT, rate=RATE, dividend=DIVIDEND, seed=3,
                              scenario=scenario)
    # jumps dominate a Merton book's hedging error, and the spike scales only the diffusion
    growth = 1.1 if model == "merton" else 1.5
    assert shocked["bs_delta"].pnl.std() > growth * calm["bs_delta"].pnl.std(), model
    assert shocked["bs_delta"].params_hash != calm["bs_delta"].params_hash


def test_garch_regime_shift_lengthens_the_half_life_and_raises_long_run_variance():
    from mjollnir.scenarios import scenario_for

    book = _book("garch")
    shifted = scenario_for("regime_shift", 3.0, book).apply(book).params
    p = book.params
    half_life = lambda phi: np.log(0.5) / np.log(phi)
    assert half_life(shifted["alpha"] + shifted["beta"]) == pytest.approx(3 * half_life(p["alpha"] + p["beta"]))
    lr = lambda q: q["omega"] / (1 - q["alpha"] - q["beta"])
    assert lr(shifted) == pytest.approx(np.sqrt(3) * lr(p))


def test_archetypes_that_mean_nothing_for_a_model_are_refused():
    from mjollnir.scenarios import scenario_for

    with pytest.raises(ValueError, match="no meaning"):
        scenario_for("correlation_breakdown", -0.9, _book("gbm"))
    with pytest.raises(ValueError, match="no meaning"):
        scenario_for("jump_cascade", 3.0, _book("heston"))


@pytest.mark.parametrize("model", ["heston", "rough_bergomi"])
def test_one_scan_for_all_hedgers_equals_separate_backtests(model):
    from mjollnir.backtest import run_backtests

    book = _book(model)
    option = EuropeanOption(strike=100.0, maturity=21 * DT)
    hedgers = standard_hedgers(option, book, dt=DT, rate=RATE, dividend=DIVIDEND)
    common = {"n_paths": 256, "horizon_steps": 21, "dt": DT, "mu": RATE - DIVIDEND, "rate": RATE,
              "dividend": DIVIDEND, "initial_cash": 2.5, "cost_bps": 5.0, "liability": option.payoff, "seed": 4}
    together = run_backtests(book, hedgers, **common)
    for name, strategy in hedgers.items():
        alone = run_backtest(book, strategy, **common)
        np.testing.assert_allclose(together[name].pnl, alone.pnl, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(together[name].positions, alone.positions, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(together[name].costs, alone.costs, rtol=1e-12, atol=1e-12)
