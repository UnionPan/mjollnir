"""Simulated worlds: drift compensation, variance laws, and the hybrid rough-volatility scheme."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.backtest import run_backtest
from mjollnir.jax import configure_runtime
from mjollnir.params import ParamSet
from mjollnir.worlds import _rough_init, hybrid_scheme_weights, market_params, world_for

configure_runtime()
DT = 1 / 252

WORLDS = {
    "gbm": {"sigma": 0.3},
    "heston": {"v0": 0.06, "kappa": 2.0, "theta": 0.05, "sigma_v": 0.6, "rho": -0.7},
    "merton": {"sigma": 0.15, "lambda_j": 4.0, "mu_j": -0.1, "sigma_j": 0.12},
    "regime_switching_gbm": {"sigma_0": 0.5, "sigma_1": 0.15, "p_00": 0.95, "p_01": 0.05,
                             "p_10": 0.01, "p_11": 0.99, "regime0": 1},
    "garch": {"omega": 4e-6, "alpha": 0.08, "beta": 0.9, "h0": 3e-4},
    "rough_bergomi": {"xi0": 0.05, "eta": 1.5, "rho": -0.7, "H": 0.1},
}


def _hold_one_share(s, state, pos, i):
    return jnp.ones_like(s)


@pytest.mark.parametrize("model", WORLDS)
def test_every_world_grows_the_spot_at_its_drift(model):
    """Jumps, regimes and GARCH variance are compensated: E[S_T] = S0 exp(mu T)."""
    ps = ParamSet.create(model, "P", WORLDS[model])
    result = run_backtest(ps, _hold_one_share, n_paths=8192, horizon_steps=126, mu=0.07, seed=11)
    terminal = result.spots[:, -1]
    error = terminal.std() / np.sqrt(terminal.size)
    assert abs(terminal.mean() - 100.0 * np.exp(0.07 * 0.5)) < 4 * error, model


def test_rough_variance_keeps_its_forward_level():
    """Under rough Bergomi E[v_t] = xi0 at every t (the exp martingale)."""
    params = market_params("rough_bergomi", WORLDS["rough_bergomi"])
    keys = jax.random.split(jax.random.PRNGKey(4), 20000)
    paths = jax.vmap(lambda k: _rough_init(params, k, 63, DT)[1])(keys)
    variance = np.asarray(paths[:, 0, :])
    for t in (1, 10, 62):
        error = variance[:, t].std() / np.sqrt(variance.shape[0])
        assert abs(variance[:, t].mean() - 0.05) < 4 * error, t


def test_hybrid_scheme_matches_the_exact_rl_variance():
    """Var W^H_t = t^(2H) for the RL fBm; the hybrid scheme gets it right at
    every step, including the first (where a Riemann sum is most wrong)."""
    H, n = 0.1, 40
    toeplitz, var_w, var_z, cov = hybrid_scheme_weights(H, n, DT)
    alpha = H - 0.5
    # Var of sqrt(2H)(exact_i + sum_j T_ij dW_j): exact term var_z, others var_w each, independent
    var = 2 * H * (var_z + var_w * np.sum(np.asarray(toeplitz) ** 2, axis=1))
    t = (np.arange(n) + 1) * DT
    assert var[0] == pytest.approx(t[0] ** (2 * H), rel=1e-9)
    assert np.allclose(var, t ** (2 * H), rtol=0.02)
    assert cov == pytest.approx(DT ** (alpha + 1) / (alpha + 1))


def test_garch_returns_have_the_long_run_variance():
    p = WORLDS["garch"]
    long_run = p["omega"] / (1 - p["alpha"] - p["beta"])
    ps = ParamSet.create("garch", "P", {**p, "h0": long_run})
    result = run_backtest(ps, _hold_one_share, n_paths=4096, horizon_steps=252, seed=2)
    returns = np.diff(np.log(result.spots), axis=1)
    assert returns.var() == pytest.approx(long_run, rel=0.05)


def test_student_t_garch_keeps_its_drift_and_variance_and_fattens_the_tails():
    p = WORLDS["garch"]
    long_run = p["omega"] / (1 - p["alpha"] - p["beta"])
    gaussian = ParamSet.create("garch", "P", {**p, "h0": long_run})
    student = ParamSet.create("garch", "P", {**p, "h0": long_run, "nu": 4.0})
    runs = [run_backtest(ps, _hold_one_share, n_paths=8192, horizon_steps=126, mu=0.07, seed=3)
            for ps in (gaussian, student)]
    terminal = runs[1].spots[:, -1]
    error = terminal.std() / np.sqrt(terminal.size)
    assert abs(terminal.mean() - 100.0 * np.exp(0.07 * 0.5)) < 4 * error
    simple = [np.diff(r.spots, axis=1) / r.spots[:, :-1] - (np.exp(0.07 * DT) - 1) for r in runs]
    assert simple[1].var() == pytest.approx(long_run, rel=0.05)
    standardized = [x / x.std() for x in simple]
    assert np.mean(np.abs(standardized[1]) > 5) > 5 * np.mean(np.abs(standardized[0]) > 5)


def test_garch_degrees_of_freedom_must_give_a_finite_variance():
    with pytest.raises(ValueError, match="nu must exceed 2"):
        market_params("garch", {**WORLDS["garch"], "nu": 2.0})


def test_regimes_follow_their_chain():
    """Starting in the calm regime, the time spent turbulent approaches the
    chain's stationary share p_10 / (p_01 + p_10) = 1/6."""
    world = world_for("regime_switching_gbm")
    params = market_params("regime_switching_gbm", WORLDS["regime_switching_gbm"])
    regimes = []
    key = jax.random.PRNGKey(0)
    state, path, key = world.init(params, key, 2000, DT)
    spot = jnp.asarray(100.0)
    for t in range(2000):
        spot, state, key = world.step(spot, state, t, params, path, DT, 0.0, key)
        regimes.append(float(state))
    assert np.mean(np.asarray(regimes[500:]) == 0) == pytest.approx(1 / 6, abs=0.08)
