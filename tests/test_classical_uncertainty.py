"""Uncertainty of the classical estimators: the helpers, and the fits that use them."""

import numpy as np
import pytest

from mjollnir.calibration.physical.garch_calibrator import GARCHCalibrator, garch_variances
from mjollnir.calibration.physical.merton_calibrator import MertonJumpCalibrator
from mjollnir.calibration.physical.uncertainty import (
    numerical_hessian,
    observed_information_std_errors,
    parametric_bootstrap,
)

DT = 1 / 252


def test_numerical_hessian_of_a_quadratic_is_exact():
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    hess = numerical_hessian(lambda x: 0.5 * x @ a @ x, np.array([0.4, -1.3]))
    assert hess == pytest.approx(a, rel=1e-6)


def test_observed_information_matches_the_gaussian_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 5000)

    def nll(p):
        mu, sigma = p
        return float(np.sum(np.log(sigma) + 0.5 * ((x - mu) / sigma) ** 2))

    mu_hat, sigma_hat = x.mean(), x.std()
    se = observed_information_std_errors(nll, np.array([mu_hat, sigma_hat]))
    assert se == pytest.approx([sigma_hat / np.sqrt(x.size), sigma_hat / np.sqrt(2 * x.size)], rel=1e-3)


def test_a_flat_likelihood_reports_infinite_errors():
    se = observed_information_std_errors(lambda p: float(p[0] ** 2), np.array([0.0, 1.0]))
    assert np.all(np.isinf(se))


def test_bootstrap_spread_of_the_gbm_vol_estimator():
    sigma, n = 0.3, 1000

    def simulate(rng, n_boot):
        returns = rng.normal(-0.5 * sigma**2 * DT, sigma * np.sqrt(DT), (n_boot, n))
        return 100.0 * np.exp(np.concatenate([np.zeros((n_boot, 1)), np.cumsum(returns, 1)], 1))

    def refit(path):
        return {"sigma": float(np.std(np.diff(np.log(path)), ddof=1) / np.sqrt(DT))}

    result = parametric_bootstrap({"sigma": sigma}, simulate, refit, n_boot=400, seed=3)
    assert result.std_errors["sigma"] == pytest.approx(sigma / np.sqrt(2 * n), rel=0.15)
    assert abs(result.bias["sigma"]) < 3 * result.std_errors["sigma"] / np.sqrt(400) + 1e-3


def _garch_prices(omega, alpha, beta, n, seed):
    rng = np.random.default_rng(seed)
    s2, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def test_garch_filter_matches_the_recursion():
    returns = np.diff(np.log(_garch_prices(2e-6, 0.08, 0.9, 300, seed=1)))
    params = np.array([1e-4, 2e-6, 0.08, 0.9])
    naive = np.empty_like(returns)
    naive[0] = 3e-4
    for t in range(1, returns.size):
        naive[t] = params[1] + params[2] * (returns[t - 1] - params[0]) ** 2 + params[3] * naive[t - 1]
    assert garch_variances(params, returns, 3e-4) == pytest.approx(naive, rel=1e-10)


def test_garch_fit_reports_errors_that_cover_the_truth():
    truth = {"omega": 2e-6, "alpha": 0.08, "beta": 0.9}
    fit = GARCHCalibrator().fit(_garch_prices(**truth, n=4000, seed=7))
    assert fit.converged
    for name, value in truth.items():
        se = fit.std_errors[name]
        assert 0 < se < np.inf and abs(getattr(fit, name) - value) < 4 * se, name
    assert fit.long_run_vol == pytest.approx(np.sqrt(2e-6 / 0.02 / DT), rel=0.25)
    assert fit.half_life_days > 10


def test_merton_fit_reports_errors_and_the_jump_share():
    rng = np.random.default_rng(11)
    sigma, lam, mu_j, sigma_j, n = 0.15, 10.0, -0.03, 0.05, 5000
    jumps = rng.poisson(lam * DT, n)
    returns = (rng.normal(-0.5 * sigma**2 * DT, sigma * np.sqrt(DT), n)
               + jumps * mu_j + np.sqrt(jumps) * sigma_j * rng.standard_normal(n))
    fit = MertonJumpCalibrator().fit(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)])))
    assert fit.converged
    assert abs(fit.sigma - sigma) < 4 * fit.std_errors["sigma"]
    share = lam * (mu_j**2 + sigma_j**2) / (sigma**2 + lam * (mu_j**2 + sigma_j**2))
    assert fit.jump_variance_share == pytest.approx(share, abs=0.12)
    assert all(np.isfinite(v) and v > 0 for v in fit.std_errors.values())


def test_rl_increment_constant_is_one_for_brownian_motion():
    from mjollnir.calibration.physical.rough_vol_variogram import rl_increment_constant

    assert rl_increment_constant(0.5) == pytest.approx(1.0, abs=1e-8)
    assert rl_increment_constant(0.1) > rl_increment_constant(0.3) > 1.0


def test_rough_vol_variogram_says_how_little_daily_data_know_about_H():
    from mjollnir.calibration.physical.rough_vol_variogram import RoughBergomiVariogramCalibrator
    from mjollnir.processes import RoughBergomi, SimulationConfig

    n = 2500
    proc = RoughBergomi(mu=0.0, xi0=0.05, eta=1.5, rho=-0.7, H=0.1)
    _, paths = proc.simulate(np.array([100.0, 0.05]), n * DT, SimulationConfig(n_paths=1, n_steps=n, random_seed=3))
    cal = RoughBergomiVariogramCalibrator()
    fit = cal.fit(paths[:, 0, 0])
    assert 0.01 <= fit.H <= 0.5 and fit.eta > 0
    assert fit.noise_floor == pytest.approx(np.pi**2 / 4, rel=0.3)  # Gaussian shocks: the theoretical floor
    # forward variance today: the trailing realized variance (rough Bergomi is not
    # stationary, so the long-run mean of squared returns would mislead)
    returns = np.diff(np.log(paths[:, 0, 0]))
    assert fit.xi0 == pytest.approx(np.mean(returns[-21:] ** 2) / DT)
    boot = cal.bootstrap(fit, n_boot=30, seed=2)
    assert boot.std_errors["H"] > 0.08  # ten years of daily closes would still not pin H down
    assert boot.n_boot >= 25


def test_heston_moment_bootstrap_reports_spread_and_bias():
    from mjollnir.calibration.physical.heston_qmle import HestonQMLECalibrator

    rng = np.random.default_rng(5)
    n, kappa, theta, sigma_v, rho = 1500, 3.0, 0.05, 0.5, -0.6
    v, log_s = theta, [0.0]
    for _ in range(n):
        z_v, z_p = rng.standard_normal(2)
        log_s.append(log_s[-1] - 0.5 * max(v, 0) * DT + np.sqrt(max(v, 0) * DT) * (rho * z_v + np.sqrt(1 - rho**2) * z_p))
        v = v + kappa * (theta - max(v, 0)) * DT + sigma_v * np.sqrt(max(v, 0) * DT) * z_v
    cal = HestonQMLECalibrator(smooth_window=10)
    fit = cal.fit(100.0 * np.exp(np.array(log_s)))
    boot = cal.bootstrap(fit, n_boot=100, seed=1)
    assert boot.n_boot == 100
    assert all(0 < se < np.inf for se in boot.std_errors.values())
    assert boot.std_errors["theta"] < 0.5 * fit.theta


def test_gbm_drift_error_scales_with_years_not_observations():
    from mjollnir.calibration.physical.gbm_calibrator import GBMCalibrator

    rng = np.random.default_rng(2)
    sigma, n = 0.3, 1260  # five years of daily closes
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma * np.sqrt(DT), n))]))
    fit = GBMCalibrator().fit(prices, DT)
    assert fit.mu_stderr == pytest.approx(fit.sigma / np.sqrt(n * DT), rel=1e-9)
    assert fit.mu_stderr == pytest.approx(sigma / np.sqrt(5.0), rel=0.1)


def test_rough_vol_variogram_ignores_stale_zero_returns():
    from mjollnir.calibration.physical.rough_vol_variogram import RoughBergomiVariogramCalibrator

    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.015, 1500)
    clean = RoughBergomiVariogramCalibrator().fit(100 * np.exp(np.concatenate([[0], np.cumsum(returns)])))
    returns[[100, 600, 1100]] = 0.0  # stale prices
    stale = RoughBergomiVariogramCalibrator().fit(100 * np.exp(np.concatenate([[0], np.cumsum(returns)])))
    assert stale.noise_floor == pytest.approx(clean.noise_floor, rel=0.05)
    assert clean.noise_floor == pytest.approx(np.pi**2 / 4, rel=0.15)


def test_garch_optimizer_leaves_its_starting_point():
    """In natural coordinates L-BFGS-B stopped at (alpha, beta) = (0.05, 0.9),
    its start, reporting convergence; the scaled fit must find the optimum."""
    truth = {"omega": 3e-6, "alpha": 0.12, "beta": 0.84}
    fit = GARCHCalibrator().fit(_garch_prices(**truth, n=3000, seed=21))
    assert fit.converged
    assert abs(fit.alpha - 0.05) > 0.02 and abs(fit.beta - 0.9) > 0.02
    for name, value in truth.items():
        assert abs(getattr(fit, name) - value) < 4 * fit.std_errors[name], name


@pytest.mark.parametrize("method", ["discretization", "exact_mle"])
def test_ou_standard_errors_match_the_spread_across_histories(method):
    """theta = a / (1 - b) needs the a-b covariance: without it the reported
    error was five times the true spread, and exact MLE reported sqrt(|gradient|)."""
    from mjollnir.calibration.physical.ou_calibrator import OUCalibrator

    dt, kappa, theta, sigma, n, reps = 1 / 252, 4.0, 0.5, 0.3, 3000, 60
    decay = np.exp(-kappa * dt)
    shock = sigma * np.sqrt((1 - decay**2) / (2 * kappa))
    rng = np.random.default_rng(3)
    x = np.empty((reps, n))
    x[:, 0] = theta
    for i in range(1, n):
        x[:, i] = theta + (x[:, i - 1] - theta) * decay + shock * rng.standard_normal(reps)
    fits = [OUCalibrator(method=method).fit(path, dt) for path in x]
    for name in ("kappa", "theta", "sigma"):
        spread = np.std([getattr(f, name) for f in fits])
        reported = np.mean([getattr(f, f"{name}_stderr") for f in fits])
        assert reported == pytest.approx(spread, rel=0.35), (name, reported, spread)
