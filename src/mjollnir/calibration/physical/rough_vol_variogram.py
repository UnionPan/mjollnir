"""Rough Bergomi from daily closes: the noise-corrected log-absolute-return variogram.

With daily returns r_t = sqrt(v_t dt) z_t,

    log|r_t| = 0.5 log v_t + 0.5 log dt + log|z_t|,

so increments of log|r| over a lag of Delta (years) have variance

    V(Delta) = a + b Delta^(2H),    b = eta^2 C_H / 4,    a = 2 Var(log|z|),

where the rough Bergomi log-variance is eta times a Riemann-Liouville
fractional Brownian motion, whose increments have variance C_H Delta^(2H) once
the process is far from its start. ``a`` is pi^2/4 for Gaussian z; it is
fitted rather than assumed, so heavier-tailed daily shocks do not bias H.

What daily data can and cannot say: over a few years the roughness signal
b Delta^(2H) is a few percent of the floor a, so H is barely identified.
:meth:`RoughBergomiVariogramCalibrator.bootstrap` measures this (a wide
standard error) instead of hiding it. The rolling-realized-variance estimator
in :mod:`rough_bergomi_calibrator` cannot resolve H at all: overlapping windows
smooth the proxy, and its estimate sits at the 0.49 clip whatever the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np
from scipy import integrate, optimize

from mjollnir.calibration.physical.uncertainty import BootstrapResult, parametric_bootstrap

PARAM_NAMES = ("H", "eta", "rho", "xi0", "mu")
H_BOUNDS = (0.01, 0.5)
GAUSSIAN_FLOOR = np.pi**2 / 4  # 2 Var(log|z|), z ~ N(0, 1)


@cache
def rl_increment_constant(H: float) -> float:
    """C_H: stationary increment variance per Delta^(2H) of the Riemann-Liouville
    fBm sqrt(2H) int (t-s)^(H-1/2) dW_s, which has Var X_t = t^(2H)."""
    integrand = lambda x: ((1.0 + x) ** (H - 0.5) - x ** (H - 0.5)) ** 2  # noqa: E731
    head, _ = integrate.quad(integrand, 0.0, 1.0, limit=200)
    tail, _ = integrate.quad(integrand, 1.0, np.inf, limit=200)
    return float(1.0 + 2.0 * H * (head + tail))


@dataclass
class RoughBergomiVariogramResult:
    mu: float              # annualized drift of log prices (educational)
    xi0: float             # forward variance today: trailing realized variance, annualized
    eta: float             # vol of log-variance
    rho: float             # leverage proxy (attenuated toward 0; see the bootstrap bias)
    H: float               # Hurst exponent of log-variance
    noise_floor: float     # fitted a; pi^2/4 for Gaussian daily shocks
    n_observations: int
    dt: float
    max_lag: int
    lags: list[float] = field(default_factory=list)       # Delta in years
    variogram: list[float] = field(default_factory=list)  # empirical V(Delta)
    fit_r2: float = float("nan")


class RoughBergomiVariogramCalibrator:
    """Fit (H, eta) to the log-absolute-return variogram; xi0, rho, mu by moments."""

    def __init__(self, max_lag: int = 30, leverage_window: int = 21, state_window: int = 21):
        self.max_lag = max_lag
        self.leverage_window = leverage_window
        # rough Bergomi is not stationary (Var log v grows like t^(2H)), so a
        # long-run mean of squared returns says little about the variance
        # level going forward; the recent level does
        self.state_window = state_window

    def fit(self, prices: np.ndarray, dt: float = 1.0 / 252.0) -> RoughBergomiVariogramResult:
        prices = np.asarray(prices, dtype=float).ravel()
        if np.any(prices <= 0):
            raise ValueError("all prices must be positive")
        returns = np.diff(np.log(prices))
        n = returns.size
        if n < 10 * self.max_lag:
            raise ValueError(f"need at least {10 * self.max_lag} returns, got {n}")

        # Exact zeros are stale prices (probability zero under the model) and
        # their log is unbounded: pairs that touch one are left out.
        valid = np.abs(returns) > 1e-10
        log_abs = np.log(np.where(valid, np.abs(returns), 1.0))
        lags = np.arange(1, self.max_lag + 1)
        variogram = np.array([
            np.var((log_abs[k:] - log_abs[:-k])[valid[k:] & valid[:-k]], ddof=1) for k in lags
        ])
        delta = lags * dt

        def residuals(x):
            a, log_b, H = x
            return a + np.exp(log_b) * delta ** (2.0 * H) - variogram

        best = None
        for H0 in (0.05, 0.15, 0.35):
            b0 = max(variogram[-1] - variogram[0], 1e-3) / max(delta[-1] ** (2 * H0) - delta[0] ** (2 * H0), 1e-6)
            x0 = np.array([max(variogram[0] - b0 * delta[0] ** (2 * H0), 0.0), np.log(b0), H0])
            sol = optimize.least_squares(residuals, x0, bounds=([0.0, -30.0, H_BOUNDS[0]], [np.inf, 30.0, H_BOUNDS[1]]))
            if best is None or sol.cost < best.cost:
                best = sol
        a, log_b, H = best.x
        eta = float(np.sqrt(4.0 * np.exp(log_b) / rl_increment_constant(float(H))))
        ss_tot = float(np.sum((variogram - variogram.mean()) ** 2))
        fit_r2 = 1.0 - 2.0 * float(best.cost) / ss_tot if ss_tot > 0 else float("nan")

        w = self.leverage_window
        sq = returns**2
        back = np.convolve(sq, np.ones(w) / w, mode="valid")  # back[i] = mean of sq[i : i + w]
        # leverage: today's return against the change from the past window to the next one
        past, future = back[: -w - 1], back[w + 1:]
        today = returns[w: w + past.size]
        change = np.log(np.maximum(future, 1e-16)) - np.log(np.maximum(past, 1e-16))
        rho = float(np.corrcoef(today, change)[0, 1]) if today.size > 3 else 0.0

        return RoughBergomiVariogramResult(
            mu=float(np.mean(returns) / dt + 0.5 * np.mean(sq) / dt),
            xi0=float(np.mean(sq[-self.state_window:]) / dt), eta=eta, rho=float(np.clip(rho, -0.99, 0.99)), H=float(H),
            noise_floor=float(a), n_observations=n, dt=dt, max_lag=self.max_lag,
            lags=delta.tolist(), variogram=variogram.tolist(), fit_r2=float(fit_r2),
        )

    def bootstrap(self, result: RoughBergomiVariogramResult, n_boot: int = 100,
                  seed: int = 0) -> BootstrapResult:
        """Spread and bias at ``result``: simulate rough Bergomi histories as long
        as the data from the fitted parameters, refit each."""
        from mjollnir.processes import RoughBergomi, SimulationConfig

        estimate = {k: float(getattr(result, k)) for k in PARAM_NAMES}
        n, dt = result.n_observations, result.dt

        def simulate(rng: np.random.Generator, size: int) -> np.ndarray:
            process = RoughBergomi(mu=estimate["mu"], xi0=estimate["xi0"], eta=estimate["eta"],
                                   rho=estimate["rho"], H=estimate["H"])
            config = SimulationConfig(n_paths=size, n_steps=n, random_seed=int(rng.integers(2**31)))
            _, paths = process.simulate(np.array([100.0, estimate["xi0"]]), n * dt, config)
            return paths[:, :, 0].T

        def refit(path: np.ndarray) -> dict[str, float]:
            fitted = self.fit(path, dt=dt)
            return {k: float(getattr(fitted, k)) for k in PARAM_NAMES}

        return parametric_bootstrap(estimate, simulate, refit, n_boot=n_boot, seed=seed)
