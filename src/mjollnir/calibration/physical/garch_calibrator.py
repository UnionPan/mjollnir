"""
GARCH(1,1) Calibration (Physical Measure)

Model:
    r_t = mu + eps_t
    eps_t = sigma_t z_t, z_t ~ N(0,1)
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.signal import lfilter

from mjollnir.calibration.physical.uncertainty import observed_information_std_errors

PARAM_NAMES = ("mu", "omega", "alpha", "beta")


def garch_variances(params: np.ndarray, returns: np.ndarray, var_init: float) -> np.ndarray:
    """Conditional variances sigma_t^2 (daily units), sigma_0^2 = ``var_init``.

    The recursion is linear in sigma^2, so it runs as one filter pass.
    """
    mu, omega, alpha, beta = params
    eps2 = (returns - mu) ** 2
    drive = np.empty_like(returns)
    drive[0] = var_init
    drive[1:] = omega + alpha * eps2[:-1]
    return np.maximum(lfilter([1.0], [1.0, -beta], drive), 1e-12)


def garch_neg_log_likelihood(params: np.ndarray, returns: np.ndarray, var_init: float) -> float:
    """Gaussian (quasi-) negative log-likelihood of GARCH(1,1) returns."""
    mu, omega, alpha, beta = params
    if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.999:
        return 1e12
    sigma2 = garch_variances(params, returns, var_init)
    eps = returns - mu
    return 0.5 * float(np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2))


@dataclass
class GARCHCalibrationResult:
    """Result of GARCH(1,1) calibration. Parameters are in daily units."""

    mu: float
    omega: float
    alpha: float
    beta: float

    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    dt: float
    converged: bool = True
    message: str = ""
    std_errors: dict[str, float] = field(default_factory=dict)  # observed information

    @property
    def persistence(self) -> float:
        """alpha + beta: how much of a variance shock survives one step."""
        return float(self.alpha + self.beta)

    @property
    def long_run_vol(self) -> float:
        """Annualized unconditional volatility, sqrt(omega / (1 - alpha - beta) / dt)."""
        return float(np.sqrt(self.omega / max(1.0 - self.persistence, 1e-12) / self.dt))

    @property
    def half_life_days(self) -> float:
        """Steps for a variance shock to halve."""
        return float(np.log(0.5) / np.log(self.persistence))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "GARCH(1,1) Calibration Result",
            "=" * 60,
            f"mu    = {self.mu:.6f}",
            f"omega = {self.omega:.6f}",
            f"alpha = {self.alpha:.6f}",
            f"beta  = {self.beta:.6f}",
            f"long-run vol = {self.long_run_vol:.4f}, half-life = {self.half_life_days:.1f} steps",
            "",
            f"logL = {self.log_likelihood:.2f}",
            f"AIC  = {self.aic:.2f}",
            f"BIC  = {self.bic:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GARCHCalibrator:
    """
    QMLE calibration for GARCH(1,1) using Gaussian likelihood.

    The optimizer works in well-scaled coordinates: the long-run variance
    relative to the sample's (log), persistence alpha + beta (logit) and
    alpha's share of it (logit). In natural coordinates omega (~1e-6) and
    alpha, beta (~0.1-0.9) differ by five orders of magnitude, and L-BFGS-B
    stopped at its starting point while reporting convergence. Several
    starts guard against a local optimum. A fit whose optimizer does not
    converge is reported as such; it never falls back to its starting values.
    """

    STARTS = ((0.90, 0.05), (0.97, 0.05), (0.99, 0.05), (0.95, 0.15), (0.98, 0.15))  # (persistence, alpha share)

    def fit(self, prices: np.ndarray, dt: float = 1.0 / 252.0) -> GARCHCalibrationResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 3:
            raise ValueError("Need at least 3 price observations")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        returns = np.diff(np.log(prices))
        n = len(returns)
        var_init = float(np.var(returns, ddof=1))
        sd = np.sqrt(var_init)

        def natural(z):
            persistence = 0.999 * _sigmoid(z[2])
            share = _sigmoid(z[3])
            long_run = var_init * np.exp(z[1])
            return np.array([z[0] * sd, long_run * (1.0 - persistence), share * persistence,
                             (1.0 - share) * persistence])

        def nll_z(z):
            return garch_neg_log_likelihood(natural(z), returns, var_init)

        def nll(x):
            return garch_neg_log_likelihood(x, returns, var_init)

        best = None
        for persistence, share in self.STARTS:
            z0 = np.array([np.mean(returns) / sd, 0.0, np.log(persistence / 0.999 / (1 - persistence / 0.999)),
                           np.log(share / (1 - share))])
            result = optimize.minimize(nll_z, z0, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
            if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
                best = result
        params = natural(best.x)
        log_likelihood = -float(best.fun)
        # BFGS reports precision loss at a flat optimum; the gradient norm is the real test
        converged = bool(best.success or np.max(np.abs(best.jac)) < 1e-3)
        std = observed_information_std_errors(nll, params) if converged else np.full(4, np.inf)

        mu, omega, alpha, beta = params
        k = 4
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return GARCHCalibrationResult(
            mu=float(mu),
            omega=float(omega),
            alpha=float(alpha),
            beta=float(beta),
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            n_observations=n,
            dt=dt,
            converged=converged,
            message=str(best.message),
            std_errors=dict(zip(PARAM_NAMES, (float(x) for x in std), strict=True)),
        )
