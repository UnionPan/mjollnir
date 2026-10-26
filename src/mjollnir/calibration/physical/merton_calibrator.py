"""
Merton Jump-Diffusion Calibration (Physical Measure)

Log return model (Merton, 1976):
    r_t = log(S_t / S_{t-1}) ~ sum of diffusion + jump components

Diffusion:
    dS_t / S_t = mu dt + sigma dW_t

Jumps:
    N_t ~ Poisson(lambda dt)
    J ~ Normal(mu_j, sigma_j^2)

Log-return distribution is a Poisson mixture of normals.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from mjollnir.calibration.physical.uncertainty import observed_information_std_errors

PARAM_NAMES = ("mu", "sigma", "lambda_", "mu_j", "sigma_j")


def merton_neg_log_likelihood(params: np.ndarray, returns: np.ndarray, dt: float, k_max: int = 5) -> float:
    """Negative log-likelihood of log returns under Merton (Poisson mixture of normals)."""
    mu, sigma, lambda_, mu_j, sigma_j = params
    if sigma <= 0 or lambda_ <= 0 or sigma_j <= 0:
        return 1e12
    k_grid = np.arange(k_max + 1)                                   # (K,)
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    base_mean = (mu - 0.5 * sigma**2 - lambda_ * kappa) * dt
    base_var = sigma**2 * dt
    weights = stats.poisson.pmf(k_grid, lambda_ * dt)              # (K,)
    mean_k = base_mean + k_grid * mu_j                              # (K,)
    var_k = base_var + k_grid * sigma_j**2                          # (K,)
    comp = stats.norm.pdf(returns[:, None], loc=mean_k[None, :], scale=np.sqrt(var_k)[None, :])  # (T, K)
    mix_prob = np.maximum(comp @ weights, 1e-300)                   # (T,)
    return -float(np.sum(np.log(mix_prob)))


@dataclass
class MertonCalibrationResult:
    """Result of Merton jump-diffusion calibration."""

    mu: float
    sigma: float
    lambda_: float
    mu_j: float
    sigma_j: float

    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    dt: float
    converged: bool = True
    message: str = ""
    std_errors: dict[str, float] = field(default_factory=dict)  # observed information

    @property
    def jump_variance_share(self) -> float:
        """Share of return variance that jumps carry; better identified than
        lambda and sigma_j separately (many small jumps look like diffusion)."""
        jump = self.lambda_ * (self.mu_j**2 + self.sigma_j**2)
        return float(jump / (self.sigma**2 + jump))

    @property
    def total_vol(self) -> float:
        """Annualized volatility of returns, diffusion and jumps together."""
        return float(np.sqrt(self.sigma**2 + self.lambda_ * (self.mu_j**2 + self.sigma_j**2)))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Merton Jump-Diffusion Calibration Result",
            "=" * 60,
            f"mu      = {self.mu:.6f}",
            f"sigma   = {self.sigma:.6f}",
            f"lambda  = {self.lambda_:.6f}",
            f"mu_j    = {self.mu_j:.6f}",
            f"sigma_j = {self.sigma_j:.6f}",
            f"jump share of variance = {self.jump_variance_share:.3f}",
            "",
            f"logL = {self.log_likelihood:.2f}",
            f"AIC  = {self.aic:.2f}",
            f"BIC  = {self.bic:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class MertonJumpCalibrator:
    """
    MLE calibration for the Merton jump-diffusion model.

    The likelihood is multimodal (rare large jumps against frequent small
    ones), so the fit starts from several jump regimes and keeps the best.
    A fit where no start converges is reported as not converged; it never
    falls back to its starting values.
    """

    # (lambda per year, sigma_j) starts: rare-large to frequent-small jumps
    STARTS = ((2.0, 0.08), (10.0, 0.04), (40.0, 0.02))

    def __init__(self, k_max: int = 5):
        """
        Args:
            k_max: Truncation level for Poisson mixture
        """
        self.k_max = k_max

    def fit(self, prices: np.ndarray, dt: float = 1.0 / 252.0) -> MertonCalibrationResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 2:
            raise ValueError("Need at least 2 price observations")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        returns = np.diff(np.log(prices))
        n = len(returns)
        mu_init = np.mean(returns) / dt
        sigma_init = np.std(returns, ddof=1) / np.sqrt(dt)
        bounds = [
            (None, None),      # mu
            (1e-6, None),      # sigma
            (1e-6, None),      # lambda
            (None, None),      # mu_j
            (1e-6, None),      # sigma_j
        ]

        def nll(x):
            return merton_neg_log_likelihood(x, returns, dt, self.k_max)

        best = None
        for lambda_init, sigma_j_init in self.STARTS:
            x0 = np.array([mu_init, 0.8 * sigma_init, lambda_init, 0.0, sigma_j_init])
            result = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
            if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
                best = result
        converged = bool(best.success)
        params = best.x
        log_likelihood = -float(best.fun)
        std = observed_information_std_errors(nll, params) if converged else np.full(5, np.inf)

        mu, sigma, lambda_, mu_j, sigma_j = params
        k = 5
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        return MertonCalibrationResult(
            mu=float(mu),
            sigma=float(sigma),
            lambda_=float(lambda_),
            mu_j=float(mu_j),
            sigma_j=float(sigma_j),
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            n_observations=n,
            dt=dt,
            converged=converged,
            message=str(best.message),
            std_errors=dict(zip(PARAM_NAMES, (float(x) for x in std), strict=True)),
        )
