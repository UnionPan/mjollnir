"""
Particle Filter for Heston Model (Physical Measure)

State:
    v_t (variance)
Observation:
    r_t = log(S_t / S_{t-1}) ~ N((mu - 0.5 v_t) dt_t, v_t dt_t)

``dt`` is one step for every interval, or one per interval: daily closes
skip weekends and holidays, and a series with gaps (or mixed sampling) is
filtered on its own clock. :func:`interval_years` turns observation dates into
those steps on a trading-day or calendar-day clock.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


def interval_years(dates, clock: str = "trading", days_per_year: float | None = None) -> np.ndarray:
    """Each interval between consecutive observation ``dates``, in years.

    ``clock="trading"`` counts business days (a weekend adds nothing, so a
    Monday close follows Friday's by one step) over 252 a year;
    ``clock="calendar"`` counts calendar days over 365.25.
    """
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(index) < 2 or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("need at least two strictly increasing dates")
    if clock == "trading":
        days = np.busday_count(index[:-1].values.astype("datetime64[D]"), index[1:].values.astype("datetime64[D]"))
        per_year = 252.0 if days_per_year is None else days_per_year
    elif clock == "calendar":
        days = np.diff(index.values.astype("datetime64[D]")).astype(float)
        per_year = 365.25 if days_per_year is None else days_per_year
    else:
        raise ValueError(f"clock must be 'trading' or 'calendar', not {clock!r}")
    days = np.asarray(days, dtype=float)
    if np.any(days <= 0):
        raise ValueError("two dates fall on the same trading day")
    return days / per_year


@dataclass
class HestonParticleFilterResult:
    """Results of Heston particle filtering."""

    filtered_variance: np.ndarray
    log_likelihood: float
    effective_sample_size: np.ndarray

    params: dict[str, float]
    n_particles: int
    dt: float | np.ndarray  # the step, or one per interval


class HestonParticleFilter:
    """
    Bootstrap particle filter for the Heston stochastic volatility model.
    """

    def __init__(
        self,
        n_particles: int = 2000,
        resample_threshold: float = 0.5,
        variance_floor: float = 1e-8,
    ):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.variance_floor = variance_floor

    def filter(
        self,
        prices: np.ndarray,
        dt: float | np.ndarray,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        v0: float | None = None,
        random_seed: int | None = None,
    ) -> HestonParticleFilterResult:
        prices = np.asarray(prices).flatten()
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to filter")
        if np.any(prices <= 0):
            raise ValueError("All prices must be positive")

        rng = np.random.default_rng(random_seed)

        returns = np.diff(np.log(prices))
        n_steps = len(returns)
        if np.ndim(dt) not in (0, 1) or (np.ndim(dt) == 1 and len(dt) != n_steps):
            raise ValueError(f"dt must be a scalar or one step per interval ({n_steps}), not shape {np.shape(dt)}")
        steps = np.broadcast_to(np.asarray(dt, dtype=float), (n_steps,))
        if np.any(~np.isfinite(steps)) or np.any(steps <= 0):
            raise ValueError("every time step must be positive")

        v_init = float(theta if v0 is None else v0)
        v_particles = np.full(self.n_particles, v_init, dtype=float)

        filtered_variance = np.zeros(n_steps)
        ess_series = np.zeros(n_steps)
        log_likelihood = 0.0

        for t in range(n_steps):
            dt_t = steps[t]
            # Propagate variance particles
            z = rng.normal(0.0, 1.0, size=self.n_particles)
            v_particles = (
                v_particles
                + kappa * (theta - v_particles) * dt_t
                + xi * np.sqrt(np.maximum(v_particles, 0.0)) * np.sqrt(dt_t) * z
            )
            v_particles = np.maximum(v_particles, self.variance_floor)

            # Observation likelihood
            mean = (mu - 0.5 * v_particles) * dt_t
            var = np.maximum(v_particles * dt_t, self.variance_floor)
            diff = returns[t] - mean
            log_w = -0.5 * (np.log(2.0 * np.pi * var) + (diff * diff) / var)

            max_log_w = np.max(log_w)
            w = np.exp(log_w - max_log_w)
            mean_w = np.mean(w)
            log_likelihood += np.log(mean_w) + max_log_w

            w_sum = np.sum(w)
            w_norm = w / w_sum
            ess = 1.0 / np.sum(w_norm**2)
            ess_series[t] = ess

            # Filtered variance (posterior mean)
            filtered_variance[t] = np.sum(w_norm * v_particles)

            # Resample if needed
            if ess < self.resample_threshold * self.n_particles:
                idx = self._systematic_resample(w_norm, rng)
                v_particles = v_particles[idx]

        return HestonParticleFilterResult(
            filtered_variance=filtered_variance,
            log_likelihood=log_likelihood,
            effective_sample_size=ess_series,
            params={
                'mu': mu,
                'kappa': kappa,
                'theta': theta,
                'xi': xi,
                'v0': v_init,
            },
            n_particles=self.n_particles,
            dt=dt,
        )

    def _systematic_resample(self, weights, rng: np.ndarray) -> np.ndarray:
        n = len(weights)
        positions = (rng.random() + np.arange(n)) / n
        cumulative_sum = np.cumsum(weights)
        idx = np.searchsorted(cumulative_sum, positions)
        return idx
