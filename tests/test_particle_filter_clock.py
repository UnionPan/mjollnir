"""The Heston particle filter on its own clock: one step per interval, from observation dates."""

import numpy as np
import pandas as pd
import pytest

from mjollnir.calibration.physical.heston_particle_filter import HestonParticleFilter, interval_years

PARAMS = {"mu": 0.05, "kappa": 3.0, "theta": 0.04, "xi": 0.4, "v0": 0.04}


def _gapped_series(seed=0, n=400):
    """Prices from a Heston path observed every trading day, with every fifth interval three times as long."""
    rng = np.random.default_rng(seed)
    steps = np.where(np.arange(n) % 5 == 4, 3.0, 1.0) / 252
    v, log_s, prices = PARAMS["v0"], 0.0, [100.0]
    for dt in steps:
        log_s += (PARAMS["mu"] - 0.5 * v) * dt + np.sqrt(v * dt) * rng.standard_normal()
        v = max(v + PARAMS["kappa"] * (PARAMS["theta"] - v) * dt + PARAMS["xi"] * np.sqrt(v * dt) * rng.standard_normal(),
                1e-8)
        prices.append(100.0 * np.exp(log_s))
    return np.array(prices), steps


def test_a_constant_clock_is_the_scalar_step():
    prices, _ = _gapped_series()
    flt = HestonParticleFilter(n_particles=300)
    scalar = flt.filter(prices, 1 / 252, random_seed=3, **PARAMS)
    array = flt.filter(prices, np.full(len(prices) - 1, 1 / 252), random_seed=3, **PARAMS)
    assert scalar.log_likelihood == array.log_likelihood
    np.testing.assert_array_equal(scalar.filtered_variance, array.filtered_variance)


def test_the_true_clock_explains_a_gapped_series_better():
    prices, steps = _gapped_series()
    flt = HestonParticleFilter(n_particles=2000)
    right = flt.filter(prices, steps, random_seed=1, **PARAMS).log_likelihood
    wrong = flt.filter(prices, 1 / 252, random_seed=1, **PARAMS).log_likelihood
    assert right > wrong


def test_steps_from_dates():
    dates = pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-26", "2026-10-28"])  # Thu, Fri, Mon, Wed
    np.testing.assert_allclose(interval_years(dates), np.array([1, 1, 2]) / 252)
    np.testing.assert_allclose(interval_years(dates, clock="calendar"), np.array([1, 3, 2]) / 365.25)
    with pytest.raises(ValueError):
        interval_years(dates[::-1])
    with pytest.raises(ValueError):
        interval_years(pd.to_datetime(["2026-10-24", "2026-10-25"]))  # Saturday to Sunday: no trading day


def test_a_clock_of_the_wrong_length_is_refused():
    prices, steps = _gapped_series(n=20)
    with pytest.raises(ValueError, match="one step per interval"):
        HestonParticleFilter(n_particles=50).filter(prices, steps[:-1], **PARAMS)
    with pytest.raises(ValueError, match="positive"):
        HestonParticleFilter(n_particles=50).filter(prices, np.zeros_like(steps), **PARAMS)
