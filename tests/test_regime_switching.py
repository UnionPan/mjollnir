"""Tests for threshold-labelled regime-switching GBM calibration."""

import numpy as np
import pytest

from mjollnir.calibration.physical import RegimeSwitchingCalibrator

DT = 1.0 / 252
BEAR, SIDEWAYS, BULL = 0, 1, 2


def _calm_crash_rally(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """200 calm days, a 40-day 35% crash, then a 60-day 45% rally."""
    rng = np.random.default_rng(seed)
    drifts = np.concatenate([np.full(200, 0.0), np.full(40, np.log(0.65) / 40), np.full(60, np.log(1.45) / 60)])
    log_prices = np.log(100.0) + np.cumsum(drifts + 0.004 * rng.standard_normal(drifts.size))
    prices = np.exp(np.concatenate([[np.log(100.0)], log_prices]))
    dates = np.arange("2024-01-01", prices.size, dtype="datetime64[D]")
    return prices, dates


def test_warm_up_window_is_not_labelled_bear():
    prices, dates = _calm_crash_rally()
    result = RegimeSwitchingCalibrator(dt=DT).fit_threshold(prices, dates)
    # The first 60 days have no lookback, so they cannot be classified as a drawdown.
    assert (result.regime_labels[:60] == SIDEWAYS).all()
    assert set(np.unique(result.regime_labels)) == {BEAR, SIDEWAYS, BULL}


def test_regime_parameters_reflect_the_episodes():
    prices, dates = _calm_crash_rally()
    result = RegimeSwitchingCalibrator(dt=DT).fit_threshold(prices, dates)
    params = result.regime_params
    assert params[BEAR].mu < 0 < params[BULL].mu
    np.testing.assert_allclose(result.transition_matrix.sum(axis=1), 1.0)


def test_summary_mean_return_is_annualized_like_mu():
    prices, dates = _calm_crash_rally()
    for regime in RegimeSwitchingCalibrator(dt=DT).fit_threshold(prices, dates).regime_params.values():
        assert regime.mean_return == pytest.approx(regime.mu)


def test_a_missing_regime_keeps_the_matrix_and_holding_times_aligned():
    """Calm then a rally still under way: no bear regime ever occurs, and the bull
    regime is entered but never left. Ids {1, 2} index the matrix by position (id 2
    is row 1), and the never-left bull row is a distribution (it stays put)."""
    rng = np.random.default_rng(1)
    drifts = np.concatenate([np.full(200, 0.0), np.full(60, np.log(1.5) / 60)])
    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(drifts + 0.004 * rng.standard_normal(drifts.size))]))
    dates = np.arange("2024-01-01", prices.size, dtype="datetime64[D]")
    result = RegimeSwitchingCalibrator(dt=DT).fit_threshold(prices, dates)
    assert set(result.regime_params) == {SIDEWAYS, BULL}
    np.testing.assert_allclose(result.transition_matrix.sum(axis=1), 1.0)
    sideways_stay = result.transition_matrix[0, 0]
    assert result.expected_holding_time(SIDEWAYS) == pytest.approx(1.0 / (1.0 - sideways_stay))
    assert result.expected_holding_time(BULL) == np.inf
