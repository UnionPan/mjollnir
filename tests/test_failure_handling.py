"""Failures are either handled for a reason or raised: never swallowed into a wrong answer."""

import numpy as np
import pytest

from mjollnir.derivatives import EuropeanCall
from mjollnir.pricer.fourier import COSPricer
from mjollnir.processes import GBM, SimulationConfig
from mjollnir.processes import _process_defs
from mjollnir.processes.variance_gamma import VarianceGamma


def _vg_with_kernel(monkeypatch, error):
    def failing_kernel(*args, **kwargs):
        raise error

    monkeypatch.setattr(_process_defs, "levy_simulate", failing_kernel)
    return VarianceGamma(theta=-0.1, sigma=0.2, nu=0.2)


def test_a_backend_that_cannot_start_falls_back_to_numpy(monkeypatch):
    vg = _vg_with_kernel(monkeypatch, RuntimeError("Unable to initialize backend 'cuda'"))
    with pytest.warns(RuntimeWarning, match="falling back to NumPy"):
        _, paths = vg.simulate(np.array([0.0]), 1.0, SimulationConfig(n_paths=8, n_steps=4, random_seed=1))
    assert np.all(np.isfinite(paths))


@pytest.mark.parametrize("error", [RuntimeError("an XLA compile error"), ValueError("a shape bug")])
def test_any_other_simulation_failure_is_raised(monkeypatch, error):
    vg = _vg_with_kernel(monkeypatch, error)
    with pytest.raises(type(error)):
        vg.simulate(np.array([0.0]), 1.0, SimulationConfig(n_paths=8, n_steps=4, random_seed=1))


def test_the_cos_pricer_refuses_a_process_without_a_transform():
    """It used to price with a characteristic function of zero, silently."""

    class NoTransform(GBM):
        def characteristic_function(self, u, X0, t):
            raise NotImplementedError("no characteristic function")

    with pytest.raises(NotImplementedError):
        COSPricer(risk_free_rate=0.0).price(EuropeanCall(strike=100.0, maturity=1.0), NoTransform(mu=0.0, sigma=0.2),
                                            np.array([100.0]))


def test_the_heston_chain_fit_reports_its_error_in_implied_vol():
    """rmse_iv was a TODO (always None): model and market prices are now inverted alike."""
    from datetime import date, timedelta

    import jax.numpy as jnp

    from mjollnir.calibration.risk_neutral.heston_calibrator import HestonCalibrator
    from mjollnir.jax import configure_runtime, fourier_price
    from mjollnir.synthetic_data.data_provider import OptionChain, OptionQuote

    configure_runtime()
    truth = {"kappa": 2.0, "theta": 0.04, "xi": 0.4, "rho": -0.6, "v0": 0.05}
    today = date(2026, 1, 2)
    options = []
    for days in (30, 91):
        for strike in (90.0, 100.0, 110.0):
            is_call = strike >= 100.0
            mid = float(fourier_price(100.0, strike, days / 365.0, 0.02, 0.0, truth["v0"], truth["kappa"],
                                      truth["theta"], truth["xi"], truth["rho"], is_call))
            options.append(OptionQuote(strike=strike, expiry=today + timedelta(days=days),
                                       option_type="call" if is_call else "put", bid=mid, ask=mid, mid=mid,
                                       last=mid, volume=10, open_interest=100))
    chain = OptionChain(underlying="SYN", spot_price=100.0, reference_date=today, risk_free_rate=0.02,
                        dividend_yield=0.0, options=options)
    result = HestonCalibrator().calibrate(chain, initial_guess=truth, method="L-BFGS-B", maxiter=5)
    assert np.isfinite(result.rmse_iv) and result.rmse_iv < 0.01
    assert jnp.isfinite(result.rmse)


R_PRICE = 0.03


def _reference_and_process(model):
    from mjollnir.jax import configure_runtime, fourier_price
    from mjollnir.model_pricing import bs_price, merton_price
    from mjollnir.processes import Heston
    from mjollnir.processes.merton import MertonJD

    configure_runtime()
    if model == "gbm":
        return (lambda K, T, c: float(bs_price(100.0, K, T, R_PRICE, 0.0, 0.25, c)),
                GBM(mu=R_PRICE, sigma=0.25), np.array([100.0]))
    if model == "merton":
        return (lambda K, T, c: float(merton_price(100.0, K, T, R_PRICE, 0.0, 0.15, 2.0, -0.08, 0.1, c)),
                MertonJD(mu=R_PRICE, sigma=0.15, lambda_jump=2.0, mu_J=-0.08, sigma_J=0.1), np.array([100.0]))
    return (lambda K, T, c: float(fourier_price(100.0, K, T, R_PRICE, 0.0, 0.05, 2.0, 0.04, 0.4, -0.6, c)),
            Heston(mu=R_PRICE, kappa=2.0, theta=0.04, sigma_v=0.4, rho=-0.6, v0=0.05), np.array([100.0, 0.05]))


@pytest.mark.parametrize("pricer", ["cos", "carr_madan"])
@pytest.mark.parametrize("model", ["gbm", "merton", "heston"])
@pytest.mark.parametrize(("strike", "maturity"), [(80.0, 0.25), (100.0, 0.1), (120.0, 1.0)])
@pytest.mark.parametrize("is_call", [True, False])
def test_the_generic_fourier_pricers_match_reference_prices(pricer, model, strike, maturity, is_call):
    """COS multiplied its payoff coefficients by the strike twice and sized its range from the
    price's variance (a Heston put worth 0.19 came out at 1677); Carr-Madan read its prices off
    a log-strike grid centred on 0, ignored puts, and returned negative prices."""
    from mjollnir.derivatives import EuropeanPut
    from mjollnir.pricer.fourier import CarrMadanPricer

    reference, process, x0 = _reference_and_process(model)
    option = (EuropeanCall if is_call else EuropeanPut)(strike=strike, maturity=maturity)
    engine = COSPricer(risk_free_rate=R_PRICE, N=256) if pricer == "cos" else CarrMadanPricer(risk_free_rate=R_PRICE)
    assert engine.price(option, process, x0).price == pytest.approx(reference(strike, maturity, is_call), abs=2e-5,
                                                                     rel=1e-5)
