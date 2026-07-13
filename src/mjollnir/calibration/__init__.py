"""
Model Calibration Module

Calibrate stochastic models to market data.

Organized by measure type:
- physical: P-measure calibrators (historical data → real-world dynamics)
- risk_neutral: Q-measure calibrators (option prices → risk-neutral dynamics)

All public names are resolved lazily (PEP 562): ``import mjollnir.calibration``
stays cheap, and heavy dependency chains (JAX pricers via the synthetic-data
layer, flax/optax via the NPE) load only when the objects that need them are
first touched.

author: Yunian Pan
email: yp1170@nyu.edu
"""

import importlib

# name -> submodule that provides it (relative to this package)
_LAZY_ATTRS = {
    # Data providers
    "YFinanceFetcher": ".marketdata.yfinance_fetcher",
    "MarketData": ".marketdata.data_provider",
    "OptionChain": ".marketdata.data_provider",
    # P-measure calibrators
    "HestonParticleFilter": ".physical.heston_particle_filter",
    "GBMCalibrator": ".physical",
    "GBMCalibrationResult": ".physical",
    "OUCalibrator": ".physical",
    "OUCalibrationResult": ".physical",
    "RegimeSwitchingCalibrator": ".physical",
    "RegimeSwitchingSimulator": ".physical",
    "RegimeSwitchingCalibrationResult": ".physical",
    "RegimeParameters": ".physical",
    # Q-measure calibrators
    "fit_heston_surface": ".risk_neutral.gradient_heston",
    "GradientHestonResult": ".risk_neutral.gradient_heston",
    "HestonCalibrator": ".risk_neutral",
    "CalibrationResult": ".risk_neutral",
    "RegimeSwitchingHestonCalibrator": ".risk_neutral",
    "RegimeSwitchingHestonSimulator": ".risk_neutral",
    "RegimeSwitchingHestonResult": ".risk_neutral",
    "RegimeHestonParameters": ".risk_neutral",
    # Joint P/Q calibration
    "JointPQCalibrator": ".joint_pq",
    "JointPQResult": ".joint_pq",
}

_LAZY_SUBMODULES = {
    "physical", "risk_neutral", "marketdata", "cli", "cross_asset", "pipeline",
}

__all__ = [*_LAZY_ATTRS, *sorted(_LAZY_SUBMODULES)]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache: resolve once
        return value
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
