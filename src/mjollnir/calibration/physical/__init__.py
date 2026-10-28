"""
Physical (P-measure) calibrators.

Calibrate models from historical price data (OHLCV).
Used for real-world forecasting, risk management, and counterfactual simulation.

Key characteristic: Drift μ is estimated from historical returns (not set to r).

Every name loads with its first use: the calibrators need only the core
install, the universe pipeline (``calibrate_universe`` and its report) the
[pipeline] extra.
"""

from mjollnir._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(__name__, globals(), {
    "GBMCalibrator": ".gbm_calibrator",
    "GBMCalibrationResult": ".gbm_calibrator",
    "OUCalibrator": ".ou_calibrator",
    "OUCalibrationResult": ".ou_calibrator",
    "VolatilityEstimate": ".volatility",
    "VolatilityEstimator": ".volatility",
    "GARCHEstimator": ".volatility",
    "compare_volatility_methods": ".volatility",
    "parkinson_volatility": ".volatility",
    "garman_klass_volatility": ".volatility",
    "rogers_satchell_volatility": ".volatility",
    "yang_zhang_volatility": ".volatility",
    "compare_ohlc_estimators": ".volatility",
    "RegimeSwitchingCalibrator": ".regime_switching_calibrator",
    "RegimeSwitchingSimulator": ".regime_switching_calibrator",
    "RegimeSwitchingCalibrationResult": ".regime_switching_calibrator",
    "RegimeParameters": ".regime_switching_calibrator",
    "RoughBergomiCalibrator": ".rough_bergomi_calibrator",
    "RoughBergomiCalibrationResult": ".rough_bergomi_calibrator",
    "RoughBergomiVariogramCalibrator": ".rough_vol_variogram",
    "RoughBergomiVariogramResult": ".rough_vol_variogram",
    "BootstrapResult": ".uncertainty",
    "observed_information_std_errors": ".uncertainty",
    "parametric_bootstrap": ".uncertainty",
    "MertonJumpCalibrator": ".merton_calibrator",
    "MertonCalibrationResult": ".merton_calibrator",
    "GARCHCalibrator": ".garch_calibrator",
    "GARCHCalibrationResult": ".garch_calibrator",
    "HestonParticleFilter": ".heston_particle_filter",
    "HestonParticleFilterResult": ".heston_particle_filter",
    "interval_years": ".heston_particle_filter",
    "HestonQMLECalibrator": ".heston_qmle",
    "HestonQMLEResult": ".heston_qmle",
    "AssetCalibration": ".multi_asset_pipeline",
    "DEFAULT_BASKET_50": ".multi_asset_pipeline",
    "calibrate_universe": ".multi_asset_pipeline",
    "calibration_report": ".multi_asset_pipeline",
    "save_calibration": ".multi_asset_pipeline",
    "RoughBergomiParticleFilter": ".rough_bergomi_particle_filter",
    "RoughBergomiParticleFilterResult": ".rough_bergomi_particle_filter",
    "CorrelationCalibrator": ".correlation",
    "CorrelationResult": ".correlation",
})
