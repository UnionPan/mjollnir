"""
Physical (P-measure) calibrators.

Calibrate models from historical price data (OHLCV).
Used for real-world forecasting, risk management, and counterfactual simulation.

Key characteristic: Drift μ is estimated from historical returns (not set to r).
"""

from .gbm_calibrator import GBMCalibrator, GBMCalibrationResult
from .ou_calibrator import OUCalibrator, OUCalibrationResult
from .volatility import (
    VolatilityEstimate,
    VolatilityEstimator,
    GARCHEstimator,
    compare_volatility_methods,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
    compare_ohlc_estimators,
)
from .regime_switching_calibrator import (
    RegimeSwitchingCalibrator,
    RegimeSwitchingSimulator,
    RegimeSwitchingCalibrationResult,
    RegimeParameters,
)
from .rough_bergomi_calibrator import (
    RoughBergomiCalibrator,
    RoughBergomiCalibrationResult,
)
from .merton_calibrator import (
    MertonJumpCalibrator,
    MertonCalibrationResult,
)
from .garch_calibrator import (
    GARCHCalibrator,
    GARCHCalibrationResult,
)
from .heston_particle_filter import (
    HestonParticleFilter,
    HestonParticleFilterResult,
)
from .heston_qmle import (
    HestonQMLECalibrator,
    HestonQMLEResult,
)
from .multi_asset_pipeline import (
    AssetCalibration,
    DEFAULT_BASKET_50,
    calibrate_universe,
    calibration_report,
    save_calibration,
)
from .rough_bergomi_particle_filter import (
    RoughBergomiParticleFilter,
    RoughBergomiParticleFilterResult,
)
from .correlation import (
    CorrelationCalibrator,
    CorrelationResult,
)

__all__ = [
    'DEFAULT_BASKET_50',
    # Multi-asset calibration pipeline
    'AssetCalibration',
    # Correlation
    'CorrelationCalibrator',
    'CorrelationResult',
    'GARCHCalibrationResult',
    # GARCH
    'GARCHCalibrator',
    'GARCHEstimator',
    'GBMCalibrationResult',
    # GBM
    'GBMCalibrator',
    # Particle filters
    'HestonParticleFilter',
    'HestonParticleFilterResult',
    # Heston QMLE (physical-measure SV calibration)
    'HestonQMLECalibrator',
    'HestonQMLEResult',
    'MertonCalibrationResult',
    # Merton jump-diffusion
    'MertonJumpCalibrator',
    'OUCalibrationResult',
    # Ornstein-Uhlenbeck
    'OUCalibrator',
    'RegimeParameters',
    'RegimeSwitchingCalibrationResult',
    # Regime-switching
    'RegimeSwitchingCalibrator',
    'RegimeSwitchingSimulator',
    'RoughBergomiCalibrationResult',
    # Rough Bergomi
    'RoughBergomiCalibrator',
    'RoughBergomiParticleFilter',
    'RoughBergomiParticleFilterResult',
    # Volatility estimators
    'VolatilityEstimate',
    'VolatilityEstimator',
    'calibrate_universe',
    'calibration_report',
    'compare_ohlc_estimators',
    'compare_volatility_methods',
    'garman_klass_volatility',
    'parkinson_volatility',
    'rogers_satchell_volatility',
    'save_calibration',
    'yang_zhang_volatility',
]
