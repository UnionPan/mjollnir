"""
Risk-neutral (Q-measure) calibrators.

Calibrate models from option market prices (bids, asks, implied volatilities).
Used for derivatives pricing, hedging, and risk-neutral scenario analysis.

Key characteristic: Drift = r (risk-free rate), not real-world drift μ.
"""

from .heston_calibrator import HestonCalibrator, CalibrationResult
from .sabr_calibrator import SABRCalibrator, SABRCalibrationResult
from .dupire_calibrator import DupireCalibrator, DupireResult
from .regime_switching_heston_calibrator import (
    RegimeSwitchingHestonCalibrator,
    RegimeSwitchingHestonSimulator,
    RegimeSwitchingHestonResult,
    RegimeHestonParameters,
)
from .cf_calibrator import (
    CFCalibrator,
    CFCalibrationResult,
    make_merton_calibrator,
    make_vg_calibrator,
    make_nig_calibrator,
    make_kou_calibrator,
    make_bates_calibrator,
)
from .svi import (
    SVIParams,
    SSVIParams,
    SVIFitResult,
    SSVIFitResult,
    fit_svi_slice,
    fit_ssvi_surface,
    fit_svi_from_chain,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
)

__all__ = [
    'CFCalibrationResult',
    # Generic CF calibrator + model-specific factories
    'CFCalibrator',
    'CalibrationResult',
    # Dupire local volatility
    'DupireCalibrator',
    'DupireResult',
    # Heston stochastic volatility
    'HestonCalibrator',
    'RegimeHestonParameters',
    # Regime-switching Heston
    'RegimeSwitchingHestonCalibrator',
    'RegimeSwitchingHestonResult',
    'RegimeSwitchingHestonSimulator',
    'SABRCalibrationResult',
    # SABR stochastic volatility
    'SABRCalibrator',
    'SSVIFitResult',
    'SSVIParams',
    'SVIFitResult',
    # SVI / SSVI volatility surface
    'SVIParams',
    'check_butterfly_arbitrage',
    'check_calendar_arbitrage',
    'fit_ssvi_surface',
    'fit_svi_from_chain',
    'fit_svi_slice',
    'make_bates_calibrator',
    'make_kou_calibrator',
    'make_merton_calibrator',
    'make_nig_calibrator',
    'make_vg_calibrator',
]
