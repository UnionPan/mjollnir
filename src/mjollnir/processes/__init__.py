"""Stochastic process models — the complete catalogue.

Every model class is exported here: this package is a menu, and a model that
exists but cannot be found from ``mjollnir.processes`` may as well not exist.
Grouped by family; all share the ``simulate(X0, T, config)`` contract with
per-call ``np.random.Generator`` seeding (see the reproducibility guide).
"""

from . import _jax_backend
from .base import (
    DriftDiffusionProcess,
    MultiFactorProcess,
    SimulationConfig,
    SingleFactorProcess,
    StochasticProcess,
)

# --- diffusions -------------------------------------------------------------
from .bachelier import Bachelier
from .cev import CEV
from .gbm import GBM
from .multi_asset_gbm import MultiAssetGBM
from .ornstein_uhlenbeck import OrnsteinUhlenbeck

# --- stochastic volatility --------------------------------------------------
from .four_half import FourHalf
from .heston import Heston
from .rough_bergomi import RoughBergomi
from .sabr import SABR
from .stochastic_local_vol import StochasticLocalVol
from .three_half import ThreeHalf

# --- jump / Levy ------------------------------------------------------------
from .bates import Bates
from .jump_diffusion import JumpDiffusionProcess
from .kou import KouJD
from .Levy import LevyProcess, SubordinatedBrownianMotion
from .merton import MertonJD
from .nig import NIG
from .variance_gamma import VarianceGamma

# --- regime switching -------------------------------------------------------
from .regime_switching import RegimeSwitchingProcess
from .regime_switching_gbm import RegimeSwitchingGBM
from .regime_switching_merton import RegimeSwitchingMerton

# --- short rate -------------------------------------------------------------
from .short_rate import CIR, HullWhite, Vasicek

__all__ = [
    "CEV",
    "CIR",
    "GBM",
    "NIG",
    "SABR",
    "Bachelier",
    "Bates",
    "DriftDiffusionProcess",
    "FourHalf",
    "Heston",
    "HullWhite",
    "JumpDiffusionProcess",
    "KouJD",
    "LevyProcess",
    "MertonJD",
    "MultiAssetGBM",
    "MultiFactorProcess",
    "OrnsteinUhlenbeck",
    "RegimeSwitchingGBM",
    "RegimeSwitchingMerton",
    "RegimeSwitchingProcess",
    "RoughBergomi",
    "SimulationConfig",
    "SingleFactorProcess",
    "StochasticLocalVol",
    "StochasticProcess",
    "SubordinatedBrownianMotion",
    "ThreeHalf",
    "VarianceGamma",
    "Vasicek",
    "_jax_backend",
]
