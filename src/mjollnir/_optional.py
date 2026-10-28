"""Optional dependencies: imported where a module needs them, named by the extra that installs them.

The core install (``pip install mjollnir``) carries what the simulators,
pricers, hedging backtests and calibrators need. Market-data fetchers, the
calibration pipeline and the gym environments pull in more, so they live in
extras::

    pip install 'mjollnir[data]'      # yfinance and exchange fetchers
    pip install 'mjollnir[pipeline]'  # price/results stores, universe runner
    pip install 'mjollnir[envs]'      # gymnasium environments, zarr cache, renderer
    pip install 'mjollnir[all]'
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import ``module``, or raise an ImportError that names the extra providing it.

    Only the dependency's own absence is translated: an ImportError raised from
    inside an installed package propagates unchanged.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] != module.split(".")[0]:
            raise
        raise ImportError(
            f"{module} is not installed; it comes with mjollnir's [{extra}] extra: "
            f"pip install 'mjollnir[{extra}]'"
        ) from error
