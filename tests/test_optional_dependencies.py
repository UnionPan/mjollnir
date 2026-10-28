"""The core install imports and runs without the extras; the extras' members say which extra they need."""

import subprocess
import sys
import textwrap

EXTRAS_ONLY = ("yfinance", "requests", "joblib", "sklearn", "pyarrow", "zarr", "gymnasium", "matplotlib")

PROGRAM = textwrap.dedent(f"""
    import importlib.abc
    import sys

    class Absent(importlib.abc.MetaPathFinder):
        # the extras' packages, as if they were not installed
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in {EXTRAS_ONLY!r}:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None

    sys.meta_path.insert(0, Absent())

    import mjollnir
    import mjollnir.backtest, mjollnir.hedging, mjollnir.inference, mjollnir.scenarios, mjollnir.worlds
    import mjollnir.calibration.marketdata as marketdata
    import mjollnir.calibration.physical as physical
    import mjollnir.calibration.risk_neutral
    import mjollnir.simulations as simulations

    physical.GBMCalibrator, physical.GARCHCalibrator, physical.RegimeSwitchingCalibrator, physical.HestonQMLECalibrator
    marketdata.OptionChain, marketdata.MarketData

    for package, name, extra in ((marketdata, "YFinanceFetcher", "data"), (marketdata, "BinanceFetcher", "data"),
                                 (physical, "calibrate_universe", "pipeline"), (simulations, "HestonEnv", "envs"),
                                 (simulations, "HestonEnvRenderer", "envs")):
        try:
            getattr(package, name)
        except ImportError as error:
            assert f"[{{extra}}]" in str(error), (name, str(error))
        else:
            raise AssertionError(f"{{name}} loaded without the [{{extra}}] extra")
    print("ok")
""")


def test_core_imports_without_the_extras():
    result = subprocess.run([sys.executable, "-c", PROGRAM], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith("ok")
