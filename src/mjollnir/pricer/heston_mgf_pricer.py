"""
Heston vanilla prices by MGF inversion, NumPy in and out.

Thin wrappers over :mod:`mjollnir.pricer._jax_mgf_pricer`, which holds the one
implementation of the inversion (contours, quadrature and their accuracy are
documented there). Adapted from StochVolModels
(https://github.com/ArturSepp/StochVolModels).
"""

import numpy as np

from ._jax_mgf_pricer import NODES, heston_price_slice_fast


def heston_price_vanilla(S: float, K: float, T: float, r: float, q: float, v0: float, theta: float, kappa: float,
                         volvol: float, rho: float, is_call: bool, max_phi: int = NODES) -> float:
    """One Heston vanilla price (``volvol`` is the vol of variance; ``T <= 0`` gives the intrinsic value)."""
    return float(heston_price_slice(S, np.array([K]), T, r, q, v0, theta, kappa, volvol, rho,
                                    np.array(["call" if is_call else "put"]), max_phi)[0])


def heston_price_slice(S: float, strikes: np.ndarray, T: float, r: float, q: float, v0: float, theta: float,
                       kappa: float, volvol: float, rho: float, option_types: np.ndarray,
                       max_phi: int = NODES) -> np.ndarray:
    """Heston vanilla prices for several strikes of one maturity; ``option_types`` holds 'call' or 'put'."""
    strikes = np.asarray(strikes, dtype=float)
    if T <= 0:
        calls = np.asarray(option_types) == "call"
        return np.where(calls, np.maximum(S - strikes, 0.0), np.maximum(strikes - S, 0.0))
    return heston_price_slice_fast(S, strikes, T, r, q, v0, theta, kappa, volvol, rho, list(option_types), max_phi)
