"""
Merton jump-diffusion vanilla prices by MGF inversion, NumPy in and out.

Thin wrappers over :mod:`mjollnir.pricer._jax_mgf_pricer`, which holds the one
implementation of the inversion.
"""

import numpy as np

from ._jax_mgf_pricer import NODES, merton_price_slice_fast


def merton_price_vanilla(S: float, K: float, T: float, r: float, q: float, sigma: float, lambda_jump: float,
                         mu_J: float, sigma_J: float, is_call: bool, max_phi: int = NODES) -> float:
    """One Merton vanilla price (``T <= 0`` gives the intrinsic value)."""
    return float(merton_price_slice(S, np.array([K]), T, r, q, sigma, lambda_jump, mu_J, sigma_J,
                                    np.array(["call" if is_call else "put"]), max_phi)[0])


def merton_price_slice(S: float, strikes: np.ndarray, T: float, r: float, q: float, sigma: float,
                       lambda_jump: float, mu_J: float, sigma_J: float, option_types: np.ndarray,
                       max_phi: int = NODES) -> np.ndarray:
    """Merton vanilla prices for several strikes of one maturity; ``option_types`` holds 'call' or 'put'."""
    strikes = np.asarray(strikes, dtype=float)
    if T <= 0:
        calls = np.asarray(option_types) == "call"
        return np.where(calls, np.maximum(S - strikes, 0.0), np.maximum(strikes - S, 0.0))
    return merton_price_slice_fast(S, strikes, T, r, q, sigma, lambda_jump, mu_J, sigma_J, list(option_types),
                                   max_phi)
