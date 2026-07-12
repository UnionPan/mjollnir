"""
mjollnir.jax — the frozen public JAX simulation kernel (RL substrate).

This is the #1 API contract: the differentiable, jittable, batchable primitives that
RL environments (deep hedging / POMARL) pin a version against. It promotes what were
private, underscore-prefixed monorepo internals into a stable public surface, so
experiments never again import ``_jax_*`` module paths whose behaviour could drift.

Reproducibility contract: every stochastic primitive takes an explicit ``jax`` PRNG
key and threads a new key out — no hidden global RNG. Given the same key and inputs,
outputs are deterministic (see ``tests/test_parity_golden.py``).

Headline primitives
-------------------
* ``configure_runtime()``  — device/precision setup (call once at import of a runner).
* ``qe_heston_step(...)``   — one Andersen quadratic-exponential Heston ``(S, v)`` step.
* ``fourier_price(...)``    — differentiable Heston COS/Fourier vanilla price (single).
* ``fourier_price_batch``   — batched COS pricer over a strike/maturity grid.

Lower-level building blocks (characteristic functions, COS coefficients, truncation)
are re-exported for consumers that compose their own pricing graphs.
"""

# --- runtime / device configuration --------------------------------------------
from mjollnir.processes._jax_backend import configure_jax_runtime as configure_runtime

# --- Heston variance-step kernel -----------------------------------------------
from mjollnir.processes._jax_qe import qe_heston_step

# --- differentiable Fourier/COS pricer -----------------------------------------
# Headline pricers: fully traced (grad/jit/vmap-safe wrt all market inputs).
from mjollnir.jax._pricing import fourier_price, fourier_price_batch

from mjollnir.pricer._jax_fourier_pricer import (
    HestonCFParams,
    jax_heston_cf,
    # numpy-orchestrated scalar conveniences (NOT differentiable; kept for parity)
    jax_cos_price_heston,
    jax_cos_price_heston_multi,
    # promote former private internals to stable public aliases:
    _jax_cos_price_multi as cos_price_multi,
    _jax_cos_price_single as cos_price_single,
    _chi_k as chi_k,
    _psi_k as psi_k,
    _cos_truncation_range as cos_truncation_range,
)

# headline aliases
heston_cf = jax_heston_cf

__all__ = [
    # headline primitives
    "configure_runtime",
    "qe_heston_step",
    "fourier_price",
    "fourier_price_batch",
    "heston_cf",
    # concrete Heston building blocks
    "HestonCFParams",
    "jax_heston_cf",
    "jax_cos_price_heston",
    "jax_cos_price_heston_multi",
    # composable COS internals (now public)
    "cos_price_multi",
    "cos_price_single",
    "chi_k",
    "psi_k",
    "cos_truncation_range",
]
