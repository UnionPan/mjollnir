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

# --- stochastic-volatility step kernels (Andersen QE family) --------------------
from mjollnir.processes._jax_qe import (
    qe_heston_step,
    qe_bates_step,        # Heston + Merton jumps
    qe_three_half_step,   # 3/2 model
    qe_four_half_step,    # 4/2 model (Grasselli 2017)
)

# --- differentiable Fourier/COS pricer -----------------------------------------
# Headline pricers: fully traced (grad/jit/vmap-safe wrt all market inputs).
from mjollnir.jax._pricing import fourier_price, fourier_price_batch

from mjollnir.pricer._jax_fourier_pricer import (
    # characteristic-function parameter containers
    HestonCFParams,
    MertonCFParams,
    BatesCFParams,
    KouCFParams,
    VGCFParams,
    NIGCFParams,
    # characteristic functions (fully traced/differentiable)
    jax_heston_cf,
    jax_merton_cf,
    jax_bates_cf,
    jax_kou_cf,
    jax_vg_cf,
    jax_nig_cf,
    # numpy-orchestrated scalar conveniences (NOT differentiable; kept for parity)
    jax_cos_price_heston,
    jax_cos_price_heston_multi,
    jax_cos_price_merton,
    jax_cos_price_merton_multi,
    jax_cos_price_bates,
    jax_cos_price_bates_multi,
    jax_cos_price_kou,
    jax_cos_price_kou_multi,
    jax_cos_price_vg,
    jax_cos_price_vg_multi,
    jax_cos_price_nig,
    jax_cos_price_nig_multi,
    # promote former private internals to stable public aliases:
    _jax_cos_price_multi as cos_price_multi,
    _jax_cos_price_single as cos_price_single,
    _chi_k as chi_k,
    _psi_k as psi_k,
    _cos_truncation_range as cos_truncation_range,
)

# --- market impact + multi-agent contract ---------------------------------------
from mjollnir.jax._impact import (
    almgren_chriss_step,
    linear_impact,
    sqrt_impact,
)
from mjollnir.jax._market import MarketState, make_market_step

# --- path signatures (feature map / market-generator brick) ----------------------
from mjollnir.jax._signature import leadlag, log_signature, signature, signature_dim

# headline aliases
heston_cf = jax_heston_cf

__all__ = [
    "BatesCFParams",
    # CF parameter containers
    "HestonCFParams",
    "KouCFParams",
    "MarketState",
    "MertonCFParams",
    "NIGCFParams",
    "VGCFParams",
    "almgren_chriss_step",
    "chi_k",
    # headline primitives
    "configure_runtime",
    # composable COS internals (now public)
    "cos_price_multi",
    "cos_price_single",
    "cos_truncation_range",
    "fourier_price",
    "fourier_price_batch",
    "heston_cf",
    "jax_bates_cf",
    "jax_cos_price_bates",
    "jax_cos_price_bates_multi",
    # COS pricers (scalar + strike-vector, per model)
    "jax_cos_price_heston",
    "jax_cos_price_heston_multi",
    "jax_cos_price_kou",
    "jax_cos_price_kou_multi",
    "jax_cos_price_merton",
    "jax_cos_price_merton_multi",
    "jax_cos_price_nig",
    "jax_cos_price_nig_multi",
    "jax_cos_price_vg",
    "jax_cos_price_vg_multi",
    # characteristic functions
    "jax_heston_cf",
    "jax_kou_cf",
    "jax_merton_cf",
    "jax_nig_cf",
    "jax_vg_cf",
    "leadlag",
    "linear_impact",
    "log_signature",
    "make_market_step",
    "psi_k",
    # QE step-kernel family
    "qe_bates_step",
    "qe_four_half_step",
    "qe_heston_step",
    "qe_three_half_step",
    "signature",
    "signature_dim",
    "sqrt_impact",
]
