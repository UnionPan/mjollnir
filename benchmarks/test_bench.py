"""Performance benchmarks (pytest-benchmark).

Run with: ``pytest benchmarks/ --benchmark-only``

Kept outside ``tests/`` so the default suite stays fast. Reference numbers
(CPU, 2026-07-12) are recorded next to each benchmark; they are machine-
dependent, so CI treats this job as informational rather than gating — but a
10x regression against the reference should be treated as a bug (that is how
an 18-minute Merton reference fit shipped unnoticed in the monorepo era).
"""

from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.jax import configure_runtime, fourier_price, fourier_price_batch, qe_heston_step

configure_runtime()

HESTON = dict(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)


@pytest.fixture(scope="module")
def warm_kernel():
    """Trigger JIT compilation outside the timed region."""
    key = jax.random.PRNGKey(0)
    s = jnp.full((1024,), 100.0)
    v = jnp.full((1024,), 0.04)
    out = qe_heston_step(s, v, 1 / 252, 0.02, **HESTON, key=key)
    jax.block_until_ready(out)
    p = fourier_price(100.0, 100.0, 0.5, 0.02, 0.0, 0.04, **HESTON)
    jax.block_until_ready(p)
    pb = fourier_price_batch(100.0, jnp.linspace(80.0, 120.0, 41), 0.5, 0.02, 0.0,
                             0.04, 2.0, 0.04, 0.3, -0.7, jnp.ones(41, bool))
    jax.block_until_ready(pb)
    return s, v


def test_qe_heston_step_batch1024(benchmark, warm_kernel):
    """Reference: ~0.1 ms/step (CPU). One QE step, batch of 1024 paths."""
    s, v = warm_kernel
    key = jax.random.PRNGKey(1)

    def step():
        out = qe_heston_step(s, v, 1 / 252, 0.02, **HESTON, key=key)
        jax.block_until_ready(out)
        return out

    benchmark(step)


def test_fourier_price_single(benchmark, warm_kernel):
    """Reference: ~1 ms (CPU). Warm scalar COS price, N=256."""
    def price():
        p = fourier_price(100.0, 100.0, 0.5, 0.02, 0.0, 0.04, **HESTON)
        jax.block_until_ready(p)
        return p

    benchmark(price)


def test_fourier_price_batch41(benchmark, warm_kernel):
    """Reference: ~2 ms (CPU). 41-strike COS slice."""
    strikes = jnp.linspace(80.0, 120.0, 41)
    is_call = jnp.ones(41, bool)

    def price():
        p = fourier_price_batch(100.0, strikes, 0.5, 0.02, 0.0,
                                0.04, 2.0, 0.04, 0.3, -0.7, is_call)
        jax.block_until_ready(p)
        return p

    benchmark(price)


def test_synthetic_chain_generation(benchmark):
    """Reference: ~19 ms warm (CPU). One full equity option chain."""
    from mjollnir.synthetic_data import (
        HestonVolatilityProfile,
        SyntheticEquityOptionChainGenerator,
    )
    prof = HestonVolatilityProfile(kappa=4.0, theta=0.04, xi=0.5, rho=-0.7,
                                   v0=0.04, atm_iv=0.20)
    gen = SyntheticEquityOptionChainGenerator(random_seed=0)
    # warm the jitted slice pricer outside the timed region
    gen.generate_single_chain(reference_date=date(2024, 1, 2),
                              spot_price=100.0, vol_profile=prof)

    benchmark(lambda: gen.generate_single_chain(
        reference_date=date(2024, 1, 2), spot_price=100.0, vol_profile=prof))


def test_batched_merton_fit(benchmark):
    """Reference: ~1.5 s warm (CPU). One-asset Merton MLE, 2000 obs,
    3-start cosine Adam."""
    from mjollnir.calibration.physical.batched import merton as bmerton
    from mjollnir.calibration.physical.batched.common import pad_returns

    rng = np.random.default_rng(0)
    rets = 0.0002 + 0.01 * rng.standard_normal(2000)
    R, M = pad_returns([rets])
    bmerton.fit_batch(R, M, 1 / 252)  # warm/compile

    benchmark(lambda: bmerton.fit_batch(R, M, 1 / 252))


def test_scipy_merton_reference_fit(benchmark):
    """Reference: ~0.3 s (CPU). Vectorized scipy L-BFGS-B reference —
    was ~18 minutes before the (T, K) broadcast rewrite."""
    from mjollnir.calibration.physical.merton_calibrator import MertonJumpCalibrator

    rng = np.random.default_rng(0)
    rets = 0.0002 + 0.01 * rng.standard_normal(2000)
    prices = 100 * np.exp(np.insert(np.cumsum(rets), 0, 0))

    benchmark(lambda: MertonJumpCalibrator(k_max=5).fit(prices, dt=1 / 252))
