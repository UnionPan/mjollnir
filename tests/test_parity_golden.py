"""Golden-key parity tests — the reproducibility contract of ``mjollnir.jax``.

These tests pin the *numerical behaviour* of the public simulation kernel to
golden values generated at extraction time (2026-07-12, from the code as it lived
in the ``option_pricer_DLC`` monorepo, CPU/float64 via ``configure_runtime``).

If any of these fail after a change, env dynamics have drifted: RL experiments
pinned to a previous version are no longer comparable. That is a breaking change —
either revert, or bump the major/minor version and regenerate the goldens
deliberately, never silently.
"""

import jax
import jax.numpy as jnp
import pytest

from mjollnir.jax import (
    configure_runtime,
    fourier_price,
    fourier_price_batch,
    jax_cos_price_heston,
    qe_heston_step,
)

configure_runtime()

# Canonical Heston test point (near-ATM, 6M, leveraged equity-like dynamics)
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
T, R, Q = 0.5, 0.02, 0.0

# ---------------------------------------------------------------------------
# Golden values (generated 2026-07-12, CPU, float64)
# ---------------------------------------------------------------------------
GOLDEN_SPOT_64 = 89.24342140181123      # 64-step QE rollout, PRNGKey(42)
GOLDEN_VAR_64 = 0.08583288257697771
GOLDEN_PRICE = 5.986438898451416        # ATM call, S0=K=100
GOLDEN_DELTA = 0.6108256957214958       # d price / d S0
GOLDEN_DPDV0 = 43.95845557345104        # d price / d v0
GOLDEN_BATCH = [12.749299984536274, 5.986438898451429, 10.818755773028288]
# strikes [90, 100, 110], is_call [True, True, False]


def _roll(key, n_steps=64, batch=4):
    s = jnp.full((batch,), 100.0)
    v = jnp.full((batch,), HESTON["v0"])
    for _ in range(n_steps):
        s, v, key = qe_heston_step(
            s, v, dt=1 / 252, mu=R,
            kappa=HESTON["kappa"], theta=HESTON["theta"],
            sigma_v=HESTON["sigma_v"], rho=HESTON["rho"], key=key,
        )
    return s, v


class TestQEHestonKernel:
    def test_golden_rollout(self):
        """Fixed key -> bitwise-stable 64-step rollout."""
        s, v = _roll(jax.random.PRNGKey(42))
        assert float(s[0]) == pytest.approx(GOLDEN_SPOT_64, abs=1e-9)
        assert float(v[0]) == pytest.approx(GOLDEN_VAR_64, abs=1e-12)

    def test_determinism_same_key(self):
        s1, v1 = _roll(jax.random.PRNGKey(7))
        s2, v2 = _roll(jax.random.PRNGKey(7))
        assert (s1 == s2).all() and (v1 == v2).all()

    def test_different_keys_differ(self):
        s1, _ = _roll(jax.random.PRNGKey(0))
        s2, _ = _roll(jax.random.PRNGKey(1))
        assert not (s1 == s2).all()

    def test_variance_nonnegative(self):
        """QE scheme guarantees v >= 0 even for high vol-of-vol."""
        key = jax.random.PRNGKey(3)
        s = jnp.full((16,), 100.0)
        v = jnp.full((16,), 0.0001)
        for _ in range(128):
            s, v, key = qe_heston_step(
                s, v, dt=1 / 52, mu=0.0, kappa=0.5, theta=0.04,
                sigma_v=1.5, rho=-0.9, key=key,
            )
        assert (v >= 0).all()


class TestDifferentiablePricer:
    def test_golden_price(self):
        p = float(fourier_price(100.0, 100.0, T, R, Q, **HESTON))
        assert p == pytest.approx(GOLDEN_PRICE, abs=1e-10)

    def test_parity_with_legacy_wrapper(self):
        """Public fourier_price must reproduce the monorepo-era scalar wrapper."""
        p_new = float(fourier_price(100.0, 100.0, T, R, Q, **HESTON))
        p_old = float(jax_cos_price_heston(100.0, 100.0, T, R, Q, **HESTON))
        assert p_new == pytest.approx(p_old, abs=1e-12)

    def test_golden_delta(self):
        delta = float(jax.grad(
            lambda s0: fourier_price(s0, 100.0, T, R, Q, **HESTON))(100.0))
        assert delta == pytest.approx(GOLDEN_DELTA, abs=1e-10)

    def test_golden_v0_sensitivity(self):
        g = float(jax.grad(lambda v0: fourier_price(
            100.0, 100.0, T, R, Q, v0=v0, kappa=HESTON["kappa"],
            theta=HESTON["theta"], sigma_v=HESTON["sigma_v"],
            rho=HESTON["rho"]))(HESTON["v0"]))
        assert g == pytest.approx(GOLDEN_DPDV0, rel=1e-9)

    def test_gradients_finite(self):
        """No NaN cotangents through the COS machinery."""
        for wrt, x0 in [(0, 100.0)]:
            g = jax.grad(lambda s0: fourier_price(s0, 100.0, T, R, Q, **HESTON))(x0)
            assert jnp.isfinite(g)

    def test_golden_batch_jit(self):
        f = jax.jit(lambda s0: fourier_price_batch(
            s0, jnp.array([90.0, 100.0, 110.0]), T, R, Q,
            HESTON["v0"], HESTON["kappa"], HESTON["theta"],
            HESTON["sigma_v"], HESTON["rho"],
            jnp.array([True, True, False])))
        out = f(100.0)
        for got, want in zip(out, GOLDEN_BATCH):
            assert float(got) == pytest.approx(want, abs=1e-9)

    def test_vmap_over_spot(self):
        f = jax.vmap(lambda s0: fourier_price(s0, 100.0, T, R, Q, **HESTON))
        p = f(jnp.array([90.0, 100.0, 110.0]))
        assert (jnp.diff(p) > 0).all()  # call price increasing in spot
