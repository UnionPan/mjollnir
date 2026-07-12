"""Golden-key tests for the widened ``mjollnir.jax`` kernel family.

Same contract as ``test_parity_golden.py``: these pin the numerics of the
newly promoted QE step kernels (Bates, 3/2, 4/2) and the per-model COS
pricers. Goldens generated 2026-07-12 (CPU, float64 via configure_runtime).
Breaking one of these = breaking simulated dynamics = never a silent change.
"""

import jax
import jax.numpy as jnp
import pytest

from mjollnir.jax import (
    configure_runtime,
    qe_bates_step,
    qe_four_half_step,
    qe_heston_step,
    qe_three_half_step,
    jax_cos_price_bates,
    jax_cos_price_kou,
    jax_cos_price_merton,
    jax_cos_price_nig,
    jax_cos_price_vg,
)

configure_runtime()

# Shared Heston-style base params (mu, kappa, theta, sigma_v, rho)
BASE = (0.02, 2.0, 0.04, 0.3, -0.7)
S0, K, T, R, Q = 100.0, 100.0, 0.5, 0.02, 0.0


def _roll(step, extra, n=64, batch=4):
    key = jax.random.PRNGKey(42)
    s = jnp.full((batch,), 100.0)
    v = jnp.full((batch,), 0.04)
    for _ in range(n):
        s, v, key = step(s, v, 1 / 252, *BASE, *extra, key=key)
    return s, v


class TestQEFamilyGolden:
    def test_bates_rollout(self):
        s, v = _roll(qe_bates_step, (0.5, -0.05, 0.08))  # lambda_j, mu_J, sigma_J
        assert float(s[0]) == pytest.approx(101.81350121243086, abs=1e-9)
        assert float(v[0]) == pytest.approx(0.044038742134618884, abs=1e-12)

    def test_three_half_rollout(self):
        s, v = _roll(qe_three_half_step, ())
        assert float(s[0]) == pytest.approx(105.46277871956336, abs=1e-9)
        assert float(v[0]) == pytest.approx(0.038681390811979466, abs=1e-12)

    def test_four_half_rollout(self):
        s, v = _roll(qe_four_half_step, (0.8, 0.2))  # a_coef, b_coef
        assert float(s[0]) == pytest.approx(112.23435948770828, abs=1e-9)
        assert float(v[0]) == pytest.approx(0.08583288257697771, abs=1e-12)

    def test_four_half_variance_matches_heston(self):
        """4/2 variance factor is the same CIR as Heston: identical key =>
        identical variance path (only the spot leg differs)."""
        _, v_heston = _roll(qe_heston_step, ())
        _, v_42 = _roll(qe_four_half_step, (0.8, 0.2))
        assert (v_heston == v_42).all()

    def test_determinism(self):
        for step, extra in [
            (qe_bates_step, (0.5, -0.05, 0.08)),
            (qe_three_half_step, ()),
            (qe_four_half_step, (0.8, 0.2)),
        ]:
            s1, v1 = _roll(step, extra, n=16)
            s2, v2 = _roll(step, extra, n=16)
            assert (s1 == s2).all() and (v1 == v2).all()


class TestCOSPricerFamilyGolden:
    def test_merton(self):
        p = float(jax_cos_price_merton(S0, K, T, R, Q, 0.2, 0.3, -0.05, 0.1))
        assert p == pytest.approx(6.339435597758064, abs=1e-10)

    def test_bates(self):
        p = float(jax_cos_price_bates(S0, K, T, R, Q,
                                      0.04, 2.0, 0.04, 0.3, -0.7,
                                      0.3, -0.05, 0.1))
        assert p == pytest.approx(6.225642936276384, abs=1e-10)

    def test_kou(self):
        p = float(jax_cos_price_kou(S0, K, T, R, Q, 0.2, 0.3, 0.4, 10.0, 5.0))
        assert p == pytest.approx(6.711946799260581, abs=1e-10)

    def test_vg(self):
        # signature: (theta_vg, sigma, nu)
        p = float(jax_cos_price_vg(S0, K, T, R, Q, -0.14, 0.2, 0.2))
        assert p == pytest.approx(6.058359498265711, abs=1e-10)

    def test_nig(self):
        p = float(jax_cos_price_nig(S0, K, T, R, Q, 15.0, -5.0, 0.5))
        assert p == pytest.approx(5.857620893854392, abs=1e-10)

    def test_jump_models_price_above_diffusion_only(self):
        """Adding jumps to the same diffusion must not cheapen an ATM call."""
        bates = float(jax_cos_price_bates(S0, K, T, R, Q,
                                          0.04, 2.0, 0.04, 0.3, -0.7,
                                          0.3, -0.05, 0.1))
        from mjollnir.jax import fourier_price
        heston = float(fourier_price(S0, K, T, R, Q, 0.04, 2.0, 0.04, 0.3, -0.7))
        assert bates > heston
