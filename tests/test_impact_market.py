"""Tests for the impact zoo and the multi-agent market contract."""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from mjollnir.jax import (
    MarketState,
    almgren_chriss_step,
    configure_runtime,
    linear_impact,
    make_market_step,
    qe_heston_step,
    sqrt_impact,
)

configure_runtime()

SPOT, ADV, SIGMA = 100.0, 1e6, 0.0126  # ~20% annual vol, daily


class TestImpactLaws:
    def test_zero_flow_is_identity(self):
        for fn in (
            lambda s, q: linear_impact(s, q, ADV),
            lambda s, q: sqrt_impact(s, q, ADV, SIGMA),
        ):
            assert float(fn(SPOT, 0.0)) == pytest.approx(SPOT)
        ep, ns = almgren_chriss_step(SPOT, 0.0, ADV, SIGMA)
        assert float(ep) == pytest.approx(SPOT) and float(ns) == pytest.approx(SPOT)

    def test_direction(self):
        """Buying moves price up, selling down, antisymmetrically."""
        for fn in (
            lambda q: linear_impact(SPOT, q, ADV),
            lambda q: sqrt_impact(SPOT, q, ADV, SIGMA),
        ):
            up, dn = float(fn(1e5)), float(fn(-1e5))
            assert up > SPOT > dn
            assert up - SPOT == pytest.approx(SPOT - dn, rel=1e-9)

    def test_sqrt_concavity(self):
        """Square-root law: 4x the flow, only 2x the impact."""
        m1 = float(sqrt_impact(SPOT, 1e5, ADV, SIGMA)) - SPOT
        m4 = float(sqrt_impact(SPOT, 4e5, ADV, SIGMA)) - SPOT
        assert m4 == pytest.approx(2.0 * m1, rel=1e-9)

    def test_linear_golden(self):
        # participation 0.1, lam 0.1 -> +1%
        assert float(linear_impact(SPOT, 1e5, ADV, lam=0.1)) == pytest.approx(101.0)

    def test_sqrt_golden(self):
        # participation 0.01 -> sqrt = 0.1; y=1, sigma=0.0126 -> +0.126%
        assert float(sqrt_impact(SPOT, 1e4, ADV, SIGMA)) == pytest.approx(
            100.0 * (1.0 + 0.0126 * 0.1), rel=1e-12)

    def test_almgren_chriss_split(self):
        """Trader pays temporary+permanent; market keeps only permanent."""
        ep, ns = almgren_chriss_step(SPOT, 1e5, ADV, SIGMA, eta=1.0, gamma=0.1)
        assert float(ep) > float(ns) > SPOT

    def test_differentiable(self):
        """Gradients flow through impact — the multi-agent coupling channel."""
        g = jax.grad(lambda q: sqrt_impact(SPOT, q, ADV, SIGMA))(1e5)
        assert jnp.isfinite(g) and g > 0
        g2 = jax.grad(lambda q: linear_impact(SPOT, q, ADV))(0.0)
        assert jnp.isfinite(g2) and g2 > 0


class TestMarketStep:
    HESTON: ClassVar[dict] = dict(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)

    def _step(self, impact=None):
        return make_market_step(dt=1 / 252, mu=0.02, **self.HESTON, impact=impact)

    def test_no_impact_matches_raw_kernel(self):
        """impact=None must reproduce qe_heston_step exactly."""
        key = jax.random.PRNGKey(3)
        state = MarketState.create(100.0, 0.04, key)
        out = self._step()(state, 0.0)
        s_ref, v_ref, k_ref = qe_heston_step(
            jnp.asarray(100.0), jnp.asarray(0.04), 1 / 252, 0.02,
            **self.HESTON, key=key)
        assert float(out.spot) == float(s_ref)
        assert float(out.variance) == float(v_ref)
        assert (out.key == k_ref).all()
        assert int(out.t) == 1

    def test_flow_moves_the_path(self):
        key = jax.random.PRNGKey(3)
        step = self._step(impact=lambda s, q: sqrt_impact(s, q, 1e6, 0.0126))
        state = MarketState.create(100.0, 0.04, key)
        quiet = step(state, 0.0)
        pushed = step(state, 2e5)
        assert float(pushed.spot) > float(quiet.spot)

    def test_jit_scan_vmap(self):
        """The contract composes with the full JAX machinery."""
        step = self._step(impact=lambda s, q: linear_impact(s, q, 1e6))

        def episode(key):
            state = MarketState.create(100.0, 0.04, key)
            def body(st, _):
                st = step(st, 1e4)
                return st, st.spot
            _final, spots = jax.lax.scan(body, state, None, length=32)
            return spots[-1]

        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        out = jax.jit(jax.vmap(episode))(keys)
        assert out.shape == (8,) and bool(jnp.isfinite(out).all())

    def test_gradient_through_market(self):
        """d(terminal spot)/d(flow) is finite and positive: agents are
        coupled through the market inside the differentiable graph."""
        step = self._step(impact=lambda s, q: linear_impact(s, q, 1e6))

        def terminal(flow):
            state = MarketState.create(100.0, 0.04, jax.random.PRNGKey(1))
            def body(st, _):
                return step(st, flow), None
            final, _ = jax.lax.scan(body, state, None, length=16)
            return final.spot

        g = jax.grad(terminal)(1e4)
        assert jnp.isfinite(g) and g > 0
