"""Property-based tests (hypothesis) — structural invariants of the kernel.

Golden tests pin exact numbers at a handful of points; these pin *laws* that
must hold across the whole sane parameter space:

- put-call parity of the differentiable COS pricer,
- monotonicity of call prices in spot and strike,
- QE variance positivity under extreme vol-of-vol,
- no-arbitrage structure of generated synthetic chains.
"""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mjollnir.jax import configure_runtime, fourier_price, qe_heston_step

configure_runtime()

# Sane Heston box: keeps the COS truncation and CF numerics in their
# well-behaved regime (this is a pricer-contract test, not a stress test).
heston_params = st.fixed_dictionaries({
    "v0": st.floats(0.01, 0.25),
    "kappa": st.floats(0.5, 6.0),
    "theta": st.floats(0.01, 0.25),
    "sigma_v": st.floats(0.05, 0.9),
    "rho": st.floats(-0.95, 0.0),
})
spots = st.floats(60.0, 150.0)
strikes = st.floats(70.0, 140.0)
maturities = st.floats(0.05, 2.0)


class TestPricerLaws:
    @settings(max_examples=60, deadline=None)
    @given(p=heston_params, S0=spots, K=strikes, T=maturities)
    def test_put_call_parity(self, p, S0, K, T):
        r, q = 0.02, 0.01
        call = float(fourier_price(S0, K, T, r, q, **p, is_call=True))
        put = float(fourier_price(S0, K, T, r, q, **p, is_call=False))
        forward = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        assert call - put == pytest.approx(forward, abs=5e-4 * S0)

    # COS is a spectral method: below ~1e-6 absolute the result is grid
    # noise, not price (e.g. 16-sigma OTM options are "zero"). Monotonicity
    # is asserted only where prices are numerically meaningful.
    PRICE_FLOOR = 1e-6

    @settings(max_examples=40, deadline=None)
    @given(p=heston_params, K=strikes, T=maturities)
    def test_call_increasing_in_spot(self, p, K, T):
        grid = np.array([80.0, 95.0, 110.0, 125.0])
        prices = [float(fourier_price(s, K, T, 0.02, 0.0, **p)) for s in grid]
        meaningful = [x for x in prices if x > self.PRICE_FLOOR]
        assert all(b >= a - 1e-8 for a, b in pairwise(meaningful))

    @settings(max_examples=40, deadline=None)
    @given(p=heston_params, S0=spots, T=maturities)
    def test_call_decreasing_in_strike(self, p, S0, T):
        grid = np.array([80.0, 95.0, 110.0, 125.0])
        prices = [float(fourier_price(S0, k, T, 0.02, 0.0, **p)) for k in grid]
        meaningful = [x for x in prices if x > self.PRICE_FLOOR]
        assert all(b <= a + 1e-8 for a, b in pairwise(meaningful))

    @settings(max_examples=40, deadline=None)
    @given(p=heston_params, S0=spots, K=strikes, T=maturities)
    def test_call_bounded_by_spot_and_intrinsic(self, p, S0, K, T):
        r, q = 0.02, 0.0
        c = float(fourier_price(S0, K, T, r, q, **p))
        intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        assert c >= intrinsic - 5e-4 * S0
        assert c <= S0 * np.exp(-q * T) + 1e-6


class TestKernelLaws:
    @settings(max_examples=25, deadline=None)
    @given(
        sigma_v=st.floats(0.5, 2.5),         # deliberately Feller-violating
        kappa=st.floats(0.1, 1.0),
        theta=st.floats(0.005, 0.1),
        v0=st.floats(1e-4, 0.02),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_qe_variance_never_negative(self, sigma_v, kappa, theta, v0, seed):
        key = jax.random.PRNGKey(seed)
        s = jnp.full((8,), 100.0)
        v = jnp.full((8,), v0)
        for _ in range(32):
            s, v, key = qe_heston_step(
                s, v, dt=1 / 52, mu=0.0, kappa=kappa, theta=theta,
                sigma_v=sigma_v, rho=-0.9, key=key,
            )
            assert bool((v >= 0.0).all())
        assert bool(jnp.isfinite(s).all())

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**31 - 1))
    def test_qe_key_threading_is_pure(self, seed):
        """Same key in => bitwise same (spot, var, key) out; no hidden state."""
        key = jax.random.PRNGKey(seed)
        args = (jnp.full((4,), 100.0), jnp.full((4,), 0.04))
        kw = dict(dt=1 / 252, mu=0.02, kappa=2.0, theta=0.04,
                  sigma_v=0.3, rho=-0.7, key=key)
        s1, v1, k1 = qe_heston_step(*args, **kw)
        s2, v2, k2 = qe_heston_step(*args, **kw)
        assert (s1 == s2).all() and (v1 == v2).all() and (k1 == k2).all()


class TestSyntheticChainLaws:
    @settings(max_examples=10, deadline=None)
    @given(
        atm_iv=st.floats(0.12, 0.45),
        spot=st.floats(50.0, 400.0),
    )
    def test_chain_is_arbitrage_free_in_strike(self, atm_iv, spot):
        """Within each (maturity, right): mids decreasing (calls) /
        increasing (puts) in strike, and all quotes positive."""
        import warnings
        from datetime import date
        from mjollnir.synthetic_data import (
            HestonVolatilityProfile,
            SyntheticEquityOptionChainGenerator,
        )
        with warnings.catch_warnings():
            # low-IV draws violate Feller with the fixed xi — expected here;
            # the profile warns (correctly), and the chain must STILL be
            # arbitrage-free, which is exactly what this test asserts
            warnings.simplefilter("ignore", UserWarning)
            prof = HestonVolatilityProfile(
                kappa=3.0, theta=atm_iv**2, xi=0.4, rho=-0.6,
                v0=atm_iv**2, atm_iv=atm_iv,
            )
        gen = SyntheticEquityOptionChainGenerator(random_seed=0)
        chain = gen.generate_single_chain(
            reference_date=date(2024, 1, 2), spot_price=spot, vol_profile=prof,
        )
        for expiry in {q.expiry for q in chain.options}:
            for right in ("call", "put"):
                qs = sorted(
                    (q for q in chain.options
                     if q.expiry == expiry and q.option_type == right),
                    key=lambda q: q.strike,
                )
                mids = [q.mid for q in qs]
                assert all(m > 0 for m in mids)
                if right == "call":
                    assert all(b <= a + 1e-6 * spot for a, b in pairwise(mids))
                else:
                    assert all(b >= a - 1e-6 * spot for a, b in pairwise(mids))
