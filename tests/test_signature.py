"""Tests for the signature kernel — closed-form laws, not just smoke."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.jax import configure_runtime
from mjollnir.jax._signature import leadlag, log_signature, signature, signature_dim

configure_runtime()


class TestSignatureLaws:
    def test_linear_path_is_tensor_exponential(self):
        """S^k of a straight line = dx^{⊗k} / k! — exact closed form."""
        dx = np.array([0.3, -0.2])
        path = jnp.asarray(np.stack([np.zeros(2), dx]))
        sig = np.asarray(signature(path, depth=3))
        lvl1 = dx
        lvl2 = np.outer(dx, dx).ravel() / 2
        lvl3 = np.einsum("i,j,k->ijk", dx, dx, dx).ravel() / 6
        np.testing.assert_allclose(sig, np.concatenate([lvl1, lvl2, lvl3]),
                                   rtol=1e-12)

    def test_chen_identity(self):
        """S(X concat Y) equals the tensor product S(X) ⊗ S(Y)."""
        rng = np.random.default_rng(0)
        X = np.cumsum(rng.standard_normal((5, 2)), axis=0)
        Y = X[-1] + np.cumsum(rng.standard_normal((4, 2)), axis=0)
        full = jnp.asarray(np.concatenate([X, Y]))
        sig_full = np.asarray(signature(full, depth=3))

        d, depth = 2, 3
        a = np.asarray(signature(jnp.asarray(X), depth=depth))
        b = np.asarray(signature(jnp.asarray(np.vstack([X[-1], Y])), depth=depth))

        def unflatten(v):
            out, i = [], 0
            for k in range(1, depth + 1):
                out.append(v[i:i + d**k])
                i += d**k
            return out

        A, B = unflatten(a), unflatten(b)
        prod = []
        for k in range(1, depth + 1):
            acc = A[k - 1] + B[k - 1]
            for i in range(1, k):
                acc = acc + np.outer(A[i - 1], B[k - i - 1]).ravel()
            prod.append(acc)
        np.testing.assert_allclose(sig_full, np.concatenate(prod), rtol=1e-10)

    def test_reparameterization_invariance(self):
        """Inserting collinear midpoints must not change the signature."""
        rng = np.random.default_rng(1)
        path = np.cumsum(rng.standard_normal((6, 2)), axis=0)
        mid = (path[:-1] + path[1:]) / 2
        refined = np.empty((11, 2))
        refined[0::2], refined[1::2] = path, mid
        np.testing.assert_allclose(
            np.asarray(signature(jnp.asarray(path), 3)),
            np.asarray(signature(jnp.asarray(refined), 3)),
            rtol=1e-10)

    def test_level2_iterated_integrals(self):
        """L-shaped path (0,0)->(1,0)->(1,1): S12 = 1, S21 = 0, Sii = 1/2."""
        path = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        sig = np.asarray(signature(path, depth=2))
        lvl2 = sig[2:].reshape(2, 2)
        np.testing.assert_allclose(lvl2, [[0.5, 1.0], [0.0, 0.5]], atol=1e-12)

    def test_log_signature_of_line_is_increment(self):
        """log of a segment exponential = the increment (higher levels zero)."""
        dx = np.array([0.4, 0.1, -0.3])
        path = jnp.asarray(np.stack([np.zeros(3), dx]))
        logsig = np.asarray(log_signature(path, depth=3))
        np.testing.assert_allclose(logsig[:3], dx, rtol=1e-12)
        np.testing.assert_allclose(logsig[3:], 0.0, atol=1e-12)

    def test_signature_dim(self):
        assert signature_dim(2, 3) == 2 + 4 + 8
        rng = np.random.default_rng(2)
        path = jnp.asarray(np.cumsum(rng.standard_normal((10, 2)), axis=0))
        assert signature(path, 3).shape == (14,)


class TestLeadLag:
    def test_shape(self):
        path = jnp.asarray(np.random.default_rng(0).standard_normal((16, 1)).cumsum(0))
        assert leadlag(path).shape == (31, 2)

    def test_levy_area_is_quadratic_variation(self):
        """Lead-lag Levy area S12 - S21 = sum of squared increments."""
        rng = np.random.default_rng(3)
        path = np.cumsum(0.1 * rng.standard_normal((20, 1)), axis=0)
        qv = float((np.diff(path, axis=0) ** 2).sum())
        sig = np.asarray(signature(leadlag(jnp.asarray(path)), depth=2))
        s12, s21 = sig[2 + 1], sig[2 + 2]
        assert (s12 - s21) == pytest.approx(qv, rel=1e-10)


class TestJaxContract:
    def test_jit_vmap_grad(self):
        rng = np.random.default_rng(4)
        batch = jnp.asarray(np.cumsum(rng.standard_normal((8, 12, 2)), axis=1))
        sigs = jax.jit(lambda p: signature(p, 3))(batch)
        assert sigs.shape == (8, signature_dim(2, 3))
        g = jax.grad(lambda p: signature(p, 3).sum())(batch[0])
        assert bool(jnp.isfinite(g).all()) and g.shape == batch[0].shape


@pytest.mark.skipif(importlib.util.find_spec("iisignature") is None,
                    reason="iisignature not installed (dev-only cross-check)")
class TestCrossCheck:
    def test_against_iisignature(self):
        import iisignature

        rng = np.random.default_rng(5)
        path = np.cumsum(rng.standard_normal((30, 3)), axis=0)
        ours = np.asarray(signature(jnp.asarray(path), depth=4))
        ref = iisignature.sig(path, 4)
        np.testing.assert_allclose(ours, ref, rtol=1e-9)
