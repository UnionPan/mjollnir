"""Truncated path signatures in pure JAX (the signature kernel).

The first brick of the signature market generator
(``docs/design/neural-calibration.md``), and independently useful as a
feature map: signatures are the canonical, reparameterization-invariant
summary of a path — for NPE calibration features and RL observations alike.

Implementation notes:

* Piecewise-linear paths: the signature of one segment is the tensor
  exponential ``exp(dx) = (dx^{⊗k} / k!)_k``; the path signature is the Chen
  product of the segment exponentials, computed with ``lax.scan``.
* Levels are stored **flat** — level ``k`` has shape ``(d**k,)`` — so every
  tensor product is one outer product + reshape: jit/vmap/grad-safe with
  static ``depth`` and ``d``.
* ``log_signature`` is the truncated tensor-algebra logarithm (full
  coordinates, *not* projected to a Lyndon basis — dimensions are those of
  the signature itself).
* ``leadlag`` provides the standard embedding that makes quadratic variation
  visible to the signature (Levy-area terms pick up realized variance).

No dependency on esig/iisignature; a dev-only cross-check against
``iisignature`` runs when that package is installed.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["leadlag", "log_signature", "signature", "signature_dim"]


def signature_dim(channels: int, depth: int) -> int:
    """Length of the flat signature vector: ``d + d^2 + ... + d^depth``."""
    return sum(channels**k for k in range(1, depth + 1))


def _outer(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Flat tensor product: (d^i,) ⊗ (d^j,) -> (d^(i+j),)."""
    return (a[:, None] * b[None, :]).reshape(-1)


def _seg_exp(dx: jnp.ndarray, depth: int) -> list[jnp.ndarray]:
    """Levels of the tensor exponential of one linear segment."""
    levels = [dx]
    for k in range(2, depth + 1):
        levels.append(_outer(levels[-1], dx) / k)
    return levels


def _chen(a: list[jnp.ndarray], b: list[jnp.ndarray], depth: int) -> list[jnp.ndarray]:
    """Truncated tensor-algebra product of two group-like elements
    (scalar level 1 implicit): ``c_k = a_k + b_k + sum_{i+j=k} a_i ⊗ b_j``."""
    out = []
    for k in range(1, depth + 1):
        acc = a[k - 1] + b[k - 1]
        for i in range(1, k):
            acc = acc + _outer(a[i - 1], b[k - i - 1])
        out.append(acc)
    return out


@partial(jax.jit, static_argnames="depth")
def signature(path: jnp.ndarray, depth: int = 3) -> jnp.ndarray:
    """Truncated signature of a piecewise-linear path.

    Args:
        path: ``(T, d)`` array of points (or ``(B, T, d)`` — batched via vmap).
        depth: truncation level (static; 2-4 is typical).

    Returns:
        Flat vector of length :func:`signature_dim` — levels concatenated
        ``[level1 (d,), level2 (d^2,), ...]``.
    """
    if path.ndim == 3:
        return jax.vmap(lambda p: signature(p, depth))(path)
    d = path.shape[-1]
    dxs = jnp.diff(path, axis=0)                      # (T-1, d)

    def body(carry, dx):
        return _chen(carry, _seg_exp(dx, depth), depth), None

    init = _seg_exp(dxs[0], depth)
    levels, _ = jax.lax.scan(body, init, dxs[1:])
    del d
    return jnp.concatenate(levels)


def _ta_mul_levels(a: list[jnp.ndarray], b: list[jnp.ndarray], depth: int) -> list[jnp.ndarray]:
    """Product of two *level-only* tensors (no scalar part)."""
    out = []
    for k in range(1, depth + 1):
        acc = jnp.zeros_like(a[k - 1])
        for i in range(1, k):
            acc = acc + _outer(a[i - 1], b[k - i - 1])
        out.append(acc)
    return out


@partial(jax.jit, static_argnames="depth")
def log_signature(path: jnp.ndarray, depth: int = 3) -> jnp.ndarray:
    """Truncated tensor-algebra logarithm of the signature.

    ``log(1 + x) = x - x⊗x/2 + x⊗x⊗x/3 - ...`` with ``x`` the signature's
    level part. Same flat layout (and length) as :func:`signature`; the
    higher-level coordinates are sparser and better conditioned for learning.
    """
    if path.ndim == 3:
        return jax.vmap(lambda p: log_signature(p, depth))(path)
    d = path.shape[-1]
    sig = signature(path, depth)
    # unflatten
    x, idx = [], 0
    for k in range(1, depth + 1):
        n = d**k
        x.append(jax.lax.dynamic_slice(sig, (idx,), (n,)))
        idx += n
    out = [lv for lv in x]                       # first series term: +x
    power = x
    for m in range(2, depth + 1):
        power = _ta_mul_levels(power, x, depth)  # x^{⊗m}
        coef = ((-1) ** (m + 1)) / m
        out = [o + coef * p for o, p in zip(out, power, strict=True)]
    return jnp.concatenate(out)


def leadlag(path: jnp.ndarray) -> jnp.ndarray:
    """Lead-lag embedding: ``(T, d) -> (2T-1, 2d)``.

    Interleaves the path with a one-step-lagged copy so the signature's
    antisymmetric (Levy-area) terms expose quadratic variation — the standard
    preprocessing for signature market generators.
    """
    lead = jnp.repeat(path, 2, axis=0)[1:]       # x1 x2 x2 x3 x3 ...
    lag = jnp.repeat(path, 2, axis=0)[:-1]       # x1 x1 x2 x2 x3 ...
    return jnp.concatenate([lead, lag], axis=-1)
