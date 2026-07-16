"""Signature market generator — nonparametric path generation.

Trains a neural generator of return paths against the **expected-signature
objective**: with the linear (truncated) signature kernel, the maximum mean
discrepancy between two path laws collapses to

    MMD^2(X, Y) = || E[sig(leadlag(X))] - E[sig(leadlag(Y))] ||^2,

so matching the expected signature — which characterizes the law of a path
under mild conditions — is a principled, closed-form training signal. The
generator emits paths *directly* (no log-signature inversion step), and the
signature enters only through the differentiable kernel in ``mjollnir.jax``.

Every trained generator carries a :class:`GeneratorCard` — the neural
analogue of a :class:`~mjollnir.params.ParamSet`: training-data fingerprint,
signature depth, architecture, seed, final MMD, library version, content
hash — so backtests against a neural market remain pinnable and attributable.

This is the deliberately-small first version of the design in
``docs/design/neural-calibration.md``: unconditional, single-asset, MLP
generator. Conditioning and CVAE variants extend it, not replace it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from mjollnir.jax._signature import leadlag, signature

__all__ = ["GeneratorCard", "SignatureMarketGenerator", "expected_signature", "sig_mmd"]


# ---------------------------------------------------------------------------
# signature statistics
# ---------------------------------------------------------------------------

def expected_signature(returns: jnp.ndarray, depth: int) -> jnp.ndarray:
    """Mean truncated signature of lead-lag-embedded cumulative-return paths.

    Args:
        returns: ``(B, T)`` batch of return sequences.
        depth: signature truncation.

    Returns:
        ``(signature_dim(2, depth),)`` expected-signature vector.
    """
    paths = jnp.cumsum(returns, axis=1)[..., None]          # (B, T, 1)
    embedded = jax.vmap(leadlag)(paths)                     # (B, 2T-1, 2)
    sigs = signature(embedded, depth)                       # (B, D)
    return sigs.mean(axis=0)


def sig_mmd(returns_a: jnp.ndarray, returns_b: jnp.ndarray, depth: int = 4) -> jnp.ndarray:
    """Signature-kernel MMD (linear kernel): distance of expected signatures."""
    ea = expected_signature(returns_a, depth)
    eb = expected_signature(returns_b, depth)
    return jnp.linalg.norm(ea - eb)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratorCard:
    """Provenance card for a trained generator (the neural ParamSet)."""

    data_fingerprint: str      # sha256 of the training returns
    n_train_paths: int
    path_length: int
    sig_depth: int
    latent_dim: int
    hidden: tuple[int, ...]
    seed: int
    train_steps: int
    final_mmd: float
    created_at: str
    mjollnir_version: str

    def content_hash(self) -> str:
        doc = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in self.__dict__.items()}
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        doc = {k: (list(v) if isinstance(v, tuple) else v)
               for k, v in self.__dict__.items()}
        doc["content_hash"] = self.content_hash()
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return path


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

class _MLPGenerator(nn.Module):
    path_length: int
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, z):
        h = z
        for width in self.hidden:
            h = nn.tanh(nn.Dense(width)(h))
        return nn.Dense(self.path_length)(h)


class SignatureMarketGenerator:
    """Unconditional neural generator of return paths, sig-MMD-trained.

    Usage::

        gen = SignatureMarketGenerator(path_length=32)
        card = gen.fit(train_returns, steps=1500, seed=0)
        fake = gen.sample(jax.random.PRNGKey(1), n_paths=1024)
    """

    def __init__(self, path_length: int, latent_dim: int = 8,
                 hidden: tuple[int, ...] = (64, 64), sig_depth: int = 4):
        self.path_length = int(path_length)
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.sig_depth = int(sig_depth)
        self._net = _MLPGenerator(self.path_length, self.hidden)
        self._params = None
        self._scale = 1.0
        self.card: GeneratorCard | None = None

    # -- core ----------------------------------------------------------------
    def _generate(self, params, key, n_paths: int) -> jnp.ndarray:
        z = jax.random.normal(key, (n_paths, self.latent_dim))
        return self._net.apply(params, z) * self._scale

    def sample(self, key, n_paths: int) -> jnp.ndarray:
        """Generate ``(n_paths, path_length)`` return sequences."""
        if self._params is None:
            raise RuntimeError("generator is not trained; call fit() first")
        return self._generate(self._params, key, n_paths)

    # -- training ------------------------------------------------------------
    def fit(self, returns: np.ndarray, *, steps: int = 1500,
            batch: int = 256, learning_rate: float = 1e-3,
            seed: int = 0) -> GeneratorCard:
        """Train against the expected-signature objective; returns the card."""
        returns = jnp.asarray(returns)
        if returns.ndim != 2 or returns.shape[1] != self.path_length:
            raise ValueError(f"expected (B, {self.path_length}) returns, "
                             f"got {returns.shape}")
        # output scale anchored to the data so tanh layers start in range
        self._scale = float(jnp.std(returns) * 2.0)

        target = expected_signature(returns, self.sig_depth)
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        params = self._net.init(init_key, jnp.zeros((1, self.latent_dim)))
        schedule = optax.cosine_decay_schedule(learning_rate, steps)
        opt = optax.adam(schedule)
        opt_state = opt.init(params)

        def loss_fn(p, k):
            fake = self._generate(p, k, batch)
            return jnp.sum((expected_signature(fake, self.sig_depth) - target) ** 2)

        @jax.jit
        def train_step(p, st, k):
            val, g = jax.value_and_grad(loss_fn)(p, k)
            updates, st = opt.update(g, st)
            return optax.apply_updates(p, updates), st, val

        for _ in range(steps):
            key, sub = jax.random.split(key)
            params, opt_state, _ = train_step(params, opt_state, sub)

        self._params = params
        key, eval_key = jax.random.split(key)
        final = float(sig_mmd(self._generate(params, eval_key, 2048),
                              returns, self.sig_depth))
        from mjollnir import __version__
        self.card = GeneratorCard(
            data_fingerprint=hashlib.sha256(
                np.asarray(returns).tobytes()).hexdigest(),
            n_train_paths=int(returns.shape[0]),
            path_length=self.path_length,
            sig_depth=self.sig_depth,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
            seed=seed,
            train_steps=steps,
            final_mmd=final,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            mjollnir_version=__version__,
        )
        return self.card
