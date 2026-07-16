"""Tests for the signature market generator (kept fast: tiny net, few steps)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.generator import GeneratorCard, SignatureMarketGenerator, sig_mmd
from mjollnir.jax import configure_runtime, qe_heston_step

configure_runtime()

PATH_LEN = 24


def _heston_returns(n_paths=512, seed=0):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_paths)

    def one(k):
        def body(carry, _):
            s, v, k = carry
            s, v, k = qe_heston_step(s, v, 1 / 252, 0.0, 2.0, 0.04, 0.5, -0.7, key=k)
            return (s, v, k), s

        _, spots = jax.lax.scan(body, (jnp.asarray(100.0), jnp.asarray(0.04), k),
                                None, length=PATH_LEN + 1)
        return jnp.diff(jnp.log(spots))

    return np.asarray(jax.vmap(one)(keys))


@pytest.fixture(scope="module")
def trained():
    returns = _heston_returns()
    gen = SignatureMarketGenerator(path_length=PATH_LEN, hidden=(48, 48),
                                   sig_depth=3)
    card = gen.fit(returns, steps=400, batch=256, seed=0)
    return gen, card, returns


class TestGenerator:
    def test_training_reduces_sig_mmd(self, trained):
        gen, card, returns = trained
        untrained = SignatureMarketGenerator(path_length=PATH_LEN, hidden=(48, 48),
                                             sig_depth=3)
        untrained._scale = gen._scale
        key = jax.random.PRNGKey(9)
        params0 = untrained._net.init(key, jnp.zeros((1, untrained.latent_dim)))
        raw = untrained._generate(params0, key, 2048)
        mmd_untrained = float(sig_mmd(raw, jnp.asarray(returns), 3))
        assert card.final_mmd < 0.5 * mmd_untrained

    def test_moments_match(self, trained):
        gen, _, returns = trained
        fake = np.asarray(gen.sample(jax.random.PRNGKey(2), 4096))
        assert fake.shape == (4096, PATH_LEN)
        assert fake.mean() == pytest.approx(returns.mean(), abs=3e-4)
        assert fake.std() == pytest.approx(returns.std(), rel=0.25)

    def test_sampling_deterministic(self, trained):
        gen, _, _ = trained
        a = gen.sample(jax.random.PRNGKey(5), 8)
        b = gen.sample(jax.random.PRNGKey(5), 8)
        assert (np.asarray(a) == np.asarray(b)).all()

    def test_card_provenance(self, trained, tmp_path):
        _gen, card, returns = trained
        assert isinstance(card, GeneratorCard)
        assert card.n_train_paths == returns.shape[0]
        assert card.sig_depth == 3 and card.train_steps == 400
        assert card.mjollnir_version
        path = card.save(tmp_path / "gen.json")
        import json
        doc = json.loads(path.read_text())
        assert doc["content_hash"] == card.content_hash()
        assert doc["data_fingerprint"] == card.data_fingerprint

    def test_untrained_sample_raises(self):
        gen = SignatureMarketGenerator(path_length=PATH_LEN)
        with pytest.raises(RuntimeError, match="not trained"):
            gen.sample(jax.random.PRNGKey(0), 4)
