"""Amortized Bayesian calibration: transforms, summaries, artifacts, posterior reweighting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mjollnir.inference import GaussianPrior, fit, load_artifact, save_artifact
from mjollnir.inference.artifact import ArtifactIntegrityError
from mjollnir.inference.check import predictive_check, predictive_p_values
from mjollnir.inference.models import MODELS
from mjollnir.inference.simulate import simulate_batch
from mjollnir.inference.summaries import FEATURE_NAMES, pad_returns, summary_features
from mjollnir.inference.train import posterior_draws, simulate_dataset, train_posterior, validate


@pytest.mark.parametrize("name", MODELS)
def test_transforms_invert_each_other_over_the_prior(name):
    model = MODELS[name]
    u = model.sample_prior(jax.random.PRNGKey(0), 256)
    back = model.to_unconstrained(model.to_natural(u))
    np.testing.assert_allclose(np.asarray(back), np.asarray(u), rtol=1e-5, atol=1e-5)
    assert model.in_prior(np.asarray(u)).all()


def test_summaries_ignore_the_drift_and_the_padding():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.015, 900)
    a, ma = pad_returns([r], 1260)
    b, mb = pad_returns([r + 0.002], 1260)  # same history, a strong drift added
    fa, fb = (np.asarray(summary_features(jnp.asarray(x), jnp.asarray(m))) for x, m in ((a, ma), (b, mb)))
    assert fa.shape == (1, len(FEATURE_NAMES))
    np.testing.assert_allclose(fa, fb, rtol=1e-6, atol=1e-8)
    c, mc = pad_returns([r], 2000)  # more padding changes nothing
    np.testing.assert_allclose(np.asarray(summary_features(jnp.asarray(c), jnp.asarray(mc))), fa, rtol=1e-9)
    assert fa[0, FEATURE_NAMES.index("log_n")] == pytest.approx(np.log(900))


@pytest.mark.parametrize("name", MODELS)
def test_every_model_simulates_finite_summaries(name):
    u, f = simulate_batch(MODELS[name], jax.random.PRNGKey(1), 64, 300)
    assert np.isfinite(np.asarray(f)).all() and np.asarray(u).shape == (64, MODELS[name].dim)


@pytest.fixture(scope="module")
def small_garch_artifact():
    model = MODELS["garch"]
    u, f = simulate_dataset(model, jax.random.PRNGKey(3), 40_000, batch=4096, max_len=600)
    artifact = train_posterior(model, u, f, epochs=6, batch=1024, hidden=(64, 64), n_components=4, log=lambda _: None)
    u_test, f_test = simulate_dataset(model, jax.random.PRNGKey(4), 2_000, batch=1000, max_len=600)
    artifact.validation = validate(artifact, u_test, f_test, n_draws=400)
    return artifact


def test_a_small_network_is_roughly_calibrated(small_garch_artifact):
    v = small_garch_artifact.validation
    for name in MODELS["garch"].params:
        assert 0.80 < v["coverage_90"][name] < 0.97, (name, v)
    assert v["contraction"]["long_run_vol"] > 0.5  # the level of volatility is well identified


def test_artifacts_round_trip_and_refuse_tampering(small_garch_artifact, tmp_path):
    path = save_artifact(small_garch_artifact, tmp_path)
    loaded = load_artifact(path)
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.012, 700)))
    a, b = fit(small_garch_artifact, prices, seed=1), fit(loaded, prices, seed=1)
    assert a.mean == b.mean
    weights = (path / "params.msgpack").read_bytes()
    (path / "params.msgpack").write_bytes(weights[:-1] + bytes([weights[-1] ^ 1]))
    with pytest.raises(ArtifactIntegrityError):
        load_artifact(path)


def test_a_narrow_prior_reweights_the_posterior(small_garch_artifact):
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(6).normal(0, 0.012, 700)))
    broad = fit(small_garch_artifact, prices, seed=2)
    model = MODELS["garch"]
    padded, mask = pad_returns([np.diff(np.log(prices))], 1260)
    features = np.asarray(summary_features(jnp.asarray(padded), jnp.asarray(mask)))
    raw = posterior_draws(small_garch_artifact, features, jax.random.PRNGKey(2), 4096)[0]
    assert broad.effective_samples == pytest.approx(model.in_prior(raw).sum())  # the training prior only drops out-of-box draws
    assert broad.in_prior_share == pytest.approx(model.in_prior(raw).mean())
    # a prior about as informative as the history (log-vol sd 0.1), first agreeing with it
    cov = np.diag([0.1, 1.0, 1.0, 1.0]) ** 2

    def prior_at(long_run_vol, label):
        point = [long_run_vol, *(broad.mean[n] for n in model.params[1:])]
        return GaussianPrior(np.asarray(model.to_unconstrained(jnp.asarray([point])))[0], cov, label)

    narrow = fit(small_garch_artifact, prices, seed=2, prior=prior_at(broad.mean["long_run_vol"], "agrees"))
    assert narrow.prior == "agrees" and 50 < narrow.effective_samples < broad.effective_samples
    assert narrow.sd["long_run_vol"] < broad.sd["long_run_vol"]
    # contraction is against the prior in use: the history adds less to an informative one
    assert narrow.contraction["long_run_vol"] < broad.contraction["long_run_vol"]
    # a prior off the data pulls the posterior toward it
    target = 1.3 * broad.mean["long_run_vol"]
    pulled = fit(small_garch_artifact, prices, seed=2, prior=prior_at(target, "higher"))
    assert broad.mean["long_run_vol"] < pulled.mean["long_run_vol"] < target


def test_predictive_p_values_place_the_observed_among_the_replicates():
    replicates = np.arange(100.0)[:, None] * np.ones((1, 3))
    p = predictive_p_values(np.array([-1.0, 49.0, 200.0]), replicates)
    np.testing.assert_allclose(p, [0.0, 0.495, 1.0])


def test_a_history_the_model_cannot_produce_fails_its_predictive_check(small_garch_artifact):
    """GARCH returns are never thinner-tailed than a Gaussian: uniform shocks (kurtosis 1.8) are a misfit."""
    rng = np.random.default_rng(8)
    model_history = 100 * np.exp(np.cumsum(rng.standard_t(6, 700) * 0.012 * np.sqrt(4 / 6)))
    thin_tailed = 100 * np.exp(np.cumsum(rng.uniform(-1, 1, 700) * 0.012 * np.sqrt(3)))
    checks = {name: predictive_check(small_garch_artifact, prices, fit(small_garch_artifact, prices, seed=3),
                                     n_replicates=64, seed=3)
              for name, prices in (("model", model_history), ("thin", thin_tailed))}
    assert checks["model"].n_replicates == 64 and "log_n" not in checks["model"].names
    assert len(checks["model"].misfits) <= 3, checks["model"].misfits
    assert "log_kurtosis" in checks["thin"].misfits


def test_an_artifact_loads_on_a_cpu_whose_logarithms_differ_in_the_last_bit(small_garch_artifact, tmp_path):
    """The stored prior box is compared with a tolerance: a one-ulp difference loads, a real change does not."""
    import json

    path = save_artifact(small_garch_artifact, tmp_path)
    meta = json.loads((path / "meta.json").read_text())
    meta["prior_low"][0] = float(np.nextafter(meta["prior_low"][0], 0.0))
    (path / "meta.json").write_text(json.dumps(meta))
    load_artifact(path)
    meta["prior_low"][0] += 1e-6
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactIntegrityError, match="different garch specification"):
        load_artifact(path)
