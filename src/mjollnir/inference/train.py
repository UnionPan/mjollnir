"""Training a posterior network, and the offline validation that comes with it.

The network is fitted by maximum likelihood on simulated (parameters,
summaries) pairs: its mixture density, evaluated at each draw's true
parameters, is the amortized posterior. Training holds out a validation set
for early stopping and then measures, on fresh simulations, what the
receipt reports:

- coverage: how often the posterior's central 50% and 90% intervals contain
  the parameters that generated the history (a calibrated posterior: 0.5
  and 0.9);
- contraction: 1 - posterior variance / prior variance, how much the data
  moved each parameter (near 0: the prior speaks, not the data);
- recovery error: the posterior mean's distance from the truth.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mjollnir.inference.artifact import NPEArtifact
from mjollnir.inference.models import InferenceModel
from mjollnir.inference.network import MixtureDensityNetwork, mixture_log_prob, sample_mixture
from mjollnir.inference.simulate import simulate_batch

__all__ = ["simulate_dataset", "train_posterior", "validate"]

FEATURE_CLIP = 10.0


def simulate_dataset(model: InferenceModel, key, n: int, *, batch: int = 8192, max_len: int = 1260):
    """``n`` training pairs (unconstrained parameters, summaries), simulated in batches."""
    us, fs = [], []
    for i in range(0, n, batch):
        u, f = simulate_batch(model, jax.random.fold_in(key, i), batch, max_len)
        us.append(np.asarray(u))
        fs.append(np.asarray(f))
    u, f = np.concatenate(us)[:n], np.concatenate(fs)[:n]
    keep = np.all(np.isfinite(f), axis=1) & np.all(np.isfinite(u), axis=1)
    return u[keep], f[keep]


def _standardize(f, mean, std):
    return np.clip((f - mean) / std, -FEATURE_CLIP, FEATURE_CLIP)


def train_posterior(model: InferenceModel, u, f, *, seed: int = 0, hidden=(256, 256, 256), n_components: int = 10,
                    epochs: int = 40, batch: int = 4096, lr: float = 2e-3, val_frac: float = 0.05,
                    patience: int = 6, log=print) -> NPEArtifact:
    """Fit the posterior network to simulated pairs; returns the best-on-validation weights."""
    from mjollnir import __version__

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(u))
    n_val = int(val_frac * len(u))
    val, tr = order[:n_val], order[n_val:]
    feature_mean, feature_std = f[tr].mean(0), f[tr].std(0) + 1e-8
    theta_mean, theta_std = u[tr].mean(0), u[tr].std(0) + 1e-8
    s = _standardize(f, feature_mean, feature_std).astype(np.float32)
    z = ((u - theta_mean) / theta_std).astype(np.float32)

    network = MixtureDensityNetwork(hidden=tuple(hidden), n_components=n_components, n_outputs=model.dim)
    params = network.init(jax.random.PRNGKey(seed), jnp.zeros((1, s.shape[1])))
    steps_per_epoch = max(len(tr) // batch, 1)
    schedule = optax.cosine_decay_schedule(lr, decay_steps=epochs * steps_per_epoch, alpha=0.01)
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    def loss(p, s_b, z_b):
        return -jnp.mean(mixture_log_prob(*network.apply(p, s_b), z_b))

    @jax.jit
    def step(p, o, s_b, z_b):
        value, grads = jax.value_and_grad(loss)(p, s_b, z_b)
        updates, o = optimizer.update(grads, o, p)
        return optax.apply_updates(p, updates), o, value

    val_loss = jax.jit(loss)
    best, best_params, stale = np.inf, params, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        perm = rng.permutation(tr)
        for i in range(steps_per_epoch):
            idx = perm[i * batch:(i + 1) * batch]
            params, opt_state, _ = step(params, opt_state, s[idx], z[idx])
        current = float(np.mean([val_loss(params, s[val[j:j + 16384]], z[val[j:j + 16384]])
                                 for j in range(0, n_val, 16384)]))
        log(f"  epoch {epoch + 1:3d}  validation NLL {current:+.4f}  ({time.perf_counter() - started:.0f} s)")
        if current < best - 1e-4:
            best, best_params, stale = current, params, 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  stopping: no improvement for {patience} epochs")
                break

    return NPEArtifact(
        model=model, params=jax.device_get(best_params), feature_mean=feature_mean, feature_std=feature_std,
        theta_mean=theta_mean, theta_std=theta_std, hidden=tuple(hidden), n_components=n_components,
        training={"n_train": len(tr), "n_validation": n_val, "epochs_run": epoch + 1, "best_val_nll": best,
                  "batch": batch, "lr": lr, "seed": seed},
        created_at=datetime.now(UTC).isoformat(timespec="seconds"), mjollnir_version=__version__,
    )


def posterior_draws(artifact: NPEArtifact, features, key, n: int = 1000):
    """(B, n, d) unconstrained posterior draws for (B, F) raw summaries."""
    s = _standardize(np.asarray(features), artifact.feature_mean, artifact.feature_std).astype(np.float32)
    logits, means, log_scales = artifact.network.apply(artifact.params, s)
    z = sample_mixture(key, logits, means, log_scales, n)
    return np.asarray(z) * artifact.theta_std + artifact.theta_mean


def validate(artifact: NPEArtifact, u, f, *, seed: int = 1, n_draws: int = 1000, chunk: int = 512) -> dict:
    """Coverage, contraction and recovery error on fresh simulations (u, f)."""
    model = artifact.model
    key = jax.random.PRNGKey(seed)
    inside50 = np.zeros(model.dim)
    inside90 = np.zeros(model.dim)
    post_var = np.zeros(model.dim)
    sq_err = np.zeros(model.dim)
    sq_err_natural = np.zeros(model.dim)
    for i in range(0, len(u), chunk):
        draws = posterior_draws(artifact, f[i:i + chunk], jax.random.fold_in(key, i), n_draws)
        truth = u[i:i + chunk]
        q = np.quantile(draws, [0.05, 0.25, 0.75, 0.95], axis=1)
        inside50 += np.sum((truth >= q[1]) & (truth <= q[2]), axis=0)
        inside90 += np.sum((truth >= q[0]) & (truth <= q[3]), axis=0)
        post_var += np.sum(draws.var(axis=1), axis=0)
        sq_err += np.sum((draws.mean(axis=1) - truth) ** 2, axis=0)
        natural_mean = np.asarray(model.to_natural(jnp.asarray(draws.reshape(-1, model.dim)))).reshape(draws.shape).mean(1)
        sq_err_natural += np.sum((natural_mean - np.asarray(model.to_natural(jnp.asarray(truth)))) ** 2, axis=0)
    n = len(u)
    prior_var = (np.asarray(model.high) - np.asarray(model.low)) ** 2 / 12.0
    names = model.params
    return {
        "n": int(n),
        "coverage_50": dict(zip(names, (inside50 / n).round(4).tolist(), strict=True)),
        "coverage_90": dict(zip(names, (inside90 / n).round(4).tolist(), strict=True)),
        "contraction": dict(zip(names, (1.0 - post_var / n / prior_var).round(4).tolist(), strict=True)),
        "rmse_unconstrained": dict(zip(names, np.sqrt(sq_err / n).round(4).tolist(), strict=True)),
        "rmse_natural": dict(zip(names, np.sqrt(sq_err_natural / n).tolist(), strict=True)),
    }
