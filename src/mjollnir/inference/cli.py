"""Train and validate a posterior network: ``mjollnir-train-posterior --model heston``.

Runs on whatever JAX device is configured (set MJOLLNIR_JAX_BACKEND=cuda for a
GPU). Simulates training pairs, trains, validates on fresh simulations, and
writes the artifact under ``--out/<model>/``.
"""

from __future__ import annotations

import argparse
import json
import time

import jax

from mjollnir.inference.artifact import save_artifact
from mjollnir.inference.models import MODELS
from mjollnir.inference.train import simulate_dataset, train_posterior, validate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--n-train", type=int, default=2_000_000)
    parser.add_argument("--n-test", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/npe")
    args = parser.parse_args(argv)

    model = MODELS[args.model]
    print(f"{model.name}: devices {jax.devices()}")
    started = time.perf_counter()
    u, f = simulate_dataset(model, jax.random.PRNGKey(args.seed), args.n_train)
    print(f"simulated {len(u):,} training histories in {time.perf_counter() - started:.0f} s")
    artifact = train_posterior(model, u, f, seed=args.seed, epochs=args.epochs)
    u_test, f_test = simulate_dataset(model, jax.random.PRNGKey(args.seed + 10_000), args.n_test)
    artifact.validation = validate(artifact, u_test, f_test, seed=args.seed + 1)
    artifact.training["simulate_seconds"] = round(time.perf_counter() - started, 1)
    path = save_artifact(artifact, args.out)
    print(json.dumps(artifact.validation, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
