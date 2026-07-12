# Reproducibility contract

`mjollnir` exists so RL experiments can pin a version and trust that market
dynamics do not drift. Three commitments back that:

## 1. Explicit randomness

Every stochastic primitive takes a JAX PRNG key and returns its successor.
There is no hidden global RNG and no mutable module state affecting dynamics.
Given the same key and inputs, outputs are deterministic.

## 2. Golden-value tests

`tests/test_parity_golden.py` pins the numerics of the public kernel:

- a fixed-key 64-step Heston QE rollout (spot & variance),
- the ATM COS price, its delta and its `v0` sensitivity,
- a jitted batch pricing result.

## 3. Versioning policy

- A change that breaks a golden test **is a breaking change of environment
  dynamics**, even if every API signature is unchanged. It must be released
  as a minor (pre-1.0) or major version bump with regenerated goldens and a
  changelog entry — never silently.
- Additive API changes (new processes, new estimators) are minor releases.
- Bug fixes that do not move golden values are patch releases.

Pin accordingly, e.g. `mjollnir==0.1.*` for an experiment series.
