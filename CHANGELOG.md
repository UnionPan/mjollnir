# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning policy is described
in `docs/guide/stability.md` (golden-test breaks = dynamics break = never a patch).

## [0.1.0] - 2026-07-12

### Added
- Initial extraction from the `option_pricer_DLC` monorepo (`src/options_desk`).
- Public `mjollnir.jax` RL-substrate API: `configure_runtime`, `qe_heston_step`,
  differentiable `fourier_price` / `fourier_price_batch`, plus COS building
  blocks (`heston_cf`, `cos_price_multi`, `cos_price_single`, `chi_k`, `psi_k`,
  `cos_truncation_range`).
- `mjollnir.synthetic`: shared option-chain value types and synthetic chain
  generators (equity, Merton), promoted out of `calibration.data`.
- Golden-key parity suite pinning kernel and pricer numerics.
- Console scripts: `mjollnir-build-universes`, `mjollnir-calibrate`,
  `mjollnir-train-npe`.

### Changed
- `fourier_price`/`fourier_price_batch` truncation grid: sized from the
  mean-reversion-aware effective variance (was `v0` alone — broke put-call
  parity for low-`v0`/high-`theta`/long-`T`), and always expanded to cover
  every strike (a strike beyond the window returned unbounded junk). Both
  found by the new property-based suite; ATM golden values unchanged.
- Batched Merton MLE: cosine-decayed Adam (4000 steps) reaches the scipy
  optimum; `MertonJumpCalibrator` reference NLL vectorized (~3600x faster).
- Gym env `reset(seed=...)` no longer seeds the global `np.random` stream
  (it had no consumer — dynamics flow through the explicitly-seeded process
  simulation); `test_env_reset_determinism` pins the contract.

### Removed
- Dead quad-based Heston pricer in `synthetic_equity` (~230 lines; the live
  path is the JAX MGF slice pricer).

### Fixed
- Rough-Bergomi simulation now accepts both the flat `(2,)` and the gym envs'
  `(n_paths, 2)` initial-state conventions.
- Heston path cache works with both zarr 2 and zarr 3.
- Bare `except:` clauses narrowed so interrupts propagate.
