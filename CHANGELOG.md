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

### Fixed
- Rough-Bergomi simulation now accepts both the flat `(2,)` and the gym envs'
  `(n_paths, 2)` initial-state conventions.
- Heston path cache works with both zarr 2 and zarr 3.
- Bare `except:` clauses narrowed so interrupts propagate.
