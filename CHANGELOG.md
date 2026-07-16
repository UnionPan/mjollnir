# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning policy is described
in `docs/guide/stability.md` (golden-test breaks = dynamics break = never a patch).

## [0.3.0] - 2026-07-16

### Added
- `mjollnir.backtest` (provenance-hashed, differentiable backtest driver)
  and `mjollnir.surface` (`ImpliedVolSurface` with static-arbitrage report
  and calibration bridge).
- Pure-JAX signature kernel (`signature`, `log_signature`, `leadlag`),
  cross-checked against iisignature; differentiable Longstaff-Schwartz
  (`lsmc_price`) with autodiff early-exercise greeks.
- `mjollnir.generator`: signature market generator trained on the
  expected-signature MMD objective, with `GeneratorCard` provenance.
- `mjollnir-params` CLI (show/verify/derive/scenario); cross-asset package
  documented and tested; complete 30-class process catalogue exported.
- CI: coverage gate, nightly jax-prerelease + pip-audit workflows;
  architecture page.

## [0.2.0] - 2026-07-13

### Changed (breaking)
- **RNG migration**: the NumPy process layer now uses per-call
  `np.random.Generator` (PCG64) instead of the global `np.random` stream.
  Sampled trajectories differ from v0.1 for the same seeds. Guarantees:
  seeded determinism per call, cross-process isolation, no global-stream
  side effects, unified `None` = fresh entropy on all backends (JAX fast
  paths previously pinned `None` to key 0). See `tests/test_rng_isolation.py`.

### Added
- Widened `mjollnir.jax`: full QE kernel family + Merton/Bates/Kou/VG/NIG
  COS pricers; market impact zoo (`linear_impact`, `sqrt_impact`,
  `almgren_chriss_step`); `MarketState` + `make_market_step` multi-agent
  contract; `mjollnir.params` (versioned ParamSet artifacts);
  `mjollnir.scenarios` (counterfactual shock library); property-based and
  benchmark suites; executable examples gallery; gradient-based
  risk-neutral Heston calibration (`fit_heston_surface`: exact jax.grad
  through the COS pricer, all five params recovered to <0.01% on clean
  surfaces, returns a Q-measure ParamSet).

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
