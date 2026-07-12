# v0.2 Roadmap — approved workstreams

Approved 2026-07-12. Ordered by dependency, committed one workstream at a time,
each verified (tests + lint + docs --strict) before the next begins.

## Phase 1 — quick wins
- [x] **W1. Widen the frozen `mjollnir.jax` API**: export `qe_bates_step`,
      `qe_three_half_step`, `qe_four_half_step` and the Merton/Bates/Kou/VG/NIG
      COS pricers; add golden-key tests for each new export.
- [x] **W3. Env vars**: `OPTIONS_DESK_JAX_*` → `MJOLLNIR_JAX_*`, no fallback;
      audit and update every consumer (incl. monorepo mirror note).
- [x] **W10. Precision policy**: explicit `dtype` parameter on batched
      calibrators (float32 default documented); stability-guide section.
- [ ] **W4. Examples gallery**: end-to-end calibrate→simulate→backtest page and
      minimal deep-hedging loop page; executed as tests so they can't rot.

## Phase 2 — reproducibility & correctness hardening
- [x] **W9. Property-based tests** (hypothesis): put–call parity, monotonicity,
      arbitrage-free synthetic chains, QE variance positivity.
- [ ] **W11. Benchmark suite** (pytest-benchmark): pin chain-gen, Merton fit,
      kernel step; regressions fail CI.
- [ ] **W12. Lazy imports**: defer flax/optax (NPE) so
      `import mjollnir.calibration` stays light.
- [ ] **W13. Lint ratchet burn-down**: re-enable one rule at a time
      (F841 → B905 → RUF013 → B007/B028 → RUF012/RUF022/RUF059).
- [ ] **W8. RNG migration**: all 30 global-`np.random` files move to
      per-call `np.random.Generator`; breaking change of sampled
      trajectories — headline of the v0.2 release notes;
      `test_env_reset_determinism` must stay green.

## Phase 3 — capabilities
- [ ] **W2. Versioned calibrated-parameter store**: `ParamSet` artifact
      (schema + semver + provenance hash), save/load, CLI integration.
- [ ] **W7. Scenario/stress module**: counterfactual transforms of a `ParamSet`
      (vol spike, jump cascade, regime shift, correlation breakdown) as
      versioned configs.
- [ ] **W5. Market impact + multi-agent contract**: `mjollnir.impact`
      (linear/sqrt/Almgren-Chriss, differentiable, golden-tested);
      `MarketState` PyTree + `make_market_step(process=..., impact=...)`;
      reference multi-agent example.
- [ ] **W6. Calibration upgrades**:
      (a) gradient-based risk-neutral Heston calibration via
          `jax.grad` through `fourier_price_batch` (exact gradients,
          replaces finite differences);
      (b) neural parametric models — extend the existing NPE (MDN) toward
          risk-neutral surfaces;
      (c) nonparametric / neural-signature market models — design doc first
          (research-grade; scope after (a) and (b) land).

Out of scope here (user-owned): GitHub push/tag/Pages, PyPI publishing, CITATION.
