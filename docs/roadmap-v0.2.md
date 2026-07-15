# v0.2 Roadmap — approved workstreams (ALL SHIPPED 2026-07-13)

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
- [x] **W4. Examples gallery**: end-to-end calibrate→simulate→backtest page and
      minimal deep-hedging loop page; executed as tests so they can't rot.

## Phase 2 — reproducibility & correctness hardening
- [x] **W9. Property-based tests** (hypothesis): put–call parity, monotonicity,
      arbitrage-free synthetic chains, QE variance positivity.
- [x] **W11. Benchmark suite** (pytest-benchmark): pin chain-gen, Merton fit,
      kernel step; regressions fail CI.
- [x] **W12. Lazy imports**: defer flax/optax (NPE) so
      `import mjollnir.calibration` stays light.
- [x] **W13. Lint ratchet burn-down**: re-enable one rule at a time
      (F841 → B905 → RUF013 → B007/B028 → RUF012/RUF022/RUF059).
- [x] **W8. RNG migration**: all 30 global-`np.random` files move to
      per-call `np.random.Generator`; breaking change of sampled
      trajectories — headline of the v0.2 release notes;
      `test_env_reset_determinism` must stay green.

## Phase 3 — capabilities
- [x] **W2. Versioned calibrated-parameter store**: `ParamSet` artifact
      (schema + semver + provenance hash), save/load, CLI integration.
- [x] **W7. Scenario/stress module**: counterfactual transforms of a `ParamSet`
      (vol spike, jump cascade, regime shift, correlation breakdown) as
      versioned configs.
- [x] **W5. Market impact + multi-agent contract**: `mjollnir.impact`
      (linear/sqrt/Almgren-Chriss, differentiable, golden-tested);
      `MarketState` PyTree + `make_market_step(process=..., impact=...)`;
      reference multi-agent example.
- [x] **W6. Calibration upgrades** (a shipped; b/c designed in docs/design/neural-calibration.md):
      (a) gradient-based risk-neutral Heston calibration via
          `jax.grad` through `fourier_price_batch` (exact gradients,
          replaces finite differences);
      (b) neural parametric models — extend the existing NPE (MDN) toward
          risk-neutral surfaces;
      (c) nonparametric / neural-signature market models — design doc first
          (research-grade; scope after (a) and (b) land).

Out of scope here (user-owned): GitHub push/tag/Pages, PyPI publishing, CITATION.

---

# v0.3 candidates (brainstormed 2026-07-15, not yet approved)

## Capabilities
- **Signature kernel first** (per docs/design/neural-calibration.md): truncated
  log-signatures in pure jnp — independently useful for NPE features and RL
  observations before the full market generator lands.
- **`mjollnir.backtest`**: a thin driver formalizing the examples' hand-rolled
  loop — (ParamSet, Scenario, strategy fn) → P&L statistics; the missing
  middle layer between the kernel and research scripts.
- **First-class `ImpliedVolSurface`**: chain → total-variance grid with
  arbitrage checks (reusing the property-test laws), bridging to the SVI/SSVI
  fitters and serving as the feature object for amortized calibration.
- **Differentiable LSMC** (American/Bermudan) on the substrate — research-grade.
- **Cross-asset surfacing**: `calibration.cross_asset` (factor model, DCC)
  exists but is undocumented and unexported — promote + document.

## Structure & ergonomics
- **Split `simulations/heston_env.py` (1331 lines)**: env core / observation
  builders / liability spec, mirroring the renderer split.
- **ParamSet CLI**: `mjollnir-params show|derive|verify` for artifact hygiene
  at the shell.
- **Architecture diagram** in docs (dependency DAG, the two consumption modes).

## CI & supply chain
- Coverage measurement + threshold; `pip-audit`/`deptry` job; nightly run
  against jax pre-releases (see the 0.10 pin).
