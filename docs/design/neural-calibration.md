# Design: neural & signature-based calibration (W6 b/c)

Status: approved direction, post-v0.2 implementation. Part (a) — gradient
risk-neutral calibration — shipped in v0.2 (`fit_heston_surface`).

## (b) Neural parametric: amortized risk-neutral calibration

**What.** Extend the existing NPE machinery (`calibration.physical.batched.npe`,
an MDN over QMLE-style summary statistics) to the risk-neutral side: a network
mapping *IV-surface features* → posterior over Heston parameters.

**Why.** `fit_heston_surface` solves one surface in ~15 s. Amortized
calibration answers in microseconds after an offline training phase — the
right tool for backtests that recalibrate daily over decades × universes, and
it quantifies parameter uncertainty (posterior, not point estimate).

**Design.**
- *Simulator*: `fourier_price_batch` on a fixed moneyness × maturity grid —
  the training loop generates (θ, surface) pairs on the fly (θ ~ prior over
  the sane Heston box already used by the property suite).
- *Features*: total-variance grid `w(k, T) = σ_imp² T` on standardized
  moneyness, flattened; optionally augmented with ATM skew/curvature per slice.
- *Head*: the existing MDN (`npe/model.py`) unchanged; training loop
  (`npe/train.py`) reused with a new simulate-batch function.
- *Contract*: `fit_heston_surface_amortized(quotes, npe=...) -> GradientHestonResult`
  with the MDN posterior mean as the ParamSet and posterior std recorded in
  `note`; falls back to (a) as a refinement step (`refine=True` runs 500 Adam
  steps from the NPE point — best of both).
- *Validation*: simulation-based calibration (SBC) ranks + recovery table on
  held-out θ, mirroring the parity-test philosophy.

## (c) Nonparametric: signature-based market generator

**What.** A conditional generator of *paths* (not parameters), following the
signature market-generator line (Buehler et al.), replacing the parametric
SDE assumption entirely: learn the distribution of truncated log-signatures
of lead-lag-embedded return paths, sample new signatures, invert to paths.

**Prior art to reuse** (`~/quant/Signature_Market_Sim`, user's own repo):
lead-lag embedding (`utils/leadlag.py`), CVAE over logsig features
(`cvae.py`), logsig-to-path inversion (`logsig_inversion.py`), and the
process discriminator for evaluation. That stack is esig/numpy/TF-era; the
mjollnir port is **JAX-native**:

1. `mjollnir.jax.signature`: truncated signature/log-signature of a batch of
   paths via iterated Chen products in pure `jnp` (depth ≤ 4, dim ≤ 4 —
   ~60 lines, jit/vmap-safe; no esig dependency). Golden-tested against
   `iisignature` in a dev-only test.
2. `SignatureMarketGenerator`: flax CVAE (or MMD generator — decide by
   discriminator score) over logsigs, conditioned on a market-state summary;
   `sample(key, n_paths)` returns paths via inversion.
3. Contract: the generator implements the same **market-step/rollout surface**
   as parametric processes where possible, and always emits provenance — a
   `ParamSet`-like `GeneratorCard` (training data window, sig depth, seed,
   architecture hash) so backtests against a neural market remain pinnable.
4. Evaluation: signature-MMD against held-out real paths + the process
   discriminator; both become part of the test suite with fixed seeds.

**Order of work**: signature kernel → generator + card → inversion →
evaluation harness. The signature kernel is independently useful (signature
features for the NPE in (b) and for RL observations) and lands first.
