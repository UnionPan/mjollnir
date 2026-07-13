# Parameter store & counterfactual scenarios

## `mjollnir.params` — pinnable calibration artifacts

The calibration pipeline writes bulk results as per-run parquet frames.
`ParamSet` is the complementary **artifact**: one frozen, hash-verified
parameter set an experiment can pin — the parameter analogue of the
golden-value contract on the kernel.

```python
from mjollnir.params import ParamSet

ps = ParamSet.create(
    "heston", "P",
    {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7, "v0": 0.04},
    asset="SPY", window="2018-01-02..2026-01-02", source="qmle close-close",
)
ps.save("data/params/spy_heston.json")   # canonical JSON + sha256
ps = ParamSet.load("data/params/spy_heston.json")   # refuses tampered files
```

Every artifact records the library version, a UTC timestamp, a schema
version, and a sha256 content hash. `load` raises `ParamSetIntegrityError`
on any silent edit.

## `mjollnir.scenarios` — counterfactuals as data

A `Scenario` is a named, documented set of parameter shocks. Applying one
produces a **provenance-chained** child (`parent_hash` = the calibration's
content hash), so every counterfactual traces back to the parameters that
spawned it.

```python
from mjollnir.scenarios import vol_spike, correlation_breakdown

crisis = (vol_spike(2.0) | correlation_breakdown(-0.9)).apply(ps)
spot, var, key = qe_heston_step(..., **crisis.as_kwargs(), key=key)
```

Curated library (severity 1.0 = no shock; rationale documented per scenario):

| Scenario | Shock |
|---|---|
| `vol_spike(s)` | spot vol ×s, long-run vol ×(1+s)/2, vol-of-vol ×√s |
| `correlation_breakdown(r)` | rho pinned to crisis level `r` (absolute) |
| `regime_shift(p)` | mean reversion ÷p, long-run variance ×√p |
| `jump_cascade(k)` | jump intensity ×k, mean jump ×√k (Bates/Merton) |

Scenarios **fail loudly** when a shock names a parameter the model lacks —
a Heston set through `jump_cascade` raises instead of silently no-opping.
Compose with `|`; composition multiplies scales and merges overrides.
