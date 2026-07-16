# Architecture

One dependency direction, two consumption modes. Everything inside the
"versioned substrate" boundary is golden-tested: changing its behaviour is a
version bump, never a silent drift.

```mermaid
graph TD
    subgraph substrate["versioned substrate (golden-tested)"]
        JAX["mjollnir.jax<br/>QE kernels · COS pricers · impact ·<br/>MarketState · signatures"]
        PROC["processes<br/>30-model catalogue (numpy)"]
        PRICER["pricer<br/>COS / Fourier / MGF"]
        SYN["synthetic_data<br/>OptionChain · chain generators"]
        SIM["simulations<br/>gym hedging envs"]
    end

    subgraph pipeline["calibration & backtesting"]
        CAL["calibration<br/>physical · risk-neutral ·<br/>cross-asset · marketdata"]
        SURF["surface<br/>ImpliedVolSurface"]
        PARAMS["params<br/>ParamSet artifacts"]
        SCEN["scenarios<br/>counterfactual shocks"]
        BT["backtest<br/>run_backtest"]
    end

    PRICER --> JAX
    PROC --> JAX
    PRICER --> SYN
    SYN --> SIM
    SYN --> CAL
    JAX --> CAL
    SURF --> CAL
    CAL --> PARAMS
    PARAMS --> SCEN
    SCEN --> BT
    PARAMS --> BT
    JAX --> BT

    RL["RL research<br/>(deep hedging / POMARL)"]
    JAX --> RL
    SIM --> RL
    BT --> RL
```

## The two modes

**Mode 1 — RL substrate.** Experiments pin a version of `mjollnir.jax`:
jittable, batchable, differentiable primitives with explicit PRNG keys.
Golden-key tests freeze the numerics; breaking them is a release event.

**Mode 2 — calibration → artifact → counterfactual → backtest.** Market
data (or synthetic chains) become an `ImpliedVolSurface`, calibrate to a
hash-verified `ParamSet`, optionally pass through a `Scenario`, and drive
`run_backtest` — every result attributable to a specific parameter artifact.
