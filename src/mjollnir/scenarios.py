"""Counterfactual market scenarios as first-class, versioned objects.

A :class:`Scenario` is *data*, not code: a named set of multiplicative and
absolute parameter shocks with a documented financial rationale. Applying one
to a :class:`~mjollnir.params.ParamSet` produces a provenance-chained child
(the parent's content hash is recorded), so every counterfactual traces back
to the calibration that spawned it — this is what makes "counterfactual
market simulation" reproducible rather than ad hoc.

The curated library below covers the classic stress archetypes for the
Heston / Bates parameter families. ``severity`` conventions are documented
per scenario; severity 1.0 always means "no shock".

Example::

    from mjollnir.params import ParamSet
    from mjollnir.scenarios import vol_spike

    base = ParamSet.load("data/params/spy_heston.json")
    crisis = vol_spike(2.0).apply(base)          # spot vol doubled
    spot, var, key = qe_heston_step(..., **crisis.as_kwargs(), key=key)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mjollnir.params import ParamSet


@dataclass(frozen=True)
class Scenario:
    """A named, documented counterfactual transform of a ParamSet."""

    name: str
    description: str
    scale: Mapping[str, float] = field(default_factory=dict)
    set_: Mapping[str, float] = field(default_factory=dict)

    def apply(self, ps: ParamSet) -> ParamSet:
        """Shock ``ps``; unknown scaled params raise, absolute sets pass through."""
        # only scale parameters the model actually has — a Heston set going
        # through a jump scenario should fail loudly, not silently no-op
        return ps.derive(
            scale=self.scale,
            set_=self.set_,
            note=f"scenario:{self.name} — {self.description}",
        )

    def __or__(self, other: Scenario) -> Scenario:
        """Compose scenarios left-to-right: ``(vol_spike(2) | corr_breakdown())``."""
        merged_scale = dict(self.scale)
        for k, v in other.scale.items():
            merged_scale[k] = merged_scale.get(k, 1.0) * v
        merged_set = {**self.set_, **other.set_}
        return Scenario(
            name=f"{self.name}|{other.name}",
            description=f"{self.description}; then {other.description}",
            scale=merged_scale,
            set_=merged_set,
        )


# ---------------------------------------------------------------------------
# Curated library
# ---------------------------------------------------------------------------

def vol_spike(severity: float = 2.0) -> Scenario:
    """Instantaneous volatility spike.

    ``severity`` is a *spot-vol multiplier*: current variance ``v0`` scales by
    ``severity**2``; the long-run level ``theta`` moves half as far in vol
    terms (crises reprice the present much more than the long run); vol-of-vol
    rises with the square root (turbulence begets turbulence, sublinearly).
    """
    s = float(severity)
    return Scenario(
        name=f"vol_spike[{s:g}x]",
        description=f"spot vol x{s:g}, long-run vol x{(1 + s) / 2:g}, vol-of-vol x{s**0.5:.3g}",
        scale={"v0": s**2, "theta": ((1 + s) / 2) ** 2, "sigma_v": s**0.5},
    )


def correlation_breakdown(rho_floor: float = -0.9) -> Scenario:
    """Spot-vol correlation pinned to a crisis level.

    In stress, the leverage effect saturates: hedges built on a calm-market
    rho are wrong-footed. Absolute override rather than scaling, because the
    base rho may be near zero (scaling near zero does nothing).
    """
    return Scenario(
        name=f"correlation_breakdown[{rho_floor:g}]",
        description=f"rho pinned to {rho_floor:g}",
        set_={"rho": float(rho_floor)},
    )


def regime_shift(persistence: float = 3.0) -> Scenario:
    """Slow-moving high-vol regime.

    Mean reversion slows (``kappa`` divided by ``persistence``) while the
    long-run level rises — the market re-anchors to a stressed equilibrium
    instead of snapping back.
    """
    p = float(persistence)
    return Scenario(
        name=f"regime_shift[{p:g}]",
        description=f"kappa /{p:g}, theta x{p ** 0.5:.3g}",
        scale={"kappa": 1.0 / p, "theta": p**0.5},
    )


def jump_cascade(intensity: float = 5.0) -> Scenario:
    """Clustered jump risk (Bates / Merton parameter families).

    Jump arrival intensity multiplies by ``intensity``; the mean jump becomes
    more negative by the square root (larger, but not proportionally, per
    jump) — the empirical signature of cascade days.
    """
    k = float(intensity)
    return Scenario(
        name=f"jump_cascade[{k:g}x]",
        description=f"lambda_j x{k:g}, mu_j x{k ** 0.5:.3g}",
        scale={"lambda_j": k, "mu_j": k**0.5},
    )


LIBRARY = {
    "vol_spike": vol_spike,
    "correlation_breakdown": correlation_breakdown,
    "regime_shift": regime_shift,
    "jump_cascade": jump_cascade,
}

__all__ = [
    "LIBRARY",
    "Scenario",
    "correlation_breakdown",
    "jump_cascade",
    "regime_shift",
    "vol_spike",
]
