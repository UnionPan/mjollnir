"""First-class implied-volatility surface with arbitrage diagnostics.

``ImpliedVolSurface`` is the bridge object between market data and
calibration: build it from an ``OptionChain`` (quoted IVs are used when
present, otherwise inverted with the batch Newton solver), inspect it for
static arbitrage using the same laws the property-test suite enforces on the
pricer, and hand it straight to :func:`~mjollnir.calibration.fit_heston_surface`
via :meth:`ImpliedVolSurface.to_quotes`.

The surface is a frozen, flat, quote-level container — no interpolation is
baked in (smoothing belongs to the SVI/SSVI fitters, not the data object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

__all__ = ["ArbitrageReport", "ImpliedVolSurface"]


@dataclass(frozen=True)
class ArbitrageReport:
    """Static-arbitrage diagnostics for a quote-level surface.

    Empty violation lists mean the corresponding no-arbitrage condition holds
    within tolerance. Entries are human-readable strings naming the offending
    quotes — diagnostics, not exceptions: real market snapshots routinely
    carry small violations inside the spread.
    """

    vertical: list[str] = field(default_factory=list)   # call mid monotone in K
    butterfly: list[str] = field(default_factory=list)  # call mid convex in K
    calendar: list[str] = field(default_factory=list)   # ATM total var nondecreasing in T

    @property
    def ok(self) -> bool:
        return not (self.vertical or self.butterfly or self.calendar)

    def summary(self) -> str:
        if self.ok:
            return "no static arbitrage detected"
        return (f"{len(self.vertical)} vertical, {len(self.butterfly)} butterfly, "
                f"{len(self.calendar)} calendar violation(s)")


@dataclass(frozen=True)
class ImpliedVolSurface:
    """Flat quote-level IV surface (arrays share one index)."""

    spot: float
    rate: float
    dividend_yield: float
    strikes: np.ndarray        # (N,)
    maturities: np.ndarray     # (N,) years
    ivs: np.ndarray            # (N,) Black-Scholes implied vols
    is_call: np.ndarray        # (N,) bool
    mids: np.ndarray           # (N,) mid prices
    asset: str | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_chain(cls, chain, rate: float | None = None,
                   dividend_yield: float | None = None) -> ImpliedVolSurface:
        """Build from an ``OptionChain``; invert IVs where quotes lack them."""
        r = chain.risk_free_rate if rate is None else rate
        q = chain.dividend_yield if dividend_yield is None else dividend_yield
        spot = float(chain.spot_price)

        quotes = [o for o in chain.options if o.mid > 0 and o.expiry > chain.reference_date]
        if not quotes:
            raise ValueError("chain contains no usable quotes")
        T = np.array([(o.expiry - chain.reference_date).days / 365.0 for o in quotes])
        K = np.array([o.strike for o in quotes], float)
        is_call = np.array([o.option_type == "call" for o in quotes])
        mids = np.array([o.mid for o in quotes], float)

        iv = np.array([o.implied_volatility if o.implied_volatility is not None
                       else np.nan for o in quotes], float)
        missing = ~np.isfinite(iv)
        if missing.any():
            from mjollnir.pricer._jax_iv import implied_vol_batch_np
            iv[missing] = implied_vol_batch_np(
                mids[missing], spot, K[missing], T[missing], r, q, is_call[missing],
            )

        keep = np.isfinite(iv) & (iv > 0)
        return cls(
            spot=spot, rate=float(r), dividend_yield=float(q),
            strikes=K[keep], maturities=T[keep], ivs=iv[keep],
            is_call=is_call[keep], mids=mids[keep], asset=chain.underlying,
        )

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------
    @property
    def log_moneyness(self) -> np.ndarray:
        """``k = log(K / forward)`` per quote."""
        fwd = self.spot * np.exp((self.rate - self.dividend_yield) * self.maturities)
        return np.log(self.strikes / fwd)

    @property
    def total_variance(self) -> np.ndarray:
        """``w = iv^2 * T`` per quote — the natural coordinate for calendar checks."""
        return self.ivs**2 * self.maturities

    def expiries(self) -> np.ndarray:
        return np.unique(np.round(self.maturities, 10))

    def atm_term_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """``(maturities, atm_iv)`` using the nearest-to-forward quote per slice."""
        out_T, out_iv = [], []
        k = np.abs(self.log_moneyness)
        for T in self.expiries():
            idx = np.where(np.isclose(self.maturities, T))[0]
            out_T.append(T)
            out_iv.append(float(self.ivs[idx[np.argmin(k[idx])]]))
        return np.asarray(out_T), np.asarray(out_iv)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def arbitrage_report(self, tol: float = 1e-8) -> ArbitrageReport:
        """Static-arbitrage checks on call mids and ATM total variance."""
        report = ArbitrageReport()
        scale = tol * self.spot
        for T in self.expiries():
            idx = np.where(np.isclose(self.maturities, T) & self.is_call)[0]
            order = idx[np.argsort(self.strikes[idx])]
            Ks, Cs = self.strikes[order], self.mids[order]
            for (k1, c1), (k2, c2) in pairwise(zip(Ks, Cs, strict=True)):
                if c2 > c1 + scale:
                    report.vertical.append(
                        f"T={T:.4f}: C(K={k2:g})={c2:.4f} > C(K={k1:g})={c1:.4f}")
            for j in range(1, len(order) - 1):
                k_lo, k_mid, k_hi = Ks[j - 1], Ks[j], Ks[j + 1]
                lam = (k_hi - k_mid) / (k_hi - k_lo)
                interp = lam * Cs[j - 1] + (1 - lam) * Cs[j + 1]
                if Cs[j] > interp + scale:
                    report.butterfly.append(
                        f"T={T:.4f}: butterfly at K={k_mid:g} "
                        f"(C={Cs[j]:.4f} > {interp:.4f})")
        ts_T, ts_iv = self.atm_term_structure()
        w = ts_iv**2 * ts_T
        for (t1, w1), (t2, w2) in pairwise(zip(ts_T, w, strict=True)):
            if w2 < w1 - tol:
                report.calendar.append(
                    f"ATM total variance decreasing: w({t2:.4f})={w2:.5f} "
                    f"< w({t1:.4f})={w1:.5f}")
        return report

    # ------------------------------------------------------------------
    # calibration bridge
    # ------------------------------------------------------------------
    def to_quotes(self) -> list[tuple[float, float, bool, float]]:
        """``(strike, maturity, is_call, mid)`` tuples for ``fit_heston_surface``."""
        return [(float(k), float(t), bool(c), float(m))
                for k, t, c, m in zip(self.strikes, self.maturities,
                                      self.is_call, self.mids, strict=True)]

    def __len__(self) -> int:
        return len(self.strikes)
