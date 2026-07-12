"""
Synthetic Equity Option Chain Generator

Equity-specific characteristics:
- Standard monthly expirations (30, 60, 90, 120, 180 days)
- Tighter strike spacing around ATM (97.5%, 100%, 102.5% + wider OTM)
- Market hours (9:30-16:00 ET, closed weekends)
- Dividends and interest rates matter
- Tighter bid-ask spreads than crypto

Uses Heston model for pricing to generate realistic IV smile/skew.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass
from scipy.stats import norm
import warnings

from .data_provider import OptionChain, OptionQuote
try:
    from mjollnir.pricer._jax_mgf_pricer import heston_price_slice_fast as heston_price_slice
except ImportError:
    from mjollnir.pricer.heston_mgf_pricer import heston_price_slice


def get_default_moneyness_by_maturity() -> dict[int, list[float]]:
    """
    Default adaptive moneyness grid: tighter for short maturities, wider for long.

    Rationale:
    - Short-dated options: Less time to move → tighter strikes around ATM
    - Long-dated options: More time to move → wider strikes for hedging

    Returns:
        Dictionary mapping maturity (days) to moneyness list
    """
    return {
        10:  [0.95, 0.975, 1.0, 1.025, 1.05],              # Very tight: ±5%
        20:  [0.95, 0.975, 1.0, 1.025, 1.05],              # Tight: ±5%
        30:  [0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10],  # Medium: ±10%
        60:  [0.90, 0.95, 1.0, 1.05, 1.10],                # Standard: ±10%
        90:  [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],    # Wide: ±15%
        120: [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15],    # Wide: ±15%
    }


@dataclass
class HestonVolatilityProfile:
    """Heston volatility parameters for option pricing."""

    # Heston parameters
    kappa: float      # Mean reversion speed
    theta: float      # Long-run variance
    xi: float         # Vol of vol (sigma_v)
    rho: float        # Correlation (spot-vol)
    v0: float         # Current variance

    # Reference
    atm_iv: float     # ATM implied vol (for reference)

    # Bounds for IV backout (lowered significantly to allow proper smile formation)
    min_iv: float = 0.001  # 0.1% floor to allow smile to form properly
    max_iv: float = 2.0

    def __post_init__(self):
        """Validate Feller condition."""
        feller_satisfied = 2 * self.kappa * self.theta > self.xi**2
        if not feller_satisfied:
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*self.kappa*self.theta:.4f} "
                f"< ξ² = {self.xi**2:.4f}. Variance process may hit zero."
            )


class SyntheticEquityOptionChainGenerator:
    """
    Generate synthetic equity option chains using Heston model.

    Equity-specific features:
    - Monthly expirations (standard cycle)
    - Tighter strike spacing
    - Dividends and interest rates
    - Tighter bid-ask spreads
    - Market hours considerations

    Example:
        >>> profile = HestonVolatilityProfile(
        ...     kappa=4.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04, atm_iv=0.20
        ... )
        >>> generator = SyntheticEquityOptionChainGenerator()
        >>> chain = generator.generate_single_chain(
        ...     reference_date=date(2024, 1, 1),
        ...     spot_price=100.0,
        ...     vol_profile=profile,
        ... )
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        dividend_yield: float = 0.01,
        # Equity-specific: standard expiries (10-120 days)
        maturities_days: list[int] | None = None,
        # Equity-specific: adaptive strike spacing by maturity
        # Can be a single list (same for all maturities) or dict mapping maturity to moneyness
        moneyness_range: list[float] = None,
        moneyness_by_maturity: dict[int, list[float]] = None,
        # Tighter spreads than crypto
        atm_spread_pct: float = 0.002,
        otm_spread_pct: float = 0.01,
        min_spread_pct: float = 0.001,
        max_spread_pct: float = 0.05,
        absolute_min_spread: float = 0.01,
        add_noise: bool = False,
        noise_level: float = 0.002,
        price_floor: float = 0.0001,  # Very low floor to not interfere with smile
        enforce_intrinsic: bool = True,
        random_seed: int | None = None,
        days_per_year: float = 365.0,
    ):
        """
        Initialize equity option chain generator.

        Args:
            risk_free_rate: Risk-free rate (annualized)
            dividend_yield: Dividend yield (annualized)
            maturities_days: Option maturities in days
            moneyness_range: Single list of strike/spot ratios (same for all maturities)
            moneyness_by_maturity: Dict mapping maturity (days) to moneyness list (adaptive)
            atm_spread_pct: Bid-ask spread at the money (fraction of mid)
            otm_spread_pct: Bid-ask spread far out of the money
            min_spread_pct: Lower bound on relative spread
            max_spread_pct: Upper bound on relative spread
            absolute_min_spread: Absolute floor on spread
            add_noise: Add small symmetric noise to mid prices
            noise_level: Noise std as fraction of price
            price_floor: Absolute minimum price
            enforce_intrinsic: Ensure price >= max(intrinsic, floor)
            random_seed: For reproducibility
            days_per_year: Convention for converting days to year fractions.
                365.0 for calendar days (default), 252.0 for trading days.
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        if maturities_days is None:
            # Equity-standard expiry ladder (10-120 days)
            maturities_days = [10, 20, 30, 60, 90, 120]
        self.maturities_days = sorted(maturities_days)

        # Setup moneyness grid (adaptive by maturity or uniform)
        # Priority: moneyness_by_maturity > moneyness_range > default adaptive
        if moneyness_by_maturity is not None:
            # Use provided maturity-specific grid
            self.moneyness_by_maturity = {
                maturity: sorted(moneyness_list)
                for maturity, moneyness_list in moneyness_by_maturity.items()
            }
        elif moneyness_range is not None:
            # Use single list for all maturities
            uniform_moneyness = sorted(moneyness_range)
            self.moneyness_by_maturity = {
                maturity: uniform_moneyness
                for maturity in self.maturities_days
            }
        else:
            # Use default adaptive grid
            self.moneyness_by_maturity = get_default_moneyness_by_maturity()

        # Ensure all maturities have a grid (use ATM if missing)
        for maturity in self.maturities_days:
            if maturity not in self.moneyness_by_maturity:
                # Fallback: use grid from closest maturity, or ATM only
                warnings.warn(
                    f"No moneyness grid for {maturity}d maturity. Using [1.0] (ATM only)."
                )
                self.moneyness_by_maturity[maturity] = [1.0]

        self.atm_spread_pct = atm_spread_pct
        self.otm_spread_pct = otm_spread_pct
        self.min_spread_pct = min_spread_pct
        self.max_spread_pct = max_spread_pct
        self.absolute_min_spread = absolute_min_spread
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.price_floor = price_floor
        self.enforce_intrinsic = enforce_intrinsic

        self.days_per_year = days_per_year
        self._rng = np.random.RandomState(random_seed)

    def generate_single_chain(
        self,
        reference_date: date,
        spot_price: float,
        vol_profile: HestonVolatilityProfile,
    ) -> OptionChain:
        """
        Generate single option chain using Heston MGF pricing.

        Args:
            reference_date: Current date
            spot_price: Current spot price
            vol_profile: Heston volatility parameters

        Returns:
            OptionChain with synthetic options
        """
        options = []

        for maturity_days in self.maturities_days:
            expiry = reference_date + timedelta(days=maturity_days)
            T = maturity_days / self.days_per_year

            # Prepare all strikes and option types for this maturity
            # Use maturity-specific moneyness grid (adaptive)
            strikes_maturity = []
            types_maturity = []
            moneyness_maturity = []

            moneyness_list = self.moneyness_by_maturity[maturity_days]
            for moneyness in moneyness_list:
                strike = spot_price * moneyness
                for is_call in [True, False]:
                    strikes_maturity.append(strike)
                    types_maturity.append('call' if is_call else 'put')
                    moneyness_maturity.append(moneyness)

            # Price all options at this maturity using MGF grid (efficient)
            strikes_array = np.array(strikes_maturity)
            types_array = np.array(types_maturity)

            prices_mid = heston_price_slice(
                S=spot_price,
                strikes=strikes_array,
                T=T,
                r=self.risk_free_rate,
                q=self.dividend_yield,
                v0=vol_profile.v0,
                theta=vol_profile.theta,
                kappa=vol_profile.kappa,
                volvol=vol_profile.xi,
                rho=vol_profile.rho,
                option_types=types_array
            )

            # Add noise (optional)
            if self.add_noise:
                noise = self._rng.normal(0, self.noise_level, len(prices_mid))
                scales = np.maximum(prices_mid, 1.0)
                prices_mid = np.maximum(prices_mid + noise * scales, self.price_floor)

            # Batch IV inversion (all options at this maturity in one JAX call)
            try:
                from mjollnir.pricer._jax_iv import implied_vol_batch_np
                is_call_array = np.array([t == 'call' for t in types_maturity])
                ivs_batch = implied_vol_batch_np(
                    S=spot_price,
                    K=np.array(strikes_maturity),
                    T=T,
                    r=self.risk_free_rate,
                    q=self.dividend_yield,
                    market_prices=prices_mid,
                    is_call=is_call_array,
                    sigma_init=np.full(len(strikes_maturity), vol_profile.atm_iv),
                    sigma_min=vol_profile.min_iv,
                    sigma_max=vol_profile.max_iv,
                )
            except ImportError:
                # Fallback to per-option IV if JAX not available
                ivs_batch = np.array([
                    self._black_scholes_iv(
                        spot_price, strike, T, self.risk_free_rate, self.dividend_yield,
                        price, opt_type == 'call', vol_profile,
                    )
                    for strike, opt_type, price in zip(strikes_maturity, types_maturity, prices_mid)
                ])

            # Process each option
            for i, (strike, opt_type, moneyness, price_mid) in enumerate(
                zip(strikes_maturity, types_maturity, moneyness_maturity, prices_mid)
            ):
                iv = float(ivs_batch[i])

                # Bid/ask spread
                spread = self._compute_bid_ask_spread(price_mid, moneyness, T)
                bid = max(price_mid - spread / 2, self.price_floor)
                ask = price_mid + spread / 2

                # Volume/OI
                volume = self._generate_volume(moneyness, T)
                oi = self._generate_open_interest(moneyness, T)

                option = OptionQuote(
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    bid=bid,
                    ask=ask,
                    mid=price_mid,
                    last=price_mid * self._rng.uniform(0.99, 1.01),
                    volume=volume,
                    open_interest=oi,
                    implied_volatility=iv,
                )

                options.append(option)

        return OptionChain(
            underlying='SPY',
            spot_price=spot_price,
            reference_date=reference_date,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=self.dividend_yield,
            options=options,
        )

    def _black_scholes_price_and_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        iv: float,
        is_call: bool,
    ) -> tuple:
        """BS price and vega."""
        d1 = (np.log(S / K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if is_call:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return price, vega

    def _black_scholes_iv(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        price: float,
        is_call: bool,
        vol_profile: HestonVolatilityProfile,
    ) -> float:
        """Robust IV inversion using Newton-Raphson with bisection fallback."""
        if T <= 0 or T > 10:
            return vol_profile.atm_iv

        # Only return min_iv if price is extremely close to floor
        if price <= self.price_floor * 1.0001:
            return vol_profile.min_iv

        # Check intrinsic - but allow some tolerance for time value
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if is_call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        # Don't immediately give up if price is close to intrinsic
        if intrinsic > 0 and price < intrinsic * 0.999:
            return vol_profile.min_iv  # Price below intrinsic is invalid

        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv

        # Initial guess
        time_value = price - intrinsic
        if time_value < 0.01 * S:
            iv = vol_profile.min_iv
        else:
            iv = vol_profile.atm_iv

        iv = np.clip(iv, iv_low, iv_high)

        # Newton-Raphson
        converged = False
        for iteration in range(50):
            model_price, vega = self._black_scholes_price_and_vega(S, K, T, r, q, iv, is_call)
            price_diff = model_price - price

            if abs(price_diff) < 1e-8 or abs(price_diff / max(price, 1e-8)) < 1e-6:
                converged = True
                break

            vega_threshold = 1e-6 * S
            if vega < vega_threshold:
                break

            newton_step = -price_diff / vega
            damping = 1.0
            iv_new = iv + newton_step

            if iv_new < iv_low or iv_new > iv_high:
                damping = 0.5
                iv_new = iv + damping * newton_step

            while (iv_new < iv_low or iv_new > iv_high) and damping > 0.01:
                damping *= 0.5
                iv_new = iv + damping * newton_step

            iv_new = np.clip(iv_new, iv_low, iv_high)

            if abs(iv_new - iv) < 1e-12:
                break

            if model_price < price:
                iv_low = max(iv_low, iv)
            else:
                iv_high = min(iv_high, iv)

            iv = iv_new

        if converged:
            return np.clip(iv, vol_profile.min_iv, vol_profile.max_iv)

        # Bisection fallback
        iv_low = vol_profile.min_iv
        iv_high = vol_profile.max_iv

        price_low, _ = self._black_scholes_price_and_vega(S, K, T, r, q, iv_low, is_call)
        price_high, _ = self._black_scholes_price_and_vega(S, K, T, r, q, iv_high, is_call)

        if price < price_low:
            return vol_profile.min_iv
        if price > price_high:
            return vol_profile.max_iv

        for iteration in range(100):
            iv_mid = 0.5 * (iv_low + iv_high)
            price_mid, _ = self._black_scholes_price_and_vega(S, K, T, r, q, iv_mid, is_call)

            if abs(price_mid - price) < 1e-8:
                return iv_mid

            if price_mid < price:
                iv_low = iv_mid
            else:
                iv_high = iv_mid

            if iv_high - iv_low < 1e-10:
                return 0.5 * (iv_low + iv_high)

        return 0.5 * (iv_low + iv_high)

    def _compute_bid_ask_spread(
        self,
        mid_price: float,
        moneyness: float,
        T: float,
    ) -> float:
        """Spread model for equity options (tighter than crypto)."""
        log_moneyness = abs(np.log(moneyness))

        if log_moneyness < 0.1:
            base_spread_pct = self.atm_spread_pct
        elif log_moneyness < 0.2:
            base_spread_pct = self.atm_spread_pct + \
                (self.otm_spread_pct - self.atm_spread_pct) * (log_moneyness / 0.2)
        else:
            base_spread_pct = self.otm_spread_pct

        # Tenor multiplier
        T_days = T * 365
        if T_days <= 30:
            tenor_mult = 1.0
        elif T_days <= 90:
            tenor_mult = 1.1
        else:
            tenor_mult = 1.3

        spread_pct = base_spread_pct * tenor_mult
        spread_pct = np.clip(spread_pct, self.min_spread_pct, self.max_spread_pct)

        dollar_spread = spread_pct * mid_price
        return max(dollar_spread, self.absolute_min_spread)

    def _generate_volume(self, moneyness: float, T: float) -> int:
        """Equity volume generation."""
        atm_distance = abs(moneyness - 1.0)
        atm_factor = np.exp(-10 * atm_distance**2)

        T_days = T * 365
        if T_days <= 30:
            maturity_factor = 2.5
        elif T_days <= 60:
            maturity_factor = 2.0
        elif T_days <= 90:
            maturity_factor = 1.5
        else:
            maturity_factor = 1.0

        base = 1000
        random_factor = self._rng.uniform(0.8, 1.2)

        volume = base * atm_factor * maturity_factor * random_factor
        return int(max(volume, 10))

    def _generate_open_interest(self, moneyness: float, T: float) -> int:
        """Equity open interest generation."""
        volume = self._generate_volume(moneyness, T)

        T_days = T * 365
        if T_days <= 30:
            oi_volume_ratio = self._rng.uniform(15, 25)
        elif T_days <= 90:
            oi_volume_ratio = self._rng.uniform(20, 35)
        else:
            oi_volume_ratio = self._rng.uniform(25, 50)

        atm_distance = abs(moneyness - 1.0)
        if atm_distance < 0.05:
            oi_boost = 1.5
        else:
            oi_boost = 1.0

        oi = volume * oi_volume_ratio * oi_boost
        return int(max(oi, 100))
