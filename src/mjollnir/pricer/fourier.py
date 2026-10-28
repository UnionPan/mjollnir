"""
Fourier-based pricing methods (COS, FFT, Lewis)

author: Yunian Pan
email: yp1170@nyu.edu
"""
import numpy as np
import time

from .base import Pricer, PricingResult


class COSPricer(Pricer):
    """
    COS method for European option pricing

    The COS (Fourier-cosine) method is extremely fast and accurate for
    European options when the characteristic function is known.

    References:
    - Fang & Oosterlee (2008) "A Novel Pricing Method for European Options
      Based on Fourier-Cosine Series Expansions"

    Works with any process whose characteristic_function(u, X0, t) is that of
    log S_t (GBM, Merton, Kou, Variance Gamma, NIG, Heston, ...).
    """

    def __init__(self, risk_free_rate: float = 0.0, N: int = 128, L: float = 10.0):
        """
        Initialize COS pricer

        Args:
            risk_free_rate: Risk-free rate for discounting
            N: Number of Fourier terms (power of 2, typically 128-512)
            L: Truncation range for log-price (typically 8-12)
        """
        super().__init__(name="COS")
        self.risk_free_rate = risk_free_rate
        self.N = N
        self.L = L

    def price(
        self,
        derivative,
        process,
        X0: float | np.ndarray,
        **kwargs
    ) -> PricingResult:
        """
        Price European option using COS method

        Args:
            derivative: European call or put
            process: Process with characteristic_function() method
            X0: Initial value(s)
            **kwargs: Additional parameters

        Returns:
            PricingResult with price
        """
        start_time = time.time()

        # Validate process has characteristic function
        if not hasattr(process, 'characteristic_function'):
            raise ValueError("Process must have characteristic_function() method for COS pricing")

        S0 = float(X0) if np.isscalar(X0) else float(X0[0])
        K = derivative.strike
        T = derivative.maturity
        r = self.risk_free_rate

        # Determine option type
        is_call = "call" in derivative.contract_type

        # COS method
        price = self._cos_price(process, S0, K, T, r, X0, is_call)

        computation_time = time.time() - start_time

        metadata = {
            "method": "COS",
            "N_terms": self.N,
            "L_truncation": self.L,
            "contract_type": derivative.contract_type,
        }

        return PricingResult(
            price=price,
            std_error=0.0,  # Deterministic method
            computation_time=computation_time,
            metadata=metadata,
        )

    def _cos_price(
        self, process, S0: float, K: float, T: float, r: float, X0, is_call: bool
    ) -> float:
        """
        COS method on the log price x = log S_T (Fang & Oosterlee 2008).

        The density of x is expanded in cosines on [a, b], sized from the first
        two cumulants of x read off the process's own characteristic function,
        so the range adapts to any process. With w_k = k pi / (b - a):

            price = e^{-rT} sum'_k Re(phi(w_k) e^{-i w_k a}) V_k

        where phi is the characteristic function of x and V_k the cosine
        coefficients of the payoff (``_payoff_coefficients``).
        """
        c1, c2 = self._cumulants(process, X0, T)
        a = c1 - self.L * np.sqrt(c2)
        b = c1 + self.L * np.sqrt(c2)

        k = np.arange(self.N)
        w = k * np.pi / (b - a)
        # a process without a characteristic function raises NotImplementedError here:
        # a price from a missing transform would be silently wrong
        char_vals = np.array([process.characteristic_function(u, X0, T) for u in w], dtype=complex)
        terms = np.real(char_vals * np.exp(-1j * w * a)) * self._payoff_coefficients(w, a, b, K, is_call)
        terms[0] *= 0.5
        return float(np.exp(-r * T) * np.sum(terms))

    @staticmethod
    def _cumulants(process, X0, T: float, h: float = 1e-4) -> tuple[float, float]:
        """Mean and variance of log S_T, by central differences of log phi at 0."""
        log_phi = [np.log(complex(process.characteristic_function(u, X0, T))) for u in (-h, 0.0, h)]
        c1 = float(((log_phi[2] - log_phi[0]) / (2 * h) / 1j).real)
        c2 = float(-((log_phi[2] - 2 * log_phi[1] + log_phi[0]) / h**2).real)
        if not (np.isfinite(c1) and np.isfinite(c2) and c2 > 0):
            raise ValueError(f"cannot size the COS range from the characteristic function (c1={c1}, c2={c2})")
        return c1, c2

    @staticmethod
    def _payoff_coefficients(w: np.ndarray, a: float, b: float, K: float, is_call: bool) -> np.ndarray:
        """Cosine coefficients V_k of a vanilla payoff on [a, b]: for a call
        2/(b-a) (chi(log K, b) - K psi(log K, b)), for a put 2/(b-a) (K psi(a, log K) - chi(a, log K)),
        with chi(c, d) = int_c^d e^x cos(w (x - a)) dx and psi(c, d) = int_c^d cos(w (x - a)) dx."""
        log_K = float(np.clip(np.log(K), a, b))

        def chi(c, d):
            def antiderivative(x):
                return np.exp(x) * (np.cos(w * (x - a)) + w * np.sin(w * (x - a))) / (1.0 + w**2)
            return antiderivative(d) - antiderivative(c)

        def psi(c, d):
            safe = np.where(w == 0, 1.0, w)
            return np.where(w == 0, d - c, (np.sin(w * (d - a)) - np.sin(w * (c - a))) / safe)

        if is_call:
            return 2.0 / (b - a) * (chi(log_K, b) - K * psi(log_K, b))
        return 2.0 / (b - a) * (K * psi(a, log_K) - chi(a, log_K))


class CarrMadanPricer(Pricer):
    """
    Carr-Madan FFT method for option pricing

    Fast Fourier Transform (FFT) method for computing option prices
    across a range of strikes simultaneously.

    Reference:
    - Carr & Madan (1999) "Option Valuation Using the Fast Fourier Transform"

    Particularly efficient when pricing many strikes at once.
    """

    def __init__(self, risk_free_rate: float = 0.0, N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        """
        Initialize Carr-Madan pricer

        Args:
            risk_free_rate: Risk-free rate
            N: Number of FFT points (power of 2)
            alpha: Dampening factor (typically 1.0-2.0 for calls)
            eta: Spacing of the integration grid (the log-strike spacing is 2 pi / (N eta))
        """
        super().__init__(name="CarrMadan")
        self.risk_free_rate = risk_free_rate
        self.N = N
        self.alpha = alpha
        self.eta = eta

    def price(
        self,
        derivative,
        process,
        X0: float | np.ndarray,
        **kwargs
    ) -> PricingResult:
        """
        Price European option using Carr-Madan FFT

        Args:
            derivative: European option
            process: Process with characteristic_function
            X0: Initial value
            **kwargs: Additional parameters

        Returns:
            PricingResult with price
        """
        start_time = time.time()

        if not hasattr(process, 'characteristic_function'):
            raise ValueError("Process must have characteristic_function() method")

        K = derivative.strike
        T = derivative.maturity
        r = self.risk_free_rate
        is_call = "call" in derivative.contract_type

        def phi(u):
            return complex(process.characteristic_function(u, X0, T))

        # Damped call transform (Carr & Madan 1999, eq. 6), phi the characteristic function of log S_T
        eta = self.eta
        v = np.arange(self.N) * eta
        alpha = self.alpha
        char_vals = np.array([phi(vj - (alpha + 1) * 1j) for vj in v])
        psi = np.exp(-r * T) * char_vals / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)

        # Log-strike grid k_u = b + lambda u, centred on log K so the target lies on it exactly
        lam = 2 * np.pi / (self.N * eta)
        b = np.log(K) - lam * self.N / 2
        simpson = eta / 3 * (3 + (-1.0) ** (np.arange(self.N) + 1))
        simpson[0] -= eta / 3
        transform = np.fft.fft(np.exp(-1j * b * v) * psi * simpson)
        call = float(np.real(np.exp(-alpha * np.log(K)) / np.pi * transform[self.N // 2]))
        if is_call:
            price = call
        else:
            # parity with the forward from the transform itself: E[S_T] = phi(-i)
            price = call - np.exp(-r * T) * phi(-1j).real + K * np.exp(-r * T)

        computation_time = time.time() - start_time

        metadata = {
            "method": "Carr-Madan FFT",
            "N_points": self.N,
            "alpha": self.alpha,
        }

        return PricingResult(
            price=price,
            std_error=0.0,
            computation_time=computation_time,
            metadata=metadata,
        )
