"""
JAX-accelerated MGF-based option pricer.

Fully vectorized across strikes, JIT-compiled for CPU/GPU. Supports Heston,
Merton, Bates, Black-Scholes, and any model that provides a log-MGF in Sepp's
convention: ``L(phi) = log E[exp(-phi * log(S_T / F))]``.

The inversion (Lewis 2001, Lee 2004). With ``k = log(F / K)`` and the
integrand ``f_c(p) = Re[exp(-k phi + L(phi)) / (-phi (phi + 1))]`` on the
contour ``phi = c + i p``, ``I_c = (1/pi) int_0^inf f_c(p) dp``:

- on Lewis's contour ``c = -1/2`` (between the integrand's poles at 0 and
  -1), a call is ``D (F - K I)``: the forward less an integral almost equal to
  it, so a deep out-of-the-money option is a difference of two large numbers;
- past the pole at -1 (``c < -1``), the residue ``F / K`` drops out and an
  out-of-the-money call is ``-D K I_c`` directly; past the pole at 0
  (``c > 0``), an out-of-the-money put likewise. Each option is priced out of
  the money this way and the other side by parity.

A shifted contour needs the MGF to exist there (a moment of order 1.25 or
-0.25 of the spot); where a model's moments explode, the MGF evaluated at the
real point ``c`` is no longer real, and the price falls back to Lewis's
contour, which every martingale model admits.

The quadrature. The integrand bends sharply within about 1/2 of ``p = 0``
(the poles) and decays over ``p ~ 1 / sigma sqrt(T)`` (Black-Scholes) or
slower (Heston's characteristic function decays only exponentially in
``p``). Simpson's rule on ``p = sinh(u)`` with ``u`` uniform resolves both:
spacing ``du`` near 0, ``p du`` in the tail, out to ``REACH / vol_scaler``.
With 2001 nodes the prices agree with Black-Scholes to 1e-14 and with a
4096-term Heston COS reference to 1e-8, from one-day to five-year options and
for strikes from half to two and a half times the spot.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..processes._jax_utils import to_numpy
from ._pricer_runtime import guard_pricer

CALL_CONTOUR = -1.25   # past the pole at -1: out-of-the-money calls, needs E[S^1.25]
PUT_CONTOUR = 0.25     # past the pole at 0: out-of-the-money puts, needs E[S^-0.25]
LEWIS_CONTOUR = -0.5   # between the poles: the fallback where a shifted contour leaves the strip
REACH = 400.0          # the grid reaches REACH / vol_scaler
NODES = 2001           # default quadrature nodes (odd, for Simpson's rule)
_REAL_TOLERANCE = 1e-9


# ============================================================================
# Quadrature
# ============================================================================
class Quadrature(NamedTuple):
    """Nodes ``p >= 0`` on the imaginary axis and their weights."""

    p: jnp.ndarray
    weights: jnp.ndarray


def make_quadrature(vol_scaler: float, n: int = NODES) -> Quadrature:
    """Simpson's rule on ``p = sinh(u)``, ``u`` uniform, ``p`` up to ``REACH / vol_scaler``.

    ``n`` is made odd (Simpson's rule needs an even number of intervals).
    """
    n = n if n % 2 == 1 else n + 1
    u = jnp.linspace(0.0, jnp.arcsinh(REACH / vol_scaler), n)
    simpson = jnp.ones(n).at[1:-1:2].set(4.0).at[2:-1:2].set(2.0)
    weights = simpson * (u[1] - u[0]) / 3.0 * jnp.cosh(u)
    return Quadrature(p=jnp.sinh(u), weights=weights)


def _integral(log_mgf: Callable[[jnp.ndarray], jnp.ndarray], c: float, quad: Quadrature,
              log_moneyness: jnp.ndarray) -> jnp.ndarray:
    """``I_c`` for each log-moneyness: shape (n_strikes,)."""
    phi = c + 1j * quad.p
    kernel = jnp.exp(-log_moneyness[:, None] * phi[None, :] + log_mgf(phi)[None, :]) / (-phi * (phi + 1.0))[None, :]
    return jnp.sum(jnp.real(kernel) * quad.weights[None, :], axis=1) / jnp.pi


def _exists_at(log_mgf: Callable[[jnp.ndarray], jnp.ndarray], c: float) -> jnp.ndarray:
    """Whether the MGF exists at the real point ``c``: there it is real and finite."""
    value = log_mgf(jnp.asarray([c + 0.0j]))[0]
    return jnp.isfinite(value.real) & (jnp.abs(value.imag) <= _REAL_TOLERANCE * (1.0 + jnp.abs(value.real)))


def price_from_log_mgf(
    log_mgf: Callable[[jnp.ndarray], jnp.ndarray],
    vol_scaler: float,
    forward: float,
    strikes: jnp.ndarray,
    is_call: jnp.ndarray,
    discfactor: float,
    n: int = NODES,
) -> jnp.ndarray:
    """Vanilla prices from a model's log-MGF (Sepp convention), one per strike.

    Args:
        log_mgf: phi -> log E[exp(-phi log(S_T / F))], vectorized over complex phi
        vol_scaler: sigma sqrt(T) scale of the log-return, sizing the grid
        forward: forward price F
        strikes: shape (n_strikes,)
        is_call: shape (n_strikes,)
        discfactor: discount factor D
        n: quadrature nodes
    """
    quad = make_quadrature(vol_scaler, n)
    log_moneyness = jnp.log(forward / strikes)
    otm_call = log_moneyness <= 0.0

    # each option out of the money, on the contour past the pole its money term comes from
    shifted = -discfactor * strikes * jnp.where(
        otm_call,
        _integral(log_mgf, CALL_CONTOUR, quad, log_moneyness),
        _integral(log_mgf, PUT_CONTOUR, quad, log_moneyness),
    )
    capped = _integral(log_mgf, LEWIS_CONTOUR, quad, log_moneyness)
    lewis = discfactor * jnp.where(otm_call, forward - strikes * capped, strikes - strikes * capped)
    admitted = jnp.where(otm_call, _exists_at(log_mgf, CALL_CONTOUR), _exists_at(log_mgf, PUT_CONTOUR))
    out_of_money = jnp.where(admitted, shifted, lewis)

    parity = discfactor * (forward - strikes)  # call - put
    call = jnp.where(otm_call, out_of_money, out_of_money + parity)
    put = jnp.where(otm_call, out_of_money - parity, out_of_money)
    prices = jnp.where(is_call, call, put)

    intrinsic = jnp.where(is_call, jnp.maximum(parity, 0.0), jnp.maximum(-parity, 0.0))
    return jnp.maximum(prices, intrinsic)


# ============================================================================
# Heston log-MGF (JAX)
# ============================================================================
class HestonMGFParams(NamedTuple):
    v0: float
    theta: float
    kappa: float
    volvol: float
    rho: float


def heston_log_mgf(params: HestonMGFParams, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute Heston log-MGF on phi grid (Sepp formulation)."""
    volvol2 = params.volvol * params.volvol

    b1 = params.kappa + params.rho * params.volvol * phi_grid
    b0 = 0.5 * phi_grid * (phi_grid + 1.0)

    zeta = jnp.sqrt(b1 * b1 - 2.0 * b0 * volvol2)
    exp_zeta_t = jnp.exp(-zeta * ttm)

    psi_p = -b1 + zeta
    psi_m = b1 + zeta

    c_p = psi_p / (2.0 * zeta)
    c_m = psi_m / (2.0 * zeta)

    b_t = -(-psi_m * c_p * exp_zeta_t + psi_p * c_m) / (volvol2 * (c_p * exp_zeta_t + c_m))
    a_t = -(params.theta * params.kappa / volvol2) * (
        psi_p * ttm + 2.0 * jnp.log(c_p * exp_zeta_t + c_m)
    )

    return a_t + b_t * params.v0


# ============================================================================
# Merton log-MGF (JAX)
# ============================================================================
class MertonMGFParams(NamedTuple):
    sigma: float
    lambda_j: float
    mu_J: float
    sigma_J: float


def merton_log_mgf(params: MertonMGFParams, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute Merton jump-diffusion log-MGF on phi grid (Sepp convention).

    Sepp convention: MGF of -log(S_T/F), so b0 = 0.5*phi*(phi+1)
    and jump uses exp(-phi*mu_J) with +k*phi compensator.
    """
    kappa = jnp.exp(params.mu_J + 0.5 * params.sigma_J ** 2) - 1.0
    # Diffusion: Sepp convention 0.5*sigma^2*T*phi*(phi+1)
    diffusion = 0.5 * params.sigma ** 2 * ttm * phi_grid * (phi_grid + 1.0)
    # Jump: Sepp convention (phi -> -phi of standard)
    jump = params.lambda_j * ttm * (
        jnp.exp(-phi_grid * params.mu_J + 0.5 * params.sigma_J ** 2 * phi_grid ** 2) - 1.0
        + kappa * phi_grid
    )
    return diffusion + jump


# ============================================================================
# Bates log-MGF (JAX) = Heston + Merton jumps
# ============================================================================
class BatesMGFParams(NamedTuple):
    v0: float
    theta: float
    kappa: float
    volvol: float
    rho: float
    lambda_j: float
    mu_J: float
    sigma_J: float


def bates_log_mgf(params: BatesMGFParams, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Compute Bates log-MGF = Heston + jump component (Sepp convention)."""
    heston_params = HestonMGFParams(
        v0=params.v0, theta=params.theta, kappa=params.kappa,
        volvol=params.volvol, rho=params.rho,
    )
    log_mgf_h = heston_log_mgf(heston_params, ttm, phi_grid)

    # Jump component in Sepp convention (phi -> -phi of standard):
    #   Sepp uses MGF of -log(S_T/F), so exp(-phi*mu_J) and +k*phi
    kappa_j = jnp.exp(params.mu_J + 0.5 * params.sigma_J ** 2) - 1.0
    log_mgf_jump = params.lambda_j * ttm * (
        jnp.exp(-phi_grid * params.mu_J + 0.5 * params.sigma_J ** 2 * phi_grid ** 2) - 1.0
        + kappa_j * phi_grid
    )

    return log_mgf_h + log_mgf_jump


# ============================================================================
# Black-Scholes log-MGF (JAX)
# ============================================================================
def bs_log_mgf(sigma: float, ttm: float, phi_grid: jnp.ndarray) -> jnp.ndarray:
    """Black-Scholes log-MGF (GBM with constant vol, Sepp convention)."""
    # Sepp convention: 0.5*sigma^2*T*phi*(phi+1)
    return 0.5 * sigma ** 2 * ttm * phi_grid * (phi_grid + 1.0)


# ============================================================================
# High-level JIT-compiled pricing functions
# ============================================================================
@guard_pricer
@jax.jit(static_argnums=(11,))
def heston_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float,
    v0: float, theta: float, kappa: float, volvol: float, rho: float,
    max_phi: int = NODES,
) -> jnp.ndarray:
    """
    Price a grid of vanilla options under Heston. Fully JIT-compiled.

    Args:
        S: Spot price
        strikes: Shape (n_strikes,)
        is_call: Boolean array, shape (n_strikes,)
        T: Time to maturity
        r, q: Risk-free rate, dividend yield
        v0, theta, kappa, volvol, rho: Heston parameters
        max_phi: quadrature nodes (made odd)

    Returns:
        Option prices, shape (n_strikes,)
    """
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt(v0 * T))
    params = HestonMGFParams(v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho)
    return price_from_log_mgf(lambda phi: heston_log_mgf(params, T, phi), vol_scaler, forward, strikes, is_call,
                              discfactor, max_phi)


@guard_pricer
@jax.jit(static_argnums=(10,))
def merton_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float,
    sigma: float, lambda_j: float, mu_J: float, sigma_J: float,
    max_phi: int = NODES,
) -> jnp.ndarray:
    """Price a grid of vanilla options under Merton JD. Fully JIT-compiled."""
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    # the grid's scale: the log-return's whole variance, jumps included (a pure-jump model has no sigma)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt((sigma ** 2 + lambda_j * (mu_J ** 2 + sigma_J ** 2)) * T))
    params = MertonMGFParams(sigma=sigma, lambda_j=lambda_j, mu_J=mu_J, sigma_J=sigma_J)
    return price_from_log_mgf(lambda phi: merton_log_mgf(params, T, phi), vol_scaler, forward, strikes, is_call,
                              discfactor, max_phi)


@guard_pricer
@jax.jit(static_argnums=(14,))
def bates_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float,
    v0: float, theta: float, kappa: float, volvol: float, rho: float,
    lambda_j: float, mu_J: float, sigma_J: float,
    max_phi: int = NODES,
) -> jnp.ndarray:
    """Price a grid of vanilla options under Bates. Fully JIT-compiled."""
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt((v0 + lambda_j * (mu_J ** 2 + sigma_J ** 2)) * T))
    params = BatesMGFParams(
        v0=v0, theta=theta, kappa=kappa, volvol=volvol, rho=rho,
        lambda_j=lambda_j, mu_J=mu_J, sigma_J=sigma_J,
    )
    return price_from_log_mgf(lambda phi: bates_log_mgf(params, T, phi), vol_scaler, forward, strikes, is_call,
                              discfactor, max_phi)


@guard_pricer
@jax.jit(static_argnums=(7,))
def bs_price_grid_jax(
    S: float, strikes: jnp.ndarray, is_call: jnp.ndarray,
    T: float, r: float, q: float, sigma: float,
    max_phi: int = NODES,
) -> jnp.ndarray:
    """Price a grid of vanilla options under Black-Scholes via MGF. Fully JIT-compiled."""
    forward = S * jnp.exp((r - q) * T)
    discfactor = jnp.exp(-r * T)
    vol_scaler = jnp.minimum(0.3, jnp.sqrt(sigma ** 2 * T))
    return price_from_log_mgf(lambda phi: bs_log_mgf(sigma, T, phi), vol_scaler, forward, strikes, is_call,
                              discfactor, max_phi)


# ============================================================================
# Numpy-returning convenience wrappers (drop-in replacements)
# ============================================================================
def heston_price_slice_fast(
    S, strikes, T, r, q, v0, theta, kappa, volvol, rho,
    option_types, max_phi=NODES,
):
    """Drop-in replacement for heston_price_slice. Returns numpy."""
    strikes_j = jnp.array(strikes, dtype=jnp.float64)
    is_call = jnp.array([t == 'call' for t in option_types], dtype=bool)
    prices = heston_price_grid_jax(
        float(S), strikes_j, is_call, float(T), float(r), float(q),
        float(v0), float(theta), float(kappa), float(volvol), float(rho),
        max_phi,
    )
    return to_numpy(prices)


def merton_price_slice_fast(
    S, strikes, T, r, q, sigma, lambda_j, mu_J, sigma_J,
    option_types, max_phi=NODES,
):
    """Drop-in replacement for merton_price_slice. Returns numpy."""
    strikes_j = jnp.array(strikes, dtype=jnp.float64)
    is_call = jnp.array([t == 'call' for t in option_types], dtype=bool)
    prices = merton_price_grid_jax(
        float(S), strikes_j, is_call, float(T), float(r), float(q),
        float(sigma), float(lambda_j), float(mu_J), float(sigma_J),
        max_phi,
    )
    return to_numpy(prices)


def bates_price_slice_fast(
    S, strikes, T, r, q, v0, theta, kappa, volvol, rho,
    lambda_j, mu_J, sigma_J, option_types, max_phi=NODES,
):
    """Drop-in replacement for bates_price_slice. Returns numpy."""
    strikes_j = jnp.array(strikes, dtype=jnp.float64)
    is_call = jnp.array([t == 'call' for t in option_types], dtype=bool)
    prices = bates_price_grid_jax(
        float(S), strikes_j, is_call, float(T), float(r), float(q),
        float(v0), float(theta), float(kappa), float(volvol), float(rho),
        float(lambda_j), float(mu_J), float(sigma_J),
        max_phi,
    )
    return to_numpy(prices)
