"""Black-Scholes-Merton pricing for European options with continuous dividends."""

from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def calc_factors(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> tuple[float, float]:
    """Calculate the ``d1`` and ``d2`` probability factors.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        q: annualized continuous dividend yield

    Returns:
        Tuple of the ``d1`` and ``d2`` factors.
    """
    # Isolate numerator and denominator for readability.
    numerator = np.log(S / K) + (r - q + (sigma**2 / 2)) * T
    denominator = sigma * np.sqrt(T)

    d1 = numerator / denominator
    d2 = d1 - denominator

    return d1, d2


def validate_inputs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> None:
    """Validate the inputs shared by the pricer and the Greeks.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0:
        raise ValueError(f"S (spot price) must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"K (strike price) must be positive, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma (volatility) must be positive, got {sigma}")
    if T < 0:
        raise ValueError(f"T (time to expiry) must be non-negative, got {T}")


def bsm_pricer(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Price a European option using the Black-Scholes-Merton formula.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Theoretical price of the option.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)

    # At expiry the option is worth its intrinsic value.
    if T == 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    d1, d2 = calc_factors(S, K, T, r, sigma, q)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return float(price)
