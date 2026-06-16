"""Analytic Black-Scholes-Merton Greeks with continuous dividend support.

Scaling conventions (also documented per function):
    delta, gamma : raw, per 1.0 change in spot
    vega         : per +1 percentage point (0.01) change in volatility
    theta        : per calendar day of decay
    rho          : per +1 percentage point (0.01) change in the rate
"""


import numpy as np
from scipy.stats import norm

from .pricer import OptionType, calc_factors, validate_inputs


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Sensitivity of option price to a change in spot (dPrice/dS).

    Returns:
        Delta, in price units per 1.0 change in spot.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)

    if T == 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1, _ = calc_factors(S, K, T, r, sigma, q)
    discount = np.exp(-q * T)

    if option_type == "call":
        return float(discount * norm.cdf(d1))
    return float(-discount * norm.cdf(-d1))


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Sensitivity of delta to a change in spot (d2Price/dS2).

    Identical for calls and puts.

    Returns:
        Gamma, in delta units per 1.0 change in spot.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)

    if T == 0:
        return 0.0

    d1, _ = calc_factors(S, K, T, r, sigma, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Sensitivity of option price to a change in volatility.

    Identical for calls and puts. Scaled per +1 percentage point of vol.

    Returns:
        Vega, in price units per +1% (0.01) change in sigma.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)

    if T == 0:
        return 0.0

    d1, _ = calc_factors(S, K, T, r, sigma, q)
    raw_vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return float(raw_vega / 100.0)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Sensitivity of option price to the passage of time (time decay).

    Scaled per calendar day.

    Returns:
        Theta, in price units per 1 day of decay.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)

    if T == 0:
        return 0.0

    d1, d2 = calc_factors(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    # Shared first term, then the type-dependent rate/dividend terms.
    decay = -(S * disc_q * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        raw_theta = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    else:
        raw_theta = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)

    return float(raw_theta / 365.0)


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> float:
    """Sensitivity of option price to a change in the risk-free rate.

    Scaled per +1 percentage point of rate.

    Returns:
        Rho, in price units per +1% (0.01) change in r.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)

    if T == 0:
        return 0.0

    _, d2 = calc_factors(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)

    if option_type == "call":
        raw_rho = K * T * disc_r * norm.cdf(d2)
    else:
        raw_rho = -K * T * disc_r * norm.cdf(-d2)

    return float(raw_rho / 100.0)


def all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """Return every Greek for the given option in a single dict."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, option_type, q),
        "vega": vega(S, K, T, r, sigma, option_type, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }
