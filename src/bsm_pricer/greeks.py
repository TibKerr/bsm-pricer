"""Analytic Black-Scholes-Merton Greeks with continuous dividend support.

Scalar interface over ``array.py``, where the math lives: takes plain floats and
``"call"`` / ``"put"``, and returns plain floats.

Scaling conventions (also documented per function):
    delta, gamma : raw, per 1.0 change in spot
    vega         : per +1 percentage point (0.01) change in volatility
    theta        : per calendar day of decay
    rho          : per +1 percentage point (0.01) change in the rate
"""

from . import array
from .pricer import OptionType, validate_inputs


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

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Delta, in price units per 1.0 change in spot.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)
    return float(array.delta(S, K, T, r, sigma, option_type == "call", q))


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

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Gamma, in delta units per 1.0 change in spot.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)
    return float(array.gamma(S, K, T, r, sigma, option_type == "call", q))


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

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Vega, in price units per +1% (0.01) change in sigma.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)
    return float(array.vega(S, K, T, r, sigma, option_type == "call", q))


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

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Theta, in price units per 1 day of decay.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)
    return float(array.theta(S, K, T, r, sigma, option_type == "call", q))


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

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Rho, in price units per +1% (0.01) change in r.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)
    return float(array.rho(S, K, T, r, sigma, option_type == "call", q))


def all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """Return every Greek for the given option in a single dict.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, ``"call"`` or ``"put"``
        q: annualized continuous dividend yield

    Returns:
        Dict mapping ``"delta"``, ``"gamma"``, ``"vega"``, ``"theta"`` and
        ``"rho"`` to their values, scaled as in the individual functions.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, option_type, q),
        "vega": vega(S, K, T, r, sigma, option_type, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }
