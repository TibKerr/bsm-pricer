"""Black-Scholes-Merton pricing for European options with continuous dividends.

Scalar interface over ``array.py``, where the math lives: takes plain floats and
``"call"`` / ``"put"``, and returns plain floats.
"""

from typing import Literal

from . import array

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
    d1, d2 = array.calc_factors(S, K, T, r, sigma, q)
    return float(d1), float(d2)


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

    # r and q may be any value, including negative.
    checks = [
        (S <= 0, f"S (spot price) must be positive, got {S}"),
        (K <= 0, f"K (strike price) must be positive, got {K}"),
        (sigma <= 0, f"sigma (volatility) must be positive, got {sigma}"),
        (T < 0, f"T (time to expiry) must be non-negative, got {T}"),
    ]
    for is_bad, message in checks:
        if is_bad:
            raise ValueError(message)


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

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    validate_inputs(S, K, T, r, sigma, option_type, q)
    return float(array.bsm_pricer(S, K, T, r, sigma, option_type == "call", q))
