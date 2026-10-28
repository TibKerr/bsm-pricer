"""Vectorized Black-Scholes-Merton pricing and Greeks over numpy arrays.

Array counterparts of ``pricer.py`` and ``greeks.py``, built for pricing many
options in one call (e.g. every trade in a backtest).

Conventions:
    - Every numeric input may be a scalar or array-like; inputs are broadcast
      together following numpy rules, and results are float arrays of the
      broadcast shape.
    - ``is_call`` is a boolean (or boolean array) instead of the ``"call"`` /
      ``"put"`` string, so calls and puts can be mixed in a single call.
    - Greek scaling matches ``greeks.py``: delta and gamma raw, vega and rho per
      +1 percentage point, theta per calendar day.
    - Invalid inputs raise ``ValueError``, as in the scalar API.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def calc_factors(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Calculate the ``d1`` and ``d2`` probability factors, element-wise.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years; must be > 0 (pass a placeholder for expired rows)
        r: annualized risk-free rate
        sigma: annualized implied volatility
        q: annualized continuous dividend yield

    Returns:
        Tuple of the ``d1`` and ``d2`` factor arrays.
    """
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))

    # Isolate numerator and denominator for readability.
    numerator = np.log(S / K) + (r - q + (sigma**2 / 2)) * T
    denominator = sigma * np.sqrt(T)

    d1 = numerator / denominator
    d2 = d1 - denominator

    return d1, d2


def _prepare_inputs(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, BoolArray, FloatArray]:
    """Convert inputs to broadcast arrays and validate them.

    The array counterpart of ``validate_inputs``: same rules, but every row is
    checked and the error names the offending indices.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        is_call: ``True`` for a call, ``False`` for a put
        q: annualized continuous dividend yield

    Returns:
        The inputs as float arrays (``is_call`` as bool), all broadcast to one shape.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    # Strings (or object columns like a pandas "call"/"put" Series) would cast
    # to True silently, so reject them instead of pricing everything as a call.
    is_call = np.asarray(is_call)
    if is_call.dtype.kind in "USO":
        raise ValueError(
            "is_call must be a bool or bool array, not strings; "
            "convert first, e.g. is_call = (option_type == 'call')"
        )

    S, K, T, r, sigma, is_call, q = np.broadcast_arrays(
        np.asarray(S, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(sigma, dtype=float),
        is_call.astype(bool),
        np.asarray(q, dtype=float),
    )

    # r and q may be any value, including negative.
    checks = [
        (S <= 0, "S (spot price) must be positive"),
        (K <= 0, "K (strike price) must be positive"),
        (sigma <= 0, "sigma (volatility) must be positive"),
        (T < 0, "T (time to expiry) must be non-negative"),
    ]
    for is_bad, message in checks:
        bad = np.flatnonzero(is_bad)
        if bad.size:
            raise ValueError(f"{message}; bad at index {bad[:5].tolist()}")

    return S, K, T, r, sigma, is_call, q


def bsm_pricer(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
    q: ArrayLike = 0.0,
) -> FloatArray:
    """Price European options using the Black-Scholes-Merton formula, element-wise.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        is_call: ``True`` for a call, ``False`` for a put
        q: annualized continuous dividend yield

    Returns:
        Theoretical option prices, in the broadcast shape of the inputs.

    Raises:
        ValueError: if any input is outside its valid domain.
    """
    S, K, T, r, sigma, is_call, q = _prepare_inputs(S, K, T, r, sigma, is_call, q)

    # Expired rows would divide by zero; give them a placeholder T and
    # swap in intrinsic value at the end.
    expired = T == 0
    T_safe = np.where(expired, 1.0, T)

    d1, d2 = calc_factors(S, K, T_safe, r, sigma, q)
    disc_q = np.exp(-q * T_safe)
    disc_r = np.exp(-r * T_safe)

    call_price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put_price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    price = np.where(is_call, call_price, put_price)

    # At expiry the option is worth its intrinsic value.
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(expired, intrinsic, price)
