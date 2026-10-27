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
from scipy.stats import norm


def _prepare_inputs(S, K, T, r, sigma, is_call, q):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Strings (or object columns like a pandas "call"/"put" Series) would cast
    # to True silently, so reject them instead of pricing everything as a call.
    is_call = np.asarray(is_call)
    if is_call.dtype.kind in "USO":
        raise ValueError(
            "is_call must be a bool or bool array, not strings; "
            "convert first, e.g. is_call = (option_type == 'call')"
        )
    is_call = is_call.astype(bool)
    q = np.asarray(q, dtype=float)

    # Broadcasting all inputs to the same shape
    S, K, T, r, sigma, is_call, q = np.broadcast_arrays(S, K, T, r, sigma, is_call, q)

    # Same rules as the scalar validate_inputs; r and q may be any value.
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


def price(S, K, T, r, sigma, is_call=True, q=0.0):
    """Price European options with the Black-Scholes-Merton formula, element-wise.

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        is_call: ``True`` for a call, ``False`` for a put
        q: annualized continuous dividend yield

    Returns:
        Array of theoretical option prices, in the broadcast shape of the inputs.
    """
    S, K, T, r, sigma, is_call, q = _prepare_inputs(S, K, T, r, sigma, is_call, q)

    # Expired rows would divide by zero; give them a placeholder T and
    # swap in intrinsic value at the end.
    expired = T == 0
    T_safe = np.where(expired, 1.0, T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)

    disc_q = np.exp(-q * T_safe)
    disc_r = np.exp(-r * T_safe)
    call_price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put_price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    bsm_price = np.where(is_call, call_price, put_price)

    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(expired, intrinsic, bsm_price)