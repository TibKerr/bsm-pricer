"""Tests for array.py: parity, finite differences, shapes, and validation.

The scalar functions are wrappers over this module, so these tests check the
math directly instead of comparing against them.
"""

import itertools

import numpy as np
import pytest

from bsm_pricer import array, greeks
from bsm_pricer.pricer import bsm_pricer

# Any RuntimeWarning (e.g. divide by zero at expiry) fails the test.
pytestmark = pytest.mark.filterwarnings("error")

# --- Reference grid --------------------------------------------------------

# Every combination of ITM/ATM/OTM, expired and live, negative rates and
# dividends, and calls and puts: 144 rows.
CASES = list(itertools.product(
    [80.0, 100.0, 120.0],  # S
    [100.0],               # K
    [0.0, 0.1, 1.0],       # T (0 = expired)
    [0.05, -0.01],         # r
    [0.2, 0.5],            # sigma
    [True, False],         # is_call
    [0.03, -0.01],         # q
))
NAMES = ("S", "K", "T", "r", "sigma", "is_call", "q")
COLUMNS = zip(*CASES, strict=True)
PARAMS = {name: np.array(column) for name, column in zip(NAMES, COLUMNS, strict=True)}

SCALAR_FUNCTIONS = {
    "bsm_pricer": bsm_pricer,
    "delta": greeks.delta,
    "gamma": greeks.gamma,
    "vega": greeks.vega,
    "theta": greeks.theta,
    "rho": greeks.rho,
}


def _scalar_results(scalar_function):
    """Run the scalar function row by row over CASES."""
    return np.array([
        scalar_function(S, K, T, r, sigma, "call" if is_call else "put", q)
        for S, K, T, r, sigma, is_call, q in CASES
    ])


# --- Pricing ---------------------------------------------------------------

def test_put_call_parity():
    calls = array.bsm_pricer(**{**PARAMS, "is_call": True})
    puts = array.bsm_pricer(**{**PARAMS, "is_call": False})
    S, K, T, r, q = (PARAMS[name] for name in ("S", "K", "T", "r", "q"))
    np.testing.assert_allclose(calls - puts, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


# --- Greeks vs finite differences -----------------------------------------

# Rows with time left; the finite-difference bumps need T > 0.
LIVE = {name: values[PARAMS["T"] > 0] for name, values in PARAMS.items()}


def _price(**overrides):
    return array.bsm_pricer(**{**LIVE, **overrides})


@pytest.mark.parametrize("greek, bumped, scale", [
    ("delta", "S", 1.0),       # raw
    ("vega", "sigma", 100.0),  # per 1% move
    ("rho", "r", 100.0),       # per 1% move
    ("theta", "T", -365.0),    # theta = -dPrice/dT, per day
])
def test_greek_matches_finite_difference(greek, bumped, scale):
    h = 1e-4
    fd = (_price(**{bumped: LIVE[bumped] + h})
          - _price(**{bumped: LIVE[bumped] - h})) / (2 * h)
    result = getattr(array, greek)(**LIVE) * scale
    np.testing.assert_allclose(result, fd, rtol=1e-5, atol=1e-6)


def test_gamma_matches_finite_difference():
    h = 1e-3
    fd = (_price(S=LIVE["S"] + h) - 2 * _price()
          + _price(S=LIVE["S"] - h)) / h**2
    np.testing.assert_allclose(array.gamma(**LIVE), fd, rtol=1e-5, atol=1e-6)


# --- Scalar wrappers agree with arrays -------------------------------------

@pytest.mark.parametrize("name", list(SCALAR_FUNCTIONS))
def test_matches_scalar(name):
    result = getattr(array, name)(**PARAMS)
    expected = _scalar_results(SCALAR_FUNCTIONS[name])
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_all_greeks_matches_scalar():
    result = array.all_greeks(**PARAMS)
    assert list(result) == ["delta", "gamma", "vega", "theta", "rho"]
    for name, values in result.items():
        expected = _scalar_results(SCALAR_FUNCTIONS[name])
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


# --- Shapes ----------------------------------------------------------------

def test_scalar_inputs_give_zero_dim_array():
    assert array.bsm_pricer(100, 100, 1, 0.05, 0.2).shape == ()


def test_inputs_broadcast_to_a_grid():
    spots = np.array([[90.0], [100.0], [110.0]])    # shape (3, 1)
    strikes = np.array([95.0, 100.0, 105.0, 110.0])  # shape (4,)
    assert array.bsm_pricer(spots, strikes, 1, 0.05, 0.2).shape == (3, 4)


def test_rejects_incompatible_shapes():
    with pytest.raises(ValueError):
        array.bsm_pricer([100, 100, 100], [90, 100], 1, 0.05, 0.2)


# --- Validation ------------------------------------------------------------

def test_validation_names_bad_index():
    with pytest.raises(ValueError, match=r"S \(spot price\).*\[1\]"):
        array.bsm_pricer([100, -1, 100], 100, 1, 0.05, 0.2)


@pytest.mark.parametrize("is_call", ["put", ["call", "put"], np.array(["call"], dtype=object)])
def test_validation_rejects_string_is_call(is_call):
    with pytest.raises(ValueError, match="is_call"):
        array.bsm_pricer(100, 100, 1, 0.05, 0.2, is_call)
