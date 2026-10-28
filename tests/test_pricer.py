"""Tests for pricer.py: reference values, parity, expiry, and validation."""

import numpy as np
import pytest

from bsm_pricer.pricer import bsm_pricer, calc_factors, validate_inputs

# --- Reference parameters --------------------------------------------------

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


# --- Pricing ---------------------------------------------------------------

def test_known_textbook_values():
    # ATM, 1y, r=5%, vol=20%, no dividend -> classic BSM values.
    assert bsm_pricer(**PARAMS, option_type="call") == pytest.approx(10.4506, abs=1e-4)
    assert bsm_pricer(**PARAMS, option_type="put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_no_dividend():
    c = bsm_pricer(**PARAMS, option_type="call")
    p = bsm_pricer(**PARAMS, option_type="put")
    S, K, T, r = PARAMS["S"], PARAMS["K"], PARAMS["T"], PARAMS["r"]
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)


def test_put_call_parity_with_dividend():
    params = dict(S=100.0, K=95.0, T=0.75, r=0.04, sigma=0.25, q=0.03)
    c = bsm_pricer(**params, option_type="call")
    p = bsm_pricer(**params, option_type="put")
    S, K, T, r, q = params["S"], params["K"], params["T"], params["r"], params["q"]
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-9)


def test_expiry_returns_intrinsic_value():
    assert bsm_pricer(110, 100, 0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert bsm_pricer(90, 100, 0, 0.05, 0.2, "put") == pytest.approx(10.0)
    assert bsm_pricer(90, 100, 0, 0.05, 0.2, "call") == pytest.approx(0.0)


def test_d2_relationship():
    d1, d2 = calc_factors(**PARAMS)
    assert d1 - d2 == pytest.approx(PARAMS["sigma"] * np.sqrt(PARAMS["T"]))


# --- Validation ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(S=0), dict(K=0), dict(sigma=0), dict(sigma=-0.1), dict(T=-1),
])
def test_validation_rejects_bad_numeric_input(bad):
    with pytest.raises(ValueError):
        bsm_pricer(**{**PARAMS, **bad})


def test_validation_rejects_bad_option_type():
    with pytest.raises(ValueError):
        bsm_pricer(**PARAMS, option_type="banana")


def test_validation_accepts_negative_rate_and_dividend():
    assert validate_inputs(**{**PARAMS, "r": -0.01}, option_type="call", q=-0.02) is None


def test_validate_inputs_returns_none_on_valid():
    assert validate_inputs(**PARAMS, option_type="call") is None
