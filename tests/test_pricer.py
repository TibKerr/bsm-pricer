import numpy as np
import pytest

from bsm_pricer import greeks as G
from bsm_pricer.pricer import bsm_pricer, calc_factors, validate_inputs

# --- Reference parameters ---
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


# --- Pricing ---------------------------------------------------------------

def test_known_textbook_values():
    # ATM, 1y, r=5%, vol=20%, no dividend -> classic BSM values.
    assert bsm_pricer(**BASE, option_type="call") == pytest.approx(10.4506, abs=1e-4)
    assert bsm_pricer(**BASE, option_type="put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_no_dividend():
    c = bsm_pricer(**BASE, option_type="call")
    p = bsm_pricer(**BASE, option_type="put")
    S, K, T, r = BASE["S"], BASE["K"], BASE["T"], BASE["r"]
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)


def test_put_call_parity_with_dividend():
    args = dict(S=100, K=95, T=0.75, r=0.04, sigma=0.25)
    q = 0.03
    c = bsm_pricer(**args, option_type="call", q=q)
    p = bsm_pricer(**args, option_type="put", q=q)
    S, K, T, r = args["S"], args["K"], args["T"], args["r"]
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert c - p == pytest.approx(expected, abs=1e-9)


def test_expiry_returns_intrinsic_value():
    assert bsm_pricer(110, 100, 0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert bsm_pricer(90, 100, 0, 0.05, 0.2, "put") == pytest.approx(10.0)
    assert bsm_pricer(90, 100, 0, 0.05, 0.2, "call") == pytest.approx(0.0)


def test_d2_relationship():
    d1, d2 = calc_factors(**BASE)
    assert d1 - d2 == pytest.approx(BASE["sigma"] * np.sqrt(BASE["T"]))


# --- Validation ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(S=0), dict(K=0), dict(sigma=0), dict(sigma=-0.1), dict(T=-1),
])
def test_validation_rejects_bad_numeric_input(bad):
    args = {**BASE, **bad}
    with pytest.raises(ValueError):
        bsm_pricer(**args)


def test_validation_rejects_bad_option_type():
    with pytest.raises(ValueError):
        bsm_pricer(**BASE, option_type="banana")


def test_validate_inputs_returns_none_on_valid():
    assert validate_inputs(**BASE, option_type="call") is None


# --- Greeks vs finite differences -----------------------------------------

GP = dict(S=100.0, K=95.0, T=0.75, r=0.04, sigma=0.25, q=0.03)


def _price(option_type, **overrides):
    p = {**GP, **overrides}
    return bsm_pricer(p["S"], p["K"], p["T"], p["r"], p["sigma"], option_type, p["q"])


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    h = 1e-4
    fd = (_price(option_type, S=GP["S"] + h) - _price(option_type, S=GP["S"] - h)) / (2 * h)
    assert G.delta(**GP, option_type=option_type) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(option_type):
    h = 1e-3
    fd = (_price(option_type, S=GP["S"] + h) - 2 * _price(option_type)
          + _price(option_type, S=GP["S"] - h)) / h**2
    assert G.gamma(**GP, option_type=option_type) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    h = 1e-4
    fd = (_price(option_type, sigma=GP["sigma"] + h)
          - _price(option_type, sigma=GP["sigma"] - h)) / (2 * h)
    # vega is scaled per 1% move, fd is raw (per 1.0)
    assert G.vega(**GP, option_type=option_type) * 100 == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type):
    h = 1e-4
    # theta = -dPrice/dT, scaled per day
    fd = -(_price(option_type, T=GP["T"] + h) - _price(option_type, T=GP["T"] - h)) / (2 * h)
    assert G.theta(**GP, option_type=option_type) * 365 == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    h = 1e-4
    fd = (_price(option_type, r=GP["r"] + h) - _price(option_type, r=GP["r"] - h)) / (2 * h)
    assert G.rho(**GP, option_type=option_type) * 100 == pytest.approx(fd, abs=1e-3)


def test_gamma_and_vega_independent_of_option_type():
    assert G.gamma(**GP, option_type="call") == G.gamma(**GP, option_type="put")
    assert G.vega(**GP, option_type="call") == G.vega(**GP, option_type="put")


def test_greeks_at_expiry():
    assert G.gamma(100, 100, 0, 0.05, 0.2) == 0.0
    assert G.vega(100, 100, 0, 0.05, 0.2) == 0.0
    assert G.theta(100, 100, 0, 0.05, 0.2) == 0.0
    assert G.delta(110, 100, 0, 0.05, 0.2, "call") == 1.0
    assert G.delta(90, 100, 0, 0.05, 0.2, "call") == 0.0
