"""Tests for greeks.py: every Greek checked against a finite-difference bump."""

import pytest

from bsm_pricer import greeks
from bsm_pricer.pricer import bsm_pricer

# --- Reference parameters --------------------------------------------------

PARAMS = dict(S=100.0, K=95.0, T=0.75, r=0.04, sigma=0.25, q=0.03)


def _price(option_type, **overrides):
    p = {**PARAMS, **overrides}
    return bsm_pricer(p["S"], p["K"], p["T"], p["r"], p["sigma"], option_type, p["q"])


# --- Greeks vs finite differences -----------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    h = 1e-4
    fd = (_price(option_type, S=PARAMS["S"] + h)
          - _price(option_type, S=PARAMS["S"] - h)) / (2 * h)
    assert greeks.delta(**PARAMS, option_type=option_type) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(option_type):
    h = 1e-3
    fd = (_price(option_type, S=PARAMS["S"] + h) - 2 * _price(option_type)
          + _price(option_type, S=PARAMS["S"] - h)) / h**2
    assert greeks.gamma(**PARAMS, option_type=option_type) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    h = 1e-4
    fd = (_price(option_type, sigma=PARAMS["sigma"] + h)
          - _price(option_type, sigma=PARAMS["sigma"] - h)) / (2 * h)
    # vega is scaled per 1% move, fd is raw (per 1.0)
    assert greeks.vega(**PARAMS, option_type=option_type) * 100 == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(option_type):
    h = 1e-4
    # theta = -dPrice/dT, scaled per day
    fd = -(_price(option_type, T=PARAMS["T"] + h)
           - _price(option_type, T=PARAMS["T"] - h)) / (2 * h)
    assert greeks.theta(**PARAMS, option_type=option_type) * 365 == pytest.approx(fd, abs=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    h = 1e-4
    fd = (_price(option_type, r=PARAMS["r"] + h)
          - _price(option_type, r=PARAMS["r"] - h)) / (2 * h)
    # rho is scaled per 1% move, fd is raw (per 1.0)
    assert greeks.rho(**PARAMS, option_type=option_type) * 100 == pytest.approx(fd, abs=1e-3)


# --- Structure and expiry --------------------------------------------------

def test_gamma_and_vega_independent_of_option_type():
    assert greeks.gamma(**PARAMS, option_type="call") == greeks.gamma(**PARAMS, option_type="put")
    assert greeks.vega(**PARAMS, option_type="call") == greeks.vega(**PARAMS, option_type="put")


def test_all_greeks_matches_individual_functions():
    result = greeks.all_greeks(**PARAMS, option_type="call")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert result[name] == getattr(greeks, name)(**PARAMS, option_type="call")


def test_greeks_at_expiry():
    assert greeks.gamma(100, 100, 0, 0.05, 0.2) == 0.0
    assert greeks.vega(100, 100, 0, 0.05, 0.2) == 0.0
    assert greeks.theta(100, 100, 0, 0.05, 0.2) == 0.0
    assert greeks.delta(110, 100, 0, 0.05, 0.2, "call") == 1.0
    assert greeks.delta(90, 100, 0, 0.05, 0.2, "call") == 0.0
