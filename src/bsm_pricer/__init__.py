"""bsm_pricer: Black-Scholes-Merton option pricing with analytic Greeks."""

from .greeks import all_greeks, delta, gamma, rho, theta, vega
from .pricer import OptionType, bsm_pricer, calc_factors, validate_inputs

__version__ = "0.1.0"

__all__ = [
    "OptionType",
    "bsm_pricer",
    "calc_factors",
    "validate_inputs",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "all_greeks",
]
