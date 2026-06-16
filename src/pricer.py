import numpy as np
from scipy.stats import norm
from typing import Literal

# S=spot price, K=strike, T=time to expiry, r=risk-free rate, sigma=volatility
def calc_factors(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> tuple[float, float]:
    '''
    Calculates the d1 and d2 probability factors

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        q: annualized continuous dividend yield

    Returns
         tuple of d1 and d2 factors
    '''
    # Isolate numerator and denominator for readability
    numerator = np.log(S / K) + (r - q + (sigma**2 / 2)) * T
    denominator = sigma * np.sqrt(T)

    d1 = numerator / denominator
    d2 = d1 - denominator

    return d1, d2

def bsm_pricer(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal['call', 'put'] = 'call',
        q: float = 0.0,
    ) -> float:
    '''
    Calculates the theoretical price of a European option using Black-Scholes

    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, call or put
        q: annualized continuous dividend yield

     Returns:
          theoretical price of the option
    '''

def validate_inputs(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal['call', 'put'],
        q: float = 0.0,
    ) -> None:
    '''
    Validates the inputs for the pricer
    
    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: type of the option, call or put
        q: annualized continuous dividend yield

    Returns:
        None
    '''

    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0:
        raise ValueError(f"S (spot price) must be positive, got {S}")
    if K <= 0:
        raise ValueError(f"K (strike price) must be positive, got {K}")
    if sigma <= 0:
        raise ValueError(f"sigma (volatility) must be positive, got {sigma}")
    if T < 0:
        raise ValueError(f"T (time to expiry) must be non-negative, got {T}")

    if T == 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)

    d1, d2 = calc_factors(S, K, T, r, sigma, q)

    if option_type == 'call':
        price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
    
    return float(price)

if __name__ == "__main__":
    # Example usage
    S = 100  # Spot price
    K = 100  # Strike price
    T = 1    # Time to expiry in years
    r = 0.05 # Risk-free rate
    sigma = 0.2 # Volatility

    call_price = bsm_pricer(S, K, T, r, sigma, option_type='call')
    put_price = bsm_pricer(S, K, T, r, sigma, option_type='put')

    print(f"Call Option Price: {call_price:.2f}")
    print(f"Put Option Price: {put_price:.2f}")