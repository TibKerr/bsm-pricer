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
    ) -> tuple[float, float]:
    '''
    Calculates teh d1 and d2 probability factors
    
    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        
    Returns
         Tuple of d1 and d2 factors
    '''
    # Isolate numerator and denominator for readability
    numerator = np.log(S / K) + (r + (sigma**2 / 2)) * T
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
        option_type: Literal['call', 'put'] = 'call'
    ) -> float:
    '''
    Calculates the theoretical price of a European option using Black-Scholes
    
    Args:
        S: spot price of asset
        K: strike price of option
        T: time to expiry in years
        r: annualized risk-free rate
        sigma: annualized implied volatility
        option_type: 'type of the option, call or put
        
     Returns:
          theoretical price of the option
    '''
    d1, d2 = calc_factors(S, K, T, r, sigma)

    if T <= 0:
        return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)

    if option_type == 'call':
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
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