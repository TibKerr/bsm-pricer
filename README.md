# bsm-pricer

A small, well-tested Black–Scholes–Merton option pricer with analytic Greeks and
continuous dividend-yield support.

## Features

- European call/put pricing via the Black–Scholes–Merton formula
- Continuous dividend yield `q` (set `q=0` for the classic no-dividend model)
- Analytic Greeks: delta, gamma, vega, theta, rho (plus an `all_greeks` helper)
- Input validation with clear error messages
- Finite-difference-verified test suite

## Requirements

- Python ≥ 3.12
- numpy, scipy, pandas (managed with [uv](https://docs.astral.sh/uv/))

```bash
uv sync
```

## Usage

```python
import sys; sys.path.insert(0, "src")   # or install the package
from pricer import bsm_pricer
from greeks import delta, gamma, vega, theta, rho, all_greeks

# Price a 1-year ATM call: spot=100, strike=100, r=5%, vol=20%
price = bsm_pricer(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
# -> 10.4506

# With a 3% continuous dividend yield
price = bsm_pricer(100, 100, 1, 0.05, 0.2, option_type="call", q=0.03)

# Greeks
g = all_greeks(100, 100, 1, 0.05, 0.2, option_type="call")
# {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ..., 'rho': ...}
```

You can also run either module directly for a demo:

```bash
uv run python src/pricer.py
uv run python src/greeks.py
```

## Parameters

| Name          | Meaning                                   | Constraint            |
| ------------- | ----------------------------------------- | --------------------- |
| `S`           | spot price of the underlying              | `> 0`                 |
| `K`           | strike price                              | `> 0`                 |
| `T`           | time to expiry, in years                  | `>= 0`                |
| `r`           | annualized risk-free rate (continuous)    | any (may be negative) |
| `sigma`       | annualized volatility                     | `> 0`                 |
| `option_type` | `"call"` or `"put"`                       | —                     |
| `q`           | annualized continuous dividend yield      | any (default `0.0`)   |

Invalid inputs raise `ValueError`. At `T = 0` the price returns the option's
intrinsic value, `max(S - K, 0)` for a call and `max(K - S, 0)` for a put.

## The math

With continuous dividend yield `q`, `N` the standard normal CDF and `n` its PDF:

```
d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Call = S * e^(-qT) * N(d1)  - K * e^(-rT) * N(d2)
Put  = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
```

## Greeks and scaling conventions

The Greeks return values in the following units (documented in each function's
docstring). This matters when interpreting the numbers:

| Greek   | Sensitivity to | Returned unit                          |
| ------- | -------------- | -------------------------------------- |
| `delta` | spot           | per `1.0` change in spot (raw)         |
| `gamma` | spot (2nd)     | per `1.0` change in spot (raw)         |
| `vega`  | volatility     | per **+1%** (`0.01`) change in `sigma` |
| `theta` | time           | per **calendar day** of decay          |
| `rho`   | rate           | per **+1%** (`0.01`) change in `r`      |

`gamma` and `vega` are identical for calls and puts.

## Tests

```bash
uv run pytest tests/ -q
```

The suite checks textbook reference values, put–call parity (with and without
dividends), expiry/intrinsic behavior, input validation, and verifies every Greek
against a central finite-difference bump of the pricer.

## Project layout

```
src/
  pricer.py   # bsm_pricer, calc_factors, validate_inputs
  greeks.py   # delta, gamma, vega, theta, rho, all_greeks
tests/
  test_pricer.py
```
