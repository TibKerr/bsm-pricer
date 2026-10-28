# bsm-pricer

A small, well-tested **Black–Scholes–Merton** option pricer with analytic Greeks and
continuous dividend-yield support, packaged as an installable Python library with a CLI.

## Features

- European call/put pricing via the Black–Scholes–Merton formula
- Continuous dividend yield `q` (set `q=0` for the classic no-dividend model)
- Analytic Greeks: delta, gamma, vega, theta, rho (plus an `all_greeks` helper)
- Input validation with clear error messages
- Finite-difference-verified test suite (every Greek checked against a bump of the pricer)
- Linted (ruff), type-checked (mypy), and run in CI (GitHub Actions)

## Requirements

- Python ≥ 3.12
- numpy, scipy — managed with [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv sync          # creates the venv and installs the package + dev tools
```

This installs `bsm_pricer` into the environment, so it imports cleanly with no path hacks.

## Usage

### Python

```python
from bsm_pricer import bsm_pricer, all_greeks

# Price a 1-year ATM call: spot=100, strike=100, r=5%, vol=20%
price = bsm_pricer(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
# -> 10.4506

# With a 3% continuous dividend yield
price = bsm_pricer(100, 100, 1, 0.05, 0.2, option_type="call", q=0.03)

# Greeks (single dict)
g = all_greeks(100, 100, 1, 0.05, 0.2, option_type="call")
# {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ..., 'rho': ...}
```

### Arrays

`bsm_pricer.array` has the same functions, but every input can be an array, so
thousands of options are priced in one call. Inputs broadcast together, and
`is_call` takes a bool (or bool array) instead of `"call"` / `"put"`.

```python
import numpy as np
from bsm_pricer import array

spots = np.array([95.0, 100.0, 105.0])
prices = array.bsm_pricer(spots, 100, 1, 0.05, 0.2, is_call=[True, False, True])
g = array.all_greeks(spots, 100, 1, 0.05, 0.2)  # dict of arrays
```

### Command line

```bash
uv run bsm-pricer -S 100 -K 100 -T 1 -r 0.05 -v 0.2 --type call --greeks
```

```
call price: 10.4506
  delta: +0.636831
  gamma: +0.018762
   vega: +0.375240
  theta: -0.017573
    rho: +0.532325
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

The Greeks return values in the following units (also documented in each function's
docstring). This matters when interpreting the numbers:

| Greek   | Sensitivity to | Returned unit                          |
| ------- | -------------- | -------------------------------------- |
| `delta` | spot           | per `1.0` change in spot (raw)         |
| `gamma` | spot (2nd)     | per `1.0` change in spot (raw)         |
| `vega`  | volatility     | per **+1%** (`0.01`) change in `sigma` |
| `theta` | time           | per **calendar day** of decay          |
| `rho`   | rate           | per **+1%** (`0.01`) change in `r`      |

`gamma` and `vega` are identical for calls and puts.

## Development

```bash
uv run ruff check .     # lint
uv run mypy             # type check
uv run pytest -q        # tests
```

The test suite checks textbook reference values, put–call parity (with and without
dividends), expiry/intrinsic behavior, input validation, and verifies every Greek
against a central finite-difference bump of the pricer.

## Project layout

```
src/bsm_pricer/
  __init__.py   # public API
  pricer.py     # bsm_pricer, calc_factors, validate_inputs
  greeks.py     # delta, gamma, vega, theta, rho, all_greeks
  array.py      # vectorized bsm_pricer and calc_factors over numpy arrays
  cli.py        # bsm-pricer command-line entry point
tests/
  test_pricer.py
  test_greeks.py
  test_array.py
.github/workflows/ci.yml
```
