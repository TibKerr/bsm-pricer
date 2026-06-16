"""Command-line interface for the bsm_pricer package."""

import argparse

from .greeks import all_greeks
from .pricer import bsm_pricer


def main(argv: list[str] | None = None) -> None:
    """Price an option and print its value and Greeks from the command line."""
    parser = argparse.ArgumentParser(
        prog="bsm-pricer",
        description="Price a European option with Black-Scholes-Merton.",
    )
    parser.add_argument("-S", "--spot", type=float, required=True, help="spot price")
    parser.add_argument("-K", "--strike", type=float, required=True, help="strike price")
    parser.add_argument("-T", "--expiry", type=float, required=True, help="time to expiry (years)")
    parser.add_argument("-r", "--rate", type=float, required=True, help="risk-free rate")
    parser.add_argument("-v", "--vol", type=float, required=True, help="volatility (sigma)")
    parser.add_argument("-q", "--dividend", type=float, default=0.0, help="dividend yield")
    parser.add_argument(
        "-t", "--type", choices=("call", "put"), default="call", help="option type"
    )
    parser.add_argument("--greeks", action="store_true", help="also print the Greeks")
    args = parser.parse_args(argv)

    price = bsm_pricer(
        args.spot, args.strike, args.expiry, args.rate, args.vol, args.type, args.dividend
    )
    print(f"{args.type} price: {price:.4f}")

    if args.greeks:
        greeks = all_greeks(
            args.spot, args.strike, args.expiry, args.rate, args.vol, args.type, args.dividend
        )
        for name, value in greeks.items():
            print(f"  {name:>5}: {value:+.6f}")


if __name__ == "__main__":
    main()
