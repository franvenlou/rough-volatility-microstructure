"""Generate deterministic toy quotes for offline execution, not market validation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_snapshots(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Use Gaussian log returns and independent positive quote sizes.

    Spread, return scale, and volume distribution are illustrative defaults.
    This does not simulate an exchange or a rough-volatility process.
    """
    if rows < 2:
        raise ValueError("rows must be at least two.")
    rng = np.random.default_rng(seed)
    midpoint = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0002, rows)))
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="100ms", tz="UTC"),
        "bid_price": midpoint - 0.005,
        "bid_vol": rng.lognormal(1.0, 0.5, rows),
        "ask_price": midpoint + 0.005,
        "ask_vol": rng.lognormal(1.0, 0.5, rows),
        "source": "synthetic",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_lob.parquet"))
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = generate_synthetic_snapshots(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote {len(frame)} synthetic quotes to {args.output}.")


if __name__ == "__main__":
    main()
