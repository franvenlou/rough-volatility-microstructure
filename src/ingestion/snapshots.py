"""Shared schema checks and features for ordered best-quote snapshots."""

from pathlib import Path

import numpy as np
import pandas as pd

QUOTE_COLUMNS = ("bid_price", "bid_vol", "ask_price", "ask_vol")
RETURN_FLOOR = 1e-8


def validate_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate quotes without reordering or resampling observations.

    Timestamps are optional. Estimators use observation index, so irregular
    arrival intervals are not accounted for.
    """
    missing = set(QUOTE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing quote columns: {', '.join(sorted(missing))}")
    if len(frame) < 2:
        raise ValueError("At least two quote snapshots are required.")
    values = frame.loc[:, list(QUOTE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Quote prices and volumes must be finite.")
    if np.any(values[:, [0, 2]] <= 0):
        raise ValueError("Quote prices must be positive.")
    if np.any(values[:, [1, 3]] < 0):
        raise ValueError("Quote volumes must be nonnegative.")
    if np.any(values[:, 0] > values[:, 2]):
        raise ValueError("Bid prices must not exceed ask prices.")
    return frame


def load_snapshots(path: str | Path) -> pd.DataFrame:
    """Read a Parquet capture with the four required best-quote columns."""
    return validate_snapshots(pd.read_parquet(path))


def snapshot_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute log returns (N-1) and total best-quote volumes (N).

    Exactly zero returns use the legacy 1e-8 floor before logarithms. This
    arbitrary floor can affect scaling estimates; it is not a noise model.
    Align volume with returns using volumes[1:] for rolling features.
    """
    validate_snapshots(frame)
    bid = frame["bid_price"].to_numpy(dtype=np.float64)
    ask = frame["ask_price"].to_numpy(dtype=np.float64)
    midpoint = bid / 2.0 + ask / 2.0
    absolute_returns = np.abs(np.diff(np.log(midpoint)))
    absolute_returns = np.where(absolute_returns == 0, RETURN_FLOOR, absolute_returns)
    volumes = (
        frame["bid_vol"].to_numpy(dtype=np.float64)
        + frame["ask_vol"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(volumes).all():
        raise ValueError("Total quote volumes must be finite.")
    return absolute_returns, volumes
