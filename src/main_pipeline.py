"""Report scaling and volume-concentration diagnostics for best-quote snapshots."""

import argparse
from pathlib import Path

import numpy as np

from .ingestion.snapshots import load_snapshots, snapshot_features
from .microstruct.core_math import (
    calculate_hurst_variogram,
    nadaraya_watson_smoother,
    shannon_entropy,
)


def run_microstructure_analysis(
    parquet_path: str | Path, bandwidth: float = 2.0, max_lag: int = 20
) -> dict[str, float | str]:
    """Smooth absolute midpoint returns, then estimate scaling of their logarithm.

    Bandwidth and lags use observations, not seconds. The inherited defaults are
    uncalibrated. Gaussian smoothing uses both past and future observations,
    so this is an offline diagnostic.
    """
    frame = load_snapshots(parquet_path)
    absolute_returns, volumes = snapshot_features(frame)
    print(f"Loaded {len(frame)} best-quote snapshots from {parquet_path}.")
    time_index = np.arange(len(absolute_returns), dtype=np.float64)
    smoothed = nadaraya_watson_smoother(time_index, absolute_returns, bandwidth)
    hurst = float(calculate_hurst_variogram(np.log(smoothed), max_lag))
    entropy = float(shannon_entropy(volumes))

    # Preserve the previous numeric bands as descriptive, untested heuristics.
    if 0.01 <= hurst <= 0.30:
        label = "low-H band (roughness heuristic)"
    elif 0.40 <= hurst <= 0.60:
        label = "near-0.5 band"
    else:
        label = "outside the configured heuristic bands"

    print(f"Hurst scaling estimate: {hurst:.5f} (first absolute moment)")
    print(f"Diagnostic label: {label}")
    print(f"Best-quote volume entropy across snapshots: {entropy:.5f} nats")
    print("These diagnostics do not test a volatility model or establish tail-risk behavior.")
    return {"hurst": hurst, "entropy_nats": entropy, "diagnostic_label": label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/lob_snapshot.parquet"))
    parser.add_argument("--bandwidth", type=float, default=2.0)
    parser.add_argument("--max-lag", type=int, default=20)
    args = parser.parse_args()
    run_microstructure_analysis(args.input, args.bandwidth, args.max_lag)


if __name__ == "__main__":
    main()
