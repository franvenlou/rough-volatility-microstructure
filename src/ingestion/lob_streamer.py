"""Collect best quotes from Binance's partial-depth stream into one Parquet file.

The subscription requests 20 levels at 100 ms updates; only the first bid and ask
are retained. Records are buffered in memory and written after capture. This
prototype has no reconnect, sequence-gap recovery, or exchange timestamp audit.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

from .snapshots import validate_snapshots


async def stream_binance_lob(
    symbol: str, duration_seconds: float = 60,
    output_path: str | Path = "data/lob_snapshot.parquet",
) -> Path:
    """Capture for a bounded interval, timestamping local UTC receipt time.

    Connection setup has its own timeout; capture starts after connecting.
    An idle stream cannot block indefinitely in recv().
    """
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive.")
    if not symbol or not symbol.isascii() or not symbol.isalnum():
        raise ValueError("symbol must contain ASCII letters and digits only.")
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth20@100ms"
    records = []
    loop = asyncio.get_running_loop()
    print(f"Connecting to the {symbol.upper()} partial-depth stream.")
    async with websockets.connect(uri, open_timeout=10) as websocket:
        deadline = loop.time() + duration_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            data = json.loads(response)
            if not data.get("bids") or not data.get("asks"):
                continue
            bid_price, bid_volume = map(float, data["bids"][0])
            ask_price, ask_volume = map(float, data["asks"][0])
            records.append({
                "timestamp": datetime.now(timezone.utc),
                "bid_price": bid_price, "bid_vol": bid_volume,
                "ask_price": ask_price, "ask_vol": ask_volume,
            })
    if len(records) < 2:
        raise ValueError("Captured fewer than two usable quotes; no file was written.")
    frame = validate_snapshots(pd.DataFrame(records))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), output, compression="snappy")
    print(f"Wrote {len(frame)} best-quote snapshots to {output}.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("data/lob_snapshot.parquet"))
    args = parser.parse_args()
    asyncio.run(stream_binance_lob(args.symbol, args.duration, args.output))


if __name__ == "__main__":
    main()
