"""Offline schema, buffer, stream-adapter and pipeline regression checks."""

import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ingestion.lob_streamer import stream_binance_lob
from src.ingestion.polling_engine import LOBEvent, MarketDataEngine, TradeEvent
from src.ingestion.snapshots import load_snapshots, snapshot_features, validate_snapshots
from src.ingestion.synthetic_snapshot import generate_synthetic_snapshots
from src.main_pipeline import run_microstructure_analysis


class SnapshotTests(unittest.TestCase):
    def test_midpoint_returns_and_volume_alignment(self):
        frame = pd.DataFrame({
            "bid_price": [9.0, 19.0, 19.0], "ask_price": [11.0, 21.0, 21.0],
            "bid_vol": [1.0, 2.0, 3.0], "ask_vol": [4.0, 5.0, 6.0],
        })
        returns, volumes = snapshot_features(frame)
        np.testing.assert_allclose(returns, [np.log(2), 1e-8])
        np.testing.assert_array_equal(volumes, [5, 7, 9])

    def test_invalid_quotes_are_rejected(self):
        good = generate_synthetic_snapshots(10)
        for column, value in (("bid_price", -1), ("ask_vol", -1),
                              ("bid_price", np.nan), ("bid_price", 1000)):
            bad = good.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_snapshots(bad)
        with self.assertRaisesRegex(ValueError, "Missing quote"):
            validate_snapshots(good.drop(columns="ask_price"))
        with self.assertRaisesRegex(ValueError, "At least two"):
            validate_snapshots(good.iloc[:1])

    def test_parquet_pipeline_reports_diagnostic_with_known_uniform_entropy(self):
        frame = generate_synthetic_snapshots(150)
        frame["bid_vol"] = 1.0
        frame["ask_vol"] = 1.0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.parquet"
            frame.to_parquet(path, index=False)
            pd.testing.assert_frame_equal(load_snapshots(path), frame)
            output = StringIO()
            with redirect_stdout(output):
                result = run_microstructure_analysis(path)
        self.assertTrue(np.isfinite(result["hurst"]))
        self.assertAlmostEqual(result["entropy_nats"], np.log(150))
        self.assertIn("Diagnostic label:", output.getvalue())
        self.assertIn("do not test a volatility model", output.getvalue())

    def test_constant_price_does_not_produce_a_fabricated_hurst_value(self):
        frame = generate_synthetic_snapshots(60)
        frame["bid_price"] = 99.0
        frame["ask_price"] = 101.0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "constant.parquet"
            frame.to_parquet(path, index=False)
            with self.assertRaises(ValueError), redirect_stdout(StringIO()):
                run_microstructure_analysis(path)


class BufferTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_and_wrapped_buffers_are_chronological_independent_copies(self):
        engine = MarketDataEngine(["TEST"], window_size=3)
        seen = []
        engine.register_dispatcher(lambda *args: seen.append(args))
        engine._insert_lob_vector(LOBEvent(1, "TEST", 9, 1, 11, 2))
        await engine._dispatch_to_quant_modules("TEST")
        await asyncio.sleep(0)
        self.assertEqual(seen[0][1].shape, (1, 5))
        self.assertEqual(seen[0][2].shape, (0, 4))
        for timestamp in range(2, 6):
            engine._insert_lob_vector(LOBEvent(timestamp, "TEST", 9, 1, 11, 2))
        engine._insert_trade_vector(TradeEvent(4, "TEST", 10, 3, -1))
        await engine._dispatch_to_quant_modules("TEST")
        await asyncio.sleep(0)
        np.testing.assert_array_equal(seen[-1][1][:, 0], [3, 4, 5])
        np.testing.assert_array_equal(seen[0][1][:, 0], [1])
        np.testing.assert_array_equal(seen[-1][2][0], [4, 10, 3, -1])
        self.assertFalse(seen[-1][1].flags.writeable)

    async def test_invalid_buffer_configuration(self):
        for symbols, window in (([], 3), (["X", "X"], 3), (["X"], 0)):
            with self.subTest(symbols=symbols, window=window), self.assertRaises(ValueError):
                MarketDataEngine(symbols, window)


class FakeConnection:
    """Two partial-depth messages followed by an idle stream, with no network."""

    def __init__(self, messages):
        self.messages = iter(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def recv(self):
        try:
            return json.dumps(next(self.messages))
        except StopIteration:
            await asyncio.Future()


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_retains_only_best_quotes_and_times_out_when_idle(self):
        messages = [
            {"bids": [["99", "2"], ["98", "9"]], "asks": [["101", "3"]]},
            {"bids": [["100", "4"]], "asks": [["102", "5"]]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "quotes.parquet"
            with patch("src.ingestion.lob_streamer.websockets.connect", return_value=FakeConnection(messages)) as connect:
                with redirect_stdout(StringIO()):
                    result = await asyncio.wait_for(stream_binance_lob("BTCUSDT", 0.03, path), 2)
            connect.assert_called_once_with(
                "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms", open_timeout=10
            )
            self.assertEqual(result, path)
            frame = load_snapshots(path)
        np.testing.assert_array_equal(frame["bid_price"], [99, 100])
        np.testing.assert_array_equal(frame["bid_vol"], [2, 4])
        self.assertIsNotNone(frame["timestamp"].dt.tz)

    async def test_empty_capture_does_not_write_a_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.parquet"
            with patch("src.ingestion.lob_streamer.websockets.connect", return_value=FakeConnection([])):
                with self.assertRaisesRegex(ValueError, "fewer than two"), redirect_stdout(StringIO()):
                    await stream_binance_lob("btcusdt", 0.01, path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
