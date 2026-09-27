"""Exploratory HTTP polling and rolling buffers, separate from Binance capture.

This preserves the root orchestrator's custom JSON contract. No server or real
exchange adapter for that contract is provided, and this engine is not wired to
the Parquet pipeline. Poll intervals are scheduling defaults, not latency claims.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable

import aiohttp
import numpy as np

DispatcherCallback = Callable[[str, np.ndarray, np.ndarray], None]


@dataclass(slots=True)
class LOBEvent:
    """Best bid and ask, with timestamp units defined by the supplied endpoint."""

    timestamp: float
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float


@dataclass(slots=True)
class TradeEvent:
    """Executed trade; direction is +1 for an ask lift and -1 for a bid hit."""

    timestamp: float
    symbol: str
    price: float
    volume: float
    direction: int


class MarketDataEngine:
    """Poll a caller-supplied endpoint and dispatch independent chronological arrays.

    Callback arrays contain only observed rows. Callbacks run on the event-loop
    thread and must return promptly; this class does not offload numerical work.
    Quote and trade arrivals are independent and are not timestamp-aligned.
    """

    def __init__(self, symbols: list[str], window_size: int = 10_000) -> None:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if not symbols or any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be nonempty and unique.")
        self.symbols = list(symbols)
        self.window_size = window_size
        # Columns: timestamp, bid price, bid size, ask price, ask size.
        self._lob_tensors = {symbol: np.empty((window_size, 5)) for symbol in symbols}
        # Columns: timestamp, price, volume, direction.
        self._trade_tensors = {symbol: np.empty((window_size, 4)) for symbol in symbols}
        self._lob_indices = dict.fromkeys(symbols, 0)
        self._trade_indices = dict.fromkeys(symbols, 0)
        self._dispatchers: list[DispatcherCallback] = []

    def register_dispatcher(self, callback: DispatcherCallback) -> None:
        """Register a synchronous callback receiving symbol, quote rows, trade rows."""
        self._dispatchers.append(callback)

    def _insert_lob_vector(self, event: LOBEvent) -> None:
        index = self._lob_indices[event.symbol] % self.window_size
        self._lob_tensors[event.symbol][index] = (
            event.timestamp, event.bid_price, event.bid_size, event.ask_price, event.ask_size
        )
        self._lob_indices[event.symbol] += 1

    def _insert_trade_vector(self, event: TradeEvent) -> None:
        index = self._trade_indices[event.symbol] % self.window_size
        self._trade_tensors[event.symbol][index] = (
            event.timestamp, event.price, event.volume, event.direction
        )
        self._trade_indices[event.symbol] += 1

    @staticmethod
    def _snapshot(buffer: np.ndarray, count: int) -> np.ndarray:
        """Copy observed rows in arrival order so later writes cannot alter callbacks."""
        if count < len(buffer):
            snapshot = buffer[:count].copy()
        else:
            start = count % len(buffer)
            snapshot = np.concatenate((buffer[start:], buffer[:start]))
        snapshot.flags.writeable = False
        return snapshot

    async def _dispatch_to_quant_modules(self, symbol: str) -> None:
        quotes = self._snapshot(self._lob_tensors[symbol], self._lob_indices[symbol])
        trades = self._snapshot(self._trade_tensors[symbol], self._trade_indices[symbol])
        for callback in self._dispatchers:
            asyncio.get_running_loop().call_soon(callback, symbol, quotes, trades)

    async def fetch_lob_stream(
        self, session: aiohttp.ClientSession, symbol: str, endpoint: str
    ) -> None:
        """Poll /lob/<symbol> JSON with keys ts, bp, bs, ap, and as.

        HTTP failures propagate to the caller. No reconnect/backoff policy is
        implemented; task cancellation propagates normally.
        """
        while True:
            async with session.get(f"{endpoint.rstrip('/')}/lob/{symbol}") as response:
                response.raise_for_status()
                data = await response.json()
                self._insert_lob_vector(LOBEvent(
                    data["ts"], symbol, data["bp"], data["bs"], data["ap"], data["as"]
                ))
                await self._dispatch_to_quant_modules(symbol)
            await asyncio.sleep(0.001)

    async def fetch_trade_stream(
        self, session: aiohttp.ClientSession, symbol: str, endpoint: str
    ) -> None:
        """Poll /trades/<symbol> JSON with keys ts, p, v, and dir."""
        while True:
            async with session.get(f"{endpoint.rstrip('/')}/trades/{symbol}") as response:
                response.raise_for_status()
                data = await response.json()
                self._insert_trade_vector(TradeEvent(
                    data["ts"], symbol, data["p"], data["v"], data["dir"]
                ))
                await self._dispatch_to_quant_modules(symbol)
            await asyncio.sleep(0.005)

    async def run_orchestrator(self, endpoint: str) -> None:
        """Poll quotes and trades concurrently until cancelled or a request fails."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = [
                asyncio.create_task(fetch(session, symbol, endpoint))
                for symbol in self.symbols
                for fetch in (self.fetch_lob_stream, self.fetch_trade_stream)
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
