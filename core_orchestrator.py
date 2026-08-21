import asyncio
import aiohttp
import numpy as np
import numpy.typing as npt
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any, Tuple

# Type aliases for strict mypy static analysis
TensorView = npt.NDArray[np.float64]
DispatcherCallback = Callable[[str, TensorView, TensorView], None]

@dataclass(slots=True)
class LOBEvent:
    """
    Data structure representing a snapshot of the Limit Order Book (Level-2).
    Tracks the best bid (P_B) and best ask (P_A) prices and sizes[cite: 1].
    """
    timestamp: float
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

@dataclass(slots=True)
class TradeEvent:
    """
    Data structure representing an executed transaction (liquidity taking)[cite: 1].
    """
    timestamp: float
    symbol: str
    price: float
    volume: float
    direction: int  # 1 for Buy (Ask-lift), -1 for Sell (Bid-hit)

class MarketDataEngine:
    """
    Core Orchestrator for asynchronous, low-latency ingestion of market data.
    Maintains pre-allocated continuous memory tensors for rolling windows of N events.
    """
    
    def __init__(self, symbols: List[str], window_size: int = 10_000) -> None:
        self.symbols: List[str] = symbols
        self.window_size: int = window_size
        
        # In-memory queues for raw event orchestration before tensor vectorization
        self._lob_queue: Dict[str, deque[LOBEvent]] = {
            sym: deque(maxlen=self.window_size) for sym in symbols
        }
        self._trade_queue: Dict[str, deque[TradeEvent]] = {
            sym: deque(maxlen=self.window_size) for sym in symbols
        }
        
        # Pre-allocated uninitialized multi-dimensional arrays for zero-copy math[cite: 5]
        # LOB Features: [timestamp, bid_price, bid_size, ask_price, ask_size]
        self._lob_tensors: Dict[str, TensorView] = {
            sym: np.empty((self.window_size, 5), dtype=np.float64) for sym in symbols
        }
        # Trade Features: [timestamp, price, volume, direction]
        self._trade_tensors: Dict[str, TensorView] = {
            sym: np.empty((self.window_size, 4), dtype=np.float64) for sym in symbols
        }
        
        # Circular buffer indices
        self._lob_indices: Dict[str, int] = {sym: 0 for sym in symbols}
        self._trade_indices: Dict[str, int] = {sym: 0 for sym in symbols}
        
        # Dispatcher callbacks
        self._dispatchers: List[DispatcherCallback] = []

    def register_dispatcher(self, callback: DispatcherCallback) -> None:
        """Registers a non-blocking callback to pipe tensors to math/quant modules."""
        self._dispatchers.append(callback)

    def _insert_lob_vector(self, event: LOBEvent) -> None:
        """Vectorizes a single LOB event into the pre-allocated circular tensor."""
        idx: int = self._lob_indices[event.symbol] % self.window_size
        self._lob_tensors[event.symbol][idx] = (
            event.timestamp, event.bid_price, event.bid_size, event.ask_price, event.ask_size
        )
        self._lob_indices[event.symbol] += 1

    def _insert_trade_vector(self, event: TradeEvent) -> None:
        """Vectorizes a single Trade event into the pre-allocated circular tensor."""
        idx: int = self._trade_indices[event.symbol] % self.window_size
        self._trade_tensors[event.symbol][idx] = (
            event.timestamp, event.price, event.volume, float(event.direction)
        )
        self._trade_indices[event.symbol] += 1

    async def _dispatch_to_quant_modules(self, symbol: str) -> None:
        """
        Asynchronously triggers registered callbacks, passing memory views of the tensors
        to prevent blocking the main asyncio Event Loop during heavy mathematical processing.
        """
        # Create zero-copy views to prevent accidental mutation by downstream models
        lob_view: TensorView = self._lob_tensors[symbol].view()
        trade_view: TensorView = self._trade_tensors[symbol].view()
        lob_view.flags.writeable = False
        trade_view.flags.writeable = False

        for callback in self._dispatchers:
            # Dispatch to thread pool or directly invoke non-blocking JIT-compiled functions
            asyncio.get_running_loop().call_soon(callback, symbol, lob_view, trade_view)

    async def fetch_lob_stream(self, session: aiohttp.ClientSession, symbol: str, endpoint: str) -> None:
        """
        Simulates an asynchronous WebSocket or HTTP streaming connection to ingest L2 updates.
        """
        try:
            # Mocking a continuous async stream 
            while True:
                async with session.get(f"{endpoint}/lob/{symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        event = LOBEvent(
                            timestamp=data['ts'],
                            symbol=symbol,
                            bid_price=data['bp'],
                            bid_size=data['bs'],
                            ask_price=data['ap'],
                            ask_size=data['as']
                        )
                        self._lob_queue[symbol].append(event)
                        self._insert_lob_vector(event)
                        
                        # Trigger async dispatch matrix
                        await self._dispatch_to_quant_modules(symbol)
                    
                await asyncio.sleep(0.001)  # Simulate 1ms tick resolution network I/O
        except asyncio.CancelledError:
            print(f"LOB ingestion stream for {symbol} terminated.")

    async def fetch_trade_stream(self, session: aiohttp.ClientSession, symbol: str, endpoint: str) -> None:
        """
        Simulates an asynchronous connection to ingest executed transactions.
        """
        try:
            while True:
                async with session.get(f"{endpoint}/trades/{symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        event = TradeEvent(
                            timestamp=data['ts'],
                            symbol=symbol,
                            price=data['p'],
                            volume=data['v'],
                            direction=data['dir']
                        )
                        self._trade_queue[symbol].append(event)
                        self._insert_trade_vector(event)
                        
                        await self._dispatch_to_quant_modules(symbol)

                await asyncio.sleep(0.005) # Trades are structurally less frequent than quotes
        except asyncio.CancelledError:
            print(f"Trade ingestion stream for {symbol} terminated.")

    async def run_orchestrator(self, mock_endpoint: str) -> None:
        """
        Main entry point for the asyncio event loop.
        Initializes aiohttp session and spawns concurrent tasks.
        """
        async with aiohttp.ClientSession() as session:
            tasks: List[asyncio.Task[Any]] = []
            
            for symbol in self.symbols:
                tasks.append(asyncio.create_task(self.fetch_lob_stream(session, symbol, mock_endpoint)))
                tasks.append(asyncio.create_task(self.fetch_trade_stream(session, symbol, mock_endpoint)))
            
            # Run all ingestion streams concurrently
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Example execution configuration
    target_symbols = ["AAPL", "ES_F"]
    
    # Initialize Core Engine with N=10,000 constraint
    engine = MarketDataEngine(symbols=target_symbols, window_size=10_000)
    
    # Example quantitative callback (e.g., passing data to a Numba JIT-compiled C-struct)
    def dummy_quant_model(symbol: str, lob_matrix: TensorView, trade_matrix: TensorView) -> None:
        pass # To be replaced by C++/Numba FFI bindings
        
    engine.register_dispatcher(dummy_quant_model)
    
    try:
        # Python 3.7+ Asyncio execution
        asyncio.run(engine.run_orchestrator("http://mock-market-data-api.local"))
    except KeyboardInterrupt:
        print("Orchestrator gracefully shutting down.")