import asyncio
import websockets
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import os

async def stream_binance_lob(symbol: str, duration_seconds: int = 60):
    # Aseguramos que el directorio 'data' exista antes de guardar
    os.makedirs('data', exist_ok=True)
    
    uri = f"wss://stream.binance.com:9443/ws/{symbol}@depth20@100ms"
    records = []
    start_time = asyncio.get_event_loop().time()
    
    print(f"Iniciando conexión WebSocket con Binance para {symbol.upper()}...")
    
    async with websockets.connect(uri) as websocket:
        print(f"Conexión establecida. Capturando LOB durante {duration_seconds} segundos...")
        while (asyncio.get_event_loop().time() - start_time) < duration_seconds:
            response = await websocket.recv()
            data = json.loads(response)
            
            # API de Binance para @depth20 usa 'bids' y 'asks'
            best_bid_price, best_bid_qty = float(data['bids'][0][0]), float(data['bids'][0][1])
            best_ask_price, best_ask_qty = float(data['asks'][0][0]), float(data['asks'][0][1])
            
            records.append({
                "timestamp": datetime.utcnow(),
                "bid_price": best_bid_price, "bid_vol": best_bid_qty,
                "ask_price": best_ask_price, "ask_vol": best_ask_qty
            })
            
    print(f"Captura finalizada. Procesando {len(records)} snapshots de alta frecuencia...")
    
    # Guardado ultra-rápido en formato columnar (Parquet)
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, 'data/lob_snapshot.parquet', compression='snappy')
    print("Datos compilados exitosamente en 'data/lob_snapshot.parquet'.")

if __name__ == "__main__":
    asyncio.run(stream_binance_lob("btcusdt", 60))