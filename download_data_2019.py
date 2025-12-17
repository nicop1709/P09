"""Script pour télécharger les données BTC/USDC 2019-2025"""
import ccxt
import pandas as pd
from datetime import datetime, timezone

def to_ms(dt):
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_ohlcv_binance(pair, timeframe, start_year, end_year, limit=1000):
    ex = ccxt.binance({"enableRateLimit": True})
    all_rows = []
    since = to_ms(datetime(start_year, 1, 1))
    end_ms = to_ms(datetime(end_year, 12, 31, 23, 59))

    print(f"Téléchargement {pair} {timeframe} de {start_year} à {end_year}...")

    while since < end_ms:
        batch = ex.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + 1

        if len(all_rows) % 10000 == 0:
            print(f"  {len(all_rows)} lignes téléchargées...")

        if len(batch) < 10:
            break

    df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = fetch_ohlcv_binance("BTC/USDC", "1h", 2019, 2025)
    filename = "btc_usdc_1h_2019_2025.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ {len(df)} lignes sauvegardées dans {filename}")
