import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# ==== User config ====
symbol = "BTCUSDT"
interval = "1h"   # 1 hour bars
limit = 1000      # Max per API call
output_csv = "BTCUSDT_1h_ohlcv_2y_prior.csv"
# Set these to your desired range
end_time_str = "2023-06-27 08:00:00"  # Earliest timestamp in your current file
years = 2
# ======================

end_dt = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
start_dt = end_dt - timedelta(days=years*365)

start_time = int(start_dt.timestamp() * 1000)
end_time = int(end_dt.timestamp() * 1000)

url = "https://api.binance.com/api/v3/klines"
all_data = []

print(f"Downloading {symbol} {interval} data from {start_dt} to {end_dt}...")

while start_time < end_time:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(url, params=params)
    data = response.json()

    if not data or "code" in data:
        print("No data or API error:", data)
        break

    all_data.extend(data)
    last_time = data[-1][0]
    start_time = last_time + 1

    print(f"Downloaded {len(all_data)} records so far...")

    # Sleep to avoid hitting Binance rate limits
    time.sleep(0.25)

# Format to your target columns
df = pd.DataFrame(all_data, columns=[
    "Open Time", "Open", "High", "Low", "Close", "Volume",
    "Close Time", "Quote Asset Volume", "Number of Trades",
    "Taker Buy Base", "Taker Buy Quote", "Ignore"
])

df["Timestamp"] = pd.to_datetime(df["Open Time"], unit='ms').dt.strftime("%Y-%m-%d %H:%M:%S")
df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

out_df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
out_df = out_df.sort_values("Timestamp").reset_index(drop=True)
out_df.to_csv(output_csv, index=False)

print(f"Saved {len(out_df)} rows to {output_csv}")
print(out_df.head(2))
print(out_df.tail(2))
