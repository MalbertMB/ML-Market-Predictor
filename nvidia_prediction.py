import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Tells Matplotlib to use the window-friendly backend
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY and API_SECRET must be set in environment variables.")

SYMBOL = "NVDA"
START_DATE = "2022-01-01"
END_DATE = "2026-01-01"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

request_params = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
)

bars = client.get_stock_bars(request_params)
df = bars.df.reset_index()

df = df[df['symbol'] == SYMBOL]

# FEATURES
df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(window=5).std()
df["ma_5"] = df["close"].rolling(window=5).mean()
df["ma_10"] = df["close"].rolling(window=10).mean()
df["ma_20"] = df["close"].rolling(window=20).mean()

df["target"] = df["close"].shift(-1)
df = df.dropna()

FEATURES = ["close", "volume", "return", "volatility", "ma_5", "ma_10", "ma_20"]

X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae}")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.title(f"{SYMBOL} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# Predict the next day's closing price
latest_data = X.iloc[-1:] 
prediction = model.predict(latest_data)[0]
print(f"Predicted next closing price for {SYMBOL}: {prediction}")
