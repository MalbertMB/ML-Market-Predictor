"""
Project Name: ML-Predictor
File Name: data_loader.py
Description: 
    This module defines the AlpacaDataLoader class, which is responsible for fetching historical stock data from the Alpaca API.
    It includes functionality to cache the raw data locally to avoid redundant API calls and to ensure that the data is properly
    formatted for subsequent processing steps in the ML-Predictor project.
    
Author: Albert Marín Blasco
"""


import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

class AlpacaDataLoader:
    """Handles fetching historical stock data from the Alpaca API with local caching."""
    
    def __init__(self, raw_data_dir="data/raw"):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.raw_data_dir = raw_data_dir
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
    def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches historical stock data for a given symbol and date range. It first checks if the data is cached locally;
        if not, it fetches from the Alpaca API and saves it for future use.
        """
        filename = f"{symbol}_{start_date}_to_{end_date}.csv"
        file_path = os.path.join(self.raw_data_dir, filename)
        
        if use_cache and os.path.exists(file_path):
            print(f"Loading cached raw data for {symbol} from {file_path}...")
            return pd.read_csv(file_path, parse_dates=['timestamp'])
            
        print(f"Data not found locally. Fetching {symbol} from Alpaca API...")
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )
        
        bars = self.client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df = df[df['symbol'] == symbol].copy()
        
        print(f"Saving raw data to {file_path}...")
        df.to_csv(file_path, index=False)
        
        print(f"Data fetched successfully. Total records: {len(df)}")
        return df