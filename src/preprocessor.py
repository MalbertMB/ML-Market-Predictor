"""
Project Name: ML-Predictor
File Name: preprocessor.py
Description: 
    This module defines the DataPreprocessor class, which is responsible for feature engineering, dataset splitting, and saving the processed data for the ML-Predictor project.
    It includes methods to create technical indicators, handle missing values, and prepare the dataset for model training and evaluation.
    
Author: Albert Marín Blasco
"""


import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles feature engineering, dataset splitting, and saving processed data."""
    
    def __init__(self, test_size=0.2, shuffle=False, processed_data_dir="data/processed"):
        self.test_size = test_size
        self.shuffle = shuffle
        self.features = ["close", "volume", "return", "volatility", "ma_5", "ma_10", "ma_20"]
        self.processed_data_dir = processed_data_dir
        
        # Ensure the processed data directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
    def engineer_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        print("Engineering features...")
        df = df.copy()
        
        df["return"] = df["close"].pct_change()
        df["volatility"] = df["return"].rolling(window=5).std()
        df["ma_5"] = df["close"].rolling(window=5).mean()
        df["ma_10"] = df["close"].rolling(window=10).mean()
        df["ma_20"] = df["close"].rolling(window=20).mean()
        df["target"] = df["close"].shift(-1)
        df = df.dropna()
        
        # Save the finalized, clean dataset
        file_path = os.path.join(self.processed_data_dir, f"{symbol}_processed.csv")
        print(f"Saving processed feature set to {file_path}...")
        df.to_csv(file_path, index=False)
        
        return df
        
    def split_data(self, df: pd.DataFrame):
        print(f"Splitting data into train and test sets (Test size: {self.test_size * 100}%)...")
        X = df[self.features]
        y = df["target"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=self.shuffle
        )
        return X_train, X_test, y_train, y_test, X