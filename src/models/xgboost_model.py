"""
Project Name: ML-Predictor
File Name: xgboost_model.py
Description: 
    This module defines the XGBoostStockModel class, which implements an XGBoost regression model for stock price prediction.
    It inherits from BaseStockModel and provides specific implementations for training and prediction using the XGBoost library.

Author: Albert Marín Blasco
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from .base_model import BaseStockModel

class XGBoostStockModel(BaseStockModel):
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective='reg:squarederror',
            **kwargs
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("Training XGBoost Model...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)