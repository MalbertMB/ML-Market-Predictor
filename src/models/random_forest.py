"""
Project Name: ML-Predictor
File Name: random_forest.py
Description: 
    This module defines the RandomForestStockModel class, which implements a Random Forest regression model for stock price prediction.
    It inherits from BaseStockModel and provides specific implementations for training and prediction using scikit-learn's RandomForestRegressor.
    
Author: Albert Marín Blasco
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseStockModel

class RandomForestStockModel(BaseStockModel):
    def __init__(self, n_estimators=300, max_depth=10, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_test)