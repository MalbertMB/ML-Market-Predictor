"""
Project Name: ML-Predictor
File Name: metrics.py
Description: 
    This module defines the calculate_metrics function, which computes simplified evaluation metrics for stock price predictions.
    It includes Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).

Author: Albert Marín Blasco
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculates simplified evaluation metrics for stock predictions."""
    
    # Absolute error: On average, how many dollars is the prediction off by?
    mae = mean_absolute_error(y_true, y_pred)
    
    # Percentage error: On average, how much does the prediction differ from the actual result?
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        "Mean Absolute Error (USD)": round(mae, 4),
        "Average Percentage Error": f"{round(mape * 100, 2)}%"
    }

def print_metrics(metrics_dict: dict, model_name: str):
    """Utility function to print metrics in a clean format."""
    print(f"\n--- {model_name} Evaluation Metrics ---")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")
    print("-" * 40)