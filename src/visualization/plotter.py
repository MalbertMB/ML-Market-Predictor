"""
Project Name: ML-Predictor
File Name: plotter.py
Description: 
    This module defines the ResultPlotter class, which is responsible for creating professional, interactive visualizations
    of the model's predictions and backtesting results using Plotly.

Author: Albert Marín Blasco
"""


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ResultPlotter:
    """Handles professional, interactive data visualization using Plotly."""
    
    @staticmethod
    def plot_results(y_test: pd.Series, predictions: np.ndarray, backtest_df: pd.DataFrame, symbol: str, model_name: str):
        print(f"Generating interactive dashboard for {model_name}...")
        
        # Create a 2-row subplot layout sharing the same X-axis
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1,
            subplot_titles=(f"<b>{symbol} Price Predictions</b>", "<b>Backtest: Portfolio Value (USD)</b>")
        )
        
        # --- ROW 1: Actual vs Predicted Prices ---
        fig.add_trace(go.Scatter(
            y=y_test.values, mode='lines', name='Actual Price',
            line=dict(color='#1f77b4', width=2) # Blue
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            y=predictions, mode='lines', name='Predicted Price',
            line=dict(color='#ff7f0e', width=2, dash='dash') # Orange dashed
        ), row=1, col=1)
        
        # --- ROW 2: Trading Simulation (Backtest) ---
        fig.add_trace(go.Scatter(
            y=backtest_df['portfolio_value'], mode='lines', name='Strategy Value',
            line=dict(color='#2ca02c', width=2) # Green
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            y=backtest_df['buy_and_hold_value'], mode='lines', name='Buy & Hold Value',
            line=dict(color='#d62728', width=2, dash='dot') # Red dotted
        ), row=2, col=1)
        
        # --- Layout Formatting ---
        fig.update_layout(
            title=f"Model Evaluation Dashboard: {model_name}",
            height=800, # Taller figure to accommodate both charts
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1
            )
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
        fig.update_xaxes(title_text="Trading Days (Test Set)", row=2, col=1)
        
        fig.show()