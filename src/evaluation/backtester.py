"""
Project Name: ML-Predictor
File Name: backtester.py
Description:
    This module defines the SimpleBacktester class, which simulates a basic trading strategy based on model predictions.
    It evaluates the strategy's performance against a buy-and-hold approach and calculates metrics like final portfolio value and ROI.

Author: Albert Marín Blasco
"""


import pandas as pd
import numpy as np

class SimpleBacktester:
    """Simulates trading based on model predictions without executing real orders."""
    
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital

    def simulate_trading(self, current_prices: pd.Series, predicted_prices: np.ndarray, actual_future_prices: pd.Series) -> pd.DataFrame:
        """
        Runs a simulation where we buy the stock if the model predicts the price will go up,
        and sell/hold cash if the model predicts the price will go down.
        """
        print(f"\nRunning trading simulation with ${self.initial_capital}...")
        
        # Create a DataFrame to hold our simulation data
        df = pd.DataFrame({
            'current_price': current_prices.values,
            'predicted_price': predicted_prices,
            'actual_future_price': actual_future_prices.values
        }, index=current_prices.index)
        
        # Trading Signal: 1 (Buy/Hold) if predicted price > current price, else 0 (Sell/Hold Cash)
        df['signal'] = np.where(df['predicted_price'] > df['current_price'], 1, 0)
        
        # Calculate the actual percentage return of the stock for that day
        df['actual_return'] = (df['actual_future_price'] - df['current_price']) / df['current_price']
        
        # Strategy return: we only capture the return if our signal was 1
        df['strategy_return'] = df['signal'] * df['actual_return']
        
        # Calculate cumulative portfolio values over time
        df['portfolio_value'] = self.initial_capital * (1 + df['strategy_return']).cumprod()
        df['buy_and_hold_value'] = self.initial_capital * (1 + df['actual_return']).cumprod()
        
        # Extract final metrics
        final_portfolio_value = df['portfolio_value'].iloc[-1]
        buy_and_hold_value = df['buy_and_hold_value'].iloc[-1]
        strategy_roi = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"Final Strategy Value: ${final_portfolio_value:.2f} (ROI: {strategy_roi:.2f}%)")
        print(f"Buy & Hold Value:     ${buy_and_hold_value:.2f}")
        
        return df