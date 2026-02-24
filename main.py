"""
Project Name: ML-Predictor
File Name: main.py
Description: 
    This is the main entry point for the ML-Predictor project, which implements a complete machine learning pipeline for stock price prediction and backtesting.
    The script orchestrates data loading, preprocessing, model training, evaluation, backtesting, and visualization.
    
Author: Albert Marín Blasco
"""


import pandas as pd

from src import BaseStockModel, AlpacaDataLoader, DataPreprocessor, ResultPlotter, SimpleBacktester, RandomForestStockModel, XGBoostStockModel
from src import calculate_metrics, print_metrics


def main():
    # 1. Configuration
    SYMBOL = "NVDA"
    START_DATE = "2022-01-01"
    END_DATE = "2026-01-01"
    INITIAL_CAPITAL = 10000.0
    
    # 2. Initialize Pipeline Components
    data_loader = AlpacaDataLoader()
    preprocessor = DataPreprocessor(test_size=0.2, shuffle=False)
    backtester = SimpleBacktester(initial_capital=INITIAL_CAPITAL)
    plotter = ResultPlotter()
    
    # 3. Fetch and Preprocess Data
    raw_df = data_loader.fetch_historical_data(SYMBOL, START_DATE, END_DATE)
    processed_df = preprocessor.engineer_features(raw_df, SYMBOL)
    
    X_train, X_test, y_train, y_test, full_X = preprocessor.split_data(processed_df)
    
    # 4. Define Models to Evaluate
    models_to_test: dict[str, BaseStockModel] = {
        "Random Forest": RandomForestStockModel(n_estimators=300, max_depth=10),
        "XGBoost": XGBoostStockModel(n_estimators=300, learning_rate=0.05),
    }
    
    # 5. Train, Evaluate, Backtest, and Plot Each Model
    for model_name, model in models_to_test.items():
        print(f"\n========== Evaluating {model_name} ==========")
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, predictions)
        print_metrics(metrics, model_name)
        
        current_prices = X_test['close']
        backtest_results = backtester.simulate_trading(current_prices, predictions, y_test)
        
        plotter.plot_results(y_test, predictions, backtest_results, SYMBOL, model_name)
        
        latest_data = full_X.iloc[-1:]
        tomorrow_prediction = model.predict(latest_data)[0]
        print(f"--> Predicted next closing price for {SYMBOL} using {model_name}: ${tomorrow_prediction:.2f}")

if __name__ == "__main__":
    main()