# Quantitative Stock Market Predictor & Backtesting Engine

## Overview
This project is a modular machine learning pipeline designed to forecast daily stock closing prices and simulate algorithmic trading strategies. By utilizing historical market data, the architecture evaluates predictive models based on absolute dollar error and percentage error, while simultaneously running a backtest to compare the model's trading Return on Investment (ROI) against a standard buy-and-hold strategy.

## Key Features
* **Automated Data Pipeline:** Fetches and caches historical daily bars via the Alpaca API to optimize speed and reduce API calls.
* **Feature Engineering:** Calculates rolling volatility, daily percentage returns, and standard moving averages (5-day, 10-day, 20-day).
* **Predictive Modeling:** Implements robust two-based machine learning algorithms (Random Forest and XGBoost) through a unified model interface.
* **Financial Metrics:** Evaluates accuracy using Mean Absolute Error (USD) and Average Percentage Error.
* **Strategy Backtesting:** Simulates portfolio growth based on model predictions to measure practical trading viability.
* **Interactive Visualization:** Generates stacked, interactive Plotly dashboards displaying price predictions alongside the simulated equity curve.

## Project Architecture

```text
stock_prediction_project/
├── data/                  
│   ├── raw/               # Cached API data pulls (.csv)
│   └── processed/         # Engineered feature sets ready for training
├── configs/               
│   └── config.py          # Environment variable and API key management
├── src/                   
│   ├── data_loader.py     # Alpaca API integration and local caching logic
│   ├── preprocessor.py    # Feature engineering and train/test splitting
│   ├── models/            
│   │   ├── base_model.py  # Abstract base class for model contracts
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   ├── evaluation/        
│   │   ├── metrics.py     # MAE and MAPE calculations
│   │   └── backtester.py  # Trading simulation and ROI calculation
│   └── visualization/     
│       └── plotter.py     # Plotly dashboard generation
├── .env                   # Secret API credentials (git-ignored)
├── requirements.txt       # Project dependencies
└── main.py                # Pipeline orchestrator
```

## Installation and Setup

### 1. Clone the repository:
```bash
git clone <repository-url>
cd stock_prediction_project
```

### 2. Install dependencies:
Ensure you are using Python 3.10+ and install the required packages.
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables:
Create a `.env` file in the root directory of the project and add your Alpaca API credentials.
```plaintext
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER_URL=https://paper-api.alpaca.markets
```

## Usage

To execute the full pipeline—which will fetch data, engineer features, train the models, run the backtest, and open the visualization dashboards in your browser—run the orchestrator script:

```bash
python main.py
```

By default, the script is configured to analyze NVIDIA (NVDA) from January 2022 to January 2026. You can modify the `SYMBOL`, `START_DATE`, `END_DATE`, and `INITIAL_CAPITAL` constants directly within `main.py` to test different assets and timeframes.

## Disclaimer

This project is for educational and research purposes only. The predictions and backtesting simulations do not constitute financial advice. Algorithmic trading involves significant risk.
