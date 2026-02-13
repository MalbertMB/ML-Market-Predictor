# Stock Price Prediction with Random Forest (Alpaca API)

A machine learning pipeline for predicting the next-day closing price of a stock using historical market data from the Alpaca API and a Random Forest regression model.

The script retrieves daily price data, performs feature engineering using technical indicators, trains a supervised learning model, evaluates performance using Mean Absolute Error (MAE), and generates a visualization comparing predicted and actual prices.

---

## Overview

This project implements a structured workflow:

1. Retrieve historical stock data from Alpaca.
2. Engineer time-series features derived from price and volatility.
3. Train a Random Forest regression model.
4. Evaluate predictive performance.
5. Visualize out-of-sample predictions.
6. Estimate the next trading day’s closing price.

The current configuration predicts daily closing prices for **NVDA** between 2022-01-01 and 2026-01-01.

---

## Data Source

Historical data is retrieved using:

- `alpaca.data.historical.StockHistoricalDataClient`
- Daily timeframe (`TimeFrame.Day`)
- API authentication via environment variables

The dataset includes:

- Open, high, low, close
- Volume
- Timestamped daily bars

---

## Feature Engineering

The following features are derived from raw market data:

- **return**: Daily percentage return (`close.pct_change()`)
- **volatility**: Rolling 5-day standard deviation of returns
- **ma_5**: 5-day moving average of closing price
- **ma_10**: 10-day moving average
- **ma_20**: 20-day moving average
- **volume**
- **close**

### Target Variable

- **target**: Next-day closing price (`close.shift(-1)`)

Rows with insufficient rolling-window history are removed prior to training.

---

## Model

The regression model used:

- **Algorithm**: RandomForestRegressor
- **n_estimators**: 300
- **max_depth**: 10
- **random_state**: 42
- **Train/Test Split**: 80/20 (chronological split, no shuffling)

A chronological split is used to preserve time-series integrity and prevent forward-looking bias.

---

## Evaluation

Model performance is measured using:

- **Mean Absolute Error (MAE)**

MAE is computed on the held-out test set and printed to the console.

---

## Visualization

The script generates a Matplotlib plot comparing:

- Actual closing prices (test set)
- Predicted closing prices

Backend configuration:

```python
matplotlib.use('QtAgg')
```

This ensures compatibility with windowed environments for interactive plotting.

---

## Predicting the Next Closing Price

After training, the model:

1. Extracts the most recent feature vector.
2. Predicts the next trading day’s closing price.
3. Outputs the forecasted value to the console.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-project-directory>
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

- pandas
- numpy
- matplotlib
- python-dotenv
- alpaca-py
- scikit-learn

---

## Environment Configuration

Create a `.env` file in the project root:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

The script validates that both variables are set before execution.

---

## How to Run

Execute the script directly:

```bash
python main.py
```

The program will:

1. Download historical stock data.
2. Train the model.
3. Print the Mean Absolute Error.
4. Display a prediction plot.
5. Output the predicted next closing price.

---

## Limitations

- This is a supervised regression model, not a trading strategy.
- No transaction costs, slippage, or execution constraints are modeled.
- Random Forest does not explicitly model temporal dependencies.
- Performance depends heavily on selected features and time window.

This implementation is intended for educational and experimental purposes, not financial advice.
