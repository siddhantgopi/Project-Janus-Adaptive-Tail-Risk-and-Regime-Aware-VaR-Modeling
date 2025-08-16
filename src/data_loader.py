# src/data_loader.py

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# Assets (13 including oil)
ASSETS = [
    "SPY",  # S&P 500 ETF
    "QQQ",  # Nasdaq ETF
    "TLT",  # Long-term Treasuries
    "HYG",  # High Yield Bonds
    "GLD",  # Gold
    "XLE",  # Energy ETF
    "UUP",  # USD Index
    "AAPL",
    "JPM",
    "TSLA",
    "NVDA",
    "META",
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_data(start="2010-01-01", end=None):
    """Fetch adjusted close prices from yfinance."""
    data = yf.download(ASSETS, start=start, end=end, progress=False, auto_adjust=False)
    # Keep only Adj Close
    prices = data["Adj Close"].dropna(how="all")
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns."""
    log_rets = np.log(prices / prices.shift(1))
    return log_rets.dropna()

def construct_portfolio(returns: pd.DataFrame, weights=None):
    """Compute equal or custom weighted portfolio returns."""
    if weights is None:
        weights = np.ones(returns.shape[1]) / returns.shape[1]
    weights = np.array(weights)
    portfolio_rets = returns.dot(weights)
    return portfolio_rets

def save_data(prices, returns, portfolio_returns):
    """Save cleaned datasets."""
    prices.to_csv(DATA_DIR / "prices.csv")
    returns.to_csv(DATA_DIR / "returns.csv")
    portfolio_returns.to_csv(DATA_DIR / "portfolio_returns.csv")

def run_pipeline():
    prices = fetch_data()
    returns = compute_log_returns(prices)

    # Handle missing tickers dynamically
    valid_assets = returns.columns
    weights = np.ones(len(valid_assets)) / len(valid_assets)

    portfolio_returns = construct_portfolio(returns[valid_assets], weights)

    save_data(prices, returns, portfolio_returns)
    print("✅ Data pipeline complete. Files saved in /data")

if __name__ == "__main__":
    run_pipeline()
