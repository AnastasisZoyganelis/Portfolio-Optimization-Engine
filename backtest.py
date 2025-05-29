# --- Step 1: Import Libraries ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting, objective_functions

# --- Step 2: Define Your Portfolio Assets ---
tickers = ["MSFT", "AMD", "NVDA", "CVX", "KO","AAPL","NKE","WMT","LLY","JPM","SIRI","GC=F","BTC-USD"]
start_date = "2023-05-20"
end_date = "2025-05-20"
initial_investment = 1

# --- Step 3: Download Tickers One-by-One to Avoid Rate Limits ---
def download_single_assets(tickers, start, end):
    all_data = []
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)[['Close']]
            df.rename(columns={"Close": ticker}, inplace=True)
            all_data.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
    return pd.concat(all_data, axis=1)

# --- Step 4: Download and Clean Data ---
data = download_single_assets(tickers, start_date, end_date)
data.dropna(inplace=True)

# --- Step 5: Calculate Returns and Covariance ---
mu = expected_returns.mean_historical_return(data)   # Annualized returns
S = risk_models.sample_cov(data)                     # Sample covariance matrix
risk_free_rate = 0.04

# --- Step 6: Show Asset-Level Risk Metrics ---


print("\nAsset Correlation Matrix:")
print(data.pct_change().corr().round(2))

# --- Step 7: Optimization for Different Objectives with Constraints ---
ef_sharpe = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.4))
ef_sharpe.add_objective(objective_functions.L2_reg, gamma=0.1)
weights_sharpe = ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
ret_sharpe, vol_sharpe, _ = ef_sharpe.portfolio_performance()
cleaned_weights_sharpe = ef_sharpe.clean_weights()

# Store weights for backtesting
weights_sharpe_array = np.array([v for v in cleaned_weights_sharpe.values()])

# --- Step 8: Backtest Strategy (Max Sharpe Portfolio, Dollar Growth vs SPY) ---
returns = data.pct_change().dropna()
portfolio_returns = (returns * weights_sharpe_array).sum(axis=1)
portfolio_growth = (1 + portfolio_returns).cumprod() * initial_investment

# Download SPY benchmark
benchmark = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True)[['Close']]
benchmark.rename(columns={"Close": "SPY"}, inplace=True)
benchmark = benchmark.pct_change().dropna()
benchmark_growth = (1 + benchmark).cumprod() * initial_investment

# Align index
aligned_growth = pd.concat([portfolio_growth, benchmark_growth], axis=1).dropna()
aligned_growth.columns = ["Portfolio", "SPY"]

# Plot
plt.figure(figsize=(10, 6))
aligned_growth.plot(ax=plt.gca())
plt.title("Dollar Growth: Max Sharpe Portfolio vs SPY")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Step 9: Output Weights as Percentages ---
def print_weights(label, weights):
    print(f"\nOptimal Weights ({label}):")
    print({k: f"{v*100:.2f}%" for k, v in weights.items()})

print_weights("Max Sharpe", cleaned_weights_sharpe)

# --- Step 10: Portfolio Performance Summary ---
summary_df = pd.DataFrame({
    "Strategy": ["Max Sharpe"],
    "Expected Return": [ret_sharpe],
    "Volatility": [vol_sharpe],
    "Sharpe Ratio": [(ret_sharpe - risk_free_rate) / vol_sharpe]
})
print("\nPortfolio Performance Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv("portfolio_metrics_summary.csv", index=False)

# --- Step 11: Plot Allocation Pie Chart ---
def plot_pie(weights, title):
    labels = [k for k, v in weights.items() if v > 0]
    sizes = [v for k, v in weights.items() if v > 0]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

plot_pie(cleaned_weights_sharpe, "Asset Allocation: Max Sharpe")
