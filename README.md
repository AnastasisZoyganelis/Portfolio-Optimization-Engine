# 📈 Portfolio Optimization Engine

This project is a fully functional Python tool for **optimizing financial portfolios** using modern portfolio theory. It allows you to compare strategies like maximizing the Sharpe ratio, maximizing return, and minimizing risk — with real market data.

## 🚀 Features

- Downloads historical price data from Yahoo Finance
- Calculates expected returns and sample covariance
- Optimizes portfolio for:
  - ✅ Maximum Sharpe Ratio
  - 📈 Maximum Return
  - 🛡️ Minimum Volatility
- Applies realistic constraints (5–40% per asset)
- Adds L2 regularization to reduce turnover
- Shows:
  - 📊 Efficient frontiers with optimal point and asset labels
  - 📌 Allocation pie charts
  - 📋 Portfolio performance summary
  - 🧮 Asset volatilities and correlation matrix

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
