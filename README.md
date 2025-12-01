# Advanced Time Series Forecasting on Crypto High-Frequency Data

**Subtitle:** Predicting Bitcoin (BTCUSDT) 1-minute price movements using Attention-LSTM and SARIMA baseline models with rolling-origin cross-validation.

---

## 📌 Project Overview

This project implements a complete pipeline for high-frequency cryptocurrency time series forecasting. It focuses on predicting Bitcoin (BTCUSDT) 1-minute price movements using both statistical and deep learning models.

**Key Features:**

- **Data ingestion:** Downloads 1-minute OHLCV klines from Binance public API.
- **Baseline modeling:** SARIMA forecasting for comparison.
- **Deep learning model:** LSTM with self-attention for multi-step prediction.
- **Evaluation:** Rolling-origin cross-validation, metrics (RMSE, MAE, MAPE), and plots.
- **Outputs:** Predictions CSV, attention weights, and forecast comparison plots.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib requests torch statsmodels scikit-learn
