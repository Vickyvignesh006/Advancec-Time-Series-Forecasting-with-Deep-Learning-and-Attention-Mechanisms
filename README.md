# Advancec-Time-Series-Forecasting-with-Deep-Learning-and-Attention-Mechanisms
Stock price forecasting using LSTM with self-attention on high-frequency data. Predicts closing prices, visualizes attention weights, and evaluates performance with RMSE, MAE, and MAPE.
# Stock Price Forecasting with LSTM + Attention

## 📌 Project Overview
This project predicts high-frequency stock prices using a combination of **LSTM (Long Short-Term Memory)** networks and **self-attention mechanisms**. The model is trained on 5-minute interval stock data to forecast future prices, providing both predictions and interpretability via attention weights.

---

## 🛠️ Features
- Fetches historical stock data using `yfinance`.
- Normalizes data with `MinMaxScaler`.
- Builds sequences for time-series prediction.
- Implements **LSTM + Self-Attention** model in PyTorch.
- Visualizes:
  - Predicted vs actual stock prices.
  - Attention weights to interpret model focus.
- Evaluates predictions using **RMSE**, **MAE**, and **MAPE**.

---

## 📂 Libraries & Dependencies
```bash
numpy
pandas
matplotlib
seaborn
torch
scikit-learn
yfinance
