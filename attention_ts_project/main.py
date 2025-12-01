"""
main.py

Advanced Time Series Forecasting (Crypto high-frequency data)
- Downloads 1m klines from Binance for a chosen symbol (default BTCUSDT)
- Builds SARIMA baseline and LSTM + Self-Attention model (PyTorch)
- Rolling-origin cross-validation, training, evaluation, plots
- Saves dataset (crypto_klines.csv) and attention weights (attention_weights.txt)

Requirements:
pip install numpy pandas matplotlib requests torch statsmodels scikit-learn
"""

import time
import math
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta, timezone

# -------------------------
# 1) BINANCE KLINES FETCHER
# -------------------------
# Reference: Binance public market data: GET /api/v3/klines, limit default 500, max 1000. (docs)
# We'll paginate to collect many 1-minute bars. See Binance docs for details and limits.
# Docs: Binance REST API (Kline/Candlestick): GET /api/v3/klines (public, no API key required).
# (This script uses the public endpoint; don't hammer the API in production.)

BINANCE_BASE = "https://api.binance.com"

def ms_now():
    return int(time.time() * 1000)

def fetch_klines(symbol="BTCUSDT", interval="1m", start_time_ms=None, end_time_ms=None, limit=1000):
    """
    Fetch one page of klines from Binance public REST API.
    Returns: list of kline lists (openTime, open, high, low, close, volume, ...).
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def download_klines(symbol="BTCUSDT", interval="1m", min_bars=6000, out_csv="crypto_klines.csv"):
    """
    Download at least `min_bars` 1-minute klines by paginating backwards from "now".
    Saves CSV with columns: open_time, open, high, low, close, volume, close_time, ...
    """
    print(f"Downloading >={min_bars} {interval} klines for {symbol} from Binance...")
    all_klines = []
    limit = 1000  # max per request per Binance doc
    end_time = ms_now()  # current time in ms

    # We'll request pages going backwards: get latest up to `limit`, then set end_time = first_bar_open_time - 1 ms
    while len(all_klines) < min_bars:
        page = fetch_klines(symbol=symbol, interval=interval, end_time_ms=end_time, limit=limit)
        if not page:
            break
        # page is ordered oldest->newest for the requested window
        all_klines = page + all_klines  # prepend newest page to earlier pages (we're paging backwards)
        first_open = page[0][0]
        # move end_time to just before this page's first open time to page earlier data
        end_time = first_open - 1
        # safety: avoid infinite loop
        if len(page) < 1:
            break
        # small sleep so we don't risk rate limiting too aggressively
        time.sleep(0.2)

    # Keep only the most recent min_bars (end of list)
    if len(all_klines) > min_bars:
        all_klines = all_klines[-min_bars :]

    # Build DataFrame
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=cols)
    # convert ms timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    # numeric conversions
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("open_time").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} bars to {out_csv}")
    return df

# -------------------------
# 2) Sequence preparation
# -------------------------
def create_sequences(series, seq_len, horizon):
    X, y = [], []
    for i in range(len(series) - seq_len - horizon + 1):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len : i + seq_len + horizon])
    X = np.array(X)  # shape (N, seq_len)
    y = np.array(y)  # shape (N, horizon)
    return X, y

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # expect X shape (N, seq_len), y shape (N, horizon)
        self.X = torch.tensor(X[..., np.newaxis], dtype=torch.float32)  # add feature dim
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# 3) Attention + LSTM model
# -------------------------
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, H)
        Q = self.W_q(x)  # (B, T, H)
        K = self.W_k(x)
        V = self.W_v(x)
        # scores: (B, T, T)
        scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        out = torch.bmm(weights, V)  # (B, T, H)
        return out, weights

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.att = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (B, T, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (B, T, H)
        att_out, att_weights = self.att(lstm_out)  # (B, T, H), (B, T, T)
        # Use last time step's attended representation
        last = att_out[:, -1, :]  # (B, H)
        out = self.fc(last)  # (B, horizon)
        return out, att_weights

# -------------------------
# 4) Training / CV / Eval
# -------------------------
def rolling_origin_cv(make_model_fn, X_train, y_train, folds=5, epochs=3, batch_size=64, lr=1e-3, device="cpu"):
    """
    Rolling-origin CV suitable for time-series:
    For i=1..folds:
       use data up to fold_i_end as training
       train model for a few epochs and record last validation loss on the fold end (no separate val set here)
    This is a lightweight CV to gauge hyperparams; for production use a proper holdout inside each fold.
    """
    n = len(X_train)
    fold_size = n // (folds + 1)  # leave final chunk for final training
    losses = []
    criterion = nn.MSELoss()
    for i in range(1, folds+1):
        end = fold_size * (i+1)  # grow training set
        Xt = X_train[:end]
        yt = y_train[:end]
        ds = TimeSeriesDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model = make_model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        last_loss = None
        for ep in range(epochs):
            model.train()
            batch_losses = []
            for Xb, yb in loader:
                Xb = Xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                pred, _ = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            last_loss = np.mean(batch_losses)
        print(f"Fold {i}/{folds} — training size {len(Xt)} — last epoch loss {last_loss:.6f}")
        losses.append(last_loss)
    return np.mean(losses), losses

def train_model(model, train_loader, epochs=10, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        batch_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        print(f"Epoch {ep+1}/{epochs} — train loss {np.mean(batch_losses):.6f}")
    return model

def evaluate_metrics(y_true, y_pred):
    # y_true, y_pred: (N, horizon)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # compute metrics for first horizon step as primary (commonly done)
    rmse = np.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0]))
    mae = mean_absolute_error(y_true[:,0], y_pred[:,0])
    # MAPE: avoid division by zero
    denom = np.where(y_true[:,0]==0, 1e-8, y_true[:,0])
    mape = np.mean(np.abs((y_true[:,0] - y_pred[:,0]) / denom)) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# -------------------------
# 5) MAIN pipeline
# -------------------------
def main():
    # --------------------
    # PARAMETERS
    # --------------------
    SYMBOL = "BTCUSDT"            # crypto symbol on Binance
    INTERVAL = "1m"              # 1-minute bars (high-frequency)
    MIN_BARS = 6000              # ensure >= 5000; default 6000 for some margin
    CSV_OUT = "crypto_klines.csv"

    SEQ_LEN = 60                 # use past 60 minutes
    HORIZON = 5                  # predict next 5 minutes (multi-step)
    TEST_RATIO = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # 1) Download / load data
    # --------------------
    try:
        df = download_klines(symbol=SYMBOL, interval=INTERVAL, min_bars=MIN_BARS, out_csv=CSV_OUT)
    except Exception as e:
        print("Error downloading klines:", e)
        print("If running repeatedly, you may be hitting Binance rate limits. Try again later or reduce requests.")
        return

    # We'll use the close price as the series
    series = df["close"].values.astype(float)
    times = df["open_time"].values

    # --------------------
    # 2) Preprocess: scaling (MinMax on train only)
    # --------------------
    split_idx = int(len(series) * (1 - TEST_RATIO))
    train_series = series[:split_idx]
    test_series = series[split_idx - SEQ_LEN:]  # include overlapping past for sequences

    # MinMax scaling based on train only
    min_val = train_series.min()
    max_val = train_series.max()
    eps = 1e-8
    def scale(x): return (x - min_val) / (max_val - min_val + eps)
    def inv_scale(x): return x * (max_val - min_val + eps) + min_val

    scaled = scale(series)
    # --------------------
    # 3) Create sequences
    # --------------------
    X, y = create_sequences(scaled, seq_len=SEQ_LEN, horizon=HORIZON)
    # Split into train/test aligned with earlier split index
    n_total = len(X)
    # compute index in X that corresponds to split_idx in original series:
    # A sequence starting at i uses data up to i+SEQ_LEN+HORIZON-1. We used create_sequences up to len(series)-SEQ_LEN-HORIZON+1
    # Simpler: compute the time index of the first y sample: y sample at index j corresponds to series index j+SEQ_LEN
    # We want to split such that the forecast horizon start >= split_idx
    # So find split_j where (j + SEQ_LEN) >= split_idx
    split_j = max(0, split_idx - SEQ_LEN)
    X_train, y_train = X[:split_j], y[:split_j]
    X_test, y_test = X[split_j:], y[split_j:]

    print(f"Total sequences: {len(X)}, Training seq: {len(X_train)}, Testing seq: {len(X_test)}")

    # --------------------
    # 4) Baseline: SARIMA on raw (train) close price forecasting first horizon step
    # --------------------
    # We'll fit SARIMA on the training portion of the raw close series and forecast len(y_test) steps ahead (first-step)
    sarima_order = (2,1,2)  # example; tune if desired
    seasonal_order = (0,0,0,0)
    try:
        print("Training SARIMA baseline (this may take some time)...")
        sar_train_series = series[:split_idx]
        sar_model = SARIMAX(sar_train_series, order=sarima_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        sar_fit = sar_model.fit(disp=False)
        # Forecast the next len(y_test) * HORIZON minutes. We'll forecast horizon step-by-step for fair comparison.
        sar_forecast = sar_fit.get_forecast(steps=len(y_test) + HORIZON - 1)
        sar_pred_all = sar_forecast.predicted_mean.values
        # We will align by taking sar_pred_all[i] corresponds to prediction for series index split_idx + i
        # For evaluation comparing first horizon step predictions for each test sample:
        sar_first_step_preds = []
        for i in range(len(y_test)):
            # prediction for series index split_idx + i (first step)
            sar_first_step_preds.append(sar_pred_all[i])
        sar_first_step_preds = np.array(sar_first_step_preds)
        # compute baseline metrics (on actual first-step of y_test)
        y_test_first = inv_scale(y_test[:,0])
        baseline_metrics = evaluate_metrics(y_test_first.reshape(-1,1), sar_first_step_preds.reshape(-1,1))
        print("SARIMA baseline metrics:", baseline_metrics)
    except Exception as e:
        print("SARIMA baseline failed:", e)
        baseline_metrics = None

    # --------------------
    # 5) Rolling-origin CV for model selection (quick)
    # --------------------
    hidden_dim = 64
    num_layers = 1
    batch_size = 128
    epochs_cv = 2

    def make_model():
        return AttentionLSTM(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, horizon=HORIZON)

    print("Running rolling-origin cross-validation (quick)...")
    cv_score, cv_folds = rolling_origin_cv(make_model, X_train, y_train, folds=4, epochs=epochs_cv, batch_size=batch_size, lr=1e-3, device=DEVICE)
    print("CV avg loss:", cv_score)

    # --------------------
    # 6) Train final model on full training set
    # --------------------
    train_ds = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = make_model()
    print("Training final Attention-LSTM model...")
    model = train_model(model, train_loader, epochs=6, lr=1e-3, device=DEVICE)

    # --------------------
    # 7) Evaluate on test set
    # --------------------
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test[..., np.newaxis], dtype=torch.float32).to(DEVICE)
        preds_scaled, att_weights = model(X_test_t)
        preds_scaled = preds_scaled.cpu().numpy()  # (Ntest, HORIZON)
    # invert scale
    preds = inv_scale(preds_scaled)
    y_test_unscaled = inv_scale(y_test)

    metrics = evaluate_metrics(y_test_unscaled, preds)
    print("Attention-LSTM metrics (first horizon step):", metrics)

    # --------------------
    # 8) Plots comparing actual vs predictions (first horizon)
    # --------------------
    plt.figure(figsize=(12,5))
    # choose a plotting window: last 500 test points or all
    plot_n = min(500, len(y_test_unscaled))
    idxs = np.arange(plot_n)
    plt.plot(idxs, y_test_unscaled[:plot_n,0], label="Actual (first-step)")
    plt.plot(idxs, preds[:plot_n,0], label="Attention-LSTM (first-step)")
    if baseline_metrics is not None:
        plt.plot(idxs, sar_first_step_preds[:plot_n], label="SARIMA (first-step)")
    plt.title(f"{SYMBOL} — Forecast Comparison (first horizon)")
    plt.xlabel("Test sample index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_comparison.png")
    plt.show()

    # --------------------
    # 9) Save attention weights (save average over test set)
    # --------------------
    # att_weights: (B, T, T)
    if att_weights is not None:
        att_np = att_weights.cpu().numpy()
        # compute mean attention across batches (if B>1)
        mean_att = np.mean(att_np, axis=0)  # (T, T)
        np.savetxt("attention_weights.txt", mean_att, fmt="%.6f")
        print("Saved attention_weights.txt (mean across test samples)")
    else:
        print("No attention weights available")

    # --------------------
    # 10) Save final predictions to CSV
    # --------------------
    out_df = pd.DataFrame({
        "timestamp": df["open_time"].iloc[split_idx + SEQ_LEN - 1 : split_idx + SEQ_LEN - 1 + len(preds)].reset_index(drop=True),
        "actual_first": y_test_unscaled[:,0],
        "pred_attention_first": preds[:,0]
    })
    if baseline_metrics is not None:
        out_df["pred_sarima_first"] = sar_first_step_preds[:len(preds)]
    out_df.to_csv("predictions_compare.csv", index=False)
    print("Saved predictions_compare.csv")

    print("Done. Files produced: crypto_klines.csv, predictions_compare.csv, forecast_comparison.png, attention_weights.txt (if available)")

if __name__ == "__main__":
    main()
