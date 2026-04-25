import numpy as np
import pandas as pd
from indicators import *

LOOKBACK = 60 # lookback window of past 60 days
# HORIZON = 10 # predict 1 day ahead
HORIZON = 5 # predict 1 day ahead
CSV_PATH = "data/spy_adj_close.csv"

def load_series(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # assume second column is SPY
    price = df.iloc[:, 0] # all rows -> 1st column
    price.name = "SPY_adj_close"

    return price

def get_returns(price_series):
    # daily returns
    daily_returns = price_series.pct_change().dropna()

    daily_returns.name = "SPY_daily_return"
    
    return daily_returns

def build_features(price):
    returns = get_returns(price)

    _, _, _, bbp = bollinger_bands(price)
    rsi = relative_strength_index(price)
    macd, macd_signal = moving_average_convergence_divergence(price)

    features = pd.DataFrame({
        "returns": returns,
        "bbp": bbp,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
    })

    features = features.dropna()
    return features

def set_window(price, lookback=LOOKBACK, horizon=HORIZON):
    X = []
    y = []

    features = build_features(price)
    feature_values = features.values

    # Use returns column for future label
    returns = features["returns"].values

    for i in range(lookback, len(features) - horizon + 1):
        # shape: (lookback, num_features)
        window = feature_values[i - lookback : i]

        # cumulative future return over next horizon days
        future_returns = (1 + returns[i : i + horizon]).prod() - 1

        if future_returns > 0.002:
            label = 1
        elif future_returns < -0.002:
            label = 0
        else:
            continue

        # transpose for Conv1d: (channels, sequence_length)
        X.append(window.T)
        y.append(label)

    X = np.array(X)  # (num_samples, num_features, lookback)
    y = np.array(y)

    return X, y

def set_window(price, lookback=LOOKBACK, horizon=HORIZON):
    X = []
    y = []
    # daily_returns = returns.values

    features = build_features(price)
    feature_values = features.values

    # Use returns column for future label
    returns = features["returns"].values
    
    for i in range(lookback, len(features) - horizon + 1):
    # i.e. - window size = 3, horizon = 1, returns = [r0, r1, r2, r3, r4, r5, r6]
    # i.e. - i goes from 3 to 5 (because len=7, horizon=1 -> 7-1 = 6, stop before 6)

        # take past "window size" returns at position i - 1
        # window = daily_returns[i - lookback : i]
        # shape: (lookback, num_features)
        window = feature_values[i - lookback : i]

        # pick return horizon days as what we are classifying
        # future_returns = daily_returns[i + horizon - 1] # target/try to predict

        # past lookback returns

        # cumulative return over the next `horizon` days
        # future_returns = (1 + daily_returns[i : i + horizon]).prod() - 1
        # cumulative future return over next horizon days
        future_returns = (1 + returns[i : i + horizon]).prod() - 1

        if future_returns > 0.002:     # +0.2%
            label = 1
        elif future_returns < -0.002:  # -0.2%
            label = 0
        else:
            continue

        X.append(window.T) # collect all windows
        y.append(label) # collect all labels

    X = np.array(X) # shape: (num_samples, window_size)
    y = np.array(y) # shape: (num_samples, )

    # reshape X for pytorch TCN: (batch, channels, length)
    # ref -> https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
    # X = X[:, np.newaxis, :] # (num_samples, 1, window_size) ; 1 channel since only using returns

    return X, y

if __name__ == "__main__":
    price = load_series()
    # rets = get_returns(price)
    X, y = set_window(price)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Positive labels:", y.sum(), "out of", len(y))
