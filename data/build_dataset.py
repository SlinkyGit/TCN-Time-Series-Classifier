import numpy as np
import pandas as pd
import fetch_data as fd

LOOKBACK = 60 # lookback window of past 60 days
HORIZON = 1 # predict 1 day ahead
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

def set_window(returns, lookback=LOOKBACK, horizon=HORIZON):
    X = []
    y = []
    daily_returns = returns.values
    
    for i in range(lookback, len(daily_returns) - horizon):
    # i.e. - window size = 3, horizon = 1, returns = [r0, r1, r2, r3, r4, r5, r6]
    # i.e. - i goes from 3 to 5 (because len=7, horizon=1 -> 7-1 = 6, stop before 6)

        # take past "window size" returns at position i - 1
        window = daily_returns[i - lookback : i]

        # pick return horizon days as what we are classifying
        future_returns = daily_returns[i + horizon] # target/try to predict

        # binary classification for up/down
        label = 1 if future_returns > 0 else 0

        X.append(window) # collect all windows
        y.append(label) # collect all labels

    X = np.array(X) # shape: (num_samples, window_size)
    y = np.array(y) # shape: (num_samples, )

    # reshape X for pytorch TCN: (batch, channels, length)
    # ref -> https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
    X = X[:, np.newaxis, :] # (num_samples, 1, window_size) ; 1 channel since only using returns

    return X, y

if __name__ == "__main__":
    price = load_series()
    rets = get_returns(price)
    X, y = set_window(rets)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Positive labels:", y.sum(), "out of", len(y))
