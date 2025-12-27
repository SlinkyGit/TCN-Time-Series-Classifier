import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["SPY"]
DEFAULT_START = "2014-01-01"
DEFAULT_END = None

def fetch_prices(tickers, start, end):

    df = yf.download(tickers=tickers, start=start, end=end, interval="1d", auto_adjust=False, progress=False, group_by="column")

    # ensure DatetimeIndex, business days only (yfinance already returns trading days)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # For now we only care about 1 ticker (SPY)
    if isinstance(tickers, (list, tuple)):
        ticker = tickers[0]
    else:
        ticker = tickers

    # This is a 1-column DataFrame; take that one column as a Series
    adj_df = df["Adj Close"].dropna()
    adj = adj_df[ticker] if isinstance(adj_df, pd.DataFrame) else adj_df

    adj = adj.astype(float)
    adj.name = f"{ticker}_adj_close"


    # adj = df["Adj Close"].dropna()
    # # if single ticker, yfinance may return Series; we want DF
    # if isinstance(adj, pd.Series):
    #     adj = adj.to_frame(name=tickers[0])

    # given series a name
    # adj.name = f"{tickers}_adj_close"

    return adj


if __name__ == "__main__":
    # fetches adjusted close
    series = fetch_prices(DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END)
    print(series.head())
    print(series.tail())
    print(series.shape)

    series.to_csv("data/spy_adj_close.csv", index_label="Date")