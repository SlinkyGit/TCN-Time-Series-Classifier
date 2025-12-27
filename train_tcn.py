import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from data.dataset import ReturnDataset
from data.build_dataset import load_series, get_returns, set_window


price = load_series()
returns = get_returns(price)
X, y = set_window(returns)

# 70% train, 15% val, 15% test
n = len(X)
n_train = int(0.7 * n)
n_val = int(0.85 * n)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_val], y[n_train: n_val]
X_test, y_test = X[n_val:], y[n_val:]


train_dataset = ReturnDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

validation_dataset = ReturnDataset(X_val, y_val)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

test_dataset = ReturnDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == "__main__":
    print("Train_dataset X:", train_dataset.X)
    print("Train_dataset y:", train_dataset.y)