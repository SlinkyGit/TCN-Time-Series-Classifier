import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from data.dataset import ReturnDataset
from data.build_dataset import load_series, get_returns, set_window

from models.tcn import TemporalConvolutionalNetwork


def init_model():
    model = TemporalConvolutionalNetwork(input_channels=1, output_size=2)

    return model

def build_loaders():

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

    return train_loader, validation_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer):

    # set model to training mode
    model.train()

    # accumulate loss to compute an average
    running_loss = 0.0 

    for i, data in enumerate(loader):
        # every data instance is an input + label pair
        inputs, labels = data

        #zero gradients for every batch
        optimizer.zero_grad()

        # forward pass: compute the model output
        predictions = model(inputs)

        # compute loss and gradients (backwards pass)
        loss = criterion(predictions, labels) # labels = y_batch
        loss.backward()

        # adjust learning weights
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    return avg_loss







if __name__ == "__main__":
    # print("Train_dataset X:", train_dataset.X)
    # print("Train_dataset y:", train_dataset.y)

    # init models + loaders
    model = init_model()
    train_loader, validation_loader, test_loader = build_loaders()

    X_batch, y_batch = next(iter(train_loader))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)

    # print the important sanity info (shapes + dtypes + label range)
    print("X_batch shape:", X_batch.shape)     # expected: (batch, 1, 60)
    print("X_batch dtype:", X_batch.dtype)     # expected: torch.float32 (usually)

    print("y_batch shape:", y_batch.shape)     # expected: (batch,)
    print("y_batch dtype:", y_batch.dtype)     # expected: torch.int64 (torch.long)

    # label min/max helps confirm your classes are correct (e.g., 0/1)
    print("y_batch min/max:", y_batch.min().item(), y_batch.max().item())

    print("AVG LOSS = ", avg_loss)

    # run a forward pass
    model.eval()
    with torch.no_grad():
        logits = model(X_batch)

    print("logits shape: ", logits.shape)
    print("logits dtype: ", logits.dtype)


