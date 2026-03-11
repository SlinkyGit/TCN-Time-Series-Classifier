import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.dataset import ReturnDataset
from data.build_dataset import load_series, get_returns, set_window

from models.tcn import TemporalConvolutionalNetwork


def init_model():
    model = TemporalConvolutionalNetwork(input_channels=1, output_size=2)

    return model

def split_data():
    price = load_series()
    returns = get_returns(price)
    X, y = set_window(returns)

    # 70% train, 30% temp
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, shuffle=False) # shuffle = False since time-series data

    # split temp into 15% val / 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False) # shuffle = False since time-series data

    return X_train, X_tmp, y_train, y_tmp, X_val, X_test, y_val, y_test


def build_loaders():

    # price = load_series()
    # returns = get_returns(price)
    # X, y = set_window(returns)

    # # 70% train, 15% val, 15% test
    # # n = len(X)
    # # n_train = int(0.7 * n)
    # # n_val = int(0.85 * n)

    # # X_train, y_train = X[:n_train], y[:n_train]
    # # X_val, y_val = X[n_train:n_val], y[n_train: n_val]
    # # X_test, y_test = X[n_val:], y[n_val:]

    # # 70% train, 30% temp
    # X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, shuffle=False) # shuffle = False since time-series data

    # # split temp into 15% val / 15% test
    # X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False) # shuffle = False since time-series data

    X_train, X_tmp, y_train, y_tmp, X_val, X_test, y_val, y_test = split_data()

    count_0_train = np.sum(y_train == 0)
    count_1_train = np.sum(y_train == 1)
    print(f"y_train has {count_0_train} zeros and {count_1_train} ones")

    count_0_val = np.sum(y_val == 0)
    count_1_val = np.sum(y_val == 1)
    print(f"y_val has {count_0_val} zeros and {count_1_val} ones")

    count_0_test = np.sum(y_test == 0)
    count_1_test = np.sum(y_test == 1)
    print(f"y_test has {count_0_test} zeros and {count_1_test} ones")



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


def validation_metrics(model, dataloader, criterion):
    # F1 and Accuracy Score
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for Xi, yi in dataloader:
            outputs = model(Xi) # logits

            # if task == "binary":
            #     loss = criterion(outputs, yi)

            #     # converts the output logits to probabilities (reshape to flat vector for BCE)
            #     probs = torch.sigmoid(outputs).view(-1).numpy()

            #     # classification threshold 0.5 (boolean to ints)
            #     preds = (probs >= 0.5).astype(int)

            #     true  = yi.view(-1).numpy().astype(int)
            # else:

            loss = criterion(outputs, yi)

            # pick largest score
            preds = np.argmax(outputs.numpy(), axis=1)

            true  = yi.numpy()

            curr_loss = loss.item() * Xi.size(0)
            total_loss += curr_loss

            y_true.append(true)
            y_pred.append(preds)

    avg_loss = total_loss / len(dataloader.dataset)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, average="macro")


    unique_preds, pred_counts = np.unique(y_pred, return_counts=True)

    print("Validation prediction distribution:")
    print("Predicted 0s:", np.sum(y_pred == 0))
    print("Predicted 1s:", np.sum(y_pred == 1))

    for cls, count in zip(unique_preds, pred_counts):
        print(f"Predicted class {cls}: {count}")

    return avg_loss, accuracy, f1






if __name__ == "__main__":
    # print("Train_dataset X:", train_dataset.X)
    # print("Train_dataset y:", train_dataset.y)

    # init models + loaders
    model = init_model()
    train_loader, validation_loader, test_loader = build_loaders()

    X_batch, y_batch = next(iter(train_loader))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)

    # # print the important sanity info (shapes + dtypes + label range)
    # print("X_batch shape:", X_batch.shape)     # expected: (batch, 1, 60)
    # print("X_batch dtype:", X_batch.dtype)     # expected: torch.float32 (usually)
    # print("y_batch shape:", y_batch.shape)     # expected: (batch,)
    # print("y_batch dtype:", y_batch.dtype)     # expected: torch.int64 (torch.long)

    # # label min/max helps confirm your classes are correct (e.g., 0/1)
    # print("y_batch min/max:", y_batch.min().item(), y_batch.max().item())

    # print("AVG LOSS = ", avg_loss)

    # # run a forward pass
    # model.eval()
    # with torch.no_grad():
    #     logits = model(X_batch)
    # print("logits shape: ", logits.shape)
    # print("logits dtype: ", logits.dtype)

    # epochs = 45
    epochs = 1
    train_loss = 0.0
    val_loss = 0.0

    history = {
        "epoch" : [],
        "train_loss" : [],
        "val_loss" : [],
        "accuracy" : [],
        "f1" : []
    }

    for ep in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, accuracy, f1 = validation_metrics(model, validation_loader, criterion)

        history["epoch"].append(ep+1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(accuracy)
        history["f1"].append(f1)

        print(f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f} | Val F1: {f1:.4f}")

        