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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np



def init_model():
    # model = TemporalConvolutionalNetwork(input_channels=1, output_size=2)
    model = TemporalConvolutionalNetwork(input_channels=5, output_size=2)

    return model

def split_data():
    price = load_series()
    # returns = get_returns(price)
    # X, y = set_window(returns)
    X, y = set_window(price)

    # 70% train, 30% temp
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, shuffle=False) # shuffle = False since time-series data

    # print("Mean return in class 0 windows:", X_train[y_train == 0].mean())
    # print("Mean return in class 1 windows:", X_train[y_train == 1].mean())

    # print("Std return in class 0 windows:", X_train[y_train == 0].std())
    # print("Std return in class 1 windows:", X_train[y_train == 1].std())

    # split temp into 15% val / 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, shuffle=False) # shuffle = False since time-series data

    # normalize each channel using train-set stats only
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

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

    # count_0_train = np.sum(y_train == 0)
    # count_1_train = np.sum(y_train == 1)

    class_weights = torch.tensor(
        [len(y_train) / (2 * count_0_train), len(y_train) / (2 * count_1_train)],
        dtype=torch.float32
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)


    train_dataset = ReturnDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    validation_dataset = ReturnDataset(X_val, y_val)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    test_dataset = ReturnDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, validation_loader, test_loader, criterion


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

    print("True 0s:", np.sum(y_true == 0))
    print("True 1s:", np.sum(y_true == 1))

    baseline = max(np.mean(y_true), 1 - np.mean(y_true))
    print("Baseline accuracy:", baseline)

    for cls, count in zip(unique_preds, pred_counts):
        print(f"Predicted class {cls}: {count}")

    return avg_loss, accuracy, f1

# Dummy Baseline
X_train, X_tmp, y_train, y_tmp, X_val, X_test, y_val, y_test = split_data()

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    )
)

clf.fit(X_train_flat, y_train)
preds = clf.predict(X_val_flat)

print("LogReg Val Acc:", accuracy_score(y_val, preds))
print("LogReg Val Macro F1:", f1_score(y_val, preds, average="macro"))
print("LogReg Pred 0s:", np.sum(preds == 0))
print("LogReg Pred 1s:", np.sum(preds == 1))


if __name__ == "__main__":
    # print("Train_dataset X:", train_dataset.X)
    # print("Train_dataset y:", train_dataset.y)

    # init models + loaders
    model = init_model()
    train_loader, validation_loader, test_loader, criterion = build_loaders()

    X_batch, y_batch = next(iter(train_loader))

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 30
    # epochs = 1
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

        