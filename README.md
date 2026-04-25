# TCN-Time-Series-Classifier
Temporal Convolutional Network (TCN)–based classifier for time-series sequence classification, built in PyTorch.

## Overview

This project explores whether short-horizon market direction (SPY) can be predicted using deep learning on time-series data.

A Temporal Convolutional Network (TCN) was trained on rolling windows of historical price data and technical indicators.

---

## Key Findings

- Raw return-based features led to model collapse (predicting a single class)
- Feature engineering (RSI, MACD, Bollinger Bands) improved class balance but produced unstable performance
- After normalization, predictive performance dropped, suggesting earlier gains were driven by scale artifacts rather than true signal
- Logistic regression baseline confirmed weak predictive signal in short-horizon market data

---

## Conclusion

This project demonstrates the difficulty of extracting reliable short-term predictive signals from financial time series, even with deep learning models.

The focus shifted from maximizing performance to validating whether meaningful signal exists.

## Status
Work in progress : continuing to explore alternative feature sets and modeling approaches.
