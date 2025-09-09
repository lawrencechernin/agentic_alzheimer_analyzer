#!/usr/bin/env python3
import numpy as np
import pandas as pd


def decision_curve(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    N = len(y_true)
    out = []
    for pt in thresholds:
        preds = (y_proba >= pt).astype(int)
        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))
        NB = (TP / N) - (FP / N) * (pt / (1 - pt))
        out.append({'threshold': float(pt), 'net_benefit': float(NB)})
    df = pd.DataFrame(out)
    prev = float(np.mean(y_true))
    df['all'] = (prev - (1 - prev) * (df['threshold'] / (1 - df['threshold'])))
    df['none'] = 0.0
    return df 