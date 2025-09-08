#!/usr/bin/env python3
"""
Anti-leakage utilities for modeling:
- Stratified train/test split helper
- Train-only spline fitting applied to test
- Out-of-fold (OOF) calibrated probabilities for stacking
- Basic evaluation helpers
"""
from typing import List, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import SplineTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score


def stratified_holdout_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_tr, X_te, y_tr, y_te


def add_splines_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame, cols: List[str], n_knots: int = 5, degree: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for col in cols:
        if col in Xtr.columns:
            try:
                st = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
                vals_tr = Xtr[[col]].to_numpy()
                vals_te = Xte[[col]].to_numpy() if col in Xte.columns else None
                if np.isfinite(vals_tr).sum() >= len(vals_tr):
                    st.fit(vals_tr)
                    spl_tr = st.transform(vals_tr)
                    spl_te = st.transform(vals_te) if vals_te is not None else None
                    for i in range(spl_tr.shape[1]):
                        Xtr[f'{col}_spline_{i+1}'] = spl_tr[:, i]
                        if spl_te is not None:
                            Xte[f'{col}_spline_{i+1}'] = spl_te[:, i]
            except Exception:
                continue
    return Xtr, Xte


def oof_calibrated_probas(estimator_builder: Callable[[], object], X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5, method: str = 'isotonic', random_state: int = 42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y_train))
    for tr_idx, val_idx in kf.split(X_train, y_train):
        est = estimator_builder()
        cal = CalibratedClassifierCV(est, cv=3, method=method)
        cal.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = cal.predict_proba(X_train.iloc[val_idx])[:, 1]
    final_est = estimator_builder()
    cal_full = CalibratedClassifierCV(final_est, cv=3, method=method)
    cal_full.fit(X_train, y_train)
    return oof, cal_full


def evaluate_holdout(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_proba)
    pr = average_precision_score(y_true, y_proba)
    return {'auc': float(auc), 'pr_auc': float(pr)} 