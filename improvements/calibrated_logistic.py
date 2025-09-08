#!/usr/bin/env python3
"""
Calibrated Logistic Regression Recipe
- Impute (median), scale, MI feature selection, logistic with class_weight
- Isotonic calibration and threshold optimization for F1
"""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def train_calibrated_logistic(X: pd.DataFrame, y: pd.Series,
                              k_features: int = 200,
                              random_state: int = 42) -> Tuple[CalibratedClassifierCV, Dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scale", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=min(k_features, X.shape[1]))),
        ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
    ])

    params = {"clf__C": np.logspace(-2, 2, 10)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipe, params, n_iter=10, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=random_state)
    search.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(search.best_estimator_, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    y_proba = calibrated.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    t_best = thresholds[int(np.argmax(f1s))]
    y_pred = (y_proba >= t_best).astype(int)

    metrics = {
        'cv_auc': float(search.best_score_),
        'test_auc': float(roc_auc_score(y_test, y_proba)),
        'test_f1': float(f1_score(y_test, y_pred)),
        'test_precision': float(precision_score(y_test, y_pred)),
        'test_recall': float(recall_score(y_test, y_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'threshold': float(t_best),
        'best_C': float(search.best_params_.get('clf__C', 1.0)),
    }
    return calibrated, metrics 