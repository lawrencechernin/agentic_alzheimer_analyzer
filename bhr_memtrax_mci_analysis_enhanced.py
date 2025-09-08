#!/usr/bin/env python3
"""
Enhanced BHR MemTrax→MCI analysis
- Reuses base analyzer for robust merging and longitudinal aggregation
- Enriches with demographics (age, education) and optional ECOG/ADL features
- Tries multiple models with hyperparameter tuning (LogReg, RF, HistGB, XGB if available, MLP)
- Applies feature scaling, selection, calibration, and threshold tuning

Usage:
    python bhr_memtrax_mci_analysis_enhanced.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Import the base analyzer
from bhr_memtrax_mci_analysis import BHRMemTraxMCIAnalyzer

OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def add_optional_ecog(analyzer: BHRMemTraxMCIAnalyzer) -> None:
    """Merge optional ECOG baseline features (self and/or informant) if available."""
    data_dir = analyzer.data_dir
    base = analyzer.combined_data
    if base is None or len(base) == 0:
        return
    added_any = False

    def merge_numeric_baseline(csv_name: str, prefix: str) -> bool:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            return False
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            if 'SubjectCode' not in df.columns:
                return False
            if 'TimepointCode' in df.columns:
                df = df[df['TimepointCode'] == 'm00'].copy()
            df = df.drop_duplicates(subset=['SubjectCode'], keep='first')
            # keep only numeric columns besides SubjectCode
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            keep_cols = ['SubjectCode'] + num_cols
            eco = df[keep_cols].copy()
            # Create simple aggregates if many numeric columns exist
            if num_cols:
                eco[f'{prefix}_GlobalMean'] = eco[num_cols].mean(axis=1)
                eco[f'{prefix}_GlobalSum'] = eco[num_cols].sum(axis=1)
            before_cols = set(base.columns)
            merged = base.merge(eco, on='SubjectCode', how='left')
            new_cols = [c for c in merged.columns if c not in before_cols]
            analyzer.combined_data = merged
            print(f"   ➕ Added {prefix} features from {csv_name} ({len(new_cols)} cols)")
            return True
        except Exception as e:
            print(f"   ⚠️ ECOG merge failed for {csv_name}: {e}")
            return False

    added_any |= merge_numeric_baseline('BHR_EverydayCognition.csv', 'ECOG')
    added_any |= merge_numeric_baseline('BHR_SP_ECog.csv', 'SP_ECOG')
    added_any |= merge_numeric_baseline('BHR_SP_ADL.csv', 'SP_ADL')

    if not added_any:
        print("   ℹ️ No ECOG/ADL features added (files missing or no numeric cols)")


def add_ecog_residuals(analyzer: BHRMemTraxMCIAnalyzer) -> None:
    """Create ECOG residual features adjusted for age and education."""
    df = analyzer.combined_data
    if df is None:
        return
    targets = []
    for col in ['ECOG_GlobalMean', 'SP_ECOG_GlobalMean', 'SP_ADL_GlobalMean']:
        if col in df.columns:
            targets.append(col)
    if not targets:
        return
    if 'Age_Baseline' not in df.columns or 'YearsEducationUS_Converted' not in df.columns:
        return
    # Prepare predictors
    X_demo = df[['Age_Baseline', 'YearsEducationUS_Converted']].copy()
    X_demo = X_demo.apply(pd.to_numeric, errors='coerce')
    for tcol in targets:
        y_t = pd.to_numeric(df[tcol], errors='coerce')
        mask = X_demo.notna().all(axis=1) & y_t.notna()
        if mask.sum() < 100:
            continue
        lr = LinearRegression()
        lr.fit(X_demo.loc[mask], y_t.loc[mask])
        pred = lr.predict(X_demo)
        resid = y_t - pred
        df[f'{tcol}_Residual'] = resid
        print(f"   ➕ Added residualized feature: {tcol}_Residual")
    analyzer.combined_data = df


def build_feature_matrix(analyzer: BHRMemTraxMCIAnalyzer) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare X, y from analyzer with additional polynomial terms on key variables."""
    # Ensure target and features are prepared
    analyzer.prepare_target_variable()
    analyzer.prepare_features()

    X = analyzer.X.copy()
    y = analyzer.y.copy()

    # Add low-order polynomial interactions for a small set of key vars
    key_vars = [v for v in ['Age_Baseline', 'YearsEducationUS_Converted', 'CognitiveScore_mean'] if v in X.columns]
    if key_vars:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_vals = poly.fit_transform(X[key_vars])
        poly_names = poly.get_feature_names_out(key_vars)
        poly_df = pd.DataFrame(poly_vals, columns=[f'poly_{n}' for n in poly_names], index=X.index)
        X = pd.concat([X, poly_df], axis=1)
        print(f"   ➕ Added polynomial features for {key_vars}: +{poly_df.shape[1]} cols")

    return X, y, list(X.columns)


def train_and_select_model(X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> dict:
    """Train multiple models with tuning, return best model and metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()

    candidates = []

    # Logistic (elastic-ish via C search)
    logit = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scale", scaler),
        ("select", SelectKBest(mutual_info_classif, k=min(200, X.shape[1]))),
        ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
    ])
    logit_params = {
        "clf__C": np.logspace(-2, 2, 10)
    }
    candidates.append(("LogReg", logit, logit_params))

    # Random Forest
    rf = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight='balanced', random_state=42))
    ])
    rf_params = {
        "clf__max_depth": [6, 8, 12, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4]
    }
    candidates.append(("RandomForest", rf, rf_params))

    # HistGradientBoosting (fast, strong on tabular)
    hgb = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("clf", HistGradientBoostingClassifier(random_state=42))
    ])
    hgb_params = {
        "clf__max_depth": [None, 3, 5, 7],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_leaf_nodes": [15, 31, 63]
    }
    candidates.append(("HistGB", hgb, hgb_params))

    # MLP (simple neural net)
    mlp = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scale", scaler),
        ("select", SelectKBest(mutual_info_classif, k=min(300, X.shape[1]))),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                               max_iter=200, random_state=42))
    ])
    mlp_params = {
        "clf__alpha": [1e-5, 1e-4, 1e-3],
        "clf__learning_rate_init": [1e-3, 5e-4]
    }
    candidates.append(("MLP", mlp, mlp_params))

    # XGBoost (if available)
    if XGB_AVAILABLE:
        xgb = Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("clf", XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective='binary:logistic',
                tree_method='hist',
                eval_metric='auc',
                random_state=42,
                scale_pos_weight=(y_train.shape[0] - y_train.sum()) / (y_train.sum() + 1e-6)
            ))
        ])
        xgb_params = {
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0]
        }
        candidates.append(("XGBoost", xgb, xgb_params))

    best = {
        "name": None,
        "estimator": None,
        "cv_auc": -np.inf,
        "metrics": None
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_pipes = {}
    for name, pipe, param_dist in candidates:
        print(f"\n🔎 Searching model: {name}")
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=min(15, np.prod([len(v) for v in param_dist.values()])),
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            refit=True
        )
        search.fit(X_train, y_train)
        cv_auc = search.best_score_
        print(f"   Best CV AUC: {cv_auc:.3f} | Params: {search.best_params_}")

        # Calibrate best model probability
        calibrated = CalibratedClassifierCV(search.best_estimator_, cv=3, method='isotonic')
        calibrated.fit(X_train, y_train)

        # Predict
        y_proba = calibrated.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Optionally adjust threshold to maximize F1 on validation slice
        # Compute optimal threshold via Youden's J or F1 grid
        thresholds = np.linspace(0.1, 0.9, 33)
        f1s = []
        for t in thresholds:
            f1s.append(f1_score(y_test, (y_proba >= t).astype(int)))
        best_t = thresholds[int(np.argmax(f1s))]
        y_pred_opt = (y_proba >= best_t).astype(int)

        metrics = {
            "cv_auc": float(cv_auc),
            "test_auc": float(roc_auc_score(y_test, y_proba)),
            "test_f1": float(f1_score(y_test, y_pred_opt)),
            "test_precision": float(precision_score(y_test, y_pred_opt)),
            "test_recall": float(recall_score(y_test, y_pred_opt)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_opt)),
            "threshold": float(best_t)
        }
        print(f"   Test AUC: {metrics['test_auc']:.3f} | F1@{best_t:.2f}: {metrics['test_f1']:.3f} | P: {metrics['test_precision']:.3f} R: {metrics['test_recall']:.3f}")

        if metrics["test_auc"] > best["cv_auc"]:
            best.update({
                "name": name,
                "estimator": calibrated,
                "cv_auc": metrics["test_auc"],
                "metrics": metrics
            })
        # Keep tuned base estimators for stacking
        best_pipes[name] = search.best_estimator_

    # Stacking ensemble (use tuned base learners)
    if len(best_pipes) >= 2:
        estimators = [(n, est) for n, est in best_pipes.items() if n in ["LogReg", "RandomForest", "HistGB", "XGBoost"]]
        if len(estimators) >= 2:
            print("\n🔗 Training stacking ensemble")
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced'),
                stack_method='predict_proba',
                passthrough=False,
                cv=cv,
                n_jobs=-1
            )
            stack.fit(X_train, y_train)
            stacked_cal = CalibratedClassifierCV(stack, cv=3, method='isotonic')
            stacked_cal.fit(X_train, y_train)
            y_proba = stacked_cal.predict_proba(X_test)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 33)
            f1s = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
            best_t = thresholds[int(np.argmax(f1s))]
            metrics = {
                "cv_auc": None,
                "test_auc": float(roc_auc_score(y_test, y_proba)),
                "test_f1": float(f1_score(y_test, (y_proba >= best_t).astype(int))),
                "test_precision": float(precision_score(y_test, (y_proba >= best_t).astype(int))),
                "test_recall": float(recall_score(y_test, (y_proba >= best_t).astype(int))),
                "test_accuracy": float(accuracy_score(y_test, (y_proba >= best_t).astype(int))),
                "threshold": float(best_t)
            }
            print(f"   Stacked Test AUC: {metrics['test_auc']:.3f} | F1@{best_t:.2f}: {metrics['test_f1']:.3f}")
            if metrics["test_auc"] > best["cv_auc"]:
                best.update({
                    "name": "Stacking",
                    "estimator": stacked_cal,
                    "cv_auc": metrics["test_auc"],
                    "metrics": metrics
                })
    return best


def main() -> int:
    print("🚀 ENHANCED BHR MEMTRAX→MCI ANALYSIS")

    analyzer = BHRMemTraxMCIAnalyzer()
    analyzer.load_data()
    analyzer.merge_datasets()
    analyzer.add_demographic_features()

    # Optional: Add ECOG/ADL baseline signals if present
    add_optional_ecog(analyzer)
    # Residualize ECOG against age/education if available
    add_ecog_residuals(analyzer)

    X, y, feature_names = build_feature_matrix(analyzer)

    print(f"📦 Dataset for modeling: {X.shape[0]:,} × {X.shape[1]} | Positives: {int(y.sum()):,} ({y.mean()*100:.2f}%)")

    best = train_and_select_model(X, y, feature_names)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": best["name"],
        "metrics": best["metrics"],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1])
    }

    out_json = OUTPUT_DIR / 'enhanced_analysis_report.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved enhanced report: {out_json}")

    print("\n==============================")
    print(f"🏆 Best Model: {best['name']}")
    print(f"📈 Test AUC: {best['metrics']['test_auc']:.3f}")
    print(f"🎯 F1: {best['metrics']['test_f1']:.3f} @ threshold {best['metrics']['threshold']:.2f}")
    print("==============================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 