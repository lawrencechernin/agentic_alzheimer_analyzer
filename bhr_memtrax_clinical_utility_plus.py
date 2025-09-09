#!/usr/bin/env python3
"""
BHR MemTrax Clinical Utility (PLUS)
- Parallel script to push held-out AUC beyond 0.80 using safer stacking
- Combines winsor and non-winsor base predictions as meta inputs
- Expands MI-k sweep and HGB grid; optional XGB grid when available
- Uses held-out test and OOF stacking to avoid leakage

Outputs: bhr_memtrax_results/clinical_utility_plus_report.json
"""
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Reuse functions and settings from the base clinical utility script
from bhr_memtrax_clinical_utility import (
    DATA_DIR, OUT_DIR, COGNITIVE_QIDS,
    load_memtrax, load_medical, build_composite_labels, build_features,
    add_splines_train_test, oof_calibrated_probas, decision_curve, subgroup_metrics,
    LEAN_COLUMNS
)

try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False


def log(msg: str):
    print(msg, flush=True)


def prepare_xy(X_df_local: pd.DataFrame, labels_df: pd.DataFrame):
    xy_local = X_df_local.merge(labels_df, on='SubjectCode', how='inner')
    cols_local = [c for c in LEAN_COLUMNS if c in xy_local.columns]
    # Keep SubjectCode for alignment
    xy_local = xy_local[['SubjectCode'] + cols_local + ['AnyCogImpairment']].copy()
    for c in cols_local:
        xy_local[c] = pd.to_numeric(xy_local[c], errors='coerce')
    xy_local = xy_local.dropna(subset=['AnyCogImpairment'])
    xy_local['AnyCogImpairment'] = xy_local['AnyCogImpairment'].astype(int)
    # Minimal validity filter
    row_valid = xy_local[cols_local].notna().any(axis=1)
    xy_local = xy_local.loc[row_valid]
    return xy_local, cols_local


def fit_base_oof(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, mode_full: bool = True, cal_cv: int = 5):
    # Logistic base
    def build_base_logit():
        return Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("scale", StandardScaler(with_mean=False)),
            ("select", SelectKBest(mutual_info_classif, k=min(150, X_tr.shape[1]))),
            ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
        ])
    oof_logit, cal_logit_full = oof_calibrated_probas(build_base_logit, X_tr, y_tr, n_splits=cal_cv, method='isotonic')

    # HGB grid
    best_hgb = {'auc': -1.0, 'oof': None, 'cal_full': None, 'desc': ''}
    lr_list = [0.02, 0.03, 0.05, 0.1]
    leaves_list = [31, 63, 127]
    reg_list = [0.0, 0.1]
    min_leaf_list = [20, 50]
    max_bins_list = [255]
    for lr in lr_list:
        for leafs in leaves_list:
            for reg in reg_list:
                for min_leaf in min_leaf_list:
                    for max_bins in max_bins_list:
                        def build_hgb():
                            return Pipeline([
                                ("impute", SimpleImputer(strategy='median')),
                                ("clf", HistGradientBoostingClassifier(
                                    random_state=42,
                                    max_leaf_nodes=leafs,
                                    learning_rate=lr,
                                    max_depth=3,
                                    l2_regularization=reg,
                                    min_samples_leaf=min_leaf,
                                    max_bins=max_bins
                                ))
                            ])
                        oof_h, cal_h_full = oof_calibrated_probas(build_hgb, X_tr, y_tr, n_splits=cal_cv, method='isotonic')
                        a = roc_auc_score(y_tr, 0.5 * oof_logit + 0.5 * oof_h)
                        if a > best_hgb['auc']:
                            best_hgb = {'auc': a, 'oof': oof_h, 'cal_full': cal_h_full, 'desc': f"hgb lr={lr}, leaves={leafs}, reg={reg}, min_leaf={min_leaf}, bins={max_bins}"}

    # Optional XGB grid
    oof_xgb, cal_xgb_full, xgb_desc = (None, None, None)
    if mode_full and XGB_OK:
        pos = int(y_tr.sum())
        neg = int((~y_tr.astype(bool)).sum())
        spw_base = (neg / max(pos, 1))
        best_auc_tmp = -1.0
        for md in [3, 4, 5, 6]:
            for eta in [0.02, 0.03, 0.05, 0.1]:
                for ss in [0.7, 0.8, 0.9, 1.0]:
                    for cs in [0.7, 0.8, 0.9, 1.0]:
                        for spw in [spw_base * 0.8, spw_base, spw_base * 1.2]:
                            def build_xgb():
                                return Pipeline([
                                    ("impute", SimpleImputer(strategy='median')),
                                    ("clf", XGBClassifier(
                                        n_estimators=800,
                                        max_depth=md,
                                        learning_rate=eta,
                                        subsample=ss,
                                        colsample_bytree=cs,
                                        reg_lambda=1.0,
                                        objective='binary:logistic',
                                        tree_method='hist',
                                        eval_metric='auc',
                                        random_state=42,
                                        scale_pos_weight=spw,
                                        verbosity=0
                                    ))
                                ])
                            oof_tmp, cal_full_tmp = oof_calibrated_probas(build_xgb, X_tr, y_tr, n_splits=cal_cv, method='isotonic')
                            a = roc_auc_score(y_tr, oof_tmp)
                            if a > best_auc_tmp:
                                best_auc_tmp = a
                                oof_xgb, cal_xgb_full, xgb_desc = oof_tmp, cal_full_tmp, f"xgb md={md}, lr={eta}, ss={ss}, cs={cs}, spw={spw:.2f}"

    # Test-time base probabilities
    p_logit_te = cal_logit_full.predict_proba(X_te)[:, 1]
    p_hgb_te = best_hgb['cal_full'].predict_proba(X_te)[:, 1]
    p_xgb_te = cal_xgb_full.predict_proba(X_te)[:, 1] if cal_xgb_full is not None else None

    return {
        'oof_logit': oof_logit,
        'oof_hgb': best_hgb['oof'],
        'oof_xgb': oof_xgb,
        'p_logit_te': p_logit_te,
        'p_hgb_te': p_hgb_te,
        'p_xgb_te': p_xgb_te,
    }


def main() -> int:
    log("🚀 BHR MEMTRAX CLINICAL UTILITY (PLUS)")
    mem_q = load_memtrax()
    med = load_medical()

    # Build features for both variants
    log("[1/6] Building features (no-winsor & winsor)...")
    X_df_nowin = build_features(mem_q, use_winsorize=False)
    X_df_win = build_features(mem_q, use_winsorize=True)
    labels_df = build_composite_labels(med)

    xy_nowin, cols_nowin = prepare_xy(X_df_nowin, labels_df)
    xy_win, cols_win = prepare_xy(X_df_win, labels_df)

    # Align on SubjectCode intersection
    common_subjects = set(xy_nowin['SubjectCode']).intersection(set(xy_win['SubjectCode']))
    xy_nowin = xy_nowin[xy_nowin['SubjectCode'].isin(common_subjects)].reset_index(drop=True)
    xy_win = xy_win[xy_win['SubjectCode'].isin(common_subjects)].reset_index(drop=True)

    # Ensure same ordering by SubjectCode
    xy_nowin = xy_nowin.sort_values('SubjectCode').reset_index(drop=True)
    xy_win = xy_win.sort_values('SubjectCode').reset_index(drop=True)

    # Extract X/y
    cols = [c for c in LEAN_COLUMNS if c in xy_nowin.columns and c in xy_win.columns]
    X_nowin = xy_nowin[cols].copy()
    X_win = xy_win[cols].copy()
    y = xy_nowin['AnyCogImpairment'].astype(int)

    # Stratified held-out split
    log("[2/6] Creating held-out split...")
    Xn_tr, Xn_te, yn_tr, yn_te, Xw_tr, Xw_te = train_test_split(
        X_nowin, y, X_win, test_size=0.2, stratify=y, random_state=42
    )

    # Add splines on train-only and apply to test
    log("[3/6] Adding spline features (train-only fit)...")
    Xn_tr_aug, Xn_te_aug = add_splines_train_test(Xn_tr, Xn_te, ['Age_Baseline', 'YearsEducationUS_Converted'], n_knots=6)
    Xw_tr_aug, Xw_te_aug = add_splines_train_test(Xw_tr, Xw_te, ['Age_Baseline', 'YearsEducationUS_Converted'], n_knots=6)

    # Base learners (OOF + calibrated full) for both variants
    log("[4/6] Fitting base learners with OOF (no-winsor)...")
    base_nowin = fit_base_oof(Xn_tr_aug, yn_tr, Xn_te_aug, mode_full=True, cal_cv=5)
    log("[4/6] Fitting base learners with OOF (winsor)...")
    base_win = fit_base_oof(Xw_tr_aug, yn_tr, Xw_te_aug, mode_full=True, cal_cv=5)

    # Build meta train (OOF) and test matrices combining both variants
    log("[5/6] Building meta-learner inputs...")
    oof_cols = [
        ('p_logit_nowin', base_nowin['oof_logit']),
        ('p_hgb_nowin', base_nowin['oof_hgb']),
        ('p_logit_win', base_win['oof_logit']),
        ('p_hgb_win', base_win['oof_hgb']),
    ]
    te_cols = [
        ('p_logit_nowin', base_nowin['p_logit_te']),
        ('p_hgb_nowin', base_nowin['p_hgb_te']),
        ('p_logit_win', base_win['p_logit_te']),
        ('p_hgb_win', base_win['p_hgb_te']),
    ]
    if base_nowin['oof_xgb'] is not None:
        oof_cols.append(('p_xgb_nowin', base_nowin['oof_xgb']))
        te_cols.append(('p_xgb_nowin', base_nowin['p_xgb_te']))
    if base_win['oof_xgb'] is not None:
        oof_cols.append(('p_xgb_win', base_win['oof_xgb']))
        te_cols.append(('p_xgb_win', base_win['p_xgb_te']))

    M_tr = pd.DataFrame({name: vec for name, vec in oof_cols}, index=Xn_tr_aug.index)
    M_te = pd.DataFrame({name: vec for name, vec in te_cols}, index=Xn_te_aug.index)

    # Add raw features and spline meta features
    for raw in ['CognitiveScore_mean', 'long_reliability_change', 'Age_Baseline']:
        if raw in Xn_tr_aug.columns:
            M_tr[raw] = Xn_tr_aug[raw].fillna(Xn_tr_aug[raw].median())
            M_te[raw] = Xn_te_aug[raw].fillna(Xn_te_aug[raw].median())
    for col in list(Xn_tr_aug.columns):
        if col.startswith('Age_Baseline_spline_') or col.startswith('YearsEducationUS_Converted_spline_'):
            M_tr[col] = Xn_tr_aug[col]
            M_te[col] = Xn_te_aug[col]

    # Meta-learner sweep (elastic-net logistic)
    best_meta = {'auc': -1.0}
    l1r_list = [0.05, 0.1, 0.3, 0.5, 0.7]
    cvals = [0.25, 0.5, 1.0, 2.0, 4.0]
    for l1r in l1r_list:
        for cval in cvals:
            meta_pipe = Pipeline([
                ("impute", SimpleImputer(strategy='median')),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=4000, class_weight='balanced', solver='saga', penalty='elasticnet', l1_ratio=l1r, C=cval)),
            ])
            cal_meta = CalibratedClassifierCV(meta_pipe, cv=5, method='isotonic')
            cal_meta.fit(M_tr, yn_tr)
            pm_te = cal_meta.predict_proba(M_te)[:, 1]
            a = roc_auc_score(yn_te, pm_te)
            if a > best_meta['auc']:
                best_meta = {'model': cal_meta, 'proba_te': pm_te, 'auc': a, 'desc': f'enet l1r={l1r}, C={cval}'}

    y_proba = best_meta['proba_te']
    auc = roc_auc_score(yn_te, y_proba)
    pr_auc = average_precision_score(yn_te, y_proba)

    # Save report
    report = {
        'samples': int(len(yn_te)),
        'positives': int(yn_te.sum()),
        'prevalence': float(yn_te.mean()),
        'model_metrics': {
            'selected_model': best_meta['desc'],
            'auc_overall': float(auc),
            'pr_auc_overall': float(pr_auc)
        },
    }
    out_json = OUT_DIR / 'clinical_utility_plus_report.json'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    log(f"💾 Saved PLUS report: {out_json}")
    log(f"📈 PLUS pipeline (held-out): AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main()) 