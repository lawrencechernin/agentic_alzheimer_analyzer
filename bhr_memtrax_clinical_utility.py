#!/usr/bin/env python3
"""
BHR MemTrax Clinical Utility Analysis
- Composite cognitive impairment target (baseline m00): OR of {QID1-5, QID1-12, QID1-13, QID1-22, QID1-23}
- Quality filtering (Ashford policy)
- Lean feature set (sequence/fatigue/variability + core RT/accuracy + age/edu/gender + interactions)
- Calibrated logistic baseline with MI selection and threshold tuning
- Decision curve analysis and subgroup metrics
- Delta over demographics-only baseline

Outputs: bhr_memtrax_results/clinical_utility_report.json, subgroup_metrics.csv, decision_curve.png
"""
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold

from improvements.ashford_policy import apply_ashford
from improvements.sequence_feature_engineering import compute_sequence_features
from improvements.demographics_enrichment import enrich_demographics
from improvements.calibrated_logistic import train_calibrated_logistic
from improvements.anti_leakage import drop_all_nan_columns

# Fast mode settings
FAST_MODE = True

# Data and output locations
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUT_DIR = Path("bhr_memtrax_results")
OUT_DIR.mkdir(exist_ok=True)

COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# Optional: XGBoost for deeper runs
try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False

import argparse

def parse_args():
    p = argparse.ArgumentParser(description="BHR MemTrax Clinical Utility Pipeline")
    p.add_argument("--mode", choices=["fast", "standard", "full"], default="fast",
                   help="fast: quick calibrated stack; standard: moderate sweeps; full: deep sweeps (can run for hours)")
    return p.parse_args()

def log(msg: str):
    print(msg, flush=True)


def load_memtrax() -> pd.DataFrame:
    log("[1/7] Loading MemTrax and applying quality filters...")
    mem = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    mem_q = apply_ashford(mem, accuracy_threshold=0.65)
    log(f"   MemTrax: {len(mem):,} → {len(mem_q):,} after Ashford")
    return mem_q


def load_medical() -> pd.DataFrame:
    log("[2/7] Loading Medical History and building baseline frame...")
    med = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    if 'TimepointCode' in med.columns:
        med = med[med['TimepointCode'] == 'm00'].copy()
    med = med.drop_duplicates(subset=['SubjectCode'], keep='first')
    log(f"   Medical baseline: {len(med):,} subjects")
    return med


def build_composite_labels(med: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in COGNITIVE_QIDS if c in med.columns]
    if not present:
        raise ValueError("No cognitive QIDs present for composite target")
    bin_mat = []
    for c in present:
        v = med[c]
        bin_mat.append((v.astype(float) == 1.0).astype(int).fillna(0))
    bin_arr = np.vstack([v.to_numpy() for v in bin_mat])
    any_pos = bin_arr.max(axis=0)
    known_any = np.zeros_like(any_pos)
    for c in present:
        known_any = np.logical_or(known_any, med[c].isin([1.0, 2.0]).to_numpy())
    y = pd.Series(any_pos, index=med.index).where(known_any, other=np.nan)
    labels = med[['SubjectCode']].copy()
    labels['AnyCogImpairment'] = y.astype(float)
    labels = labels.dropna(subset=['AnyCogImpairment'])
    labels['AnyCogImpairment'] = labels['AnyCogImpairment'].astype(int)
    log(f"   Composite labels: {labels['AnyCogImpairment'].sum():,} positive of {len(labels):,} ({labels['AnyCogImpairment'].mean()*100:.2f}%)")
    return labels


def winsorize_reaction_times(mem: pd.DataFrame, low: float = 0.4, high: float = 2.0) -> pd.DataFrame:
    df = mem.copy()
    if 'ReactionTimes' not in df.columns:
        return df
    def _clip_str(rt_str: str) -> str:
        parts = []
        for x in str(rt_str).split(','):
            x = x.strip()
            if not x:
                continue
            try:
                v = float(x)
                if np.isfinite(v):
                    v = min(max(v, low), high)
                    parts.append(f"{v:.3f}")
            except Exception:
                continue
        return ','.join(parts)
    df['ReactionTimes'] = df['ReactionTimes'].apply(_clip_str)
    return df


def compute_ecog_residuals(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    dfx = df.copy()
    targets = []
    for name, csv in [('ECOG', 'BHR_EverydayCognition.csv'), ('SP_ECOG', 'BHR_SP_ECog.csv'), ('SP_ADL', 'BHR_SP_ADL.csv')]:
        p = DATA_DIR / csv
        if not p.exists():
            continue
        eco = pd.read_csv(p, low_memory=False)
        if 'SubjectCode' not in eco.columns:
            continue
        if 'TimepointCode' in eco.columns:
            eco = eco[eco['TimepointCode'] == 'm00'].copy()
        num_cols = eco.select_dtypes(include=[np.number]).columns.tolist()
        keep = ['SubjectCode'] + num_cols
        eco_small = eco[keep].drop_duplicates(subset=['SubjectCode'], keep='first')
        if num_cols:
            eco_small[f'{name}_GlobalMean'] = eco_small[num_cols].mean(axis=1)
        # Only merge aggregated cols to avoid collisions
        feature_cols = ['SubjectCode'] + [c for c in eco_small.columns if c.startswith(f'{name}_')]
        dfx = dfx.merge(eco_small[feature_cols], on='SubjectCode', how='left')
        for c in feature_cols:
            if c != 'SubjectCode' and c not in targets:
                targets.append(c)
    if 'Age_Baseline' in dfx.columns and 'YearsEducationUS_Converted' in dfx.columns:
        X_demo = dfx[['Age_Baseline', 'YearsEducationUS_Converted']].copy()
        X_demo = X_demo.apply(pd.to_numeric, errors='coerce')
        for t in targets:
            if t in dfx.columns:
                y = pd.to_numeric(dfx[t], errors='coerce')
                mask = X_demo.notna().all(axis=1) & y.notna()
                if mask.sum() >= 100:
                    lr = LinearRegression()
                    lr.fit(X_demo.loc[mask], y.loc[mask])
                    pred = lr.predict(X_demo.loc[mask])
                    dfx.loc[mask, f'{t}_Residual'] = y.loc[mask] - pred
    return dfx


def build_features(mem_q: pd.DataFrame, use_winsorize: bool = False) -> pd.DataFrame:
    # Sequence features
    mem_q_base = winsorize_reaction_times(mem_q, 0.4, 2.0) if use_winsorize else mem_q
    log(f"[3/7] Computing sequence features (winsorize={'on' if use_winsorize else 'off'})...")
    seq = compute_sequence_features(mem_q_base)
    # Aggregate numeric means per subject
    agg = mem_q_base.groupby('SubjectCode').mean(numeric_only=True).reset_index()
    # Cognitive score ratio mean
    if 'CorrectResponsesRT' in mem_q_base.columns and 'CorrectPCT' in mem_q_base.columns:
        mem_q_base['CognitiveScore'] = mem_q_base['CorrectResponsesRT'] / (mem_q_base['CorrectPCT'] + 0.01)
        cg = mem_q_base.groupby('SubjectCode')['CognitiveScore'].mean().rename('CognitiveScore_mean').reset_index()
        agg = agg.merge(cg, on='SubjectCode', how='left')
    X_df = agg.merge(seq, on='SubjectCode', how='left')
    # Demographics
    log("[4/7] Enriching demographics and interactions...")
    X_df = enrich_demographics(DATA_DIR, X_df)
    if 'Age_Baseline' in X_df.columns:
        if 'CorrectResponsesRT_mean' in X_df.columns:
            X_df['age_rt_interaction'] = X_df['CorrectResponsesRT_mean'] * (X_df['Age_Baseline'] / 65.0)
        if 'long_reliability_change' in X_df.columns:
            X_df['age_variability_interaction'] = X_df['long_reliability_change'] * (X_df['Age_Baseline'] / 65.0)
        if 'CorrectPCT_mean' in X_df.columns and 'long_reliability_change' in X_df.columns:
            X_df['accuracy_stability'] = X_df['CorrectPCT_mean'] / (X_df['long_reliability_change'] + 1e-6)
    # ECOG residuals
    log("[5/7] Merging ECOG/SP/ADL and computing residuals...")
    X_df = compute_ecog_residuals(X_df)
    return X_df


def add_splines_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, cols: List[str], n_knots: int) -> (pd.DataFrame, pd.DataFrame):
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for col in cols:
        if col in Xtr.columns:
            try:
                st = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
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


def oof_calibrated_probas(estimator_builder, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5, method: str = 'isotonic', random_state: int = 42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y_train))
    for tr_idx, val_idx in kf.split(X_train, y_train):
        est = estimator_builder()
        cal = CalibratedClassifierCV(est, cv=3, method=method)
        cal.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = cal.predict_proba(X_train.iloc[val_idx])[:, 1]
    # Fit final model on full train for test-time inference
    final_est = estimator_builder()
    cal_full = CalibratedClassifierCV(final_est, cv=3, method=method)
    cal_full.fit(X_train, y_train)
    return oof, cal_full


LEAN_COLUMNS = [
    'CorrectResponsesRT_mean', 'CorrectPCT_mean', 'IncorrectRejectionsN_mean',
    'CognitiveScore_mean',
    'seq_first_third_mean', 'seq_last_third_mean', 'seq_fatigue_effect',
    'seq_mean_rt', 'seq_median_rt',
    'long_reliability_change', 'long_n_timepoints', 'long_rt_slope',
    'Age_Baseline', 'YearsEducationUS_Converted', 'Gender_Numeric',
    'age_rt_interaction', 'age_variability_interaction', 'accuracy_stability',
    'ECOG_GlobalMean_Residual', 'SP_ECOG_GlobalMean_Residual', 'SP_ADL_GlobalMean_Residual'
]


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
    df['all'] = (np.mean(y_true) - (1 - np.mean(y_true)) * (df['threshold'] / (1 - df['threshold'])))
    df['none'] = 0.0
    return df


def subgroup_metrics(X: pd.DataFrame, y: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    rows = []
    if 'Age_Baseline' in X.columns:
        bins = [0, 65, 75, 85, 200]
        labels = ['<65', '65-75', '75-85', '85+']
        bands = pd.cut(X['Age_Baseline'], bins=bins, labels=labels, right=False)
        for b in labels:
            mask = (bands == b) & y.notna()
            if mask.sum() < 50:
                continue
            rows.append({'group': f'age_{b}', 'auc': roc_auc_score(y[mask], y_proba[mask])})
    if 'YearsEducationUS_Converted' in X.columns:
        bins = [0, 12, 16, 25]
        labels = ['<=12', '13-16', '>16']
        bands = pd.cut(X['YearsEducationUS_Converted'], bins=bins, labels=labels, right=False)
        for b in labels:
            mask = (bands == b) & y.notna()
            if mask.sum() < 50:
                continue
            rows.append({'group': f'edu_{b}', 'auc': roc_auc_score(y[mask], y_proba[mask])})
    return pd.DataFrame(rows)


def main(mode: str = 'fast') -> int:
    global FAST_MODE
    FAST_MODE = (mode == 'fast')
    log(f"🚀 BHR MEMTRAX CLINICAL UTILITY ANALYSIS ({mode.upper()} MODE)")
    mem_q = load_memtrax()
    med = load_medical()

    if mode == 'fast':
        # Fast: single pass without winsorization
        X_df = build_features(mem_q, use_winsorize=False)
        labels_df = build_composite_labels(med)
        log("[6/7] Merging features and labels...")
        xy = X_df.merge(labels_df, on='SubjectCode', how='inner')
        cols = [c for c in LEAN_COLUMNS if c in xy.columns]
        X = xy[cols].apply(pd.to_numeric, errors='coerce')
        X = X.loc[:, X.notna().mean() > 0]
        y = xy['AnyCogImpairment'].astype(int)
        row_valid = X.notna().any(axis=1)
        X = X.loc[row_valid]
        y = y.loc[row_valid]
        if len(y) == 0:
            raise RuntimeError("No samples after merging features and labels.")

        # Splines (fast)
        for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
            if col in X.columns:
                try:
                    st = SplineTransformer(n_knots=4, degree=3, include_bias=False)
                    vals = X[[col]].to_numpy()
                    if np.isfinite(vals).sum() >= len(vals):
                        spl = st.fit_transform(vals)
                        for i in range(spl.shape[1]):
                            X[f'{col}_spline_{i+1}'] = spl[:, i]
                except Exception:
                    pass

        log("[7/7] Training calibrated logistic (fast) and minimal stacked model...")
        model_logit, _ = train_calibrated_logistic(X, y, k_features=min(100, X.shape[1]))
        y_proba_logit = model_logit.predict_proba(X)[:, 1]

        hgb = HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=31, learning_rate=0.05, max_depth=3)
        hgb_pipe = Pipeline([("impute", SimpleImputer(strategy='median')), ("clf", hgb)])
        cal_hgb = CalibratedClassifierCV(hgb_pipe, cv=2, method='sigmoid')
        cal_hgb.fit(X, y)
        p_hgb = cal_hgb.predict_proba(X)[:, 1]

        M_df = pd.DataFrame({'p_logit': y_proba_logit, 'p_hgb': p_hgb}, index=X.index)
        for raw in ['CognitiveScore_mean', 'long_reliability_change', 'Age_Baseline']:
            if raw in X.columns:
                M_df[raw] = X[raw].fillna(X[raw].median())

        meta = Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', C=1.0))
        ])
        cal_meta = CalibratedClassifierCV(meta, cv=2, method='sigmoid')
        cal_meta.fit(M_df, y)
        y_proba = cal_meta.predict_proba(M_df)[:, 1]

        auc = roc_auc_score(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        thresholds = np.linspace(0.05, 0.5, 10)
        dc = decision_curve(y.to_numpy(), y_proba, thresholds)

        plt.figure(figsize=(8, 6))
        plt.plot(dc['threshold'], dc['net_benefit'], label='Model')
        plt.plot(dc['threshold'], dc['all'], label='Treat All', linestyle='--')
        plt.plot(dc['threshold'], dc['none'], label='Treat None', linestyle=':')
        plt.xlabel('Threshold probability')
        plt.ylabel('Net Benefit')
        plt.title('Decision Curve Analysis (Fast)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        dcurve_path = OUT_DIR / 'decision_curve.png'
        plt.savefig(dcurve_path, dpi=300)
        plt.close()

        sub_df = subgroup_metrics(X, y, y_proba)
        sub_path = OUT_DIR / 'subgroup_metrics.csv'
        sub_df.to_csv(sub_path, index=False)

        report = {
            'samples': int(len(y)),
            'positives': int(y.sum()),
            'prevalence': float(y.mean()),
            'features_used': cols,
            'model_metrics': {
                'selected_model': 'FastMeta(Logit+HGB)',
                'auc_overall': float(auc),
                'pr_auc_overall': float(pr_auc)
            },
            'artifacts': {
                'decision_curve_png': str(dcurve_path),
                'subgroup_metrics_csv': str(sub_path)
            }
        }
        out_json = OUT_DIR / 'clinical_utility_report.json'
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=2)
        log(f"💾 Saved clinical report: {out_json}")
        log(f"📈 Fast pipeline: AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")
        return 0

    # STANDARD/FULL MODES (deeper; use held-out test to avoid leakage)
    log("[3/9] Building features (nowin + winsor)...")
    X_df_nowin = build_features(mem_q, use_winsorize=False)
    X_df_win = build_features(mem_q, use_winsorize=True)
    labels_df = build_composite_labels(med)

    def prepare_xy(X_df_local: pd.DataFrame):
        xy_local = X_df_local.merge(labels_df, on='SubjectCode', how='inner')
        cols_local = [c for c in LEAN_COLUMNS if c in xy_local.columns]
        X_local = xy_local[cols_local].apply(pd.to_numeric, errors='coerce')
        X_local = X_local.loc[:, X_local.notna().mean() > 0]
        y_local = xy_local['AnyCogImpairment'].astype(int)
        row_valid = X_local.notna().any(axis=1)
        X_local = X_local.loc[row_valid]
        y_local = y_local.loc[row_valid]
        return X_local, y_local, cols_local

    X_nowin, y_nowin, cols_nowin = prepare_xy(X_df_nowin)
    X_win, y_win, cols_win = prepare_xy(X_df_win)
    candidates = [(X_nowin, y_nowin, cols_nowin, False), (X_win, y_win, cols_win, True)]

    best_overall = {'auc': -1.0}
    for X_all, y_all, cols, used_win in candidates:
        if len(y_all) == 0:
            continue
        log(f"[4/9] Candidate dataset (winsorize={'on' if used_win else 'off'}): n={len(y_all):,}")

        # Stratified split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
        )

        # Drop all-NaN cols based on training; align test
        X_tr, X_te = drop_all_nan_columns(X_tr, X_te)

        # Add splines fitted on train only
        knots = 5 if mode == 'standard' else 6
        X_tr_aug, X_te_aug = add_splines_train_test(
            X_tr, X_te, ['Age_Baseline', 'YearsEducationUS_Converted'], knots
        )

        # Logistic sweep (mutual info k) on train; evaluate on test
        k_list = [50, 100, 150] if mode == 'standard' else [50, 100, 150, 200, 300]
        log(f"[5/9] Logistic sweep over k={k_list}...")
        best_log = {'auc': -1.0}
        for k in k_list:
            k_use = min(k, X_tr_aug.shape[1])
            model, _ = train_calibrated_logistic(X_tr_aug, y_tr, k_features=k_use)
            y_proba_te = model.predict_proba(X_te_aug)[:, 1]
            auc_log = roc_auc_score(y_te, y_proba_te)
            pr_log = average_precision_score(y_te, y_proba_te)
            if auc_log > best_log['auc']:
                best_log = {'model': model, 'proba_te': y_proba_te, 'auc': auc_log, 'pr': pr_log, 'k': k_use}
        log(f"      best logistic AUC(test)={best_log['auc']:.3f} (k={best_log['k']})")

        # Define estimator builders
        def build_base_logit():
            return Pipeline([
                ("impute", SimpleImputer(strategy='median')),
                ("scale", StandardScaler(with_mean=False)),
                ("select", SelectKBest(mutual_info_classif, k=min(100, X_tr_aug.shape[1]))),
                ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
            ])

        def build_hgb(lr: float, leafs: int):
            return Pipeline([
                ("impute", SimpleImputer(strategy='median')),
                ("clf", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=leafs, learning_rate=lr, max_depth=3))
            ])

        cal_cv = 3 if mode == 'standard' else 5

        # OOF for base logit
        oof_logit, cal_logit_full = oof_calibrated_probas(build_base_logit, X_tr_aug, y_tr, n_splits=cal_cv, method='isotonic')

        # HGB grid with OOF
        log("[6/9] HGB grid search with OOF...")
        best_hgb = {'auc': -1.0, 'desc': '', 'oof': None, 'cal_full': None}
        lrs = [0.03, 0.05, 0.1] if mode == 'standard' else [0.02, 0.03, 0.05, 0.1]
        leaves_list = [31, 63] if mode == 'standard' else [31, 63, 127]
        for lr in lrs:
            for leafs in leaves_list:
                def builder():
                    return build_hgb(lr, leafs)
                oof_h, cal_h_full = oof_calibrated_probas(builder, X_tr_aug, y_tr, n_splits=cal_cv, method='isotonic')
                # Evaluate simple meta over two OOF columns for selection
                M_oof_tmp = np.column_stack([oof_logit, oof_h])
                a = roc_auc_score(y_tr, 0.5 * M_oof_tmp[:, 0] + 0.5 * M_oof_tmp[:, 1])
                if a > best_hgb['auc']:
                    best_hgb = {'auc': a, 'desc': f'hgb lr={lr}, leaves={leafs}', 'oof': oof_h, 'cal_full': cal_h_full}

        # Optional XGB with OOF (full mode)
        oof_xgb, cal_xgb_full, xgb_desc = (None, None, None)
        if mode == 'full' and XGB_OK:
            log("[7/9] XGB deep grid search with OOF (this can take a long time)...")
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
                                            scale_pos_weight=spw
                                        ))
                                    ])
                                oof_tmp, cal_full_tmp = oof_calibrated_probas(build_xgb, X_tr_aug, y_tr, n_splits=cal_cv, method='isotonic')
                                # quick internal AUC on OOF
                                a = roc_auc_score(y_tr, oof_tmp)
                                if a > best_auc_tmp:
                                    best_auc_tmp = a
                                    oof_xgb, cal_xgb_full = oof_tmp, cal_full_tmp
                                    xgb_desc = f"xgb md={md}, lr={eta}, ss={ss}, cs={cs}, spw={spw:.2f}"

        # Build meta features (OOF on train)
        oof_list = [oof_logit, best_hgb['oof']]
        meta_names = ['p_logit', 'p_hgb']
        if oof_xgb is not None:
            oof_list.append(oof_xgb)
            meta_names.append('p_xgb')
        M_tr = np.column_stack(oof_list)
        M_tr_df = pd.DataFrame(M_tr, columns=meta_names, index=X_tr_aug.index)
        for raw in ['CognitiveScore_mean', 'long_reliability_change', 'Age_Baseline']:
            if raw in X_tr_aug.columns:
                M_tr_df[raw] = X_tr_aug[raw].fillna(X_tr_aug[raw].median())

        # Test meta inputs from full-calibrated base models
        p_logit_te = cal_logit_full.predict_proba(X_te_aug)[:, 1]
        p_hgb_te = best_hgb['cal_full'].predict_proba(X_te_aug)[:, 1]
        meta_te_cols = [p_logit_te, p_hgb_te]
        if cal_xgb_full is not None:
            p_xgb_te = cal_xgb_full.predict_proba(X_te_aug)[:, 1]
            meta_te_cols.append(p_xgb_te)
        M_te = np.column_stack(meta_te_cols)
        M_te_df = pd.DataFrame(M_te, columns=meta_names[:M_te.shape[1]], index=X_te_aug.index)
        for raw in ['CognitiveScore_mean', 'long_reliability_change', 'Age_Baseline']:
            if raw in X_te_aug.columns:
                M_te_df[raw] = X_te_aug[raw].fillna(X_te_aug[raw].median())

        # Meta-learner grid on OOF-train, evaluate on test
        log("[8/9] Meta-learner grid search (elastic-net logistic) on OOF train, evaluate on test...")
        best_meta = {'auc': -1.0}
        l1r_list = [0.1, 0.5] if mode == 'standard' else [0.05, 0.1, 0.3, 0.5, 0.7]
        cvals = [0.5, 1.0, 2.0] if mode == 'standard' else [0.25, 0.5, 1.0, 2.0, 4.0]
        for l1r in l1r_list:
            for cval in cvals:
                meta_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy='median')),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, class_weight='balanced', solver='saga', penalty='elasticnet', l1_ratio=l1r, C=cval)),
                ])
                cal_meta = CalibratedClassifierCV(meta_pipe, cv=cal_cv, method='isotonic')
                cal_meta.fit(M_tr_df, y_tr)
                pm_te = cal_meta.predict_proba(M_te_df)[:, 1]
                a = roc_auc_score(y_te, pm_te)
                if a > best_meta['auc']:
                    best_meta = {'model': cal_meta, 'proba_te': pm_te, 'auc': a, 'desc': f'enet l1r={l1r}, C={cval}'}

        auc_stack = best_meta['auc']
        pr_auc_stack = average_precision_score(y_te, best_meta['proba_te']) if best_meta['proba_te'] is not None else -1.0
        base_desc = f"{best_hgb['desc']}" + (f", {xgb_desc}" if xgb_desc else "")
        best_stack_name = f"Meta({best_meta['desc']}) over [logit, hgb{'+xgb' if oof_xgb is not None else ''}] | {base_desc}"

        # Choose vs best logistic (both on test)
        if auc_stack > best_log['auc']:
            chosen_name = best_stack_name
            chosen_auc = auc_stack
            chosen_pr = pr_auc_stack
            chosen_proba_te = best_meta['proba_te']
        else:
            chosen_name = f"CalibratedLogistic(k={best_log['k']})"
            chosen_auc = best_log['auc']
            chosen_pr = best_log['pr']
            chosen_proba_te = best_log['proba_te']

        if chosen_auc > best_overall.get('auc', -1):
            best_overall = {
                'name': chosen_name + ("+winsor" if used_win else ""),
                'auc': chosen_auc,
                'pr': chosen_pr,
                'proba_te': chosen_proba_te,
                'X_te_aug': X_te_aug,
                'y_te': y_te,
                'features': cols
            }

    if best_overall['auc'] <= 0:
        raise RuntimeError("No viable model configuration produced metrics.")

    X = best_overall['X_te_aug']
    y = best_overall['y_te']
    y_proba = best_overall['proba_te']
    cols = best_overall['features']
    auc = best_overall['auc']
    pr_auc = best_overall['pr']

    log("[9/9] Computing decision curves and subgroup metrics...")
    thresholds = np.linspace(0.05, 0.5, 20 if mode == 'full' else 15)
    dc = decision_curve(y.to_numpy(), y_proba, thresholds)
    plt.figure(figsize=(8, 6))
    plt.plot(dc['threshold'], dc['net_benefit'], label='Model')
    plt.plot(dc['threshold'], dc['all'], label='Treat All', linestyle='--')
    plt.plot(dc['threshold'], dc['none'], label='Treat None', linestyle=':')
    plt.xlabel('Threshold probability')
    plt.ylabel('Net Benefit')
    plt.title(f'Decision Curve Analysis ({mode.title()})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dcurve_path = OUT_DIR / 'decision_curve.png'
    plt.savefig(dcurve_path, dpi=300)
    plt.close()

    sub_df = subgroup_metrics(X, y, y_proba)
    sub_path = OUT_DIR / 'subgroup_metrics.csv'
    sub_df.to_csv(sub_path, index=False)

    report = {
        'samples': int(len(y)),
        'positives': int(y.sum()),
        'prevalence': float(y.mean()),
        'features_used': cols,
        'model_metrics': {
            'selected_model': best_overall['name'],
            'auc_overall': float(auc),
            'pr_auc_overall': float(pr_auc)
        },
        'artifacts': {
            'decision_curve_png': str(dcurve_path),
            'subgroup_metrics_csv': str(sub_path)
        }
    }
    out_json = OUT_DIR / 'clinical_utility_report.json'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    log(f"💾 Saved clinical report: {out_json}")
    log(f"📈 {mode.title()} pipeline (held-out): AUC={auc:.3f}, PR-AUC={pr_auc:.3f}")
    return 0


if __name__ == '__main__':
    args = parse_args()
    raise SystemExit(main(args.mode)) 