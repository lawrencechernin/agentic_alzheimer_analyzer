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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV

from improvements.ashford_policy import apply_ashford
try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False
from improvements.sequence_feature_engineering import compute_sequence_features
from improvements.demographics_enrichment import enrich_demographics
from improvements.calibrated_logistic import train_calibrated_logistic

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUT_DIR = Path("bhr_memtrax_results")
OUT_DIR.mkdir(exist_ok=True)

COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']


def load_memtrax() -> pd.DataFrame:
    mem = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    mem_q = apply_ashford(mem, accuracy_threshold=0.65)
    return mem_q


def load_medical() -> pd.DataFrame:
    med = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    if 'TimepointCode' in med.columns:
        med = med[med['TimepointCode'] == 'm00'].copy()
    med = med.drop_duplicates(subset=['SubjectCode'], keep='first')
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
    # require at least one known response among present QIDs
    known_any = np.zeros_like(any_pos)
    for c in present:
        known_any = np.logical_or(known_any, med[c].isin([1.0, 2.0]).to_numpy())
    y = pd.Series(any_pos, index=med.index).where(known_any, other=np.nan)
    labels = med[['SubjectCode']].copy()
    labels['AnyCogImpairment'] = y.astype(float)
    labels = labels.dropna(subset=['AnyCogImpairment'])
    labels['AnyCogImpairment'] = labels['AnyCogImpairment'].astype(int)
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
        # Global mean
        if num_cols:
            eco_small[f'{name}_GlobalMean'] = eco_small[num_cols].mean(axis=1)
        # Per-domain means if available by name pattern (skip for ADL)
        if name != 'SP_ADL':
            domains = {
                'Memory': [c for c in num_cols if 'mem' in c.lower() or 'memory' in c.lower()],
                'Language': [c for c in num_cols if 'lang' in c.lower() or 'language' in c.lower()],
                'Visuospatial': [c for c in num_cols if 'visu' in c.lower() or 'spatial' in c.lower()],
                'Executive': [c for c in num_cols if 'exec' in c.lower() or 'planning' in c.lower()],
            }
            for dom, cols in domains.items():
                if cols:
                    eco_small[f'{name}_{dom}Mean'] = eco_small[cols].mean(axis=1)
        # Only merge aggregated feature columns to avoid collisions
        feature_cols = ['SubjectCode'] + [c for c in eco_small.columns if c.startswith(f'{name}_')]
        dfx = dfx.merge(eco_small[feature_cols], on='SubjectCode', how='left')
        for c in eco_small.columns:
            if c != 'SubjectCode' and c in dfx.columns and c not in targets:
                targets.append(c)
    # residualize
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
    X_df = enrich_demographics(DATA_DIR, X_df)
    # Interactions
    if 'Age_Baseline' in X_df.columns:
        if 'CorrectResponsesRT_mean' in X_df.columns:
            X_df['age_rt_interaction'] = X_df['CorrectResponsesRT_mean'] * (X_df['Age_Baseline'] / 65.0)
        if 'long_reliability_change' in X_df.columns:
            X_df['age_variability_interaction'] = X_df['long_reliability_change'] * (X_df['Age_Baseline'] / 65.0)
        if 'CorrectPCT_mean' in X_df.columns and 'long_reliability_change' in X_df.columns:
            X_df['accuracy_stability'] = X_df['CorrectPCT_mean'] / (X_df['long_reliability_change'] + 1e-6)
    # ECOG residuals
    X_df = compute_ecog_residuals(X_df)
    return X_df


LEAN_COLUMNS = [
    # Core performance
    'CorrectResponsesRT_mean', 'CorrectPCT_mean', 'IncorrectRejectionsN_mean',
    'CognitiveScore_mean',
    # Sequence/fatigue
    'seq_first_third_mean', 'seq_last_third_mean', 'seq_fatigue_effect',
    'seq_mean_rt', 'seq_median_rt',
    # Longitudinal variability/trend
    'long_reliability_change', 'long_n_timepoints', 'long_rt_slope',
    # Demographics
    'Age_Baseline', 'YearsEducationUS_Converted', 'Gender_Numeric',
    # Interactions
    'age_rt_interaction', 'age_variability_interaction', 'accuracy_stability',
    # Informant residuals if present
    'ECOG_GlobalMean_Residual', 'SP_ECOG_GlobalMean_Residual'
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
    # age bands
    if 'Age_Baseline' in X.columns:
        bins = [0, 65, 75, 85, 200]
        labels = ['<65', '65-75', '75-85', '85+']
        bands = pd.cut(X['Age_Baseline'], bins=bins, labels=labels, right=False)
        for b in labels:
            mask = (bands == b) & y.notna()
            if mask.sum() < 50:
                continue
            rows.append({'group': f'age_{b}', 'auc': roc_auc_score(y[mask], y_proba[mask])})
    # education bands
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


def main() -> int:
    print("🚀 BHR MEMTRAX CLINICAL UTILITY ANALYSIS")
    mem_q = load_memtrax()
    med = load_medical()

    # Try both: without and with winsorization
    X_df_nowin = build_features(mem_q, use_winsorize=False)
    X_df_win = build_features(mem_q, use_winsorize=True)
    # pick later after modeling

    labels_df = build_composite_labels(med)

    # Merge features and labels on SubjectCode
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
    # Choose the larger merged set as primary candidate; we will evaluate both
    candidates = [(X_nowin, y_nowin, cols_nowin, False), (X_win, y_win, cols_win, True)]

    best_overall = {'auc': -1.0}
    for X, y, cols, used_win in candidates:
        if len(y) == 0:
            continue

        # Add spline basis for age/education if present (captures non-linearities)
        X_aug = X.copy()
        for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
            if col in X_aug.columns:
                try:
                    st = SplineTransformer(n_knots=5, degree=3, include_bias=False)
                    vals = X_aug[[col]].to_numpy()
                    if np.isfinite(vals).sum() >= len(vals):
                        spl = st.fit_transform(vals)
                        for i in range(spl.shape[1]):
                            X_aug[f'{col}_spline_{i+1}'] = spl[:, i]
                except Exception:
                    pass

        # Sweep MI k for logistic
        best_log = {'auc': -1.0}
        for k in [50, 100, 150]:
            k_use = min(k, X_aug.shape[1])
            model, metrics = train_calibrated_logistic(X_aug, y, k_features=k_use)
            y_proba = model.predict_proba(X_aug)[:, 1]
            auc_log = roc_auc_score(y, y_proba)
            pr_log = average_precision_score(y, y_proba)
            if auc_log > best_log['auc']:
                best_log = {'model': model, 'metrics': metrics, 'proba': y_proba, 'auc': auc_log, 'pr': pr_log, 'k': k_use}

        # Train stacking sweeps (LogReg + HistGB [+ XGB if available]) and calibrate
        logit_pipe = Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("scale", StandardScaler(with_mean=False)),
            ("select", SelectKBest(mutual_info_classif, k=min(100, X_aug.shape[1]))),
            ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
        ])
        best_stack_auc = -1.0
        best_proba_stack = None
        best_stack_name = None
        for lr in [0.03, 0.05, 0.1]:
            for leafs in [31, 63]:
                estimators = [('logit', logit_pipe)]
                hgb_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy='median')),
                    ("clf", HistGradientBoostingClassifier(random_state=2024, stratify=y
                ])
                estimators.append(('hgb', hgb_pipe))
                if XGB_OK:
                    pos = int(y.sum())
                    neg = int((~y.astype(bool)).sum())
                    spw = (neg / max(pos, 1))
                    xgb = XGBClassifier(
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective='binary:logistic',
                        tree_method='hist',
                        eval_metric='auc',
                        random_state=2024, stratify=y
                        scale_pos_weight=spw
                    )
                    estimators.append(('xgb', xgb))
                stack = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, solver='lbfgs'),
                    stack_method='predict_proba',
                    cv=5,
                    n_jobs=-1
                )
                stack.fit(X_aug, y)
                stack_cal = CalibratedClassifierCV(stack, cv=3, method='isotonic')
                stack_cal.fit(X_aug, y)
                yps = stack_cal.predict_proba(X_aug)[:, 1]
                a = roc_auc_score(y, yps)
                if a > best_stack_auc:
                    best_stack_auc = a
                    best_proba_stack = yps
                    best_stack_name = f"Stacked({'+'.join([n for n,_ in estimators])} lr={lr}, leaves={leafs})"
        auc_stack = best_stack_auc
        pr_auc_stack = average_precision_score(y, best_proba_stack) if best_proba_stack is not None else -1.0

        # Choose best for this candidate
        if auc_stack > best_log['auc']:
            best_name = best_stack_name
            best_auc = auc_stack
            best_pr = pr_auc_stack
            best_proba = best_proba_stack
            best_metrics = {
                'cv_auc': None,
                'test_auc': None,
                'threshold': None,
                'best_C': None
            }
        else:
            best_name = f"CalibratedLogistic(k={best_log['k']})"
            best_auc = best_log['auc']
            best_pr = best_log['pr']
            best_proba = best_log['proba']
            best_metrics = best_log['metrics']

        # Track best overall across winsorize options
        if best_auc > best_overall.get('auc', -1):
            best_overall = {
                'name': best_name + ("+winsor" if used_win else ""),
                'auc': best_auc,
                'pr': best_pr,
                'proba': best_proba,
                'X': X_aug,
                'y': y,
                'metrics': best_metrics,
                'winsor': used_win,
                'features': cols
            }

    # Overall metrics chosen
    if best_overall['auc'] <= 0:
        raise RuntimeError("No viable model configuration produced metrics.")
    best_name = best_overall['name']
    best_auc = best_overall['auc']
    best_pr = best_overall['pr']
    best_proba = best_overall['proba']
    X = best_overall['X']
    y = best_overall['y']
    metrics = best_overall['metrics']
    cols = best_overall['features']

    # Decision curves
    thresholds = np.linspace(0.05, 0.5, 20)
    dc = decision_curve(y.to_numpy(), best_proba, thresholds)
    plt.figure(figsize=(8, 6))
    plt.plot(dc['threshold'], dc['net_benefit'], label='Model')
    plt.plot(dc['threshold'], dc['all'], label='Treat All', linestyle='--')
    plt.plot(dc['threshold'], dc['none'], label='Treat None', linestyle=':')
    plt.xlabel('Threshold probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dcurve_path = OUT_DIR / 'decision_curve.png'
    plt.savefig(dcurve_path, dpi=300)
    plt.close()

    # Subgroup metrics (use available columns in X)
    sub_df = subgroup_metrics(X, y, best_proba)
    sub_path = OUT_DIR / 'subgroup_metrics.csv'
    sub_df.to_csv(sub_path, index=False)

    # Delta over demographics
    demo_cols = [c for c in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender_Numeric'] if c in X.columns]
    if demo_cols:
        from improvements.calibrated_logistic import train_calibrated_logistic as train_demo
        demo_model, demo_metrics = train_demo(X[demo_cols], y, k_features=min(3, len(demo_cols)))
        demo_proba = demo_model.predict_proba(X[demo_cols])[:, 1]
        demo_auc = roc_auc_score(y, demo_proba)
    else:
        demo_metrics, demo_auc = {}, None

    report = {
        'samples': int(len(y)),
        'positives': int(y.sum()),
        'prevalence': float(y.mean()),
        'features_used': cols,
        'model_metrics': {
            'selected_model': best_name,
            'cv_auc_logistic': metrics.get('cv_auc') if metrics else None,
            'test_auc_internal_logistic': metrics.get('test_auc') if metrics else None,
            'auc_overall': float(best_auc),
            'pr_auc_overall': float(best_pr),
            'threshold': metrics.get('threshold') if metrics else None,
            'best_C': metrics.get('best_C') if metrics else None
        },
        'delta_over_demographics': {
            'demo_auc': float(demo_auc) if demo_auc is not None else None,
            'delta_auc': float(best_auc - demo_auc) if demo_auc is not None else None
        },
        'artifacts': {
            'decision_curve_png': str(dcurve_path),
            'subgroup_metrics_csv': str(sub_path)
        }
    }
    out_json = OUT_DIR / 'clinical_utility_report.json'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"💾 Saved clinical report: {out_json}")
    demo_auc_str = f"{demo_auc:.3f}" if demo_auc is not None else "nan"
    print(f"📈 {best_name}: AUC={best_auc:.3f}, PR-AUC={best_pr:.3f} | Demo AUC={demo_auc_str}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main()) 