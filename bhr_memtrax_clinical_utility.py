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

from improvements.ashford_policy import apply_ashford
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


def build_composite_labels(med: pd.DataFrame) -> pd.Series:
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
    y = pd.Series(any_pos, index=med.index).where(known_any, other=np.nan).dropna().astype(int)
    return y


def compute_ecog_residuals(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    dfx = df.copy()
    targets = []
    for name, csv in [('ECOG', 'BHR_EverydayCognition.csv'), ('SP_ECOG', 'BHR_SP_ECog.csv')]:
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
        eco_small[f'{name}_GlobalMean'] = eco_small[num_cols].mean(axis=1) if num_cols else np.nan
        dfx = dfx.merge(eco_small[['SubjectCode', f'{name}_GlobalMean']], on='SubjectCode', how='left')
        targets.append(f'{name}_GlobalMean')
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
                    pred = lr.predict(X_demo)
                    dfx[f'{t}_Residual'] = y - pred
    return dfx


def build_features(mem_q: pd.DataFrame) -> pd.DataFrame:
    # Sequence features
    seq = compute_sequence_features(mem_q)
    # Aggregate numeric means per subject
    agg = mem_q.groupby('SubjectCode').mean(numeric_only=True).reset_index()
    # Cognitive score ratio mean
    if 'CorrectResponsesRT' in mem_q.columns and 'CorrectPCT' in mem_q.columns:
        mem_q['CognitiveScore'] = mem_q['CorrectResponsesRT'] / (mem_q['CorrectPCT'] + 0.01)
        cg = mem_q.groupby('SubjectCode')['CognitiveScore'].mean().rename('CognitiveScore_mean').reset_index()
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

    X_df = build_features(mem_q)
    med_y = build_composite_labels(med)

    # Merge labels
    xy = X_df.merge(med[['SubjectCode']], on='SubjectCode', how='inner')
    xy = xy.set_index('SubjectCode')
    med_y = med_y.loc[xy.index.intersection(med_y.index)]
    xy = xy.loc[med_y.index]

    # Prepare X (lean numeric columns present)
    cols = [c for c in LEAN_COLUMNS if c in xy.columns]
    X = xy[cols].apply(pd.to_numeric, errors='coerce')
    # Drop all-NaN columns
    X = X.loc[:, X.notna().mean() > 0]
    y = med_y.astype(int)

    # Train calibrated logistic
    model, metrics = train_calibrated_logistic(X, y, k_features=min(50, X.shape[1]))
    y_proba = model.predict_proba(X)[:, 1]

    # Overall metrics
    auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)

    # Decision curves
    thresholds = np.linspace(0.05, 0.5, 20)
    dc = decision_curve(y.to_numpy(), y_proba, thresholds)
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

    # Subgroup metrics
    sub_df = subgroup_metrics(X.assign(Age_Baseline=xy.get('Age_Baseline'), YearsEducationUS_Converted=xy.get('YearsEducationUS_Converted')),
                              y, y_proba)
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
            'cv_auc': metrics.get('cv_auc'),
            'test_auc_internal': metrics.get('test_auc'),
            'auc_overall': float(auc),
            'pr_auc_overall': float(pr_auc),
            'threshold': metrics.get('threshold'),
            'best_C': metrics.get('best_C')
        },
        'delta_over_demographics': {
            'demo_auc': float(demo_auc) if demo_auc is not None else None,
            'delta_auc': float(auc - demo_auc) if demo_auc is not None else None
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
    print(f"📈 AUC={auc:.3f}, PR-AUC={pr_auc:.3f} | Demo AUC={demo_auc:.3f if demo_auc is not None else float('nan')}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main()) 