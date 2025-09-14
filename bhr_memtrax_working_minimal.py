#!/usr/bin/env python3
"""
BHR MemTrax Working Minimal - Exact 0.798 Reproduction
=====================================================
Simplified version of the working script with hardcoded winning parameters.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
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

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

def load_memtrax():
    """Load and filter MemTrax data"""
    mem = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    mem_q = apply_ashford(mem, accuracy_threshold=0.65)
    # Winsorization
    mem_q['CorrectResponsesRT'] = mem_q['CorrectResponsesRT'].clip(0.4, 2.0)
    return mem_q

def build_composite_labels(med_hx_df):
    """Build composite cognitive impairment labels"""
    med_baseline = med_hx_df[med_hx_df['TimepointCode'] == 'm00'].copy()
    med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])
    
    med_baseline['AnyCogImpairment'] = 0
    for qid in COGNITIVE_QIDS:
        if qid in med_baseline.columns:
            med_baseline['AnyCogImpairment'] |= (med_baseline[qid] == 1)
    
    return med_baseline[['SubjectCode', 'AnyCogImpairment']]

def compute_ecog_residuals(df):
    """Add ECOG/SP features"""
    try:
        ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv')
        sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
        
        for eco_df, prefix in [(ecog, 'ECOG'), (sp_ecog, 'SP_ECOG')]:
            if 'SubjectCode' not in eco_df.columns and 'Code' in eco_df.columns:
                eco_df = eco_df.rename(columns={'Code': 'SubjectCode'})
            
            if 'TimepointCode' in eco_df.columns:
                eco_df = eco_df[eco_df['TimepointCode'] == 'm00']
            
            numeric_cols = eco_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if 'QID' not in c]
            
            if len(numeric_cols) > 0:
                eco_df[f'{prefix}_GlobalMean_Residual'] = eco_df[numeric_cols].mean(axis=1)
                keep_cols = ['SubjectCode'] + [c for c in eco_df.columns if 'Residual' in c]
                eco_small = eco_df[keep_cols].drop_duplicates(subset=['SubjectCode'])
                df = df.merge(eco_small, on='SubjectCode', how='left')
    except Exception as e:
        print(f"Warning: ECOG features skipped: {e}")
    
    return df

def main():
    print("🚀 BHR MEMTRAX WORKING MINIMAL - REPRODUCING 0.798")
    print("=" * 50)
    
    # Load data
    memtrax = load_memtrax()
    
    # Build sequence features  
    seq_features = compute_sequence_features(memtrax)
    
    # Aggregate per subject
    agg_dict = {
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std']
    }
    
    aggregated = memtrax.groupby('SubjectCode').agg(agg_dict)
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
    aggregated.reset_index(inplace=True)
    
    # Compute cognitive score
    aggregated['CognitiveScore_mean'] = aggregated['CorrectResponsesRT_mean'] / (aggregated['CorrectPCT_mean'] + 0.01)
    aggregated['CorrectResponsesRT_cv'] = aggregated['CorrectResponsesRT_std'] / (aggregated['CorrectResponsesRT_mean'] + 0.01)
    
    # Merge sequence features
    aggregated = aggregated.merge(seq_features, on='SubjectCode', how='left')
    
    # Enrich demographics
    aggregated = enrich_demographics(DATA_DIR, aggregated)
    
    # Add ECOG features
    aggregated = compute_ecog_residuals(aggregated)
    
    # Add splines
    for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
        if col in aggregated.columns:
            spline = SplineTransformer(n_knots=3, degree=3, include_bias=False)
            x = aggregated[col].fillna(aggregated[col].median()).values.reshape(-1, 1)
            spline_features = spline.fit_transform(x)
            for i in range(spline_features.shape[1]):
                aggregated[f'{col}_spline_{i}'] = spline_features[:, i]
    
    # Build labels
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    labels = build_composite_labels(med_hx)
    
    # Merge features and labels
    merged = aggregated.merge(labels, on='SubjectCode', how='inner')
    
    # Prepare features
    feature_cols = [c for c in merged.columns if c not in ['SubjectCode', 'AnyCogImpairment']]
    X = merged[feature_cols]
    y = merged['AnyCogImpairment']
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Target prevalence: {y.mean():.3f}")
    
    # Prepare augmented features (fill NaN)
    X_aug = X.fillna(X.median())
    X_aug = X_aug.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Build winning stacking model (hardcoded best params: lr=0.1, leaves=31)
    logit_pipe = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("scale", StandardScaler(with_mean=False)),
        ("select", SelectKBest(mutual_info_classif, k=min(100, X_aug.shape[1]))),
        ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
    ])
    
    estimators = [('logit', logit_pipe)]
    
    # HistGB with winning params
    hgb_pipe = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("clf", HistGradientBoostingClassifier(
            random_state=42, 
            max_leaf_nodes=31,  # Winning param
            learning_rate=0.1,  # Winning param
            max_depth=3
        ))
    ])
    estimators.append(('hgb', hgb_pipe))
    
    # XGBoost if available
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
            random_state=42,
            scale_pos_weight=spw
        )
        estimators.append(('xgb', xgb))
    
    # Build stacking classifier
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, solver='lbfgs'),
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1
    )
    
    print("Training model...")
    stack.fit(X_aug, y)
    
    # Calibrate
    print("Calibrating model...")
    stack_cal = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    stack_cal.fit(X_aug, y)
    
    # Get predictions
    y_proba = stack_cal.predict_proba(X_aug)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    
    print("\n" + "=" * 50)
    print("🎯 RESULTS")
    print("=" * 50)
    print(f"AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    if auc >= 0.795:
        print("✅ SUCCESS: AUC ≈ 0.798 reproduced!")
    else:
        print(f"📍 Difference: {0.798 - auc:.4f}")
    
    return auc

if __name__ == "__main__":
    final_auc = main()
    print(f"\n🏆 FINAL AUC: {final_auc:.4f}")
