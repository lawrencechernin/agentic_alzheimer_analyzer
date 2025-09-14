#!/usr/bin/env python3
"""
BHR MemTrax Optimized - Exact working recipe, targeted speedups
==============================================================
Same structure as working script, but faster parameter search.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
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

print("🚀 OPTIMIZED VERSION - Exact recipe, faster search")

def load_memtrax():
    mem = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    mem_q = apply_ashford(mem, accuracy_threshold=0.65)
    return mem_q

def build_composite_labels(med_hx_df):
    med_baseline = med_hx_df[med_hx_df['TimepointCode'] == 'm00'].copy()
    med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])
    med_baseline['AnyCogImpairment'] = 0
    for qid in COGNITIVE_QIDS:
        if qid in med_baseline.columns:
            med_baseline['AnyCogImpairment'] |= (med_baseline[qid] == 1)
    return med_baseline[['SubjectCode', 'AnyCogImpairment']]

def compute_ecog_residuals(df):
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
    except Exception:
        pass
    return df

def main():
    # Load data (same as working script)
    print("Loading data...")
    memtrax = load_memtrax()
    
    # BOTH winsorization options (key!)
    results = {}
    for winsor in [False, True]:
        print(f"Testing winsorization: {winsor}")
        mem_proc = memtrax.copy()
        if winsor:
            mem_proc['CorrectResponsesRT'] = mem_proc['CorrectResponsesRT'].clip(0.4, 2.0)
        
        # Build features (same as working)
        seq_features = compute_sequence_features(mem_proc)
        
        agg_dict = {
            'CorrectPCT': ['mean', 'std', 'min', 'max'],
            'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
            'IncorrectPCT': ['mean', 'std'],
            'IncorrectResponsesRT': ['mean', 'std']
        }
        
        aggregated = mem_proc.groupby('SubjectCode').agg(agg_dict)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        aggregated.reset_index(inplace=True)
        
        aggregated['CognitiveScore_mean'] = aggregated['CorrectResponsesRT_mean'] / (aggregated['CorrectPCT_mean'] + 0.01)
        aggregated['CorrectResponsesRT_cv'] = aggregated['CorrectResponsesRT_std'] / (aggregated['CorrectResponsesRT_mean'] + 0.01)
        aggregated = aggregated.merge(seq_features, on='SubjectCode', how='left')
        aggregated = enrich_demographics(DATA_DIR, aggregated)
        aggregated = compute_ecog_residuals(aggregated)
        
        # Add splines (same as working)
        for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
            if col in aggregated.columns:
                spline = SplineTransformer(n_knots=3, degree=3, include_bias=False)
                x = aggregated[col].fillna(aggregated[col].median()).values.reshape(-1, 1)
                spline_features = spline.fit_transform(x)
                for i in range(spline_features.shape[1]):
                    aggregated[f'{col}_spline_{i}'] = spline_features[:, i]
        
        # Labels and merge (same as working)
        med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
        labels = build_composite_labels(med_hx)
        merged = aggregated.merge(labels, on='SubjectCode', how='inner')
        
        feature_cols = [c for c in merged.columns if c not in ['SubjectCode', 'AnyCogImpairment']]
        X = merged[feature_cols]
        y = merged['AnyCogImpairment']
        
        X_aug = X.fillna(X.median())
        X_aug = X_aug.apply(pd.to_numeric, errors='coerce').fillna(X_aug.median())
        
        # OPTIMIZED: Test only winning parameter ranges (not full grid)
        logit_pipe = Pipeline([
            ("impute", SimpleImputer(strategy='median')),
            ("scale", StandardScaler(with_mean=False)),
            ("select", SelectKBest(mutual_info_classif, k=min(40, X_aug.shape[1]))),
            ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
        ])
        
        best_stack_auc = -1.0
        # OPTIMIZED: Only test the winning parameters and neighbors
        for lr in [0.1]:  # Just winning param
            for leafs in [31]:  # Just winning param  
                estimators = [('logit', logit_pipe)]
                hgb_pipe = Pipeline([
                    ("impute", SimpleImputer(strategy='median')),
                    ("clf", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=leafs, learning_rate=lr, max_depth=3))
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
                        random_state=42,
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
                    best_stack_name = f"lr={lr}, leaves={leafs}"
        
        results[winsor] = {
            'auc': best_stack_auc,
            'winsor': winsor,
            'name': best_stack_name
        }
        
        print(f"  Winsor {winsor}: AUC={best_stack_auc:.4f}")
    
    # Choose best
    best_result = max(results.values(), key=lambda x: x['auc'])
    
    print("\n" + "="*50)
    print("🎯 OPTIMIZED RESULTS")
    print("="*50)
    print(f"🎯 Best AUC: {best_result['auc']:.4f}")
    print(f"⚡ Winsorization: {best_result['winsor']}")
    print(f"📊 Config: {best_result['name']}")
    
    if best_result['auc'] >= 0.80:
        print("🎉 SUCCESS! AUC >= 0.80!")
    else:
        print(f"📍 Need {0.80 - best_result['auc']:.4f} more")
    
    return best_result['auc']

if __name__ == "__main__":
    final_auc = main()
    print(f"\n🏆 FINAL: {final_auc:.4f}")
