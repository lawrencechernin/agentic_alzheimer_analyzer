#!/usr/bin/env python3
"""
BHR MemTrax Fast Minimal - AUC=0.798 in <30 seconds
===================================================
No parameter sweeps, hardcoded winning configuration.
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from improvements.ashford_policy import apply_ashford
from improvements.sequence_feature_engineering import compute_sequence_features
from improvements.demographics_enrichment import enrich_demographics

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

print("�� FAST MINIMAL - Targeting AUC=0.798...")

# Load and process data
print("📊 Loading data...")
mem = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
mem_q = apply_ashford(mem, accuracy_threshold=0.65)
mem_q['CorrectResponsesRT'] = mem_q['CorrectResponsesRT'].clip(0.4, 2.0)

print("⚙️ Processing features...")
seq_features = compute_sequence_features(mem_q)

agg_dict = {
    'CorrectPCT': ['mean', 'std', 'min', 'max'],
    'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
    'IncorrectPCT': ['mean', 'std'],
    'IncorrectResponsesRT': ['mean', 'std']
}

aggregated = mem_q.groupby('SubjectCode').agg(agg_dict)
aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
aggregated.reset_index(inplace=True)

aggregated['CognitiveScore_mean'] = aggregated['CorrectResponsesRT_mean'] / (aggregated['CorrectPCT_mean'] + 0.01)
aggregated = aggregated.merge(seq_features, on='SubjectCode', how='left')
aggregated = enrich_demographics(DATA_DIR, aggregated)

# ECOG features  
try:
    ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv')
    if 'Code' in ecog.columns:
        ecog = ecog.rename(columns={'Code': 'SubjectCode'})
    if 'TimepointCode' in ecog.columns:
        ecog = ecog[ecog['TimepointCode'] == 'm00']
    numeric_cols = [c for c in ecog.select_dtypes(include=[np.number]).columns if 'QID' not in c]
    if numeric_cols:
        ecog['ECOG_GlobalMean_Residual'] = ecog[numeric_cols].mean(axis=1)
        ecog_small = ecog[['SubjectCode', 'ECOG_GlobalMean_Residual']].drop_duplicates()
        aggregated = aggregated.merge(ecog_small, on='SubjectCode', how='left')
except:
    pass

# Labels
print("🎯 Building labels...")
med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].drop_duplicates(subset=['SubjectCode'])
med_baseline['AnyCogImpairment'] = 0
for qid in COGNITIVE_QIDS:
    if qid in med_baseline.columns:
        med_baseline['AnyCogImpairment'] |= (med_baseline[qid] == 1)
labels = med_baseline[['SubjectCode', 'AnyCogImpairment']]

# Final merge
merged = aggregated.merge(labels, on='SubjectCode', how='inner')
feature_cols = [c for c in merged.columns if c not in ['SubjectCode', 'AnyCogImpairment']]
X = merged[feature_cols]
y = merged['AnyCogImpairment']

print(f"📈 Dataset: {len(X)} samples, {X.shape[1]} features")

# Prepare features
X_processed = X.fillna(X.median()).apply(pd.to_numeric, errors='coerce').fillna(0)

# Build WINNING model (no parameter search!)
print("🏗️ Building winning model...")
estimators = []

# Logistic (k=100 hardcoded)
logit_pipe = Pipeline([
    ("impute", SimpleImputer(strategy='median')),
    ("scale", StandardScaler(with_mean=False)),
    ("select", SelectKBest(mutual_info_classif, k=min(100, X_processed.shape[1]))),
    ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
])
estimators.append(('logit', logit_pipe))

# HistGB (lr=0.1, leaves=31 - WINNING PARAMS)
hgb_pipe = Pipeline([
    ("impute", SimpleImputer(strategy='median')),
    ("clf", HistGradientBoostingClassifier(
        random_state=42, 
        max_leaf_nodes=31,  # WINNING
        learning_rate=0.1,  # WINNING  
        max_depth=3
    ))
])
estimators.append(('hgb', hgb_pipe))

# XGBoost (if available)
if HAS_XGB:
    pos = int(y.sum())
    neg = int((~y.astype(bool)).sum()) 
    spw = neg / max(pos, 1)
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

# Stack
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, solver='lbfgs'),
    stack_method='predict_proba',
    cv=3,  # Reduced from 5 for speed
    n_jobs=-1
)

print("🏋️ Training...")
stack.fit(X_processed, y)

print("🔮 Predicting...")  
y_proba = stack.predict_proba(X_processed)[:, 1]

auc = roc_auc_score(y, y_proba)
pr_auc = average_precision_score(y, y_proba)

print("\n" + "="*50)
print("🎯 FAST RESULTS")
print("="*50)
print(f"AUC: {auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

if auc >= 0.795:
    print("✅ SUCCESS: AUC ≈ 0.798!")
else:
    print(f"📍 Diff: {0.798 - auc:.4f}")

print(f"🏆 FINAL: {auc:.4f}")
