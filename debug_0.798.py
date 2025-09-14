#!/usr/bin/env python3
"""
BHR MemTrax Clinical Utility Analysis - DEBUGGING VERSION
Added proper train/test splits, stability analysis, and data leakage checks
"""
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Mock imports for missing modules
class MockApplyAshford:
    def __call__(self, mem, accuracy_threshold=0.65):
        return mem

class MockComputeSequenceFeatures:
    def __call__(self, mem):
        # Return basic sequence features
        result = mem.groupby('SubjectCode').agg({
            'SubjectCode': 'first'
        }).reset_index(drop=True)
        
        # Add mock sequence features
        n_subjects = len(result)
        np.random.seed(42)  # For reproducibility
        result['seq_first_third_mean'] = np.random.normal(1.2, 0.3, n_subjects)
        result['seq_last_third_mean'] = np.random.normal(1.3, 0.3, n_subjects)
        result['seq_fatigue_effect'] = result['seq_last_third_mean'] - result['seq_first_third_mean']
        result['seq_mean_rt'] = np.random.normal(1.25, 0.25, n_subjects)
        result['seq_median_rt'] = np.random.normal(1.23, 0.25, n_subjects)
        result['long_reliability_change'] = np.random.normal(0.1, 0.05, n_subjects)
        result['long_n_timepoints'] = np.random.randint(2, 8, n_subjects)
        result['long_rt_slope'] = np.random.normal(0.02, 0.1, n_subjects)
        
        return result

class MockEnrichDemographics:
    def __call__(self, data_dir, X_df):
        # Add mock demographic features
        n_subjects = len(X_df)
        np.random.seed(42)
        X_df['Age_Baseline'] = np.random.normal(70, 12, n_subjects)
        X_df['YearsEducationUS_Converted'] = np.random.normal(14, 3, n_subjects)
        X_df['Gender_Numeric'] = np.random.randint(0, 2, n_subjects)
        return X_df

class MockTrainCalibratedLogistic:
    def __call__(self, X, y, k_features=50):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', SelectKBest(f_classif, k=min(k_features, X.shape[1]))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        pipe.fit(X, y)
        
        # Mock cross-validation scores
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
        
        metrics = {
            'cv_auc': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'test_auc': None,
            'threshold': 0.5,
            'best_C': 1.0
        }
        
        return pipe, metrics

# Initialize mock functions
apply_ashford = MockApplyAshford()
compute_sequence_features = MockComputeSequenceFeatures()
enrich_demographics = MockEnrichDemographics()
train_calibrated_logistic = MockTrainCalibratedLogistic()

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUT_DIR = Path("bhr_memtrax_results")
OUT_DIR.mkdir(exist_ok=True)

COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

def load_memtrax() -> pd.DataFrame:
    """Load or generate mock MemTrax data"""
    # Generate realistic mock data
    np.random.seed(42)
    n_samples = 2000
    subjects = [f"SUBJ_{i:04d}" for i in range(1, 401)]  # 400 unique subjects
    
    data = []
    for subj in subjects:
        n_sessions = np.random.randint(3, 12)  # 3-11 sessions per subject
        for session in range(n_sessions):
            data.append({
                'SubjectCode': subj,
                'SessionNumber': session + 1,
                'CorrectResponsesRT': np.random.lognormal(0.1, 0.3),  # ~1.1 seconds
                'CorrectPCT': np.random.beta(8, 2),  # ~80% accuracy
                'IncorrectRejectionsN': np.random.poisson(2),
                'ReactionTimes': ','.join([f"{np.random.lognormal(0.1, 0.3):.3f}" for _ in range(20)])
            })
    
    mem = pd.DataFrame(data)
    print(f"📊 Generated {len(mem)} MemTrax records for {len(subjects)} subjects")
    return apply_ashford(mem)

def load_medical() -> pd.DataFrame:
    """Load or generate mock medical data"""
    np.random.seed(43)
    subjects = [f"SUBJ_{i:04d}" for i in range(1, 401)]
    
    data = []
    for subj in subjects:
        # Generate correlated cognitive impairment indicators
        base_impairment = np.random.random() < 0.3  # 30% base rate
        
        record = {'SubjectCode': subj}
        for qid in COGNITIVE_QIDS:
            if base_impairment:
                record[qid] = np.random.choice([1, 2], p=[0.7, 0.3])  # 1=yes, 2=no
            else:
                record[qid] = np.random.choice([1, 2], p=[0.1, 0.9])
        
        data.append(record)
    
    med = pd.DataFrame(data)
    print(f"📊 Generated medical records for {len(med)} subjects")
    return med

def analyze_data_stability(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Analyze how stable the results are with different data splits"""
    print("\n🔍 STABILITY ANALYSIS")
    print("=" * 50)
    
    aucs = []
    feature_importances = []
    
    # Multiple random splits
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i, stratify=y
        )
        
        # Simple logistic regression
        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', SelectKBest(f_classif, k=min(20, X.shape[1]))),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)
        
        # Get feature names after selection
        selected_features = pipe.named_steps['select'].get_feature_names_out()
        coefficients = pipe.named_steps['clf'].coef_[0]
        
        feature_imp = dict(zip(selected_features, np.abs(coefficients)))
        feature_importances.append(feature_imp)
    
    stability_metrics = {
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'auc_min': float(np.min(aucs)),
        'auc_max': float(np.max(aucs)),
        'auc_range': float(np.max(aucs) - np.min(aucs)),
        'individual_aucs': aucs
    }
    
    print(f"AUC Mean: {stability_metrics['auc_mean']:.3f}")
    print(f"AUC Std:  {stability_metrics['auc_std']:.3f}")
    print(f"AUC Range: {stability_metrics['auc_range']:.3f} ({stability_metrics['auc_min']:.3f} - {stability_metrics['auc_max']:.3f})")
    
    # Feature stability analysis
    all_features = set()
    for fi in feature_importances:
        all_features.update(fi.keys())
    
    feature_stability = {}
    for feat in all_features:
        selections = sum(1 for fi in feature_importances if feat in fi)
        importances = [fi.get(feat, 0) for fi in feature_importances]
        feature_stability[feat] = {
            'selection_rate': selections / len(feature_importances),
            'importance_mean': float(np.mean(importances)),
            'importance_std': float(np.std(importances))
        }
    
    # Sort by selection rate
    stable_features = sorted(feature_stability.items(), 
                           key=lambda x: x[1]['selection_rate'], reverse=True)
    
    print(f"\n🎯 MOST STABLE FEATURES:")
    for feat, stats in stable_features[:10]:
        print(f"  {feat}: {stats['selection_rate']:.1%} selection, "
              f"importance {stats['importance_mean']:.3f}±{stats['importance_std']:.3f}")
    
    return stability_metrics, feature_stability

def check_data_leakage(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Check for potential data leakage issues"""
    print("\n🚨 DATA LEAKAGE ANALYSIS")
    print("=" * 50)
    
    leakage_indicators = {}
    
    # Check for suspiciously high correlations with target
    correlations = {}
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            corr = np.corrcoef(X[col].fillna(X[col].median()), y)[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
    
    high_corr_features = {k: v for k, v in correlations.items() if v > 0.5}
    
    print(f"📊 Features with |correlation| > 0.5:")
    for feat, corr in sorted(high_corr_features.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {corr:.3f}")
    
    # Check for features that perfectly separate classes
    perfect_separation = []
    for col in X.select_dtypes(include=[np.number]).columns:
        try:
            pos_vals = X.loc[y == 1, col].dropna()
            neg_vals = X.loc[y == 0, col].dropna()
            
            if len(pos_vals) > 0 and len(neg_vals) > 0:
                if pos_vals.max() <= neg_vals.min() or neg_vals.max() <= pos_vals.min():
                    perfect_separation.append(col)
        except Exception:
            continue
    
    if perfect_separation:
        print(f"⚠️  Features with perfect class separation: {perfect_separation}")
    
    # Check sample size adequacy
    n_samples = len(y)
    n_features = X.shape[1]
    n_positive = y.sum()
    
    print(f"\n📏 SAMPLE SIZE ANALYSIS:")
    print(f"  Total samples: {n_samples}")
    print(f"  Positive cases: {n_positive} ({n_positive/n_samples:.1%})")
    print(f"  Features: {n_features}")
    print(f"  Samples per feature: {n_samples/n_features:.1f}")
    print(f"  Events per variable (EPV): {n_positive/n_features:.1f}")
    
    adequacy_warnings = []
    if n_samples / n_features < 10:
        adequacy_warnings.append("Low samples-to-features ratio (< 10)")
    if n_positive / n_features < 10:
        adequacy_warnings.append("Low events-per-variable (< 10)")
    if n_positive < 100:
        adequacy_warnings.append("Low total positive cases (< 100)")
    
    if adequacy_warnings:
        print(f"⚠️  Adequacy warnings: {'; '.join(adequacy_warnings)}")
    
    return {
        'high_correlations': high_corr_features,
        'perfect_separation': perfect_separation,
        'sample_adequacy': {
            'n_samples': n_samples,
            'n_features': n_features,
            'n_positive': int(n_positive),
            'samples_per_feature': n_samples / n_features,
            'events_per_variable': n_positive / n_features
        },
        'warnings': adequacy_warnings
    }

def proper_train_test_evaluation(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Proper evaluation with held-out test set"""
    print("\n✅ PROPER TRAIN/TEST EVALUATION")
    print("=" * 50)
    
    # Single stratified split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(y_train)} samples ({y_train.sum()} positive)")
    print(f"Test set: {len(y_test)} samples ({y_test.sum()} positive)")
    
    # Simple logistic regression with proper CV on training set only
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectKBest(f_classif, k=min(20, X.shape[1]))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    # Cross-validation on training set
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Fit on full training set and evaluate on test set
    pipe.fit(X_train, y_train)
    
    # Test set predictions
    y_train_pred = pipe.predict_proba(X_train)[:, 1]
    y_test_pred = pipe.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    results = {
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores)),
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'overfit_gap': float(train_auc - test_auc),
        'cv_scores': cv_scores.tolist()
    }
    
    print(f"CV AUC: {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f}")
    print(f"Train AUC: {results['train_auc']:.3f}")
    print(f"Test AUC: {results['test_auc']:.3f}")
    print(f"Overfitting gap: {results['overfit_gap']:.3f}")
    
    if results['overfit_gap'] > 0.05:
        print("⚠️  Significant overfitting detected!")
    
    return results

# [Include the original data processing functions with modifications]
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

def build_features(mem_q: pd.DataFrame, use_winsorize: bool = False) -> pd.DataFrame:
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
        if 'CorrectResponsesRT' in X_df.columns:
            X_df['age_rt_interaction'] = X_df['CorrectResponsesRT'] * (X_df['Age_Baseline'] / 65.0)
        if 'long_reliability_change' in X_df.columns:
            X_df['age_variability_interaction'] = X_df['long_reliability_change'] * (X_df['Age_Baseline'] / 65.0)
        if 'CorrectPCT' in X_df.columns and 'long_reliability_change' in X_df.columns:
            X_df['accuracy_stability'] = X_df['CorrectPCT'] / (X_df['long_reliability_change'] + 1e-6)
    return X_df

LEAN_COLUMNS = [
    # Core performance
    'CorrectResponsesRT', 'CorrectPCT', 'IncorrectRejectionsN',
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
]

def main() -> int:
    print("🚀 BHR MEMTRAX CLINICAL UTILITY ANALYSIS - DEBUG VERSION")
    print("=" * 60)
    
    # Load data
    mem_q = load_memtrax()
    med = load_medical()
    
    # Build features and labels
    X_df = build_features(mem_q, use_winsorize=False)
    labels_df = build_composite_labels(med)
    
    # Merge features and labels
    xy = X_df.merge(labels_df, on='SubjectCode', how='inner')
    cols = [c for c in LEAN_COLUMNS if c in xy.columns]
    X = xy[cols].apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, X.notna().mean() > 0.5]  # Keep features with >50% non-missing
    y = xy['AnyCogImpairment'].astype(int)
    
    # Remove rows with too many missing features
    row_valid = X.notna().sum(axis=1) >= (X.shape[1] * 0.5)  # At least 50% features present
    X = X.loc[row_valid]
    y = y.loc[row_valid]
    
    print(f"\n📊 FINAL DATASET:")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Positive cases: {y.sum()} ({y.mean():.1%})")
    print(f"Features used: {list(X.columns)}")
    
    # Run debugging analyses
    stability_metrics, feature_stability = analyze_data_stability(X, y)
    leakage_analysis = check_data_leakage(X, y)
    proper_results = proper_train_test_evaluation(X, y)
    
    # Compile debugging report
    debug_report = {
        'dataset_info': {
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_positive': int(y.sum()),
            'prevalence': float(y.mean()),
            'features_used': list(X.columns)
        },
        'stability_analysis': stability_metrics,
        'feature_stability': {k: v for k, v in sorted(feature_stability.items(), 
                                                     key=lambda x: x[1]['selection_rate'], 
                                                     reverse=True)[:20]},
        'leakage_analysis': leakage_analysis,
        'proper_evaluation': proper_results
    }
    
    # Save debug report
    debug_path = OUT_DIR / 'debug_analysis.json'
    with open(debug_path, 'w') as f:
        json.dump(debug_report, f, indent=2)
    
    print(f"\n💾 Debug report saved: {debug_path}")
    
    print("\n🎯 KEY INSIGHTS:")
    print(f"• Stability: AUC varies {stability_metrics['auc_range']:.3f} across splits")
    print(f"• Proper test AUC: {proper_results['test_auc']:.3f} (vs CV: {proper_results['cv_auc_mean']:.3f})")
    print(f"• Overfitting gap: {proper_results['overfit_gap']:.3f}")
    
    if proper_results['overfit_gap'] > 0.05:
        print("⚠️  Model appears to be overfitting!")
    
    if stability_metrics['auc_std'] > 0.05:
        print("⚠️  Results are unstable across different data splits!")
    
    if leakage_analysis['sample_adequacy']['events_per_variable'] < 10:
        print("⚠️  Low events-per-variable - consider reducing features!")
    
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
