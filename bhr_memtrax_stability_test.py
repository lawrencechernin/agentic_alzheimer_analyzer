#!/usr/bin/env python3
"""
BHR MemTrax Stability Test
==========================
Run the current best configuration (AUC=0.798) multiple times
with different random seeds to check stability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Import our modular components
from improvements import (
    enrich_demographics,
    compute_sequence_features,
    curate_cognitive_target,
    apply_ashford
)

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUT_DIR = Path("bhr_memtrax_results")
OUT_DIR.mkdir(exist_ok=True)

def load_and_prepare_data():
    """Load data with current best configuration"""
    # Load MemTrax
    memtrax = pd.read_csv(DATA_DIR / "MemTrax.csv")
    
    # Current best: Ashford filtering (accuracy >= 0.65)
    memtrax_valid = apply_ashford(memtrax, accuracy_threshold=0.65)
    
    # Current best: Winsorization [0.4, 2.0]
    memtrax_valid['CorrectResponsesRT'] = memtrax_valid['CorrectResponsesRT'].clip(0.4, 2.0)
    
    # Extract sequence features
    seq_features = compute_sequence_features(memtrax_valid)
    
    # Aggregate per subject
    agg_dict = {
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std'],
        'NRPCT': ['mean']
    }
    
    aggregated = memtrax_valid.groupby('SubjectCode').agg(agg_dict)
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
    aggregated.reset_index(inplace=True)
    
    # Compute cognitive score
    aggregated['CognitiveScore_mean'] = aggregated['CorrectResponsesRT_mean'] / (aggregated['CorrectPCT_mean'] + 0.01)
    aggregated['CorrectResponsesRT_cv'] = aggregated['CorrectResponsesRT_std'] / (aggregated['CorrectResponsesRT_mean'] + 0.01)
    
    # Merge sequence features
    aggregated = aggregated.merge(seq_features, on='SubjectCode', how='left')
    
    # Enrich demographics
    aggregated = enrich_demographics(aggregated, DATA_DIR)
    
    # Load ECOG/SP/ADL and compute residuals
    aggregated = add_ecog_residuals(aggregated)
    
    # Add spline features
    aggregated = add_spline_features(aggregated)
    
    return aggregated

def add_ecog_residuals(df):
    """Add ECOG/SP/ADL residuals"""
    try:
        # Load ECOG files
        ecog = pd.read_csv(DATA_DIR / "BHR_EverydayCognition.csv")
        sp_ecog = pd.read_csv(DATA_DIR / "BHR_SP_ECog.csv")
        sp_adl = pd.read_csv(DATA_DIR / "BHR_SP_ADL.csv")
        
        # Process each
        for eco_df, prefix in [(ecog, 'ECOG'), (sp_ecog, 'SP_ECOG'), (sp_adl, 'SP_ADL')]:
            if 'SubjectCode' not in eco_df.columns and 'Code' in eco_df.columns:
                eco_df = eco_df.rename(columns={'Code': 'SubjectCode'})
            
            # Filter baseline
            if 'TimepointCode' in eco_df.columns:
                eco_df = eco_df[eco_df['TimepointCode'] == 'm00']
            
            # Compute means
            numeric_cols = eco_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if 'QID' not in c]
            
            if len(numeric_cols) > 0:
                eco_df[f'{prefix}_mean'] = eco_df[numeric_cols].mean(axis=1)
                
                # Per-domain means for ECOG
                if prefix == 'ECOG':
                    for domain in ['Memory', 'Language', 'Visuospatial', 'Executive']:
                        domain_cols = [c for c in numeric_cols if domain.lower() in c.lower()]
                        if domain_cols:
                            eco_df[f'{prefix}_{domain}_mean'] = eco_df[domain_cols].mean(axis=1)
                
                # Keep only SubjectCode and new features
                keep_cols = ['SubjectCode'] + [c for c in eco_df.columns if c.startswith(prefix + '_')]
                eco_small = eco_df[keep_cols].drop_duplicates(subset=['SubjectCode'])
                
                # Merge
                df = df.merge(eco_small, on='SubjectCode', how='left')
    except Exception as e:
        print(f"Warning: Could not add ECOG residuals: {e}")
    
    return df

def add_spline_features(df):
    """Add natural cubic spline features"""
    for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
        if col in df.columns:
            x = df[col].fillna(df[col].median())
            # Create 3 knot points
            knots = np.percentile(x[x.notna()], [25, 50, 75])
            
            # Create basis functions
            for i, knot in enumerate(knots):
                df[f'{col}_spline_{i}'] = np.maximum(0, x - knot) ** 3
    
    return df

def build_current_best_model():
    """Build the current best stacking configuration"""
    base_estimators = []
    
    # 1. Calibrated Logistic with MI feature selection
    log_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(mutual_info_classif, k=15)),  # Current best k=15
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.1,
            solver='lbfgs'
        ))
    ])
    base_estimators.append(('logistic', log_pipe))
    
    # 2. HistGradientBoosting with current best params
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.1,  # Current best
        max_leaf_nodes=31,  # Current best
        max_depth=8,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42
    )
    base_estimators.append(('histgb', hgb))
    
    # 3. XGBoost if available
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        base_estimators.append(('xgb', xgb))
    
    # Final estimator
    final_estimator = LogisticRegression(
        max_iter=1000,
        C=1.0
    )
    
    # Create stacking classifier
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba',
        passthrough=False
    )
    
    return stack

def run_single_test(seed=42):
    """Run a single test with given random seed"""
    # Set random seeds
    np.random.seed(seed)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Build composite target
    med_hx = pd.read_csv(DATA_DIR / "BHR_MedicalHx.csv")
    labels = curate_cognitive_target(med_hx, composite=True)
    
    # Merge
    merged = data.merge(labels, on='SubjectCode', how='inner')
    
    # Separate features and target
    feature_cols = [c for c in merged.columns if c not in ['SubjectCode', 'AnyCogImpairment']]
    X = merged[feature_cols]
    y = merged['AnyCogImpairment']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Build and train model
    model = build_current_best_model()
    model.fit(X_train_imp, y_train)
    
    # Get predictions
    y_proba = model.predict_proba(X_test_imp)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    return {
        'seed': seed,
        'auc': auc,
        'pr_auc': pr_auc,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'prevalence': y.mean()
    }

def main():
    print("🔬 BHR MEMTRAX STABILITY TEST")
    print("=" * 50)
    print("Testing current best configuration (AUC=0.798) with multiple random seeds")
    print()
    
    # Run multiple tests with different seeds
    seeds = [42, 123, 456, 789, 999, 1337, 2024, 3141, 5926, 7777]
    results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"🧪 Test {i}/10 (seed={seed})...", end=" ")
        try:
            result = run_single_test(seed)
            results.append(result)
            print(f"AUC={result['auc']:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not results:
        print("❌ No successful tests!")
        return
    
    # Analyze results
    aucs = [r['auc'] for r in results]
    pr_aucs = [r['pr_auc'] for r in results]
    
    print("\n" + "=" * 50)
    print("STABILITY ANALYSIS")
    print("=" * 50)
    print(f"✅ Successful tests: {len(results)}/10")
    print(f"📊 AUC Statistics:")
    print(f"   Mean: {np.mean(aucs):.4f}")
    print(f"   Std:  {np.std(aucs):.4f}")
    print(f"   Min:  {np.min(aucs):.4f}")
    print(f"   Max:  {np.max(aucs):.4f}")
    print(f"   Range: {np.max(aucs) - np.min(aucs):.4f}")
    
    print(f"\n📈 PR-AUC Statistics:")
    print(f"   Mean: {np.mean(pr_aucs):.4f}")
    print(f"   Std:  {np.std(pr_aucs):.4f}")
    
    # Check if we consistently hit 0.80+
    above_80 = sum(1 for auc in aucs if auc >= 0.80)
    print(f"\n🎯 Tests above 0.80: {above_80}/{len(results)} ({above_80/len(results)*100:.1f}%)")
    
    # Stability assessment
    if np.std(aucs) < 0.01:
        print("✅ EXCELLENT stability (std < 0.01)")
    elif np.std(aucs) < 0.02:
        print("✅ GOOD stability (std < 0.02)")
    elif np.std(aucs) < 0.03:
        print("⚠️  MODERATE stability (std < 0.03)")
    else:
        print("❌ POOR stability (std >= 0.03)")
    
    # Save results
    stability_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "configuration": "Current best (AUC=0.798)",
        "n_tests": len(results),
        "auc_stats": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "min": float(np.min(aucs)),
            "max": float(np.max(aucs)),
            "range": float(np.max(aucs) - np.min(aucs))
        },
        "pr_auc_stats": {
            "mean": float(np.mean(pr_aucs)),
            "std": float(np.std(pr_aucs))
        },
        "above_80_percent": float(above_80/len(results)*100),
        "individual_results": results
    }
    
    results_path = OUT_DIR / "stability_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(stability_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    return np.mean(aucs), np.std(aucs)

if __name__ == "__main__":
    mean_auc, std_auc = main()
    
    print(f"\n🎯 FINAL VERDICT:")
    if mean_auc >= 0.80:
        print(f"✅ CONFIRMED: Mean AUC = {mean_auc:.4f} >= 0.80")
    else:
        print(f"📍 Mean AUC = {mean_auc:.4f} (std={std_auc:.4f})")
        print(f"   Need {0.80 - mean_auc:.4f} more to reach clinical threshold")
