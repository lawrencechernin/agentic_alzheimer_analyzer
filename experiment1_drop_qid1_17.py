#!/usr/bin/env python3
"""
Experiment 1: Drop QID1-17 "Other Dementia" from Labels
======================================================

Based on mismatch analysis, QID1-17 appears to be over-diagnosed (56.6% of mismatch cases).
This experiment tests the impact of removing QID1-17 from the cognitive impairment labels.

Expected Impact: Should improve AUC by reducing false positive labels
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Data paths
MEMTRAX_DIR = Path('../bhr/from_paul/processed/')
DATA_DIR = Path('../bhr/BHR-ALL-EXT_Mem_2022/')
OUTPUT_DIR = Path('bhr_memtrax_results')

# Cognitive impairment QIDs (EXCLUDING QID1-17)
COGNITIVE_QIDS = [
    'QID186',  # Memory problems
    'QID187',  # Confusion
    'QID188',  # Dementia
    'QID189',  # Alzheimer's
    'QID190',  # Stroke
    'QID191',  # Parkinson's
    'QID192',  # Depression
    'QID193',  # Anxiety
    'QID194',  # Sleep problems
    'QID195',  # Head injury
    # NOTE: QID1-17 "Other Dementia" EXCLUDED due to over-diagnosis
]

def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality filter to MemTrax data"""
    # Processed data doesn't have Status column, just apply accuracy and RT filters
    return df[(df['CorrectPCT'] >= min_acc) &
              (df['CorrectResponsesRT'].between(0.5, 2.5))].copy()

def extract_sequence_features(df):
    """Extract fatigue and variability features from reaction time sequences"""
    features = []
    for subject, group in df.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        all_rts = []
        
        for _, row in group.iterrows():
            if pd.notna(row.get('ReactionTimes')):
                try:
                    rts = [float(x.strip()) for x in str(row['ReactionTimes']).split(',') 
                           if x.strip() and x.strip() != 'nan']
                    all_rts.extend([r for r in rts if 0.3 <= r <= 3.0])
                except:
                    continue
        
        if len(all_rts) >= 10:
            n = len(all_rts)
            third = max(1, n // 3)
            feat['seq_first_third'] = np.mean(all_rts[:third])
            feat['seq_last_third'] = np.mean(all_rts[-third:])
            feat['seq_fatigue'] = feat['seq_last_third'] - feat['seq_first_third']
            feat['seq_mean_rt'] = np.mean(all_rts)
            feat['seq_std_rt'] = np.std(all_rts)
            feat['seq_cv'] = feat['seq_std_rt'] / (feat['seq_mean_rt'] + 1e-6)
            
            mid = n // 2
            if mid > 1:
                feat['reliability_change'] = np.var(all_rts[mid:]) - np.var(all_rts[:mid])
                
            if n >= 3:
                slope, _ = np.polyfit(np.arange(n), all_rts, 1)
                feat['rt_slope'] = slope
                
            feat['n_tests'] = len(group)
            
        features.append(feat)
    
    return pd.DataFrame(features)

def build_composite_labels(med_hx):
    """Build composite cognitive impairment labels (EXCLUDING QID1-17)"""
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
    if not available_qids:
        print("   WARNING: No cognitive QIDs found!")
        return pd.DataFrame({'SubjectCode': [], 'cognitive_impairment': []})
    
    print(f"   Using {len(available_qids)} cognitive QIDs: {available_qids}")
    
    # Create individual condition flags
    labels = med_hx[['SubjectCode']].copy()
    for qid in available_qids:
        labels[f'{qid}_present'] = (med_hx[qid] == 1).astype(int)
    
    # Composite label: OR of all available conditions
    condition_cols = [f'{qid}_present' for qid in available_qids]
    labels['cognitive_impairment'] = labels[condition_cols].max(axis=1)
    
    # Add individual condition counts
    labels['condition_count'] = labels[condition_cols].sum(axis=1)
    
    return labels

def add_demographics(df, data_dir):
    """Add demographic features with cognitive reserve modeling"""
    print("   Loading demographics...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Merge with main data
    df = df.merge(profile[['SubjectCode', 'YearsEducationUS_Converted', 'Gender']], 
                  on='SubjectCode', how='left')
    
    # Create cognitive reserve features
    if 'YearsEducationUS_Converted' in df.columns:
        df['Education'] = df['YearsEducationUS_Converted'].fillna(16)  # Default to college
        df['Education_sq'] = df['Education'] ** 2
        df['High_Education'] = (df['Education'] >= 16).astype(int)
        df['Very_High_Education'] = (df['Education'] >= 20).astype(int)
        
        # Cognitive reserve interactions
        if 'CorrectResponsesRT_mean' in df.columns:
            df['education_rt_interact'] = df['Education'] * df['CorrectResponsesRT_mean'] / 16
        if 'CorrectPCT_mean' in df.columns:
            df['education_acc_interact'] = df['Education'] * df['CorrectPCT_mean'] / 16
    
    if 'Gender' in df.columns:
        df['Gender_Num'] = (df['Gender'] == 1).astype(int)  # 1 = Male, 0 = Female
    
    return df

def add_informant_scores(df, data_dir):
    """Add informant (SP-ECOG) and self-report (ECOG) scores as features"""
    print("   Loading informant scores...")
    
    # SP-ECOG (informant)
    try:
        sp_ecog = pd.read_csv(data_dir / 'BHR_SP_ECog.csv')
        sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
        sp_ecog_baseline = sp_ecog_baseline.drop_duplicates(subset=['SubjectCode'])
        
        # Calculate mean scores for each domain
        ecog_domains = ['QID49', 'QID50', 'QID51', 'QID52', 'QID53', 'QID54']
        for domain in ecog_domains:
            domain_cols = [col for col in sp_ecog_baseline.columns if col.startswith(f'{domain}-')]
            if domain_cols:
                # Calculate mean, excluding '8' (Don't Know)
                domain_scores = sp_ecog_baseline[domain_cols].replace(8, np.nan).mean(axis=1)
                sp_ecog_baseline[f'sp_{domain}_mean'] = domain_scores
        
        # Merge SP-ECOG features
        sp_cols = ['SubjectCode'] + [f'sp_{domain}_mean' for domain in ecog_domains if f'sp_{domain}_mean' in sp_ecog_baseline.columns]
        df = df.merge(sp_ecog_baseline[sp_cols], on='SubjectCode', how='left')
        print(f"   Added {len(sp_cols)-1} SP-ECOG features")
        
    except Exception as e:
        print(f"   SP-ECOG loading failed: {e}")
    
    # ECOG (self-report)
    try:
        ecog = pd.read_csv(data_dir / 'BHR_EverydayCognition.csv')
        ecog_baseline = ecog[ecog['TimepointCode'] == 'm00'].copy()
        ecog_baseline = ecog_baseline.drop_duplicates(subset=['SubjectCode'])
        
        # Calculate mean scores for each domain
        for domain in ecog_domains:
            domain_cols = [col for col in ecog_baseline.columns if col.startswith(f'{domain}-')]
            if domain_cols:
                domain_scores = ecog_baseline[domain_cols].replace(8, np.nan).mean(axis=1)
                ecog_baseline[f'ecog_{domain}_mean'] = domain_scores
        
        # Merge ECOG features
        ecog_cols = ['SubjectCode'] + [f'ecog_{domain}_mean' for domain in ecog_domains if f'ecog_{domain}_mean' in ecog_baseline.columns]
        df = df.merge(ecog_baseline[ecog_cols], on='SubjectCode', how='left')
        print(f"   Added {len(ecog_cols)-1} ECOG features")
        
    except Exception as e:
        print(f"   ECOG loading failed: {e}")
    
    return df

def create_best_model():
    """Create the best performing model configuration"""
    # Base models
    logistic = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectKBest(k='all')),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    rf = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    histgb = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', HistGradientBoostingClassifier(random_state=42, max_iter=100))
    ])
    
    # Stacking ensemble
    stack = StackingClassifier(
        estimators=[
            ('logistic', logistic),
            ('rf', rf),
            ('histgb', histgb)
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # Calibrated ensemble
    calibrated_stack = CalibratedClassifierCV(stack, cv=3)
    
    return calibrated_stack, [logistic, rf, histgb]

def main():
    """Main experiment function"""
    print("="*60)
    print("EXPERIMENT 1: Drop QID1-17 'Other Dementia' from Labels")
    print("="*60)
    print("Expected Impact: Should improve AUC by reducing false positive labels")
    print()
    
    # Load MemTrax data
    print("1. Loading MemTrax data...")
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"   Loaded {len(memtrax)} MemTrax records")
    
    # Apply quality filter
    print("2. Applying Ashford quality filter...")
    memtrax_q = apply_ashford_filter(memtrax)
    print(f"   After quality filter: {len(memtrax_q)} records")
    
    # Extract sequence features
    print("3. Extracting sequence features...")
    seq_feat = extract_sequence_features(memtrax_q)
    print(f"   Sequence features for {len(seq_feat)} subjects")
    
    # Aggregates
    agg_feat = memtrax_q.groupby('SubjectCode').agg({
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std'],
        'CorrectRejectionsN': ['mean', 'std'],
        'IncorrectRejectionsN': ['mean', 'std']
    })
    agg_feat.columns = ['_'.join(col) for col in agg_feat.columns]
    agg_feat = agg_feat.reset_index()
    
    # Composite scores
    agg_feat['CogScore'] = agg_feat['CorrectResponsesRT_mean'] / (agg_feat['CorrectPCT_mean'] + 0.01)
    agg_feat['RT_CV'] = agg_feat['CorrectResponsesRT_std'] / (agg_feat['CorrectResponsesRT_mean'] + 1e-6)
    agg_feat['Speed_Accuracy_Product'] = agg_feat['CorrectPCT_mean'] / (agg_feat['CorrectResponsesRT_mean'] + 0.01)
    
    # Merge features
    features = agg_feat.merge(seq_feat, on='SubjectCode', how='left')
    features = add_demographics(features, DATA_DIR)
    features = add_informant_scores(features, DATA_DIR)
    
    # Labels (EXCLUDING QID1-17)
    print("\n4. Building composite labels (EXCLUDING QID1-17)...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    labels = build_composite_labels(med_hx)
    print(f"   Labels: {len(labels)} subjects")
    print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"\n5. Final dataset: {len(data)} subjects")
    print(f"   Features: {data.shape[1]-3}")  # Exclude SubjectCode, cognitive_impairment, condition_count
    print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
    
    # Prepare features and labels
    feature_cols = [col for col in data.columns if col not in ['SubjectCode', 'cognitive_impairment', 'condition_count']]
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n6. Train/test split:")
    print(f"   Train: {len(X_train)} subjects ({y_train.mean():.1%} MCI)")
    print(f"   Test: {len(X_test)} subjects ({y_test.mean():.1%} MCI)")
    
    # Create and train model
    print(f"\n7. Training best model...")
    model, base_models = create_best_model()
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n8. Results:")
    print(f"   Test AUC: {auc:.4f}")
    print(f"   Test PR-AUC: {pr_auc:.4f}")
    
    # Compare with baseline
    baseline_auc = 0.744
    improvement = auc - baseline_auc
    print(f"\n9. Comparison with baseline (0.744):")
    print(f"   Improvement: {improvement:+.4f} AUC")
    print(f"   Relative improvement: {improvement/baseline_auc*100:+.1f}%")
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {
        'experiment': 'Drop QID1-17 from labels',
        'baseline_auc': baseline_auc,
        'test_auc': auc,
        'test_pr_auc': pr_auc,
        'improvement': improvement,
        'relative_improvement_pct': improvement/baseline_auc*100,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mci_prevalence_train': y_train.mean(),
        'mci_prevalence_test': y_test.mean(),
        'n_features': len(feature_cols)
    }
    
    with open(OUTPUT_DIR / 'experiment1_drop_qid1_17_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n10. Results saved to: {OUTPUT_DIR / 'experiment1_drop_qid1_17_results.json'}")
    
    if improvement > 0.01:
        print(f"\n🎯 SUCCESS! Dropping QID1-17 improved AUC by {improvement:.4f}")
    elif improvement > 0:
        print(f"\n✅ Minor improvement: {improvement:.4f} AUC")
    else:
        print(f"\n❌ No improvement: {improvement:.4f} AUC")

if __name__ == "__main__":
    main()
