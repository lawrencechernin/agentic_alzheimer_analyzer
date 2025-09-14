#!/usr/bin/env python3
"""
Experiment 8: Expand Cognitive Labels
====================================

This experiment tests expanding the cognitive impairment label set to include
additional cognitive-related conditions:

ORIGINAL QIDs (5 conditions):
- QID1-5: Dementia (general)
- QID1-12: Alzheimer's Disease  
- QID1-13: Mild Cognitive Impairment
- QID1-22: Frontotemporal Dementia (FTD)
- QID1-23: Lewy Body Disease (LBD)

EXPANDED QIDs (8 conditions):
- QID1-5: Dementia (general)
- QID1-12: Alzheimer's Disease  
- QID1-13: Mild Cognitive Impairment
- QID1-14: Traumatic Brain Injury (can affect cognition)
- QID1-22: Frontotemporal Dementia (FTD)
- QID1-23: Lewy Body Disease (LBD)
- QID65: Subjective memory problems ("Do you feel you have more problems with memory than most?")
- QID75: Developmental language delay (cognitive development)

Expected Impact: +0.02-0.05 AUC by capturing more cognitive impairment cases
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

# Original cognitive QIDs from best script
ORIGINAL_COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# Expanded cognitive QIDs - include additional cognitive-related conditions
EXPANDED_COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-14', 'QID1-22', 'QID1-23', 'QID65', 'QID75']

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality criteria for cognitive data validity"""
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

def build_composite_labels(med_hx, cognitive_qids):
    """Build composite cognitive impairment labels using specified QIDs"""
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in cognitive_qids if q in med_hx.columns]
    
    if not available_qids:
        raise ValueError("No cognitive QIDs found!")
    
    print(f"   Using {len(available_qids)} cognitive QIDs: {available_qids}")
    
    # OR combination of QIDs
    impairment = np.zeros(len(med_hx), dtype=int)
    valid = np.zeros(len(med_hx), dtype=bool)
    
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
        valid |= (med_hx[qid].notna()).values
    
    labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'cognitive_impairment': impairment,
        'valid': valid
    })
    
    return labels

def add_demographics(df, data_dir):
    """Add demographic features"""
    print("   Loading demographics...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Merge with main data
    df = df.merge(profile[['SubjectCode', 'YearsEducationUS_Converted', 'Gender']], 
                  on='SubjectCode', how='left')
    
    # Basic demographic features
    if 'YearsEducationUS_Converted' in df.columns:
        df['Education'] = df['YearsEducationUS_Converted'].fillna(16)  # Default to college
        df['Education_sq'] = df['Education'] ** 2
        
        # Basic interactions with MemTrax performance
        if 'CorrectResponsesRT_mean' in df.columns:
            df['education_rt_interact'] = df['Education'] * df['CorrectResponsesRT_mean'] / 16
        if 'CorrectPCT_mean' in df.columns:
            df['education_acc_interact'] = df['Education'] * df['CorrectPCT_mean'] / 16
    
    if 'Gender' in df.columns:
        df['Gender_Num'] = (df['Gender'] == 1).astype(int)  # 1 = Male, 0 = Female
    
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
    print("EXPERIMENT 8: Expand Cognitive Labels")
    print("="*60)
    print("Testing expanded cognitive impairment labels:")
    print("  ORIGINAL QIDs: QID1-5 (Dementia), QID1-12 (Alzheimer's), QID1-13 (MCI), QID1-22 (FTD), QID1-23 (LBD)")
    print("  EXPANDED QIDs: + QID1-14 (TBI), QID65 (Subjective memory), QID75 (Language delay)")
    print("  RATIONALE: Include conditions that can affect cognition but may not be captured by primary dementia labels")
    print("Expected Impact: +0.02-0.05 AUC by capturing more cognitive impairment cases")
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
    
    # Load medical history
    print("\n4. Loading medical history...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    
    # Test both label sets
    results = {}
    
    for label_name, cognitive_qids in [
        ("Original", ORIGINAL_COGNITIVE_QIDS),
        ("Expanded", EXPANDED_COGNITIVE_QIDS)
    ]:
        print(f"\n--- Testing {label_name} Labels ---")
        
        # Build labels
        labels = build_composite_labels(med_hx, cognitive_qids)
        print(f"   Labels: {len(labels)} subjects")
        print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
        
        # Merge
        data = features.merge(labels, on='SubjectCode', how='inner')
        print(f"   Final dataset: {len(data)} subjects")
        print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
        
        if len(data) == 0:
            print(f"   WARNING: No data for {label_name} labels!")
            continue
        
        # Prepare features and labels
        feature_cols = [col for col in data.columns if col not in ['SubjectCode', 'cognitive_impairment', 'valid']]
        X = data[feature_cols]
        y = data['cognitive_impairment']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"   Train: {len(X_train)} subjects ({y_train.mean():.1%} MCI)")
        print(f"   Test: {len(X_test)} subjects ({y_test.mean():.1%} MCI)")
        
        # Create and train model
        print(f"   Training model...")
        model, base_models = create_best_model()
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        print(f"   Test AUC: {auc:.4f}")
        print(f"   Test PR-AUC: {pr_auc:.4f}")
        
        results[label_name] = {
            'auc': auc,
            'pr_auc': pr_auc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'mci_prevalence_train': y_train.mean(),
            'mci_prevalence_test': y_test.mean(),
            'n_features': len(feature_cols)
        }
    
    # Compare results
    print(f"\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if "Original" in results and "Expanded" in results:
        orig_auc = results["Original"]["auc"]
        exp_auc = results["Expanded"]["auc"]
        improvement = exp_auc - orig_auc
        
        print(f"Original Labels AUC: {orig_auc:.4f}")
        print(f"Expanded Labels AUC: {exp_auc:.4f}")
        print(f"Improvement: {improvement:+.4f} AUC")
        print(f"Relative improvement: {improvement/orig_auc*100:+.1f}%")
        
        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        experiment_results = {
            'experiment': 'Expand Cognitive Labels',
            'original_qids': ORIGINAL_COGNITIVE_QIDS,
            'expanded_qids': EXPANDED_COGNITIVE_QIDS,
            'added_qids': ['QID1-14', 'QID65', 'QID75'],
            'original_auc': orig_auc,
            'expanded_auc': exp_auc,
            'improvement': improvement,
            'relative_improvement_pct': improvement/orig_auc*100,
            'results': results
        }
        
        with open(OUTPUT_DIR / 'experiment8_expand_cognitive_labels_results.json', 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"\nResults saved to: {OUTPUT_DIR / 'experiment8_expand_cognitive_labels_results.json'}")
        
        if improvement > 0.02:
            print(f"\n🎯 SUCCESS! Expanded labels improved AUC by {improvement:.4f}")
        elif improvement > 0:
            print(f"\n✅ Minor improvement: {improvement:.4f} AUC")
        else:
            print(f"\n❌ No improvement: {improvement:.4f} AUC")
    else:
        print("Could not compare results - missing data for one or both label sets")

if __name__ == "__main__":
    main()
