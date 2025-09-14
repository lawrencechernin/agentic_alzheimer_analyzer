#!/usr/bin/env python3
"""
Experiment 5: Multi-Source Label Validation
==========================================

This experiment tests the impact of using multiple sources to validate MCI labels,
requiring 2 of 3 sources to agree for a positive MCI label:

1. Self-reported cognitive complaints (QIDs)
2. Objective impairment (MemTrax performance < threshold)
3. Informant report (SP-ECOG > threshold)

Expected Impact: +0.05-0.10 AUC by reducing "worried well" false positives
and catching anosognosia cases that self-report misses.
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

# Cognitive QIDs for self-report labels
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

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

def build_self_report_labels(med_hx):
    """Build self-reported cognitive impairment labels"""
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
    if not available_qids:
        raise ValueError("No cognitive QIDs found!")
    
    # OR combination of QIDs
    impairment = np.zeros(len(med_hx), dtype=int)
    valid = np.zeros(len(med_hx), dtype=bool)
    
    for qid in available_qids:
        impairment |= (med_hx[qid] == 1).values
        valid |= (med_hx[qid].notna()).values
    
    return pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'self_mci': impairment,
        'self_valid': valid
    })

def build_objective_labels(memtrax_data):
    """Build objective impairment labels based on MemTrax performance"""
    # Calculate performance thresholds
    accuracy_threshold = memtrax_data['CorrectPCT'].quantile(0.20)  # Bottom 20%
    rt_threshold = memtrax_data['CorrectResponsesRT'].quantile(0.80)  # Top 20%
    
    # Create objective impairment labels
    objective_impairment = (
        (memtrax_data['CorrectPCT'] < accuracy_threshold) |
        (memtrax_data['CorrectResponsesRT'] > rt_threshold)
    ).astype(int)
    
    return pd.DataFrame({
        'SubjectCode': memtrax_data['SubjectCode'],
        'objective_mci': objective_impairment,
        'objective_valid': True
    })

def build_informant_labels(data_dir):
    """Build informant-based labels from SP-ECOG"""
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
        
        # Calculate overall informant score
        sp_cols = [f'sp_{domain}_mean' for domain in ecog_domains if f'sp_{domain}_mean' in sp_ecog_baseline.columns]
        if sp_cols:
            sp_ecog_baseline['sp_overall_mean'] = sp_ecog_baseline[sp_cols].mean(axis=1)
            
            # Define informant MCI (higher scores = worse performance)
            informant_threshold = sp_ecog_baseline['sp_overall_mean'].quantile(0.80)  # Top 20%
            informant_impairment = (sp_ecog_baseline['sp_overall_mean'] >= informant_threshold).astype(int)
            
            return pd.DataFrame({
                'SubjectCode': sp_ecog_baseline['SubjectCode'],
                'informant_mci': informant_impairment,
                'informant_valid': True
            })
        else:
            return pd.DataFrame(columns=['SubjectCode', 'informant_mci', 'informant_valid'])
            
    except Exception as e:
        print(f"   SP-ECOG loading failed: {e}")
        return pd.DataFrame(columns=['SubjectCode', 'informant_mci', 'informant_valid'])

def create_multi_source_labels(self_labels, objective_labels, informant_labels):
    """Create multi-source validated labels requiring 2 of 3 sources to agree"""
    # Merge all label sources
    labels = self_labels.merge(objective_labels, on='SubjectCode', how='outer')
    labels = labels.merge(informant_labels, on='SubjectCode', how='outer')
    
    # Fill missing values
    labels['self_mci'] = labels['self_mci'].fillna(0).astype(int)
    labels['objective_mci'] = labels['objective_mci'].fillna(0).astype(int)
    labels['informant_mci'] = labels['informant_mci'].fillna(0).astype(int)
    labels['self_valid'] = labels['self_valid'].fillna(False)
    labels['objective_valid'] = labels['objective_valid'].fillna(False)
    labels['informant_valid'] = labels['informant_valid'].fillna(False)
    
    # Count valid sources for each subject
    labels['n_valid_sources'] = (
        labels['self_valid'].astype(int) + 
        labels['objective_valid'].astype(int) + 
        labels['informant_valid'].astype(int)
    )
    
    # Count positive sources for each subject
    labels['n_positive_sources'] = (
        labels['self_mci'].astype(int) + 
        labels['objective_mci'].astype(int) + 
        labels['informant_mci'].astype(int)
    )
    
    # Create multi-source labels: require 2 of 3 sources to agree
    labels['multi_source_mci'] = (labels['n_positive_sources'] >= 2).astype(int)
    labels['multi_source_valid'] = (labels['n_valid_sources'] >= 2)
    
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
    print("EXPERIMENT 5: Multi-Source Label Validation")
    print("="*60)
    print("Testing multi-source label validation:")
    print("  - Self-reported cognitive complaints (QIDs)")
    print("  - Objective impairment (MemTrax performance)")
    print("  - Informant report (SP-ECOG)")
    print("  - Require 2 of 3 sources to agree for MCI label")
    print("Expected Impact: +0.05-0.10 AUC by reducing false positives")
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
    
    # Load medical history for self-report labels
    print("\n4. Building multi-source labels...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    
    # Build different label types
    self_labels = build_self_report_labels(med_hx)
    print(f"   Self-report labels: {len(self_labels)} subjects, {self_labels['self_mci'].mean():.1%} MCI")
    
    objective_labels = build_objective_labels(memtrax_q)
    print(f"   Objective labels: {len(objective_labels)} subjects, {objective_labels['objective_mci'].mean():.1%} MCI")
    
    informant_labels = build_informant_labels(DATA_DIR)
    print(f"   Informant labels: {len(informant_labels)} subjects, {informant_labels['informant_mci'].mean():.1%} MCI")
    
    # Create multi-source labels
    multi_labels = create_multi_source_labels(self_labels, objective_labels, informant_labels)
    print(f"   Multi-source labels: {len(multi_labels)} subjects, {multi_labels['multi_source_mci'].mean():.1%} MCI")
    
    # Add demographics
    print("5. Adding demographics...")
    features = add_demographics(features, DATA_DIR)
    
    # Test both label types
    results = {}
    
    for label_name, label_col in [
        ("Self-Report Only", "self_mci"),
        ("Multi-Source", "multi_source_mci")
    ]:
        print(f"\n--- Testing {label_name} Labels ---")
        
        # Merge with labels
        data = features.merge(multi_labels[['SubjectCode', label_col]], on='SubjectCode', how='inner')
        print(f"   Final dataset: {len(data)} subjects")
        print(f"   MCI prevalence: {data[label_col].mean():.1%}")
        
        if len(data) == 0:
            print(f"   WARNING: No data for {label_name} labels!")
            continue
        
        # Prepare features and labels
        feature_cols = [col for col in data.columns if col not in ['SubjectCode', label_col]]
        X = data[feature_cols]
        y = data[label_col]
        
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
    
    if "Self-Report Only" in results and "Multi-Source" in results:
        self_auc = results["Self-Report Only"]["auc"]
        multi_auc = results["Multi-Source"]["auc"]
        improvement = multi_auc - self_auc
        
        print(f"Self-Report Only AUC: {self_auc:.4f}")
        print(f"Multi-Source AUC: {multi_auc:.4f}")
        print(f"Improvement: {improvement:+.4f} AUC")
        print(f"Relative improvement: {improvement/self_auc*100:+.1f}%")
        
        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        experiment_results = {
            'experiment': 'Multi-Source Label Validation',
            'self_report_auc': self_auc,
            'multi_source_auc': multi_auc,
            'improvement': improvement,
            'relative_improvement_pct': improvement/self_auc*100,
            'results': results
        }
        
        with open(OUTPUT_DIR / 'experiment5_multi_source_labels_results.json', 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"\nResults saved to: {OUTPUT_DIR / 'experiment5_multi_source_labels_results.json'}")
        
        if improvement > 0.02:
            print(f"\n🎯 SUCCESS! Multi-source labels improved AUC by {improvement:.4f}")
        elif improvement > 0:
            print(f"\n✅ Minor improvement: {improvement:.4f} AUC")
        else:
            print(f"\n❌ No improvement: {improvement:.4f} AUC")
    else:
        print("Could not compare results - missing data for one or both label types")

if __name__ == "__main__":
    main()

