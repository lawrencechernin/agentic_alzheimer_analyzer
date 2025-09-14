#!/usr/bin/env python3
"""
Experiment 9: Multi-Source Label Validation
==========================================

This experiment tests multi-source label validation by combining:
1. Self-reported cognitive impairment (medical history)
2. MemTrax performance (objective cognitive assessment)
3. Informant reports (SP-ECOG - study partner reports)

The idea is to require 2 out of 3 sources to agree for a positive MCI label,
which should reduce noise and improve label quality.

Expected Impact: +0.02-0.05 AUC by improving label quality through consensus
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

def build_multi_source_labels(med_hx, sp_ecog_data):
    """Build multi-source cognitive impairment labels using consensus approach"""
    
    # 1. Self-reported cognitive impairment (medical history)
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in ORIGINAL_COGNITIVE_QIDS if q in med_hx.columns]
    
    self_report = np.zeros(len(med_hx), dtype=int)
    for qid in available_qids:
        self_report |= (med_hx[qid] == 1).values
    
    self_labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'self_report_impairment': self_report
    })
    
    # 2. Informant reports (SP-ECOG)
    # Use conservative threshold: any domain with score >= 3 (moderate impairment)
    sp_ecog_data['sp_ecog_impairment'] = (
        (sp_ecog_data['QID49-1'] >= 3) |  # Memory: shopping items
        (sp_ecog_data['QID49-2'] >= 3) |  # Memory: recent events
        (sp_ecog_data['QID49-3'] >= 3) |  # Memory: conversations
        (sp_ecog_data['QID49-4'] >= 3) |  # Memory: object placement
        (sp_ecog_data['QID49-5'] >= 3) |  # Memory: repeating stories
        (sp_ecog_data['QID49-6'] >= 3) |  # Memory: current date
        (sp_ecog_data['QID49-7'] >= 3) |  # Memory: told someone
        (sp_ecog_data['QID49-8'] >= 3)    # Memory: appointments
    ).astype(int)
    
    sp_labels = sp_ecog_data[['SubjectCode', 'sp_ecog_impairment']].copy()
    
    # Merge both sources
    all_labels = self_labels.merge(sp_labels, on='SubjectCode', how='outer')
    
    # Fill missing values with 0 (no impairment)
    all_labels = all_labels.fillna(0)
    
    # Multi-source consensus: require both sources to agree (2 out of 2)
    all_labels['consensus_impairment'] = (
        (all_labels['self_report_impairment'] == 1) & 
        (all_labels['sp_ecog_impairment'] == 1)
    ).astype(int)
    
    # Also create individual source labels for comparison
    all_labels['self_only'] = all_labels['self_report_impairment']
    all_labels['sp_ecog_only'] = all_labels['sp_ecog_impairment']
    
    return all_labels

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
    print("EXPERIMENT 9: Multi-Source Label Validation")
    print("="*60)
    print("Testing multi-source label validation:")
    print("  1. Self-reported cognitive impairment (medical history)")
    print("  2. Informant reports (SP-ECOG - study partner reports)")
    print("  CONSENSUS: Require both sources to agree for positive MCI label")
    print("Expected Impact: +0.02-0.05 AUC by improving label quality through consensus")
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
    
    # Load SP-ECOG data
    print("5. Loading SP-ECOG data...")
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
    sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    print(f"   Loaded {len(sp_ecog)} SP-ECOG records")
    
    # Build multi-source labels
    print("\n6. Building multi-source labels...")
    multi_labels = build_multi_source_labels(med_hx, sp_ecog)
    
    print(f"   Multi-source labels: {len(multi_labels)} subjects")
    print(f"   Self-report prevalence: {multi_labels['self_report_impairment'].mean():.1%}")
    print(f"   SP-ECOG impairment prevalence: {multi_labels['sp_ecog_impairment'].mean():.1%}")
    print(f"   Consensus prevalence: {multi_labels['consensus_impairment'].mean():.1%}")
    
    # Test different label approaches
    results = {}
    
    for label_name, label_col in [
        ("Self-Report Only", "self_only"),
        ("SP-ECOG Only", "sp_ecog_only"),
        ("Multi-Source Consensus", "consensus_impairment")
    ]:
        print(f"\n--- Testing {label_name} Labels ---")
        
        # Merge with features
        data = features.merge(multi_labels[['SubjectCode', label_col]], on='SubjectCode', how='inner')
        data = data.rename(columns={label_col: 'cognitive_impairment'})
        
        print(f"   Final dataset: {len(data)} subjects")
        print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
        
        if len(data) == 0:
            print(f"   WARNING: No data for {label_name} labels!")
            continue
        
        # Prepare features and labels
        feature_cols = [col for col in data.columns if col not in ['SubjectCode', 'cognitive_impairment']]
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
    
    if "Self-Report Only" in results and "Multi-Source Consensus" in results:
        self_auc = results["Self-Report Only"]["auc"]
        consensus_auc = results["Multi-Source Consensus"]["auc"]
        improvement = consensus_auc - self_auc
        
        print(f"Self-Report Only AUC: {self_auc:.4f}")
        print(f"Multi-Source Consensus AUC: {consensus_auc:.4f}")
        print(f"Improvement: {improvement:+.4f} AUC")
        print(f"Relative improvement: {improvement/self_auc*100:+.1f}%")
        
        # Show all results
        print(f"\nAll Label Approaches:")
        for name, result in results.items():
            print(f"  {name}: {result['auc']:.4f} AUC")
        
        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        experiment_results = {
            'experiment': 'Multi-Source Label Validation',
            'self_report_auc': self_auc,
            'consensus_auc': consensus_auc,
            'improvement': improvement,
            'relative_improvement_pct': improvement/self_auc*100,
            'results': results
        }
        
        with open(OUTPUT_DIR / 'experiment9_multi_source_validation_results.json', 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"\nResults saved to: {OUTPUT_DIR / 'experiment9_multi_source_validation_results.json'}")
        
        if improvement > 0.02:
            print(f"\n🎯 SUCCESS! Multi-source consensus improved AUC by {improvement:.4f}")
        elif improvement > 0:
            print(f"\n✅ Minor improvement: {improvement:.4f} AUC")
        else:
            print(f"\n❌ No improvement: {improvement:.4f} AUC")
    else:
        print("Could not compare results - missing data for one or both label sets")

if __name__ == "__main__":
    main()
