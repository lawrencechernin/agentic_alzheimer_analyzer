#!/usr/bin/env python3
"""
Experiment 3: Enhanced Cognitive Reserve Features
===============================================

Based on mismatch analysis, cognitive reserve plays a major role in protecting against
cognitive impairment. This experiment tests enhanced cognitive reserve modeling:

1. More sophisticated education features
2. Education-performance interactions
3. Cognitive reserve composite scores
4. Education-stratified analysis

Expected Impact: Should improve AUC by better modeling cognitive reserve effects
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

def build_composite_labels(med_hx):
    """Build composite cognitive impairment labels"""
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
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

def add_basic_demographics(df, data_dir):
    """Add basic demographic features (original approach)"""
    print("   Loading basic demographics...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Merge with main data
    df = df.merge(profile[['SubjectCode', 'YearsEducationUS_Converted', 'Gender']], 
                  on='SubjectCode', how='left')
    
    # Basic features
    if 'YearsEducationUS_Converted' in df.columns:
        df['Education'] = df['YearsEducationUS_Converted'].fillna(16)  # Default to college
        df['Education_sq'] = df['Education'] ** 2
        
        # Basic interactions
        if 'CorrectResponsesRT_mean' in df.columns:
            df['education_rt_interact'] = df['Education'] * df['CorrectResponsesRT_mean'] / 16
        if 'CorrectPCT_mean' in df.columns:
            df['education_acc_interact'] = df['Education'] * df['CorrectPCT_mean'] / 16
    
    if 'Gender' in df.columns:
        df['Gender_Num'] = (df['Gender'] == 1).astype(int)  # 1 = Male, 0 = Female
    
    return df

def add_enhanced_cognitive_reserve(df, data_dir):
    """Add enhanced cognitive reserve features"""
    print("   Loading enhanced cognitive reserve features...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Merge with main data
    df = df.merge(profile[['SubjectCode', 'YearsEducationUS_Converted', 'Gender']], 
                  on='SubjectCode', how='left')
    
    if 'YearsEducationUS_Converted' in df.columns:
        # Fill missing education with median
        median_education = df['YearsEducationUS_Converted'].median()
        df['Education'] = df['YearsEducationUS_Converted'].fillna(median_education)
        
        # Enhanced education features
        df['Education_sq'] = df['Education'] ** 2
        df['Education_cubed'] = df['Education'] ** 3
        df['Education_log'] = np.log(df['Education'] + 1)
        
        # Education categories (as numeric for ML)
        df['Education_Level_Num'] = pd.cut(df['Education'], 
                                          bins=[0, 12, 16, 20, 25], 
                                          labels=[1, 2, 3, 4],  # Numeric labels
                                          include_lowest=True).astype(float)
        
        # Binary education indicators
        df['High_School_Only'] = (df['Education'] <= 12).astype(int)
        df['Some_College'] = ((df['Education'] > 12) & (df['Education'] < 16)).astype(int)
        df['College_Graduate'] = ((df['Education'] >= 16) & (df['Education'] < 20)).astype(int)
        df['Graduate_Degree'] = (df['Education'] >= 20).astype(int)
        df['Very_High_Education'] = (df['Education'] >= 22).astype(int)
        
        # Cognitive reserve composite score
        # Higher education = higher reserve
        df['Cognitive_Reserve_Score'] = (
            df['Education'] * 0.4 +  # Education weight
            (df['Education'] >= 16).astype(int) * 2 +  # College bonus
            (df['Education'] >= 20).astype(int) * 3 +  # Graduate bonus
            (df['Education'] >= 22).astype(int) * 2    # Advanced degree bonus
        )
        
        # Education-performance interactions
        if 'CorrectResponsesRT_mean' in df.columns:
            df['education_rt_interact'] = df['Education'] * df['CorrectResponsesRT_mean'] / 16
            df['education_rt_squared'] = df['Education'] * (df['CorrectResponsesRT_mean'] ** 2) / 256
            df['reserve_rt_interact'] = df['Cognitive_Reserve_Score'] * df['CorrectResponsesRT_mean'] / 20
            
        if 'CorrectPCT_mean' in df.columns:
            df['education_acc_interact'] = df['Education'] * df['CorrectPCT_mean'] / 16
            df['education_acc_squared'] = df['Education'] * (df['CorrectPCT_mean'] ** 2) / 256
            df['reserve_acc_interact'] = df['Cognitive_Reserve_Score'] * df['CorrectPCT_mean'] / 20
            
        # Education-stratified performance
        if 'CorrectPCT_mean' in df.columns and 'CorrectResponsesRT_mean' in df.columns:
            # Calculate performance relative to education level
            for edu_level in ['High_School_Only', 'Some_College', 'College_Graduate', 'Graduate_Degree']:
                if edu_level in df.columns:
                    mask = df[edu_level] == 1
                    if mask.sum() > 10:  # Only if we have enough samples
                        edu_acc_mean = df.loc[mask, 'CorrectPCT_mean'].mean()
                        edu_rt_mean = df.loc[mask, 'CorrectResponsesRT_mean'].mean()
                        
                        df[f'{edu_level}_acc_relative'] = df['CorrectPCT_mean'] - edu_acc_mean
                        df[f'{edu_level}_rt_relative'] = df['CorrectResponsesRT_mean'] - edu_rt_mean
        
        # Education-performance ratio features
        if 'CorrectPCT_mean' in df.columns and 'CorrectResponsesRT_mean' in df.columns:
            df['education_performance_ratio'] = df['Education'] / (df['CorrectResponsesRT_mean'] + 0.1)
            df['reserve_performance_ratio'] = df['Cognitive_Reserve_Score'] / (df['CorrectResponsesRT_mean'] + 0.1)
    
    if 'Gender' in df.columns:
        df['Gender_Num'] = (df['Gender'] == 1).astype(int)  # 1 = Male, 0 = Female
        
        # Gender-education interactions
        if 'Education' in df.columns:
            df['gender_education_interact'] = df['Gender_Num'] * df['Education'] / 16
    
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
    print("EXPERIMENT 3: Enhanced Cognitive Reserve Features")
    print("="*60)
    print("Testing enhanced cognitive reserve modeling:")
    print("  - Sophisticated education features")
    print("  - Education-performance interactions")
    print("  - Cognitive reserve composite scores")
    print("  - Education-stratified analysis")
    print("Expected Impact: Should improve AUC by better modeling cognitive reserve")
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
    
    # Load medical history
    print("\n4. Loading medical history...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    labels = build_composite_labels(med_hx)
    print(f"   Labels: {len(labels)} subjects")
    print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Test both approaches
    results = {}
    
    for approach_name, demo_func in [
        ("Basic Demographics", add_basic_demographics),
        ("Enhanced Cognitive Reserve", add_enhanced_cognitive_reserve)
    ]:
        print(f"\n--- Testing {approach_name} ---")
        
        # Create features with specified demographic approach
        features_copy = features.copy()
        features_copy = demo_func(features_copy, DATA_DIR)
        features_copy = add_informant_scores(features_copy, DATA_DIR)
        
        # Merge with labels
        data = features_copy.merge(labels, on='SubjectCode', how='inner')
        print(f"   Final dataset: {len(data)} subjects")
        print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
        print(f"   Features: {data.shape[1]-3}")  # Exclude SubjectCode, cognitive_impairment, valid
        
        if len(data) == 0:
            print(f"   WARNING: No data for {approach_name}!")
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
        
        results[approach_name] = {
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
    
    if "Basic Demographics" in results and "Enhanced Cognitive Reserve" in results:
        basic_auc = results["Basic Demographics"]["auc"]
        enhanced_auc = results["Enhanced Cognitive Reserve"]["auc"]
        improvement = enhanced_auc - basic_auc
        
        print(f"Basic Demographics AUC: {basic_auc:.4f}")
        print(f"Enhanced Cognitive Reserve AUC: {enhanced_auc:.4f}")
        print(f"Improvement: {improvement:+.4f} AUC")
        print(f"Relative improvement: {improvement/basic_auc*100:+.1f}%")
        
        # Save results
        OUTPUT_DIR.mkdir(exist_ok=True)
        experiment_results = {
            'experiment': 'Enhanced Cognitive Reserve Features',
            'basic_auc': basic_auc,
            'enhanced_auc': enhanced_auc,
            'improvement': improvement,
            'relative_improvement_pct': improvement/basic_auc*100,
            'results': results
        }
        
        with open(OUTPUT_DIR / 'experiment3_enhanced_cognitive_reserve_results.json', 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        print(f"\nResults saved to: {OUTPUT_DIR / 'experiment3_enhanced_cognitive_reserve_results.json'}")
        
        if improvement > 0.01:
            print(f"\n🎯 SUCCESS! Enhanced cognitive reserve improved AUC by {improvement:.4f}")
        elif improvement > 0:
            print(f"\n✅ Minor improvement: {improvement:.4f} AUC")
        else:
            print(f"\n❌ No improvement: {improvement:.4f} AUC")
    else:
        print("Could not compare results - missing data for one or both approaches")

if __name__ == "__main__":
    main()
