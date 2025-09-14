#!/usr/bin/env python3
"""
Debug Model Predictions: BHR-ALL-49397
=====================================

This script investigates why models are giving low probabilities despite high RT.
We'll examine:
- Exact feature values for this subject
- Model coefficients and feature importance
- Training data distribution for similar cases
- Feature scaling and preprocessing effects
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
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

def build_consensus_labels(med_hx, sp_ecog_data):
    """Build multi-source consensus cognitive impairment labels"""
    
    # 1. Self-reported cognitive impairment (medical history)
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    available_qids = [q for q in COGNITIVE_QIDS if q in med_hx.columns]
    
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
    all_labels['cognitive_impairment'] = (
        (all_labels['self_report_impairment'] == 1) & 
        (all_labels['sp_ecog_impairment'] == 1)
    ).astype(int)
    
    return all_labels[['SubjectCode', 'cognitive_impairment', 'self_report_impairment', 'sp_ecog_impairment']]

def add_demographics(df, data_dir):
    """Add demographic features including age"""
    print("   Loading demographics...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Merge with main data
    df = df.merge(profile[['SubjectCode', 'YearsEducationUS_Converted', 'Gender', 'AgeRange']], 
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

def create_individual_models():
    """Create individual models for analysis"""
    # Base models
    logistic = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
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
    
    return logistic, rf, histgb

def analyze_logistic_regression(logistic, X, y, subject_idx, feature_names):
    """Analyze logistic regression coefficients and predictions"""
    print("\n=== LOGISTIC REGRESSION ANALYSIS ===")
    
    # Get the logistic regression model
    lr_model = logistic.named_steps['clf']
    scaler = logistic.named_steps['scale']
    imputer = logistic.named_steps['impute']
    
    # Get subject's features
    subject_features = X.iloc[subject_idx:subject_idx+1]
    
    # Transform features (impute + scale)
    subject_imputed = imputer.transform(subject_features)
    subject_scaled = scaler.transform(subject_imputed)
    
    # Get coefficients
    coefficients = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    # Calculate log-odds
    log_odds = intercept + np.sum(coefficients * subject_scaled[0])
    probability = 1 / (1 + np.exp(-log_odds))
    
    print(f"Intercept: {intercept:.6f}")
    print(f"Log-odds: {log_odds:.6f}")
    print(f"Probability: {probability:.6f}")
    
    # Show top contributing features
    feature_contributions = coefficients * subject_scaled[0]
    contribution_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'scaled_value': subject_scaled[0],
        'contribution': feature_contributions
    }).sort_values('contribution', key=abs, ascending=False)
    
    print(f"\nTop 10 contributing features:")
    for i, row in contribution_df.head(10).iterrows():
        print(f"  {row['feature']:30s} | Coef: {row['coefficient']:8.4f} | Value: {row['scaled_value']:8.4f} | Contrib: {row['contribution']:8.4f}")
    
    # Show MemTrax-related features
    print(f"\nMemTrax-related features:")
    memtrax_features = contribution_df[contribution_df['feature'].str.contains('rt|correct|memtrax', case=False, na=False)]
    for i, row in memtrax_features.iterrows():
        print(f"  {row['feature']:30s} | Coef: {row['coefficient']:8.4f} | Value: {row['scaled_value']:8.4f} | Contrib: {row['contribution']:8.4f}")
    
    return contribution_df

def analyze_random_forest(rf, X, y, subject_idx, feature_names):
    """Analyze random forest feature importance and predictions"""
    print("\n=== RANDOM FOREST ANALYSIS ===")
    
    # Get the random forest model
    rf_model = rf.named_steps['clf']
    imputer = rf.named_steps['impute']
    
    # Get subject's features
    subject_features = X.iloc[subject_idx:subject_idx+1]
    subject_imputed = imputer.transform(subject_features)
    
    # Get feature importance
    feature_importance = rf_model.feature_importances_
    
    # Get prediction
    prediction = rf_model.predict_proba(subject_imputed)[0, 1]
    
    print(f"Prediction probability: {prediction:.6f}")
    
    # Show feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
        'value': subject_imputed[0]
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s} | Importance: {row['importance']:8.4f} | Value: {row['value']:8.4f}")
    
    # Show MemTrax-related features
    print(f"\nMemTrax-related features:")
    memtrax_features = importance_df[importance_df['feature'].str.contains('rt|correct|memtrax', case=False, na=False)]
    for i, row in memtrax_features.iterrows():
        print(f"  {row['feature']:30s} | Importance: {row['importance']:8.4f} | Value: {row['value']:8.4f}")
    
    return importance_df

def compare_with_training_data(X_train, y_train, subject_features, feature_names):
    """Compare subject with similar cases in training data"""
    print("\n=== TRAINING DATA COMPARISON ===")
    
    # Find subjects with similar RT
    rt_col = 'CorrectResponsesRT_mean'
    if rt_col in X_train.columns:
        subject_rt = subject_features[rt_col].iloc[0]
        similar_rt_mask = (X_train[rt_col] >= subject_rt * 0.9) & (X_train[rt_col] <= subject_rt * 1.1)
        similar_cases = X_train[similar_rt_mask]
        similar_labels = y_train[similar_rt_mask]
        
        print(f"Subject RT: {subject_rt:.4f}")
        print(f"Similar RT cases (90-110% of subject RT): {len(similar_cases)}")
        if len(similar_cases) > 0:
            print(f"MCI rate in similar RT cases: {similar_labels.mean():.3f}")
            
            # Show RT distribution
            print(f"RT range in similar cases: {similar_cases[rt_col].min():.4f} - {similar_cases[rt_col].max():.4f}")
            print(f"RT mean in similar cases: {similar_cases[rt_col].mean():.4f}")
            print(f"RT std in similar cases: {similar_cases[rt_col].std():.4f}")
    
    # Find subjects with similar accuracy
    acc_col = 'CorrectPCT_mean'
    if acc_col in X_train.columns:
        subject_acc = subject_features[acc_col].iloc[0]
        similar_acc_mask = (X_train[acc_col] >= subject_acc * 0.9) & (X_train[acc_col] <= subject_acc * 1.1)
        similar_acc_cases = X_train[similar_acc_mask]
        similar_acc_labels = y_train[similar_acc_mask]
        
        print(f"\nSubject Accuracy: {subject_acc:.4f}")
        print(f"Similar Accuracy cases (90-110% of subject accuracy): {len(similar_acc_cases)}")
        if len(similar_acc_cases) > 0:
            print(f"MCI rate in similar accuracy cases: {similar_acc_labels.mean():.3f}")
    
    # Find subjects with similar education
    edu_col = 'Education'
    if edu_col in X_train.columns:
        subject_edu = subject_features[edu_col].iloc[0]
        similar_edu_mask = (X_train[edu_col] >= subject_edu - 2) & (X_train[edu_col] <= subject_edu + 2)
        similar_edu_cases = X_train[similar_edu_mask]
        similar_edu_labels = y_train[similar_edu_mask]
        
        print(f"\nSubject Education: {subject_edu:.1f} years")
        print(f"Similar Education cases (±2 years): {len(similar_edu_cases)}")
        if len(similar_edu_cases) > 0:
            print(f"MCI rate in similar education cases: {similar_edu_labels.mean():.3f}")

def main():
    """Main function"""
    print("="*60)
    print("Debug Model Predictions: BHR-ALL-49397")
    print("="*60)
    print("Why are models giving low probabilities despite high RT?")
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
    print("4. Computing aggregated features...")
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
    
    # Load medical history and SP-ECOG
    print("\n5. Loading medical history and SP-ECOG...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
    sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    print(f"   Loaded {len(sp_ecog)} SP-ECOG records")
    
    # Build consensus labels
    print("6. Building consensus labels...")
    labels = build_consensus_labels(med_hx, sp_ecog)
    print(f"   Consensus labels: {len(labels)} subjects")
    print(f"   MCI prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Merge features and labels
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data)} subjects")
    print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
    
    # Prepare features and labels
    feature_cols = [col for col in data.columns if col not in ['SubjectCode', 'cognitive_impairment', 'self_report_impairment', 'sp_ecog_impairment']]
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    # Train/test split
    print("\n7. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Get test subject codes
    test_subjects = data.iloc[X_test.index]['SubjectCode'].values
    
    # Create and train individual models
    print("8. Training individual models...")
    logistic, rf, histgb = create_individual_models()
    logistic.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    histgb.fit(X_train, y_train)
    
    # Find BHR-ALL-49397
    print("\n9. Analyzing BHR-ALL-49397...")
    target_subject = 'BHR-ALL-49397'
    
    # Check if subject is in test set
    subject_idx = None
    for i, subject in enumerate(test_subjects):
        if subject == target_subject:
            subject_idx = i
            break
    
    if subject_idx is None:
        print(f"   Subject {target_subject} not found in test set!")
        return
    
    # Get subject data
    subject_data = data[data['SubjectCode'] == target_subject].iloc[0]
    subject_features = X_test.iloc[subject_idx:subject_idx+1]
    actual_label = y_test.iloc[subject_idx]
    
    print(f"   Subject: {target_subject}")
    print(f"   Actual label: {actual_label}")
    
    # Show key feature values
    print(f"\n10. Key feature values:")
    key_features = ['CorrectResponsesRT_mean', 'CorrectPCT_mean', 'CogScore', 'RT_CV', 'Education', 'AgeRange']
    for feat in key_features:
        if feat in subject_features.columns:
            value = subject_features[feat].iloc[0]
            print(f"   {feat}: {value}")
    
    # Analyze each model
    lr_contrib = analyze_logistic_regression(logistic, X_test, y_test, subject_idx, feature_cols)
    rf_contrib = analyze_random_forest(rf, X_test, y_test, subject_idx, feature_cols)
    
    # Compare with training data
    compare_with_training_data(X_train, y_train, subject_features, feature_cols)
    
    # Show feature scaling effects
    print(f"\n=== FEATURE SCALING ANALYSIS ===")
    scaler = logistic.named_steps['scale']
    imputer = logistic.named_steps['impute']
    
    subject_imputed = imputer.transform(subject_features)
    subject_scaled = scaler.transform(subject_imputed)
    
    print(f"Original RT: {subject_features['CorrectResponsesRT_mean'].iloc[0]:.4f}")
    print(f"Scaled RT: {subject_scaled[0][feature_cols.index('CorrectResponsesRT_mean')]:.4f}")
    print(f"RT scaling factor: {scaler.scale_[feature_cols.index('CorrectResponsesRT_mean')]:.4f}")
    print(f"RT mean (training): {scaler.mean_[feature_cols.index('CorrectResponsesRT_mean')]:.4f}")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
