#!/usr/bin/env python3
"""
Low Probability Case Analysis: BHR-ALL-49397
===========================================

This script investigates why BHR-ALL-49397 has such a low MCI probability (0.020167)
despite having high MemTrax RT (1.25). We'll examine:
- All feature values and their contributions
- Model feature importance
- Individual model predictions
- Feature interactions
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

def create_best_model():
    """Create the best performing model configuration"""
    logistic, rf, histgb = create_individual_models()
    
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
    
    return calibrated_stack, logistic, rf, histgb

def analyze_feature_contributions(model, X, feature_names, subject_idx):
    """Analyze feature contributions for a specific subject"""
    # Get the base models from the stacking classifier
    base_models = model.base_estimator.estimators_
    meta_learner = model.base_estimator.final_estimator_
    
    # Get predictions from each base model
    base_predictions = []
    for name, base_model in base_models:
        pred = base_model.predict_proba(X[subject_idx:subject_idx+1])[:, 1]
        base_predictions.append(pred[0])
        print(f"  {name}: {pred[0]:.6f}")
    
    # Get meta-learner prediction
    base_pred_array = np.array(base_predictions).reshape(1, -1)
    meta_pred = meta_learner.predict_proba(base_pred_array)[0, 1]
    print(f"  Meta-learner: {meta_pred:.6f}")
    
    # Get feature importance from Random Forest
    rf_model = base_models[1]  # Random Forest
    feature_importance = rf_model.named_steps['clf'].feature_importances_
    
    # Get subject's feature values
    subject_features = X.iloc[subject_idx]
    
    # Create feature analysis
    feature_analysis = []
    for i, (feature, importance, value) in enumerate(zip(feature_names, feature_importance, subject_features)):
        feature_analysis.append({
            'feature': feature,
            'importance': importance,
            'value': value,
            'contribution': importance * value  # Simplified contribution
        })
    
    # Sort by importance
    feature_analysis.sort(key=lambda x: x['importance'], reverse=True)
    
    return feature_analysis, base_predictions, meta_pred

def main():
    """Main function"""
    print("="*60)
    print("Low Probability Case Analysis: BHR-ALL-49397")
    print("="*60)
    print("Why does this subject have low MCI probability despite high RT?")
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
    
    # Create and train models
    print("8. Training models...")
    model, logistic, rf, histgb = create_best_model()
    model.fit(X_train, y_train)
    
    # Train individual models
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
    prediction_prob = model.predict_proba(X_test.iloc[subject_idx:subject_idx+1])[:, 1][0]
    actual_label = y_test.iloc[subject_idx]
    
    print(f"   Subject: {target_subject}")
    print(f"   Prediction probability: {prediction_prob:.6f}")
    print(f"   Actual label: {actual_label}")
    
    # Get individual model predictions
    print(f"\n10. Individual model predictions:")
    logistic_pred = logistic.predict_proba(X_test.iloc[subject_idx:subject_idx+1])[:, 1][0]
    rf_pred = rf.predict_proba(X_test.iloc[subject_idx:subject_idx+1])[:, 1][0]
    histgb_pred = histgb.predict_proba(X_test.iloc[subject_idx:subject_idx+1])[:, 1][0]
    
    print(f"   Logistic Regression: {logistic_pred:.6f}")
    print(f"   Random Forest: {rf_pred:.6f}")
    print(f"   HistGradientBoosting: {histgb_pred:.6f}")
    print(f"   Ensemble (Calibrated): {prediction_prob:.6f}")
    
    # Analyze feature contributions
    print(f"\n11. Feature analysis:")
    feature_analysis, base_predictions, meta_pred = analyze_feature_contributions(
        model, X_test, feature_cols, subject_idx
    )
    
    print(f"\n12. Top 20 most important features for this subject:")
    for i, feat in enumerate(feature_analysis[:20]):
        print(f"   {i+1:2d}. {feat['feature']:30s} | Importance: {feat['importance']:.4f} | Value: {feat['value']:8.4f}")
    
    # Focus on MemTrax-related features
    print(f"\n13. MemTrax-related features:")
    memtrax_features = [f for f in feature_analysis if 'memtrax' in f['feature'].lower() or 'rt' in f['feature'].lower() or 'correct' in f['feature'].lower()]
    for feat in memtrax_features:
        print(f"   {feat['feature']:30s} | Importance: {feat['importance']:.4f} | Value: {feat['value']:8.4f}")
    
    # Focus on demographic features
    print(f"\n14. Demographic features:")
    demo_features = [f for f in feature_analysis if 'education' in f['feature'].lower() or 'gender' in f['feature'].lower() or 'age' in f['feature'].lower()]
    for feat in demo_features:
        print(f"   {feat['feature']:30s} | Importance: {feat['importance']:.4f} | Value: {feat['value']:8.4f}")
    
    # Compare with other MCI cases
    print(f"\n15. Comparing with other MCI cases:")
    mci_cases = X_test[y_test == 1]
    mci_predictions = model.predict_proba(mci_cases)[:, 1]
    
    print(f"   MCI cases in test set: {len(mci_cases)}")
    print(f"   Mean MCI prediction probability: {mci_predictions.mean():.6f}")
    print(f"   Median MCI prediction probability: {np.median(mci_predictions):.6f}")
    print(f"   Min MCI prediction probability: {mci_predictions.min():.6f}")
    print(f"   Max MCI prediction probability: {mci_predictions.max():.6f}")
    print(f"   BHR-ALL-49397 percentile: {np.percentile(mci_predictions, (prediction_prob > mci_predictions).sum() / len(mci_predictions) * 100):.1f}%")
    
    # Save detailed analysis
    print(f"\n16. Saving analysis...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    analysis_results = {
        'subject': target_subject,
        'prediction_probability': float(prediction_prob),
        'actual_label': int(actual_label),
        'individual_predictions': {
            'logistic': float(logistic_pred),
            'random_forest': float(rf_pred),
            'histgradientboosting': float(histgb_pred),
            'ensemble': float(prediction_prob)
        },
        'feature_analysis': [
            {
                'feature': feat['feature'],
                'importance': float(feat['importance']),
                'value': float(feat['value']) if not pd.isna(feat['value']) else None,
                'contribution': float(feat['contribution']) if not pd.isna(feat['contribution']) else None
            }
            for feat in feature_analysis
        ],
        'mci_comparison': {
            'n_mci_cases': len(mci_cases),
            'mean_probability': float(mci_predictions.mean()),
            'median_probability': float(np.median(mci_predictions)),
            'min_probability': float(mci_predictions.min()),
            'max_probability': float(mci_predictions.max())
        }
    }
    
    with open(OUTPUT_DIR / 'bhr_49397_low_probability_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"   Analysis saved to: {OUTPUT_DIR / 'bhr_49397_low_probability_analysis.json'}")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"BHR-ALL-49397 has low probability despite high RT because:")
    print(f"1. Individual models show different predictions")
    print(f"2. Other features may be counteracting the high RT signal")
    print(f"3. The ensemble calibration may be reducing the probability")
    print(f"4. Check the detailed feature analysis for specific factors")
    print("="*60)

if __name__ == "__main__":
    main()
