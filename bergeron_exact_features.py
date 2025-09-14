#!/usr/bin/env python3
"""
Bergeron Exact Features - Medical History Focus
==============================================

This script uses Bergeron's EXACT demographic features:
- MemTrax: Percent correct + Response time (2 features)
- Demographics: Age, Sex, Education, Hypertension, Diabetes, 
  Hyperlipidemia, Stroke, Heart Disease (8 features)
- Total: 10 features

This focuses on medical history features that are more predictive of cognitive health.
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
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

warnings.filterwarnings('ignore')

# Data paths
MEMTRAX_DIR = Path('../bhr/from_paul/processed/')
DATA_DIR = Path('../bhr/BHR-ALL-EXT_Mem_2022/')
OUTPUT_DIR = Path('bhr_memtrax_results')

# Cognitive QIDs for labels only
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality criteria for cognitive data validity"""
    return df[(df['CorrectPCT'] >= min_acc) &
              (df['CorrectResponsesRT'].between(0.5, 2.5))].copy()

def extract_bergeron_memtrax_features(df):
    """Extract ONLY the exact MemTrax features Bergeron used"""
    features = []
    for subject, group in df.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Bergeron used only 2 MemTrax features:
        # 1. Percent correct
        # 2. Response time
        
        # Calculate mean percent correct across all sessions
        correct_pcts = group['CorrectPCT'].dropna()
        if len(correct_pcts) > 0:
            feat['memtrax_percent_correct'] = correct_pcts.mean()
        else:
            feat['memtrax_percent_correct'] = np.nan
        
        # Calculate mean response time across all sessions
        response_times = group['CorrectResponsesRT'].dropna()
        if len(response_times) > 0:
            feat['memtrax_response_time'] = response_times.mean()
        else:
            feat['memtrax_response_time'] = np.nan
        
        # Only include if we have both features
        if not (pd.isna(feat['memtrax_percent_correct']) or pd.isna(feat['memtrax_response_time'])):
            features.append(feat)
    
    return pd.DataFrame(features)

def extract_bergeron_medical_features(df, data_dir):
    """Extract Bergeron's EXACT 8 demographic/medical features"""
    print("   Loading Bergeron's exact demographic/medical features...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Load medical history for medical conditions
    med_hx = pd.read_csv(data_dir / 'BHR_MedicalHx.csv')
    if 'TimepointCode' in med_hx.columns:
        med_hx = med_hx[med_hx['TimepointCode'] == 'm00']
    med_hx = med_hx.drop_duplicates(subset=['SubjectCode'])
    
    # Merge with main data
    df = df.merge(profile, on='SubjectCode', how='left')
    df = df.merge(med_hx, on='SubjectCode', how='left')
    
    # Extract Bergeron's EXACT 8 features:
    
    # 1. Age (from AgeRange)
    if 'AgeRange' in df.columns:
        age_mapping = {
            '18-24': 21, '25-34': 29.5, '35-44': 39.5, '45-54': 49.5,
            '55-64': 59.5, '65-74': 69.5, '75-84': 79.5, '85+': 85
        }
        df['age'] = df['AgeRange'].map(age_mapping)
    else:
        df['age'] = 65  # Default age
    
    # 2. Sex (Gender: 1 = Male, 0 = Female)
    if 'Gender' in df.columns:
        df['sex_male'] = (df['Gender'] == 1).astype(int)
    else:
        df['sex_male'] = 0
    
    # 3. Education Level (YearsEducationUS_Converted)
    if 'YearsEducationUS_Converted' in df.columns:
        df['education_years'] = df['YearsEducationUS_Converted'].fillna(16)
    else:
        df['education_years'] = 16
    
    # 4. History of Hypertension (QID3 - High Blood Pressure)
    if 'QID3' in df.columns:
        df['hypertension'] = (df['QID3'] == 1).astype(int)
    else:
        df['hypertension'] = 0
    
    # 5. History of Diabetes (QID4 - Diabetes)
    if 'QID4' in df.columns:
        df['diabetes'] = (df['QID4'] == 1).astype(int)
    else:
        df['diabetes'] = 0
    
    # 6. History of Hyperlipidemia (QID5 - High Cholesterol)
    if 'QID5' in df.columns:
        df['hyperlipidemia'] = (df['QID5'] == 1).astype(int)
    else:
        df['hyperlipidemia'] = 0
    
    # 7. History of Stroke (QID1-3 - Stroke)
    if 'QID1-3' in df.columns:
        df['stroke'] = (df['QID1-3'] == 1).astype(int)
    else:
        df['stroke'] = 0
    
    # 8. History of Heart Disease (QID6 - Heart Disease)
    if 'QID6' in df.columns:
        df['heart_disease'] = (df['QID6'] == 1).astype(int)
    else:
        df['heart_disease'] = 0
    
    # Select only the 8 features Bergeron used
    bergeron_features = [
        'age', 'sex_male', 'education_years', 'hypertension',
        'diabetes', 'hyperlipidemia', 'stroke', 'heart_disease'
    ]
    
    return df[['SubjectCode'] + bergeron_features]

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

def create_bergeron_model():
    """Create a model similar to Bergeron's approach"""
    # Simple logistic regression (most common in cognitive studies)
    logistic = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Random Forest (for comparison)
    rf = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
    ])
    
    # Gradient Boosting (for comparison)
    gb = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', HistGradientBoostingClassifier(random_state=42, max_iter=100))
    ])
    
    # Stacking ensemble (like Bergeron might have used)
    stack = StackingClassifier(
        estimators=[
            ('logistic', logistic),
            ('rf', rf),
            ('gb', gb)
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # Calibrated ensemble
    calibrated_stack = CalibratedClassifierCV(stack, cv=3)
    
    return calibrated_stack, [logistic, rf, gb]

def main():
    """Main experiment function"""
    print("="*60)
    print("BERGERON EXACT MEDICAL FEATURES")
    print("="*60)
    print("Features used (Bergeron's exact medical focus):")
    print("  - MemTrax: Percent correct + Response time (2 features)")
    print("  - Medical: Age, Sex, Education, Hypertension, Diabetes,")
    print("    Hyperlipidemia, Stroke, Heart Disease (8 features)")
    print("  - Total: 10 features")
    print("Target: Medical history MCI labels")
    print()
    
    # Load MemTrax data
    print("1. Loading MemTrax data...")
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"   Loaded {len(memtrax)} MemTrax records")
    
    # Apply quality filter
    print("2. Applying Ashford quality filter...")
    memtrax_q = apply_ashford_filter(memtrax)
    print(f"   After quality filter: {len(memtrax_q)} records")
    
    # Extract Bergeron's exact MemTrax features
    print("3. Extracting Bergeron's exact MemTrax features...")
    memtrax_feat = extract_bergeron_memtrax_features(memtrax_q)
    print(f"   MemTrax features for {len(memtrax_feat)} subjects")
    
    # Extract Bergeron's exact medical features
    print("4. Extracting Bergeron's exact medical features...")
    medical_feat = extract_bergeron_medical_features(memtrax_feat, DATA_DIR)
    print(f"   Medical features for {len(medical_feat)} subjects")
    
    # Load medical history for labels only
    print("5. Loading medical history for labels...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    labels = build_composite_labels(med_hx)
    print(f"   Labels: {len(labels)} subjects")
    print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Merge all features
    features = memtrax_feat.merge(medical_feat, on='SubjectCode', how='inner')
    
    # Merge with labels
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data)} subjects")
    print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
    
    if len(data) == 0:
        print("   ERROR: No data after merging!")
        return
    
    # Prepare features and labels
    feature_cols = [col for col in data.columns if col not in ['SubjectCode', 'cognitive_impairment', 'valid']]
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    print(f"\n6. Feature Analysis:")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   MemTrax features: {len([col for col in feature_cols if 'memtrax' in col])}")
    print(f"   Medical features: {len([col for col in feature_cols if 'memtrax' not in col])}")
    print(f"   Features: {feature_cols}")
    
    # Show feature distributions
    print(f"\n7. Feature Distributions:")
    for col in feature_cols:
        if col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                print(f"   {col}: mean={data[col].mean():.3f}, std={data[col].std():.3f}")
            else:
                print(f"   {col}: {data[col].value_counts().to_dict()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n8. Train/Test Split:")
    print(f"   Train: {len(X_train)} subjects ({y_train.mean():.1%} MCI)")
    print(f"   Test: {len(X_test)} subjects ({y_test.mean():.1%} MCI)")
    
    # Create and train model
    print(f"\n9. Training Bergeron-style model...")
    model, base_models = create_bergeron_model()
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n10. Results:")
    print(f"   Test AUC: {auc:.4f}")
    print(f"   Test PR-AUC: {pr_auc:.4f}")
    
    # Compare to Bergeron's results
    print(f"\n11. Comparison to Bergeron:")
    print(f"   Bergeron AUC: 0.91 (MemTrax + medical features → MOCA-defined MCI)")
    print(f"   Our AUC: {auc:.4f} (MemTrax + medical features → Medical history MCI)")
    print(f"   Difference: {0.91 - auc:.4f}")
    
    if auc > 0.80:
        print(f"   ✅ Good performance! Close to Bergeron's results")
    elif auc > 0.70:
        print(f"   ⚠️  Moderate performance. Label quality may be limiting factor")
    else:
        print(f"   ❌ Lower performance. Significant difference from Bergeron")
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    experiment_results = {
        'experiment': 'Bergeron Exact Medical Features',
        'auc': auc,
        'pr_auc': pr_auc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mci_prevalence_train': y_train.mean(),
        'mci_prevalence_test': y_test.mean(),
        'n_features': len(feature_cols),
        'features_used': feature_cols,
        'bergeron_features': {
            'memtrax_features': ['memtrax_percent_correct', 'memtrax_response_time'],
            'medical_features': ['age', 'sex_male', 'education_years', 'hypertension', 
                               'diabetes', 'hyperlipidemia', 'stroke', 'heart_disease'],
            'total_features': 10
        },
        'bergeron_comparison': {
            'bergeron_auc': 0.91,
            'our_auc': auc,
            'difference': 0.91 - auc,
            'bergeron_target': 'MOCA-defined MCI',
            'our_target': 'Medical history MCI'
        }
    }
    
    with open(OUTPUT_DIR / 'bergeron_exact_medical_features_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR / 'bergeron_exact_medical_features_results.json'}")

if __name__ == "__main__":
    main()
