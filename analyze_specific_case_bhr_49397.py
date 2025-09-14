#!/usr/bin/env python3
"""
Specific Case Analysis: BHR-ALL-49397
====================================

This script analyzes why BHR-ALL-49397 wasn't predicted to have MCI despite
having high MemTrax RT (1.25). We'll examine:
- All QID values with proper descriptions
- ECOG scores by domain
- MemTrax performance metrics
- Demographics including age
- Model prediction probability and threshold
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

def create_best_model():
    """Create the best performing model configuration"""
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
    
    return calibrated_stack

def optimize_threshold(y_true, y_pred_proba):
    """Optimize decision threshold for clinical utility"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Youden's J statistic (maximize sensitivity + specificity - 1)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    youden_threshold = thresholds[optimal_idx]
    
    return {
        'youden_threshold': youden_threshold,
        'youden_sensitivity': tpr[optimal_idx],
        'youden_specificity': 1 - fpr[optimal_idx]
    }

def load_data_dictionary():
    """Load the data dictionary to get QID descriptions"""
    try:
        data_dict = pd.read_csv(DATA_DIR / 'DataDictionary/DataDictionary.csv')
        return data_dict
    except Exception as e:
        print(f"Could not load data dictionary: {e}")
        return None

def get_qid_descriptions(data_dict, qids):
    """Get descriptions for QIDs from data dictionary"""
    descriptions = {}
    if data_dict is not None:
        for qid in qids:
            qid_info = data_dict[data_dict['ColumnName'] == qid]
            if not qid_info.empty:
                descriptions[qid] = qid_info.iloc[0]['Description']
            else:
                descriptions[qid] = f"Description not found for {qid}"
    else:
        for qid in qids:
            descriptions[qid] = f"Data dictionary not available for {qid}"
    return descriptions

def main():
    """Main function"""
    print("="*60)
    print("Specific Case Analysis: BHR-ALL-49397")
    print("="*60)
    print("Why wasn't this subject predicted to have MCI despite high RT?")
    print()
    
    # Load data dictionary
    print("1. Loading data dictionary...")
    data_dict = load_data_dictionary()
    
    # Load MemTrax data
    print("2. Loading MemTrax data...")
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"   Loaded {len(memtrax)} MemTrax records")
    
    # Apply quality filter
    print("3. Applying Ashford quality filter...")
    memtrax_q = apply_ashford_filter(memtrax)
    print(f"   After quality filter: {len(memtrax_q)} records")
    
    # Extract sequence features
    print("4. Extracting sequence features...")
    seq_feat = extract_sequence_features(memtrax_q)
    print(f"   Sequence features for {len(seq_feat)} subjects")
    
    # Aggregates
    print("5. Computing aggregated features...")
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
    print("\n6. Loading medical history and SP-ECOG...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv')
    sp_ecog = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
    sp_ecog = sp_ecog.drop_duplicates(subset=['SubjectCode'])
    print(f"   Loaded {len(sp_ecog)} SP-ECOG records")
    
    # Load self-reported ECOG
    print("7. Loading self-reported ECOG...")
    try:
        self_ecog = pd.read_csv(DATA_DIR / 'BHR_EverydayCognition.csv')
        self_ecog = self_ecog[self_ecog['TimepointCode'] == 'm00'].copy()
        self_ecog = self_ecog.drop_duplicates(subset=['SubjectCode'])
        print(f"   Loaded {len(self_ecog)} self-ECOG records")
    except Exception as e:
        print(f"   Could not load self-ECOG: {e}")
        self_ecog = pd.DataFrame()
    
    # Build consensus labels
    print("8. Building consensus labels...")
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
    print("\n9. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Get test subject codes
    test_subjects = data.iloc[X_test.index]['SubjectCode'].values
    
    # Create and train model
    print("10. Training model...")
    model = create_best_model()
    model.fit(X_train, y_train)
    
    # Get predictions
    print("11. Getting predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold
    print("12. Optimizing threshold...")
    threshold_results = optimize_threshold(y_test, y_pred_proba)
    optimal_threshold = threshold_results['youden_threshold']
    
    print(f"   Youden's J threshold: {optimal_threshold:.6f}")
    print(f"   Youden's sensitivity: {threshold_results['youden_sensitivity']:.3f}")
    print(f"   Youden's specificity: {threshold_results['youden_specificity']:.3f}")
    
    # Find BHR-ALL-49397
    print("\n13. Analyzing BHR-ALL-49397...")
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
    prediction_prob = y_pred_proba[subject_idx]
    actual_label = y_test.iloc[subject_idx]
    prediction = 1 if prediction_prob > optimal_threshold else 0
    
    print(f"   Subject: {target_subject}")
    print(f"   Prediction probability: {prediction_prob:.6f}")
    print(f"   Optimal threshold: {optimal_threshold:.6f}")
    print(f"   Model prediction: {prediction}")
    print(f"   Actual label: {actual_label}")
    print(f"   Above threshold: {prediction_prob > optimal_threshold}")
    
    # Get QID descriptions
    print("\n14. Getting QID descriptions...")
    qid_descriptions = get_qid_descriptions(data_dict, COGNITIVE_QIDS)
    
    # Get all QIDs from medical history
    all_qids = [col for col in med_hx.columns if col.startswith('QID1-')]
    
    # Create detailed analysis for this subject
    print("\n15. Creating detailed analysis...")
    
    analysis_data = {
        'subject_code': target_subject,
        'prediction_probability': prediction_prob,
        'optimal_threshold': optimal_threshold,
        'model_prediction': prediction,
        'actual_label': actual_label,
        'above_threshold': prediction_prob > optimal_threshold
    }
    
    # MemTrax performance
    analysis_data.update({
        'memtrax_accuracy_mean': subject_data.get('CorrectPCT_mean', np.nan),
        'memtrax_rt_mean': subject_data.get('CorrectResponsesRT_mean', np.nan),
        'memtrax_cogscore': subject_data.get('CogScore', np.nan),
        'memtrax_rt_cv': subject_data.get('RT_CV', np.nan),
        'memtrax_speed_accuracy_product': subject_data.get('Speed_Accuracy_Product', np.nan),
        'memtrax_seq_fatigue': subject_data.get('seq_fatigue', np.nan),
        'memtrax_seq_cv': subject_data.get('seq_cv', np.nan),
        'memtrax_rt_slope': subject_data.get('rt_slope', np.nan),
        'memtrax_n_tests': subject_data.get('n_tests', np.nan)
    })
    
    # Demographics
    analysis_data.update({
        'age_range': subject_data.get('AgeRange', np.nan),
        'education_years': subject_data.get('Education', np.nan),
        'gender': subject_data.get('Gender_Num', np.nan)
    })
    
    # Consensus label components
    analysis_data.update({
        'self_report_impairment': subject_data.get('self_report_impairment', np.nan),
        'sp_ecog_impairment': subject_data.get('sp_ecog_impairment', np.nan)
    })
    
    # Get all QID values with descriptions
    subject_med = med_hx[med_hx['SubjectCode'] == target_subject]
    if not subject_med.empty:
        for qid in all_qids:
            if qid in subject_med.columns:
                value = subject_med.iloc[0][qid]
                analysis_data[f'qid_{qid}'] = value
                analysis_data[f'qid_{qid}_description'] = qid_descriptions.get(qid, f"Description not found for {qid}")
            else:
                analysis_data[f'qid_{qid}'] = np.nan
                analysis_data[f'qid_{qid}_description'] = qid_descriptions.get(qid, f"Description not found for {qid}")
    else:
        for qid in all_qids:
            analysis_data[f'qid_{qid}'] = np.nan
            analysis_data[f'qid_{qid}_description'] = qid_descriptions.get(qid, f"Description not found for {qid}")
    
    # Get SP-ECOG domain scores
    subject_sp_ecog = sp_ecog[sp_ecog['SubjectCode'] == target_subject]
    ecog_descriptions = {
        'QID49-1': 'Memory: Remembering a few shopping items without a list',
        'QID49-2': 'Memory: Remembering things that happened recently (such as recent outings, events in the news)',
        'QID49-3': 'Memory: Recalling conversations a few days later',
        'QID49-4': 'Memory: Remembering where I have placed objects',
        'QID49-5': 'Memory: Repeating stories and/or questions',
        'QID49-6': 'Memory: Remembering the current date or day of the week',
        'QID49-7': 'Memory: Remembering I have already told someone something',
        'QID49-8': 'Memory: Remembering appointments, meetings, or engagements'
    }
    
    if not subject_sp_ecog.empty:
        for domain in ['QID49-1', 'QID49-2', 'QID49-3', 'QID49-4', 'QID49-5', 'QID49-6', 'QID49-7', 'QID49-8']:
            if domain in subject_sp_ecog.columns:
                value = subject_sp_ecog.iloc[0][domain]
                analysis_data[f'sp_ecog_{domain}'] = value
                analysis_data[f'sp_ecog_{domain}_description'] = ecog_descriptions[domain]
            else:
                analysis_data[f'sp_ecog_{domain}'] = np.nan
                analysis_data[f'sp_ecog_{domain}_description'] = ecog_descriptions[domain]
    else:
        for domain in ['QID49-1', 'QID49-2', 'QID49-3', 'QID49-4', 'QID49-5', 'QID49-6', 'QID49-7', 'QID49-8']:
            analysis_data[f'sp_ecog_{domain}'] = np.nan
            analysis_data[f'sp_ecog_{domain}_description'] = ecog_descriptions[domain]
    
    # Get self-ECOG domain scores
    if not self_ecog.empty:
        subject_self_ecog = self_ecog[self_ecog['SubjectCode'] == target_subject]
        if not subject_self_ecog.empty:
            for domain in ['QID49-1', 'QID49-2', 'QID49-3', 'QID49-4', 'QID49-5', 'QID49-6', 'QID49-7', 'QID49-8']:
                if domain in subject_self_ecog.columns:
                    value = subject_self_ecog.iloc[0][domain]
                    analysis_data[f'self_ecog_{domain}'] = value
                    analysis_data[f'self_ecog_{domain}_description'] = ecog_descriptions[domain]
                else:
                    analysis_data[f'self_ecog_{domain}'] = np.nan
                    analysis_data[f'self_ecog_{domain}_description'] = ecog_descriptions[domain]
        else:
            for domain in ['QID49-1', 'QID49-2', 'QID49-3', 'QID49-4', 'QID49-5', 'QID49-6', 'QID49-7', 'QID49-8']:
                analysis_data[f'self_ecog_{domain}'] = np.nan
                analysis_data[f'self_ecog_{domain}_description'] = ecog_descriptions[domain]
    else:
        for domain in ['QID49-1', 'QID49-2', 'QID49-3', 'QID49-4', 'QID49-5', 'QID49-6', 'QID49-7', 'QID49-8']:
            analysis_data[f'self_ecog_{domain}'] = np.nan
            analysis_data[f'self_ecog_{domain}_description'] = ecog_descriptions[domain]
    
    # Save results
    print("16. Saving results...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    analysis_data_clean = convert_numpy_types(analysis_data)
    
    # Save detailed analysis
    with open(OUTPUT_DIR / 'bhr_49397_detailed_analysis.json', 'w') as f:
        json.dump(analysis_data_clean, f, indent=2)
    
    print(f"   Detailed analysis saved to: {OUTPUT_DIR / 'bhr_49397_detailed_analysis.json'}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("BHR-ALL-49397 ANALYSIS SUMMARY")
    print("="*60)
    print(f"Subject: {target_subject}")
    print(f"Age Range: {analysis_data.get('age_range', 'Not available')}")
    print(f"Education: {analysis_data.get('education_years', 'Not available')} years")
    print(f"Gender: {'Male' if analysis_data.get('gender') == 1 else 'Female' if analysis_data.get('gender') == 0 else 'Not available'}")
    print()
    print(f"MemTrax Performance:")
    print(f"  Accuracy: {analysis_data.get('memtrax_accuracy_mean', 'N/A'):.3f}")
    print(f"  RT Mean: {analysis_data.get('memtrax_rt_mean', 'N/A'):.3f}")
    print(f"  CogScore: {analysis_data.get('memtrax_cogscore', 'N/A'):.3f}")
    print(f"  RT CV: {analysis_data.get('memtrax_rt_cv', 'N/A'):.3f}")
    print()
    print(f"Model Prediction:")
    print(f"  Probability: {prediction_prob:.6f}")
    print(f"  Threshold: {optimal_threshold:.6f}")
    print(f"  Prediction: {prediction}")
    print(f"  Actual: {actual_label}")
    print()
    print(f"Cognitive QIDs:")
    for qid in COGNITIVE_QIDS:
        value = analysis_data.get(f'qid_{qid}', 'N/A')
        desc = analysis_data.get(f'qid_{qid}_description', 'N/A')
        print(f"  {qid}: {value} - {desc}")
    print()
    print(f"SP-ECOG Scores (Informant):")
    for domain in ['QID49-1', 'QID49-2', 'QID49-3', 'QID49-4', 'QID49-5', 'QID49-6', 'QID49-7', 'QID49-8']:
        value = analysis_data.get(f'sp_ecog_{domain}', 'N/A')
        desc = analysis_data.get(f'sp_ecog_{domain}_description', 'N/A')
        print(f"  {domain}: {value} - {desc}")
    print("="*60)

if __name__ == "__main__":
    main()
