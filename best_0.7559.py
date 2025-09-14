#!/usr/bin/env python3
"""
Experiment 4: Standalone Script - No Data Leakage
===============================================

This is a completely standalone version that removes all potential sources of data leakage:
- No ECOG/SP-ECOG features (these are cognitive assessments themselves)
- No medical history features (these could leak label information)
- Only MemTrax performance + demographics
- Proper train/test split
- No data leakage between training and testing

Expected Impact: Should give us the true, honest AUC without any leakage
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
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

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

def extract_advanced_memtrax_features(df):
    """Extract advanced MemTrax features similar to Bergeron's approach"""
    features = []
    for subject, group in df.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Collect all reaction times and responses
        all_rts = []
        all_accuracies = []
        all_responses = []  # 1 = correct, 0 = incorrect
        
        for _, row in group.iterrows():
            if pd.notna(row.get('ReactionTimes')):
                try:
                    rts = [float(x.strip()) for x in str(row['ReactionTimes']).split(',') 
                           if x.strip() and x.strip() != 'nan']
                    # Filter valid RTs (0.3-3.0 seconds)
                    valid_rts = [r for r in rts if 0.3 <= r <= 3.0]
                    all_rts.extend(valid_rts)
                    
                    # Get accuracy for each trial
                    if pd.notna(row.get('CorrectPCT')):
                        acc = row['CorrectPCT'] / 100.0
                        all_accuracies.extend([acc] * len(valid_rts))
                        # Approximate individual responses (1 = correct, 0 = incorrect)
                        n_correct = int(acc * len(valid_rts))
                        responses = [1] * n_correct + [0] * (len(valid_rts) - n_correct)
                        all_responses.extend(responses)
                except:
                    continue
        
        if len(all_rts) >= 10:
            n = len(all_rts)
            
            # === BASIC STATISTICS ===
            feat['seq_mean_rt'] = np.mean(all_rts)
            feat['seq_std_rt'] = np.std(all_rts)
            feat['seq_cv'] = feat['seq_std_rt'] / (feat['seq_mean_rt'] + 1e-6)
            feat['seq_median_rt'] = np.median(all_rts)
            feat['seq_iqr_rt'] = np.percentile(all_rts, 75) - np.percentile(all_rts, 25)
            
            # === LEARNING CURVE FEATURES ===
            # RT improvement over time (negative slope = improvement)
            if n >= 3:
                slope, intercept = np.polyfit(np.arange(n), all_rts, 1)
                feat['rt_learning_slope'] = slope
                feat['rt_learning_intercept'] = intercept
                
                # Exponential decay fit for learning curve
                try:
                    from scipy.optimize import curve_fit
                    def exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    popt, _ = curve_fit(exp_decay, np.arange(n), all_rts, maxfev=1000)
                    feat['rt_learning_rate'] = popt[1]  # b parameter
                    feat['rt_learning_plateau'] = popt[2]  # c parameter
                except:
                    feat['rt_learning_rate'] = 0
                    feat['rt_learning_plateau'] = feat['seq_mean_rt']
            
            # === FATIGUE PATTERNS ===
            # Performance in first vs last 25% of trials
            quarter = max(1, n // 4)
            feat['rt_first_quarter'] = np.mean(all_rts[:quarter])
            feat['rt_last_quarter'] = np.mean(all_rts[-quarter:])
            feat['rt_fatigue'] = feat['rt_last_quarter'] - feat['rt_first_quarter']
            
            # Performance in first vs last third
            third = max(1, n // 3)
            feat['rt_first_third'] = np.mean(all_rts[:third])
            feat['rt_last_third'] = np.mean(all_rts[-third:])
            feat['rt_fatigue_third'] = feat['rt_last_third'] - feat['rt_first_third']
            
            # === VARIABILITY MEASURES ===
            # Intra-individual variability (coefficient of variation)
            feat['rt_cv'] = feat['seq_std_rt'] / (feat['seq_mean_rt'] + 1e-6)
            
            # RT consistency (inverse of standard deviation)
            feat['rt_consistency'] = 1.0 / (feat['seq_std_rt'] + 1e-6)
            
            # Variability change over time
            mid = n // 2
            if mid > 1:
                first_half_var = np.var(all_rts[:mid])
                second_half_var = np.var(all_rts[mid:])
                feat['rt_variability_change'] = second_half_var - first_half_var
                feat['rt_variability_ratio'] = second_half_var / (first_half_var + 1e-6)
            
            # === SPEED-ACCURACY TRADEOFFS ===
            if len(all_accuracies) > 0:
                mean_acc = np.mean(all_accuracies)
                feat['speed_accuracy_efficiency'] = mean_acc / (feat['seq_mean_rt'] + 1e-6)
                feat['speed_accuracy_product'] = mean_acc * (1.0 / (feat['seq_mean_rt'] + 1e-6))
                
                # Correlation between RT and accuracy
                if len(all_rts) == len(all_accuracies):
                    corr = np.corrcoef(all_rts, all_accuracies)[0, 1]
                    feat['rt_acc_correlation'] = corr if not np.isnan(corr) else 0
            
            # === ERROR PATTERN ANALYSIS ===
            if len(all_responses) > 0:
                mean_accuracy = np.mean(all_responses)
                feat['trial_accuracy'] = mean_accuracy
                
                # Error streaks
                error_streaks = []
                current_streak = 0
                for resp in all_responses:
                    if resp == 0:  # Error
                        current_streak += 1
                    else:  # Correct
                        if current_streak > 0:
                            error_streaks.append(current_streak)
                            current_streak = 0
                if current_streak > 0:
                    error_streaks.append(current_streak)
                
                feat['max_error_streak'] = max(error_streaks) if error_streaks else 0
                feat['mean_error_streak'] = np.mean(error_streaks) if error_streaks else 0
                
                # Correct streaks
                correct_streaks = []
                current_streak = 0
                for resp in all_responses:
                    if resp == 1:  # Correct
                        current_streak += 1
                    else:  # Error
                        if current_streak > 0:
                            correct_streaks.append(current_streak)
                            current_streak = 0
                if current_streak > 0:
                    correct_streaks.append(current_streak)
                
                feat['max_correct_streak'] = max(correct_streaks) if correct_streaks else 0
                feat['mean_correct_streak'] = np.mean(correct_streaks) if correct_streaks else 0
            
            # === SEQUENCE FEATURES ===
            # Alternating patterns (alternating between fast/slow)
            if n >= 4:
                rt_diffs = np.diff(all_rts)
                alternating_patterns = np.sum(np.diff(np.sign(rt_diffs)) != 0)
                feat['rt_alternating_patterns'] = alternating_patterns / (n - 2)
            
            # RT autocorrelation (consistency over time)
            if n >= 10:
                try:
                    from scipy.stats import pearsonr
                    lag1_corr, _ = pearsonr(all_rts[:-1], all_rts[1:])
                    feat['rt_autocorrelation'] = lag1_corr if not np.isnan(lag1_corr) else 0
                except:
                    feat['rt_autocorrelation'] = 0
            
            # === COMPOSITE SCORES ===
            # Cognitive efficiency score (accuracy per unit time)
            if len(all_accuracies) > 0:
                feat['cognitive_efficiency'] = np.mean(all_accuracies) / (feat['seq_mean_rt'] + 1e-6)
            
            # Stability score (inverse of variability)
            feat['rt_stability'] = 1.0 / (feat['seq_cv'] + 1e-6)
            
            # Learning efficiency (improvement per trial)
            if n >= 3:
                feat['learning_efficiency'] = -feat['rt_learning_slope']  # Negative slope = improvement
            
            # === SESSION METADATA ===
            feat['n_tests'] = len(group)
            feat['n_trials'] = n
            
            # Test frequency (tests per day if we have dates)
            if 'StatusDateTime' in group.columns:
                try:
                    group['StatusDateTime'] = pd.to_datetime(group['StatusDateTime'])
                    days_span = (group['StatusDateTime'].max() - group['StatusDateTime'].min()).days
                    if days_span > 0:
                        feat['test_frequency'] = len(group) / days_span
                    else:
                        feat['test_frequency'] = len(group)
                except:
                    feat['test_frequency'] = len(group)
            else:
                feat['test_frequency'] = len(group)
        
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

def add_demographics_only(df, data_dir):
    """Add ONLY basic demographics - no cognitive assessments"""
    print("   Loading demographics only...")
    
    # Load Profile.csv for basic demographics
    profile = pd.read_csv(data_dir / 'Profile.csv')
    profile = profile.rename(columns={'Code': 'SubjectCode'})
    
    # Merge with main data
    df = df.merge(profile[['SubjectCode', 'YearsEducationUS_Converted', 'Gender']], 
                  on='SubjectCode', how='left')
    
    # Basic demographic features only
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
    print("EXPERIMENT 5: Advanced MemTrax Features - Bergeron Style")
    print("="*60)
    print("Features used:")
    print("  - Advanced MemTrax features (learning curves, fatigue, variability)")
    print("  - Error pattern analysis (streaks, consistency)")
    print("  - Speed-accuracy tradeoffs and efficiency scores")
    print("  - Basic demographics (education, gender)")
    print("  - NO ECOG/SP-ECOG features (cognitive assessments)")
    print("  - NO medical history features")
    print("Expected Impact: Higher AUC using sophisticated MemTrax feature engineering")
    print()
    
    # Load MemTrax data
    print("1. Loading MemTrax data...")
    memtrax = pd.read_csv(MEMTRAX_DIR / 'MemTraxRecalculated.csv')
    print(f"   Loaded {len(memtrax)} MemTrax records")
    
    # Apply quality filter
    print("2. Applying Ashford quality filter...")
    memtrax_q = apply_ashford_filter(memtrax)
    print(f"   After quality filter: {len(memtrax_q)} records")
    
    # Extract advanced MemTrax features
    print("3. Extracting advanced MemTrax features...")
    seq_feat = extract_advanced_memtrax_features(memtrax_q)
    print(f"   Advanced features for {len(seq_feat)} subjects")
    
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
    
    # Load medical history for labels only
    print("\n4. Loading medical history for labels only...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv')
    labels = build_composite_labels(med_hx)
    print(f"   Labels: {len(labels)} subjects")
    print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Add demographics only
    print("5. Adding demographics only...")
    features = add_demographics_only(features, DATA_DIR)
    
    # Merge with labels
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data)} subjects")
    print(f"   MCI prevalence: {data['cognitive_impairment'].mean():.1%}")
    print(f"   Features: {data.shape[1]-3}")  # Exclude SubjectCode, cognitive_impairment, valid
    
    if len(data) == 0:
        print("   ERROR: No data after merging!")
        return
    
    # Prepare features and labels
    feature_cols = [col for col in data.columns if col not in ['SubjectCode', 'cognitive_impairment', 'valid']]
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    print(f"\n6. Feature Analysis:")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Advanced MemTrax features: {len([col for col in feature_cols if any(x in col for x in ['seq_', 'rt_', 'cognitive_', 'learning_', 'fatigue_', 'streak_', 'efficiency_', 'stability_'])])}")
    print(f"   Basic MemTrax features: {len([col for col in feature_cols if any(x in col for x in ['Correct', 'Incorrect', 'CogScore', 'RT_CV', 'Speed_Accuracy'])])}")
    print(f"   Demographic features: {len([col for col in feature_cols if any(x in col for x in ['Education', 'Gender', 'age'])])}")
    
    # Show some example advanced features
    advanced_features = [col for col in feature_cols if any(x in col for x in ['rt_learning', 'rt_fatigue', 'cognitive_efficiency', 'rt_stability', 'error_streak', 'correct_streak'])]
    if advanced_features:
        print(f"   Example advanced features: {advanced_features[:5]}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n7. Train/Test Split:")
    print(f"   Train: {len(X_train)} subjects ({y_train.mean():.1%} MCI)")
    print(f"   Test: {len(X_test)} subjects ({y_test.mean():.1%} MCI)")
    
    # Create and train model
    print(f"\n8. Training model...")
    model, base_models = create_best_model()
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n9. Results:")
    print(f"   Test AUC: {auc:.4f}")
    print(f"   Test PR-AUC: {pr_auc:.4f}")
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    experiment_results = {
        'experiment': 'Advanced MemTrax Features - Bergeron Style',
        'auc': auc,
        'pr_auc': pr_auc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'mci_prevalence_train': y_train.mean(),
        'mci_prevalence_test': y_test.mean(),
        'n_features': len(feature_cols),
        'features_used': feature_cols,
        'advanced_features_added': [
            'Learning curve analysis (slope, exponential decay)',
            'Fatigue patterns (first vs last quarter/third)',
            'Variability measures (CV, consistency, change over time)',
            'Speed-accuracy tradeoffs and efficiency scores',
            'Error pattern analysis (streaks, clustering)',
            'Sequence features (autocorrelation, alternating patterns)',
            'Composite cognitive scores'
        ],
        'leakage_sources_removed': [
            'ECOG/SP-ECOG features (cognitive assessments)',
            'Medical history features',
            'Informant scores'
        ]
    }
    
    with open(OUTPUT_DIR / 'experiment5_advanced_memtrax_bergeron_style.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR / 'experiment5_advanced_memtrax_bergeron_style.json'}")
    
    if auc > 0.80:
        print(f"\n🎯 SUCCESS! Achieved {auc:.4f} AUC without leakage")
    elif auc > 0.75:
        print(f"\n✅ Good performance: {auc:.4f} AUC")
    else:
        print(f"\n❌ Lower than expected: {auc:.4f} AUC")

if __name__ == "__main__":
    main()
