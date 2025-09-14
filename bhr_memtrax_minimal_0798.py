#!/usr/bin/env python3
"""
BHR MemTrax Minimal Script - Reproduces AUC=0.798
==================================================
Self-contained script with winning configuration.
No external dependencies on improvements/ modules.
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Configuration
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

def apply_ashford_filter(df, accuracy_threshold=0.65):
    """Apply Ashford quality filters"""
    filtered = df[
        (df['Status'] == 'Collected') &
        (df['CorrectPCT'] >= accuracy_threshold) &
        (df['CorrectResponsesRT'] >= 0.5) &
        (df['CorrectResponsesRT'] <= 2.5)
    ].copy()
    return filtered

def extract_sequence_features(df):
    """Extract sequence and fatigue features from ReactionTimes"""
    def process_subject(group):
        result = {'SubjectCode': group.name}
        
        # Process each test
        for _, row in group.iterrows():
            if pd.isna(row['ReactionTimes']):
                continue
                
            try:
                rt_str = str(row['ReactionTimes']).strip()
                if not rt_str or rt_str == 'nan':
                    continue
                    
                # Parse reaction times
                rts = [float(x) for x in rt_str.split(',') if x.strip() and x.strip() != 'nan']
                if len(rts) < 3:
                    continue
                
                # Sequence features
                n_rts = len(rts)
                third = max(1, n_rts // 3)
                
                first_third = np.mean(rts[:third])
                last_third = np.mean(rts[-third:])
                fatigue_effect = last_third - first_third
                
                # Accumulate across tests
                for key, val in [
                    ('seq_first_third_mean', first_third),
                    ('seq_last_third_mean', last_third), 
                    ('seq_fatigue_effect', fatigue_effect),
                    ('seq_mean_rt', np.mean(rts)),
                    ('seq_median_rt', np.median(rts)),
                    ('long_n_timepoints', 1)
                ]:
                    if key in result:
                        if key == 'long_n_timepoints':
                            result[key] += val
                        else:
                            result[key] = (result[key] + val) / 2  # Average across tests
                    else:
                        result[key] = val
                        
                # Reliability change (simplified)
                if len(rts) >= 4:
                    mid = len(rts) // 2
                    first_half_var = np.var(rts[:mid]) if mid > 1 else 0
                    second_half_var = np.var(rts[mid:]) if len(rts) - mid > 1 else 0
                    rel_change = second_half_var - first_half_var
                    result['long_reliability_change'] = result.get('long_reliability_change', 0) + rel_change
                    
                # RT slope (simplified)
                if len(rts) >= 3:
                    x = np.arange(len(rts))
                    if np.var(x) > 0:
                        slope = np.polyfit(x, rts, 1)[0]
                        result['long_rt_slope'] = result.get('long_rt_slope', 0) + slope
                        
            except Exception:
                continue
                
        return pd.Series(result)
    
    seq_df = df.groupby('SubjectCode').apply(process_subject).reset_index(drop=True)
    return seq_df

def enrich_demographics(df, data_dir):
    """Add age, education, gender and derived features"""
    demo_files = ['BHR_Demographics.csv', 'Profile.csv', 'Participants.csv', 'Subjects.csv']
    
    for filename in demo_files:
        try:
            demo_path = data_dir / filename
            if not demo_path.exists():
                continue
                
            demo_df = pd.read_csv(demo_path, low_memory=False)
            
            # Normalize subject column
            if 'Code' in demo_df.columns and 'SubjectCode' not in demo_df.columns:
                demo_df = demo_df.rename(columns={'Code': 'SubjectCode'})
            
            if 'SubjectCode' not in demo_df.columns:
                continue
                
            # Merge relevant columns
            demo_cols = ['SubjectCode']
            for col in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
                if col in demo_df.columns:
                    demo_cols.append(col)
                    
            if len(demo_cols) > 1:
                demo_small = demo_df[demo_cols].drop_duplicates(subset=['SubjectCode'])
                df = df.merge(demo_small, on='SubjectCode', how='left')
                
        except Exception:
            continue
    
    # Derived features
    if 'Age_Baseline' in df.columns:
        df['Age_squared'] = df['Age_Baseline'] ** 2
        df['age_rt_interaction'] = df['Age_Baseline'] * df.get('CorrectResponsesRT_mean', 0)
        df['age_variability_interaction'] = df['Age_Baseline'] * df.get('CorrectResponsesRT_std', 0)
        
    if 'YearsEducationUS_Converted' in df.columns:
        df['Education_squared'] = df['YearsEducationUS_Converted'] ** 2
        
    if 'Age_Baseline' in df.columns and 'YearsEducationUS_Converted' in df.columns:
        df['Age_Education_interaction'] = df['Age_Baseline'] * df['YearsEducationUS_Converted']
        df['CognitiveReserve_Proxy'] = df['YearsEducationUS_Converted'] / (df['Age_Baseline'] + 1)
        
    if 'Gender' in df.columns:
        df['Gender_Numeric'] = df['Gender'].map({'Male': 1, 'Female': 0})
        
    return df

def add_ecog_features(df, data_dir):
    """Add ECOG global mean features"""
    try:
        ecog_path = data_dir / 'BHR_EverydayCognition.csv'
        if ecog_path.exists():
            ecog = pd.read_csv(ecog_path, low_memory=False)
            
            if 'Code' in ecog.columns:
                ecog = ecog.rename(columns={'Code': 'SubjectCode'})
                
            if 'TimepointCode' in ecog.columns:
                ecog = ecog[ecog['TimepointCode'] == 'm00']
                
            # Global mean
            numeric_cols = ecog.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if 'QID' not in c]
            
            if len(numeric_cols) > 0:
                ecog['ECOG_GlobalMean_Residual'] = ecog[numeric_cols].mean(axis=1)
                ecog_small = ecog[['SubjectCode', 'ECOG_GlobalMean_Residual']].drop_duplicates()
                df = df.merge(ecog_small, on='SubjectCode', how='left')
                
    except Exception:
        pass
    
    return df

def load_and_process():
    """Load and process all data"""
    print("Loading MemTrax data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    
    print("Applying quality filters...")
    memtrax_filtered = apply_ashford_filter(memtrax, accuracy_threshold=0.65)
    
    print("Winsorizing reaction times...")
    memtrax_filtered['CorrectResponsesRT'] = memtrax_filtered['CorrectResponsesRT'].clip(0.4, 2.0)
    
    print("Extracting sequence features...")
    seq_features = extract_sequence_features(memtrax_filtered)
    
    print("Aggregating per subject...")
    agg_dict = {
        'CorrectPCT': ['mean', 'std', 'min', 'max'],
        'CorrectResponsesRT': ['mean', 'std', 'min', 'max'],
        'IncorrectPCT': ['mean', 'std'],
        'IncorrectResponsesRT': ['mean', 'std']
    }
    
    aggregated = memtrax_filtered.groupby('SubjectCode').agg(agg_dict)
    aggregated.columns = ['_'.join(col) for col in aggregated.columns]
    aggregated = aggregated.reset_index()
    
    # Cognitive score
    aggregated['CognitiveScore_mean'] = (
        aggregated['CorrectResponsesRT_mean'] / (aggregated['CorrectPCT_mean'] + 0.01)
    )
    aggregated['CorrectResponsesRT_cv'] = (
        aggregated['CorrectResponsesRT_std'] / (aggregated['CorrectResponsesRT_mean'] + 0.01)
    )
    
    print("Merging sequence features...")
    aggregated = aggregated.merge(seq_features, on='SubjectCode', how='left')
    
    print("Enriching demographics...")
    aggregated = enrich_demographics(aggregated, DATA_DIR)
    
    print("Adding ECOG features...")
    aggregated = add_ecog_features(aggregated, DATA_DIR)
    
    print("Building labels...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].drop_duplicates(subset=['SubjectCode'])
    
    med_baseline['AnyCogImpairment'] = 0
    for qid in COGNITIVE_QIDS:
        if qid in med_baseline.columns:
            med_baseline['AnyCogImpairment'] |= (med_baseline[qid] == 1)
    
    labels = med_baseline[['SubjectCode', 'AnyCogImpairment']]
    
    print("Final merge...")
    final_df = aggregated.merge(labels, on='SubjectCode', how='inner')
    
    return final_df

def build_model():
    """Build the winning stacking model"""
    # Base estimators  
    estimators = []
    
    # Logistic regression with feature selection
    logit_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectKBest(mutual_info_classif, k=15)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, solver='lbfgs'))
    ])
    estimators.append(('logistic', logit_pipe))
    
    # HistGradientBoosting  
    hgb_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', HistGradientBoostingClassifier(
            random_state=42, 
            learning_rate=0.1, 
            max_leaf_nodes=31, 
            max_depth=3
        ))
    ])
    estimators.append(('histgb', hgb_pipe))
    
    # XGBoost if available
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        estimators.append(('xgb', xgb))
    
    # Stacking classifier
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=3,
        stack_method='predict_proba'
    )
    
    return stack

def main():
    print("🚀 BHR MEMTRAX MINIMAL - REPRODUCING AUC=0.798")
    print("=" * 50)
    
    # Load data
    data = load_and_process()
    print(f"Final dataset: {len(data)} samples")
    
    # Prepare features and target
    feature_cols = [c for c in data.columns if c not in ['SubjectCode', 'AnyCogImpairment']]
    X = data[feature_cols]
    y = data['AnyCogImpairment']
    
    print(f"Features: {X.shape[1]}, Positive rate: {y.mean():.3f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train)
    
    print("Making predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print("\n" + "=" * 50)
    print("🎯 RESULTS")
    print("=" * 50)
    print(f"AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    if auc >= 0.795:
        print("✅ SUCCESS: AUC ≈ 0.798 reproduced!")
    else:
        print(f"📍 AUC difference from target: {0.798 - auc:.4f}")
    
    # Save results
    results = {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "samples": len(data),
        "features": X.shape[1],
        "positives": int(y.sum()),
        "model": "Stacked(Logistic+HistGB+XGBoost)",
        "config": "lr=0.1, leaves=31, k=15"
    }
    
    with open("minimal_0798_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return auc

if __name__ == "__main__":
    final_auc = main()
    print(f"\n🏆 FINAL AUC: {final_auc:.4f}")
