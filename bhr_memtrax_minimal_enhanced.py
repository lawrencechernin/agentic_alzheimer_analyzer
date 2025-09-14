#!/usr/bin/env python3
"""
BHR MemTrax Enhanced Minimal - Targeting AUC=0.798
==================================================
Enhanced version with key missing components.
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

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

def apply_ashford_filter(df, accuracy_threshold=0.65):
    """Apply Ashford quality filters"""
    return df[
        (df['Status'] == 'Collected') &
        (df['CorrectPCT'] >= accuracy_threshold) &
        (df['CorrectResponsesRT'] >= 0.5) &
        (df['CorrectResponsesRT'] <= 2.5)
    ].copy()

def extract_sequence_features(df):
    """Enhanced sequence feature extraction"""
    results = []
    
    for subject_code, group in df.groupby('SubjectCode'):
        subject_features = {'SubjectCode': subject_code}
        all_rts = []
        
        for _, row in group.iterrows():
            if pd.isna(row['ReactionTimes']):
                continue
                
            try:
                rt_str = str(row['ReactionTimes']).strip()
                if not rt_str or rt_str == 'nan':
                    continue
                    
                rts = [float(x) for x in rt_str.split(',') if x.strip() and x.strip() != 'nan']
                if len(rts) >= 3:
                    all_rts.extend(rts)
            except:
                continue
        
        if len(all_rts) >= 6:  # Need enough data points
            # Sequence features
            n_third = len(all_rts) // 3
            subject_features.update({
                'seq_first_third_mean': np.mean(all_rts[:n_third]),
                'seq_last_third_mean': np.mean(all_rts[-n_third:]),
                'seq_fatigue_effect': np.mean(all_rts[-n_third:]) - np.mean(all_rts[:n_third]),
                'seq_mean_rt': np.mean(all_rts),
                'seq_median_rt': np.median(all_rts),
                'long_n_timepoints': len(group),
                'long_reliability_change': np.var(all_rts[-n_third:]) - np.var(all_rts[:n_third]),
            })
            
            # RT slope
            if len(all_rts) >= 4:
                x = np.arange(len(all_rts))
                try:
                    slope = np.polyfit(x, all_rts, 1)[0]
                    subject_features['long_rt_slope'] = slope
                except:
                    subject_features['long_rt_slope'] = 0
                    
        results.append(subject_features)
    
    return pd.DataFrame(results)

def enrich_demographics(df, data_dir):
    """Enhanced demographics with more sources"""
    demo_files = ['BHR_Demographics.csv', 'Profile.csv', 'Participants.csv', 'Subjects.csv']
    
    for filename in demo_files:
        try:
            demo_path = data_dir / filename
            if not demo_path.exists():
                continue
                
            demo_df = pd.read_csv(demo_path, low_memory=False)
            
            if 'Code' in demo_df.columns and 'SubjectCode' not in demo_df.columns:
                demo_df = demo_df.rename(columns={'Code': 'SubjectCode'})
            
            if 'SubjectCode' not in demo_df.columns:
                continue
                
            demo_cols = ['SubjectCode']
            for col in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
                if col in demo_df.columns:
                    demo_cols.append(col)
                    
            if len(demo_cols) > 1:
                demo_small = demo_df[demo_cols].drop_duplicates(subset=['SubjectCode'])
                df = df.merge(demo_small, on='SubjectCode', how='left')
                
        except Exception:
            continue
    
    # Enhanced derived features
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
        
    # Add splines if possible
    for col in ['Age_Baseline', 'YearsEducationUS_Converted']:
        if col in df.columns and df[col].notna().sum() > 100:
            try:
                spline = SplineTransformer(n_knots=3, degree=3, include_bias=False)
                x = df[col].fillna(df[col].median()).values.reshape(-1, 1)
                spline_features = spline.fit_transform(x)
                for i in range(spline_features.shape[1]):
                    df[f'{col}_spline_{i}'] = spline_features[:, i]
            except:
                pass
        
    return df

def add_ecog_features(df, data_dir):
    """Enhanced ECOG features"""
    ecog_files = [
        ('BHR_EverydayCognition.csv', 'ECOG'),
        ('BHR_SP_ECog.csv', 'SP_ECOG'),
        ('BHR_SP_ADL.csv', 'SP_ADL')
    ]
    
    for filename, prefix in ecog_files:
        try:
            ecog_path = data_dir / filename
            if not ecog_path.exists():
                continue
                
            ecog = pd.read_csv(ecog_path, low_memory=False)
            
            if 'Code' in ecog.columns:
                ecog = ecog.rename(columns={'Code': 'SubjectCode'})
                
            if 'TimepointCode' in ecog.columns:
                ecog = ecog[ecog['TimepointCode'] == 'm00']
                
            # Global mean
            numeric_cols = ecog.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if 'QID' not in c and c != 'SubjectCode']
            
            if len(numeric_cols) > 0:
                ecog[f'{prefix}_mean'] = ecog[numeric_cols].mean(axis=1)
                
                # Per-domain means for ECOG
                if prefix == 'ECOG':
                    for domain in ['Memory', 'Language', 'Visuospatial', 'Executive']:
                        domain_cols = [c for c in numeric_cols if domain.lower() in c.lower()]
                        if domain_cols:
                            ecog[f'{prefix}_{domain}_mean'] = ecog[domain_cols].mean(axis=1)
                
                # Keep relevant columns
                keep_cols = ['SubjectCode'] + [c for c in ecog.columns if c.startswith(f'{prefix}_')]
                ecog_small = ecog[keep_cols].drop_duplicates(subset=['SubjectCode'])
                df = df.merge(ecog_small, on='SubjectCode', how='left')
                
        except Exception:
            continue
    
    return df

def build_optimized_model():
    """Build model with parameter sweep like the original"""
    estimators = []
    
    # Logistic with optimized k
    for k in [15, 25, 35]:  # Test multiple k values
        logit_pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif, k=k)),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, solver='lbfgs'))
        ])
        estimators.append((f'logistic_k{k}', logit_pipe))
    
    # HistGradientBoosting with parameter sweep
    for lr in [0.08, 0.1, 0.12]:
        for leaves in [28, 31, 35]:
            hgb_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('clf', HistGradientBoostingClassifier(
                    random_state=42, 
                    learning_rate=lr, 
                    max_leaf_nodes=leaves, 
                    max_depth=3
                ))
            ])
            estimators.append((f'hgb_lr{lr}_l{leaves}', hgb_pipe))
    
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
    
    # Stacking classifier with best estimators
    stack = StackingClassifier(
        estimators=estimators[:5],  # Use top 5 to avoid overfitting
        final_estimator=LogisticRegression(max_iter=2000),
        cv=3,
        stack_method='predict_proba'
    )
    
    return stack

def main():
    print("🚀 BHR MEMTRAX ENHANCED MINIMAL - TARGETING AUC=0.798")
    print("=" * 55)
    
    # Load data
    print("Loading MemTrax data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    
    print("Applying quality filters...")
    memtrax_filtered = apply_ashford_filter(memtrax, accuracy_threshold=0.65)
    
    print("Winsorizing reaction times...")
    memtrax_filtered['CorrectResponsesRT'] = memtrax_filtered['CorrectResponsesRT'].clip(0.4, 2.0)
    
    print("Extracting enhanced sequence features...")
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
    
    # Enhanced features
    aggregated['CognitiveScore_mean'] = aggregated['CorrectResponsesRT_mean'] / (aggregated['CorrectPCT_mean'] + 0.01)
    aggregated['CorrectResponsesRT_cv'] = aggregated['CorrectResponsesRT_std'] / (aggregated['CorrectResponsesRT_mean'] + 0.01)
    
    print("Merging sequence features...")
    aggregated = aggregated.merge(seq_features, on='SubjectCode', how='left')
    
    print("Enhanced demographics enrichment...")
    aggregated = enrich_demographics(aggregated, DATA_DIR)
    
    print("Adding enhanced ECOG features...")
    aggregated = add_ecog_features(aggregated, DATA_DIR)
    
    print("Building composite labels...")
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].drop_duplicates(subset=['SubjectCode'])
    
    med_baseline['AnyCogImpairment'] = 0
    for qid in COGNITIVE_QIDS:
        if qid in med_baseline.columns:
            med_baseline['AnyCogImpairment'] |= (med_baseline[qid] == 1)
    
    labels = med_baseline[['SubjectCode', 'AnyCogImpairment']]
    
    print("Final merge...")
    final_df = aggregated.merge(labels, on='SubjectCode', how='inner')
    print(f"Final dataset: {len(final_df)} samples")
    
    # Prepare features
    feature_cols = [c for c in final_df.columns if c not in ['SubjectCode', 'AnyCogImpairment']]
    X = final_df[feature_cols]
    y = final_df['AnyCogImpairment']
    
    print(f"Features: {X.shape[1]}, Positive rate: {y.mean():.3f}")
    
    # Train/test split  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training enhanced model...")
    model = build_optimized_model()
    model.fit(X_train, y_train)
    
    print("Making predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print("\n" + "=" * 55)
    print("🎯 ENHANCED RESULTS")
    print("=" * 55)
    print(f"AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    if auc >= 0.795:
        print("✅ SUCCESS: AUC ≈ 0.798 achieved!")
    else:
        print(f"📍 AUC difference from target: {0.798 - auc:.4f}")
    
    return auc

if __name__ == "__main__":
    final_auc = main()
    print(f"\n🏆 FINAL AUC: {final_auc:.4f}")
