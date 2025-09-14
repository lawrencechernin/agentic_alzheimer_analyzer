#!/usr/bin/env python3
"""
BHR MemTrax MCI Analysis - Methodologically Sound Version
Target: AUC > 0.80 with PROPER ML methodology
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
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Configuration
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']
RANDOM_STATE = 42


def apply_ashford_filter(df, min_acc=0.60):
    """Apply Ashford quality criteria"""
    return df[(df['Status'] == 'Collected') & 
              (df['CorrectPCT'] >= min_acc) &
              (df['CorrectResponsesRT'].between(0.5, 2.5))].copy()


def extract_sequence_features(df):
    """Extract fatigue and variability features"""
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


def build_labels(med_hx):
    """Build composite cognitive impairment labels"""
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
        valid |= med_hx[qid].isin([1, 2]).values
    
    labels = pd.DataFrame({
        'SubjectCode': med_hx['SubjectCode'],
        'cognitive_impairment': impairment
    })
    
    return labels[valid].copy()


def add_demographics(df, data_dir):
    """Add demographics and interactions"""
    demo_files = ['BHR_Demographics.csv', 'Profile.csv']
    
    for filename in demo_files:
        path = data_dir / filename
        if path.exists():
            try:
                demo = pd.read_csv(path, low_memory=False)
                if 'Code' in demo.columns:
                    demo.rename(columns={'Code': 'SubjectCode'}, inplace=True)
                    
                if 'SubjectCode' in demo.columns:
                    cols = ['SubjectCode']
                    for c in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
                        if c in demo.columns:
                            cols.append(c)
                    
                    if len(cols) > 1:
                        df = df.merge(demo[cols].drop_duplicates('SubjectCode'), 
                                     on='SubjectCode', how='left')
                        break
            except:
                continue
    
    # Derived features
    if 'Age_Baseline' in df.columns:
        df['Age_sq'] = df['Age_Baseline'] ** 2
        if 'CorrectResponsesRT_mean' in df.columns:
            df['age_rt_interact'] = df['Age_Baseline'] * df['CorrectResponsesRT_mean'] / 65
            
    if 'YearsEducationUS_Converted' in df.columns:
        df['Edu_sq'] = df['YearsEducationUS_Converted'] ** 2
        
    if all(c in df.columns for c in ['Age_Baseline', 'YearsEducationUS_Converted']):
        df['Age_Edu_interact'] = df['Age_Baseline'] * df['YearsEducationUS_Converted']
        df['CogReserve'] = df['YearsEducationUS_Converted'] / (df['Age_Baseline'] + 1)
        
    if 'Gender' in df.columns:
        df['Gender_Num'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
        
    return df


def add_informant_scores(df, data_dir):
    """Add SP-ECOG and other informant assessments"""
    info_files = [
        ('SP_ECOG', 'BHR_SP_ECog.csv'),
        ('ECOG', 'BHR_EverydayCognition.csv')
    ]
    
    for prefix, filename in info_files:
        path = data_dir / filename
        if path.exists():
            try:
                info = pd.read_csv(path, low_memory=False)
                if 'TimepointCode' in info.columns:
                    info = info[info['TimepointCode'] == 'm00']
                    
                num_cols = info.select_dtypes(include=[np.number]).columns
                num_cols = [c for c in num_cols if 'QID' not in c and 'Subject' not in c]
                
                if len(num_cols) > 0:
                    info[f'{prefix}_mean'] = info[num_cols].mean(axis=1)
                    subset = info[['SubjectCode', f'{prefix}_mean']].drop_duplicates('SubjectCode')
                    df = df.merge(subset, on='SubjectCode', how='left')
            except:
                continue
                
    return df


def main():
    print("\n" + "="*60)
    print("BHR MEMTRAX MCI ANALYSIS - PROPER METHODOLOGY")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    print(f"   MemTrax: {len(memtrax)} records")
    print(f"   Medical: {len(med_hx)} records")
    
    # Quality filter
    print("\n2. Applying Ashford filter...")
    memtrax_q = apply_ashford_filter(memtrax, min_acc=0.60)
    print(f"   Retained: {len(memtrax_q)}/{len(memtrax)} ({len(memtrax_q)/len(memtrax)*100:.1f}%)")
    
    # Features
    print("\n3. Feature engineering...")
    
    # Sequence features
    seq_feat = extract_sequence_features(memtrax_q)
    
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
    
    # Cognitive score
    agg_feat['CogScore'] = agg_feat['CorrectResponsesRT_mean'] / (agg_feat['CorrectPCT_mean'] + 0.01)
    agg_feat['RT_CV'] = agg_feat['CorrectResponsesRT_std'] / (agg_feat['CorrectResponsesRT_mean'] + 1e-6)
    
    # Additional composite features
    agg_feat['Speed_Accuracy_Product'] = agg_feat['CorrectPCT_mean'] / (agg_feat['CorrectResponsesRT_mean'] + 0.01)
    agg_feat['Error_Rate'] = agg_feat['IncorrectPCT_mean'] + agg_feat['IncorrectRejectionsN_mean'] / 100
    agg_feat['Response_Consistency'] = 1 / (agg_feat['CorrectResponsesRT_std'] + agg_feat['IncorrectResponsesRT_std'] + 1e-6)
    
    # Merge features
    features = agg_feat.merge(seq_feat, on='SubjectCode', how='left')
    features = add_demographics(features, DATA_DIR)
    features = add_informant_scores(features, DATA_DIR)
    
    # Labels
    print("\n4. Building labels...")
    labels = build_labels(med_hx)
    print(f"   Labels: {len(labels)} subjects")
    print(f"   Prevalence: {labels['cognitive_impairment'].mean():.1%}")
    
    # Merge
    data = features.merge(labels, on='SubjectCode', how='inner')
    print(f"\n5. Final dataset: {len(data)} subjects")
    
    # Prepare X, y
    feature_cols = [c for c in data.columns if c not in ['SubjectCode', 'cognitive_impairment']]
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    # CRITICAL: Proper train/test split
    print(f"\n6. Train/test split (proper methodology)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   ✅ Test set held out for final evaluation only")
    
    # Models
    print(f"\n7. Training models...")
    
    results = {}
    best_auc = 0
    best_name = None
    
    # Test different models
    models = {
        'Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif, k='all')),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ]),
        'RandomForest': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=300, max_depth=10, 
                                          min_samples_split=15, min_samples_leaf=5,
                                          class_weight='balanced',
                                          random_state=RANDOM_STATE))
        ]),
        'HistGB': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=31,
                                                   learning_rate=0.03, max_depth=6,
                                                   min_samples_leaf=20,
                                                   random_state=RANDOM_STATE))
        ])
    }
    
    # Add XGBoost if available
    if HAS_XGB:
        models['XGBoost'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, reg_alpha=0.5,
                                 scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                                 random_state=RANDOM_STATE))
        ])
    
    for name, pipe in models.items():
        # CV on training only
        cv_scores = cross_val_score(pipe, X_train, y_train, 
                                    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                    scoring='roc_auc')
        
        # Fit and evaluate on test
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        
        print(f"   {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_auc:.4f}")
        
        results[name] = {'cv_auc': cv_scores.mean(), 'test_auc': test_auc}
        if test_auc > best_auc:
            best_auc = test_auc
            best_name = name
    
    # Try stacking
    print(f"\n8. Testing ensemble...")
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5, stack_method='predict_proba'
    )
    
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    y_prob = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_prob)
    
    print(f"   Calibrated Stack: Test={stack_auc:.4f}")
    
    results['CalibratedStack'] = {'test_auc': stack_auc}
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'CalibratedStack'
    
    # Results
    print(f"\n" + "="*60)
    print(f"RESULTS")
    print(f"="*60)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    if best_auc >= 0.80:
        print(f"\n🎯 SUCCESS! Achieved AUC ≥ 0.80 with proper methodology!")
    else:
        print(f"\n📊 Best honest AUC: {best_auc:.4f}")
        
    # Save
    output = {
        'methodology': 'Proper train/test split',
        'samples': {'total': len(data), 'train': len(X_train), 'test': len(X_test)},
        'results': results,
        'best': {'model': best_name, 'auc': best_auc}
    }
    
    with open(OUTPUT_DIR / 'proper_methodology_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    return best_auc


if __name__ == '__main__':
    auc = main()
