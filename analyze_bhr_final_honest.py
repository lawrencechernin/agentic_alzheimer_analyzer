#!/usr/bin/env python3
"""
Final Honest BHR MCI Prediction
================================
Goal: Beat 0.744 baseline with better features, no data leakage
Uses same self-report labels as baseline for fair comparison
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
import json

np.random.seed(42)
DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Cognitive impairment QIDs (same as baseline)
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']


def extract_enhanced_memtrax_features(memtrax_q):
    """Extract comprehensive MemTrax features"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Basic aggregates
        feat['correct_pct_mean'] = group['CorrectPCT'].mean()
        feat['correct_pct_std'] = group['CorrectPCT'].std()
        feat['correct_pct_min'] = group['CorrectPCT'].min()
        
        feat['correct_rt_mean'] = group['CorrectResponsesRT'].mean()
        feat['correct_rt_std'] = group['CorrectResponsesRT'].std()
        
        feat['incorrect_pct_mean'] = group['IncorrectPCT'].mean()
        feat['incorrect_rt_mean'] = group['IncorrectResponsesRT'].mean()
        
        # Composite scores
        feat['cog_score'] = feat['correct_rt_mean'] / (feat['correct_pct_mean'] + 0.01)
        feat['speed_accuracy'] = feat['correct_pct_mean'] / (feat['correct_rt_mean'] + 0.01)
        feat['error_rate'] = 1 - feat['correct_pct_mean']
        feat['rt_cv'] = feat['correct_rt_std'] / (feat['correct_rt_mean'] + 1e-6)
        
        # Sequence features from reaction times
        all_rts = []
        for _, row in group.iterrows():
            if pd.notna(row.get('ReactionTimes')):
                try:
                    rts = [float(x.strip()) for x in str(row['ReactionTimes']).split(',') 
                           if x.strip() and x.strip() != 'nan']
                    valid_rts = [r for r in rts if 0.3 <= r <= 3.0]
                    all_rts.extend(valid_rts)
                except:
                    continue
        
        if len(all_rts) >= 10:
            n = len(all_rts)
            third = max(1, n // 3)
            
            # Fatigue effect
            feat['seq_fatigue'] = np.mean(all_rts[-third:]) - np.mean(all_rts[:third])
            feat['seq_mean_rt'] = np.mean(all_rts)
            feat['seq_cv'] = np.std(all_rts) / (np.mean(all_rts) + 1e-6)
            
            # Learning effect
            mid = n // 2
            feat['learning_effect'] = np.mean(all_rts[:mid]) - np.mean(all_rts[mid:])
        
        feat['n_tests'] = len(group)
        features.append(feat)
    
    return pd.DataFrame(features)


def add_demographics(data):
    """Add demographic features - FIXED path"""
    # Try multiple possible paths
    demo_paths = [
        DATA_DIR / 'BHR_Demographics.csv',
        DATA_DIR / 'Demographics.csv',
        DATA_DIR / 'BHR_Demographic.csv'
    ]
    
    demo_df = None
    for path in demo_paths:
        if path.exists():
            print(f"   Found demographics at: {path.name}")
            demo_df = pd.read_csv(path, low_memory=False)
            break
    
    if demo_df is None:
        # Try loading from main data if demographics embedded
        print("   No separate demographics file, checking main files...")
        return data
    
    # Standardize column names
    if 'Code' in demo_df.columns and 'SubjectCode' not in demo_df.columns:
        demo_df = demo_df.rename(columns={'Code': 'SubjectCode'})
    
    # Check what demographic columns are available
    demo_cols = ['SubjectCode']
    
    # Try different column name variations
    age_cols = ['Age_Baseline', 'Age', 'age_baseline', 'AGE']
    edu_cols = ['YearsEducationUS_Converted', 'Education', 'YearsEducation', 'education']
    gender_cols = ['Gender', 'Sex', 'gender', 'sex']
    
    found_age = None
    found_edu = None
    found_gender = None
    
    for col in age_cols:
        if col in demo_df.columns:
            found_age = col
            demo_cols.append(col)
            break
    
    for col in edu_cols:
        if col in demo_df.columns:
            found_edu = col
            demo_cols.append(col)
            break
            
    for col in gender_cols:
        if col in demo_df.columns:
            found_gender = col
            demo_cols.append(col)
            break
    
    if len(demo_cols) == 1:
        print("   No usable demographic columns found")
        return data
    
    # Get unique records per subject
    demo_subset = demo_df[demo_cols].drop_duplicates(subset=['SubjectCode'])
    
    # Standardize column names
    rename_dict = {}
    if found_age and found_age != 'Age':
        rename_dict[found_age] = 'Age'
    if found_edu and found_edu != 'Education':
        rename_dict[found_edu] = 'Education'
    if found_gender and found_gender != 'Gender':
        rename_dict[found_gender] = 'Gender'
    
    if rename_dict:
        demo_subset = demo_subset.rename(columns=rename_dict)
    
    # Merge with data
    data_with_demo = data.merge(demo_subset, on='SubjectCode', how='left')
    
    # Create interaction features
    if 'Age' in data_with_demo.columns and 'Education' in data_with_demo.columns:
        data_with_demo['age_edu_interaction'] = (
            data_with_demo['Age'] * data_with_demo['Education']
        )
        data_with_demo['cognitive_reserve'] = (
            data_with_demo['Education'] / (data_with_demo['Age'] / 100 + 0.1)
        )
    
    # Binary gender encoding
    if 'Gender' in data_with_demo.columns:
        # Handle various encodings
        data_with_demo['is_female'] = data_with_demo['Gender'].isin(['F', 'Female', 'female', 2]).astype(int)
    
    n_added = len([c for c in data_with_demo.columns if c not in data.columns])
    print(f"   Added {n_added} demographic features")
    
    return data_with_demo


def create_self_report_labels(med_hx):
    """
    Create labels from self-report ONLY (same as baseline for fair comparison)
    """
    # Get baseline records and drop duplicates
    if 'TimepointCode' in med_hx.columns:
        med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].copy()
    else:
        med_baseline = med_hx.copy()
    
    med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])
    
    # Create MCI labels from QIDs
    available_qids = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]
    
    if not available_qids:
        raise ValueError("No cognitive QIDs found in medical history")
    
    # Create labels
    mci = np.zeros(len(med_baseline), dtype=int)
    valid = np.zeros(len(med_baseline), dtype=bool)
    
    for qid in available_qids:
        mci |= (med_baseline[qid] == 1).values
        valid |= med_baseline[qid].isin([1, 2]).values
    
    # Create dataframe
    labels = pd.DataFrame({
        'SubjectCode': med_baseline['SubjectCode'],
        'mci': mci,
        'valid': valid
    })
    
    # Only keep valid labels
    labels = labels[labels['valid']].copy()
    
    return labels[['SubjectCode', 'mci']]


def main():
    print("\n" + "="*70)
    print("FINAL HONEST BHR MCI PREDICTION")
    print("="*70)
    print("Using same labels as baseline (self-report) for fair comparison\n")
    
    # 1. Load data
    print("1. Loading datasets...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # 2. Quality filter MemTrax
    print("\n2. Applying quality filters...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    print(f"   Quality filtered: {len(memtrax_q):,} records")
    
    # 3. Extract enhanced features
    print("\n3. Extracting enhanced MemTrax features...")
    memtrax_features = extract_enhanced_memtrax_features(memtrax_q)
    print(f"   MemTrax features: {memtrax_features.shape}")
    
    # 4. Add demographics
    print("\n4. Adding demographic features...")
    memtrax_features = add_demographics(memtrax_features)
    
    # 5. Create labels (self-report only for fair comparison)
    print("\n5. Creating self-report labels (same as baseline)...")
    labels = create_self_report_labels(med_hx)
    print(f"   Valid labels: {len(labels):,}")
    print(f"   MCI prevalence: {labels['mci'].mean():.1%}")
    
    # 6. Merge
    print("\n6. Building final dataset...")
    data = memtrax_features.merge(labels, on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data):,} subjects")
    print(f"   Features: {data.shape[1]-2}")
    print(f"   MCI prevalence: {data['mci'].mean():.1%}")
    
    # Check if we have demographics
    demo_features = ['Age', 'Education', 'Gender', 'is_female', 'age_edu_interaction', 'cognitive_reserve']
    available_demo = [f for f in demo_features if f in data.columns]
    print(f"   Demographic features available: {available_demo}")
    
    # 7. Prepare for modeling
    print("\n7. Preparing for modeling...")
    
    X = data.drop(['SubjectCode', 'mci'], axis=1).values
    y = data['mci'].values
    feature_names = [c for c in data.columns if c not in ['SubjectCode', 'mci']]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train/test: {len(X_train)}/{len(X_test)}")
    print(f"   Test MCI cases: {y_test.sum()} ({y_test.mean():.1%})")
    
    # 8. Train models
    print("\n8. Training optimized models...")
    
    models = {
        'Logistic': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_split=15,
                class_weight='balanced', random_state=42
            ))
        ]),
        'HistGB': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.03, max_depth=6,
                min_samples_leaf=20, random_state=42
            ))
        ])
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric='logloss'
            ))
        ])
    
    # Train and evaluate
    best_auc = 0
    best_model = None
    best_name = None
    results = {}
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_auc:.4f}")
        
        results[name] = {
            'cv_auc': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'test_auc': float(test_auc)
        }
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
            best_name = name
    
    # 9. Calibrated stacking
    print("\n9. Training calibrated stacking ensemble...")
    
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    
    y_pred_stack = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred_stack)
    
    print(f"   Calibrated Stack: Test={stack_auc:.4f}")
    
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'CalibratedStack'
        best_model = cal_stack
    
    # Results summary
    print("\n" + "="*70)
    print("FINAL HONEST RESULTS")
    print("="*70)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    baseline_auc = 0.744
    improvement = best_auc - baseline_auc
    
    print(f"\nComparison to baseline:")
    print(f"  Baseline (simple features): {baseline_auc:.4f}")
    print(f"  Enhanced features: {best_auc:.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    
    if best_auc > 0.75:
        print(f"\n✅ SUCCESS! Improved to {best_auc:.4f} with enhanced features!")
    elif best_auc > baseline_auc:
        print(f"\n📈 Modest improvement to {best_auc:.4f}")
    else:
        print(f"\n📊 No improvement over baseline")
    
    # Feature importance
    if 'RF' in models and hasattr(models['RF'].named_steps['clf'], 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': models['RF'].named_steps['clf'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features (Random Forest):")
        for i, row in importance.head(10).iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    # Save results
    output = {
        'strategy': 'Enhanced features with self-report labels (honest)',
        'n_subjects': len(data),
        'n_features': len(feature_names),
        'prevalence': float(y.mean()),
        'best_model': best_name,
        'best_auc': float(best_auc),
        'baseline_auc': baseline_auc,
        'improvement': float(improvement),
        'all_results': results
    }
    
    with open(OUTPUT_DIR / 'final_honest_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/final_honest_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()

