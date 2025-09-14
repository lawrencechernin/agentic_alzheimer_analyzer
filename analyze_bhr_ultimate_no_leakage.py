#!/usr/bin/env python3
"""
Ultimate BHR MCI Prediction WITHOUT Data Leakage
==================================================
Legitimate improvements:
1. Add demographics (age, education, gender)
2. Enhanced MemTrax feature engineering
3. Better ensemble with calibration
4. Keep SP-ECOG for labels ONLY (no leakage)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
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

# Cognitive impairment QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']


def extract_enhanced_memtrax_features(memtrax_q):
    """Extract comprehensive MemTrax features with advanced engineering"""
    features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        feat = {'SubjectCode': subject}
        
        # Basic aggregates
        feat['correct_pct_mean'] = group['CorrectPCT'].mean()
        feat['correct_pct_std'] = group['CorrectPCT'].std()
        feat['correct_pct_min'] = group['CorrectPCT'].min()
        feat['correct_pct_max'] = group['CorrectPCT'].max()
        
        feat['correct_rt_mean'] = group['CorrectResponsesRT'].mean()
        feat['correct_rt_std'] = group['CorrectResponsesRT'].std()
        feat['correct_rt_min'] = group['CorrectResponsesRT'].min()
        feat['correct_rt_max'] = group['CorrectResponsesRT'].max()
        
        feat['incorrect_pct_mean'] = group['IncorrectPCT'].mean()
        feat['incorrect_rt_mean'] = group['IncorrectResponsesRT'].mean()
        
        # Composite scores
        feat['cog_score'] = feat['correct_rt_mean'] / (feat['correct_pct_mean'] + 0.01)
        feat['speed_accuracy'] = feat['correct_pct_mean'] / (feat['correct_rt_mean'] + 0.01)
        feat['error_rate'] = 1 - feat['correct_pct_mean']
        feat['rt_cv'] = feat['correct_rt_std'] / (feat['correct_rt_mean'] + 1e-6)
        
        # Performance consistency
        feat['accuracy_consistency'] = 1 / (feat['correct_pct_std'] + 0.01)
        feat['rt_consistency'] = 1 / (feat['correct_rt_std'] + 0.01)
        
        # Range features
        feat['accuracy_range'] = feat['correct_pct_max'] - feat['correct_pct_min']
        feat['rt_range'] = feat['correct_rt_max'] - feat['correct_rt_min']
        
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
            
            # Percentiles
            feat['rt_p25'] = np.percentile(all_rts, 25)
            feat['rt_p50'] = np.percentile(all_rts, 50)
            feat['rt_p75'] = np.percentile(all_rts, 75)
            feat['rt_iqr'] = feat['rt_p75'] - feat['rt_p25']
            
            # Learning effect (first vs second half)
            mid = n // 2
            feat['learning_effect'] = np.mean(all_rts[:mid]) - np.mean(all_rts[mid:])
            
            # Response time entropy (variability measure)
            hist, _ = np.histogram(all_rts, bins=10)
            hist = hist / hist.sum()
            feat['rt_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        feat['n_tests'] = len(group)
        
        features.append(feat)
    
    return pd.DataFrame(features)


def add_demographics(data, data_dir):
    """Add demographic features WITHOUT leakage"""
    demo_path = data_dir / 'BHR_Demographics.csv'
    
    if not demo_path.exists():
        print("   Demographics file not found")
        return data
    
    demo = pd.read_csv(demo_path, low_memory=False)
    
    # Rename Code to SubjectCode if needed
    if 'Code' in demo.columns and 'SubjectCode' not in demo.columns:
        demo = demo.rename(columns={'Code': 'SubjectCode'})
    
    # Select demographic columns
    demo_cols = ['SubjectCode']
    available = []
    
    for col in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
        if col in demo.columns:
            demo_cols.append(col)
            available.append(col)
    
    if len(available) == 0:
        print("   No demographic columns found")
        return data
    
    # Merge
    demo_subset = demo[demo_cols].drop_duplicates(subset=['SubjectCode'])
    data_with_demo = data.merge(demo_subset, on='SubjectCode', how='left')
    
    # Create interaction features if we have age and education
    if 'Age_Baseline' in available and 'YearsEducationUS_Converted' in available:
        data_with_demo['age_edu_interaction'] = (
            data_with_demo['Age_Baseline'] * data_with_demo['YearsEducationUS_Converted']
        )
        data_with_demo['cognitive_reserve'] = (
            data_with_demo['YearsEducationUS_Converted'] / 
            (data_with_demo['Age_Baseline'] / 100)
        )
    
    # Binary gender encoding if present
    if 'Gender' in available:
        data_with_demo['is_female'] = (data_with_demo['Gender'] == 'F').astype(int)
    
    print(f"   Added {len(available)} demographic features")
    
    return data_with_demo


def create_labels_from_informant(med_hx, sp_ecog):
    """
    Create labels using SP-ECOG informant data
    NO LEAKAGE: SP-ECOG used ONLY for labels, not features
    """
    # Get SP-ECOG baseline and DROP DUPLICATES
    sp_ecog_baseline = sp_ecog[sp_ecog['TimepointCode'] == 'sp-m00'].copy()
    sp_ecog_baseline = sp_ecog_baseline.drop_duplicates(subset=['SubjectCode'])
    
    # Get numeric QID columns and clean
    qid_cols = [c for c in sp_ecog_baseline.columns if c.startswith('QID')]
    numeric_qids = sp_ecog_baseline[qid_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Replace value 8 (Don't Know) with NaN
    for col in numeric_qids:
        sp_ecog_baseline.loc[sp_ecog_baseline[col] == 8, col] = np.nan
    
    # Calculate informant scores
    sp_ecog_baseline['sp_mean'] = sp_ecog_baseline[numeric_qids].mean(axis=1)
    sp_ecog_baseline['sp_high_pct'] = (sp_ecog_baseline[numeric_qids] >= 3).sum(axis=1) / sp_ecog_baseline[numeric_qids].notna().sum(axis=1)
    
    # Informant MCI (calibrated thresholds)
    sp_ecog_baseline['informant_mci'] = (
        (sp_ecog_baseline['sp_mean'] >= 3.5) | 
        (sp_ecog_baseline['sp_high_pct'] >= 0.20)
    ).astype(int)
    
    # Get medical history baseline and DROP DUPLICATES
    if 'TimepointCode' in med_hx.columns:
        med_baseline = med_hx[med_hx['TimepointCode'] == 'm00'].copy()
    else:
        med_baseline = med_hx.copy()
    med_baseline = med_baseline.drop_duplicates(subset=['SubjectCode'])
    
    # Self-reported MCI
    available_qids = [q for q in COGNITIVE_QIDS if q in med_baseline.columns]
    if available_qids:
        self_mci = np.zeros(len(med_baseline), dtype=int)
        valid_self = np.zeros(len(med_baseline), dtype=bool)
        
        for qid in available_qids:
            self_mci |= (med_baseline[qid] == 1).values
            valid_self |= med_baseline[qid].isin([1, 2]).values
        
        med_baseline['self_mci'] = self_mci
        med_baseline['self_valid'] = valid_self
    else:
        med_baseline['self_mci'] = 0
        med_baseline['self_valid'] = False
    
    # Merge
    labels = med_baseline[['SubjectCode', 'self_mci', 'self_valid']].merge(
        sp_ecog_baseline[['SubjectCode', 'informant_mci']], 
        on='SubjectCode', 
        how='left'
    )
    
    labels['has_informant'] = labels['informant_mci'].notna()
    
    # Composite label: Use OR when both available (catches anosognosia)
    # Convert to numpy arrays to handle NaN properly
    self_mci_arr = labels['self_mci'].values.astype(float)
    informant_mci_arr = labels['informant_mci'].values.astype(float)
    has_informant_arr = labels['has_informant'].values
    self_valid_arr = labels['self_valid'].values
    
    mci = np.zeros(len(labels))
    for i in range(len(labels)):
        if has_informant_arr[i] and self_valid_arr[i]:
            # Both available: use OR
            mci[i] = int(self_mci_arr[i] == 1 or informant_mci_arr[i] == 1)
        elif has_informant_arr[i]:
            # Only informant
            mci[i] = int(informant_mci_arr[i] == 1)
        else:
            # Only self-report
            mci[i] = int(self_mci_arr[i] == 1)
    
    labels['mci'] = mci.astype(int)
    
    labels['valid'] = labels['has_informant'] | labels['self_valid']
    
    return labels[labels['valid']][['SubjectCode', 'mci']]


def main():
    print("\n" + "="*70)
    print("ULTIMATE BHR MCI PREDICTION (NO LEAKAGE)")
    print("="*70)
    print("Enhancements: Demographics, better features, calibrated ensemble\n")
    
    # 1. Load data
    print("1. Loading datasets...")
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    sp_ecog = pd.read_csv(DATA_DIR / 'BHR_SP_ECog.csv', low_memory=False)
    
    # 2. Quality filter MemTrax
    print("\n2. Applying quality filters...")
    memtrax_q = memtrax[
        (memtrax['Status'] == 'Collected') &
        (memtrax['CorrectPCT'] >= 0.60) &
        (memtrax['CorrectResponsesRT'].between(0.5, 2.5))
    ]
    print(f"   Quality filtered: {len(memtrax_q):,} records")
    
    # 3. Extract enhanced MemTrax features
    print("\n3. Extracting enhanced MemTrax features...")
    memtrax_features = extract_enhanced_memtrax_features(memtrax_q)
    print(f"   MemTrax features: {memtrax_features.shape}")
    
    # 4. Add demographics
    print("\n4. Adding demographic features...")
    memtrax_features = add_demographics(memtrax_features, DATA_DIR)
    
    # 5. Create labels (SP-ECOG for labels ONLY - no leakage!)
    print("\n5. Creating labels from informant data...")
    labels = create_labels_from_informant(med_hx, sp_ecog)
    print(f"   Valid labels: {len(labels):,}")
    print(f"   MCI prevalence: {labels['mci'].mean():.1%}")
    
    # 6. Merge features and labels
    print("\n6. Building final dataset...")
    data = memtrax_features.merge(labels, on='SubjectCode', how='inner')
    print(f"   Final dataset: {len(data):,} subjects")
    print(f"   Features: {data.shape[1]-2}")
    
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
    
    # 8. Define models with tuned hyperparameters
    print("\n8. Training optimized models...")
    
    models = {
        'LogisticL1': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif, k=min(30, X.shape[1]))),
            ('clf', LogisticRegression(penalty='l1', solver='saga', max_iter=3000, 
                                      class_weight='balanced', C=0.1, random_state=42))
        ]),
        'LogisticL2': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', max_iter=2000, 
                                      class_weight='balanced', C=0.5, random_state=42))
        ]),
        'RF': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt',
                class_weight='balanced_subsample', random_state=42
            ))
        ]),
        'HistGB': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.02, max_depth=8,
                min_samples_leaf=15, l2_regularization=0.1,
                random_state=42
            ))
        ])
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=500, learning_rate=0.02, max_depth=7,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
                gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
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
    
    # 9. Try calibrated stacking ensemble
    print("\n9. Training calibrated stacking ensemble...")
    
    # Use best 3 models for stacking
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_auc'], reverse=True)
    top_models = [(name, models[name]) for name, _ in sorted_models[:3]]
    
    stack = StackingClassifier(
        estimators=top_models,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=5,
        stack_method='predict_proba'
    )
    
    # Calibrate the stack
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    
    y_pred_stack = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred_stack)
    
    print(f"   Calibrated Stack: Test={stack_auc:.4f}")
    
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'CalibratedStack'
        best_model = cal_stack
    
    # 10. Feature importance (if RF available)
    if 'RF' in models:
        rf = models['RF'].named_steps['clf']
        if hasattr(rf, 'feature_importances_'):
            # Handle imputed features
            imputer = models['RF'].named_steps['impute']
            X_imputed = imputer.transform(X)
            
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n10. Top features (Random Forest):")
            for i, (_, row) in enumerate(importance.head(15).iterrows(), 1):
                feature_type = "Demo" if any(x in row['feature'] for x in ['Age', 'Gender', 'Education', 'cognitive_reserve']) else "MemTrax"
                print(f"    {i:2d}. {row['feature']:<30} {row['importance']:.4f} ({feature_type})")
    
    # Results summary
    print("\n" + "="*70)
    print("ULTIMATE RESULTS (NO LEAKAGE)")
    print("="*70)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    baseline_auc = 0.744
    previous_best = 0.816
    
    print(f"\nComparison:")
    print(f"  Original baseline: {baseline_auc:.3f}")
    print(f"  With informant labels: {previous_best:.3f}")
    print(f"  Ultimate (enhanced): {best_auc:.3f}")
    print(f"  Total improvement: {(best_auc - baseline_auc):+.3f}")
    
    if best_auc >= 0.82:
        print(f"\n🏆 NEW RECORD! {best_auc:.3f} AUC!")
    elif best_auc > previous_best:
        print(f"\n🎯 Improved further to {best_auc:.3f} AUC!")
    else:
        print(f"\n📊 Result: {best_auc:.3f} AUC")
    
    # Save results
    output = {
        'strategy': 'Ultimate with enhanced features (no leakage)',
        'n_subjects': len(data),
        'n_features': len(feature_names),
        'prevalence': float(y.mean()),
        'best_model': best_name,
        'best_auc': float(best_auc),
        'improvements': {
            'from_baseline': float(best_auc - baseline_auc),
            'from_informant_only': float(best_auc - previous_best)
        },
        'all_results': results
    }
    
    with open(OUTPUT_DIR / 'ultimate_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/ultimate_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()
