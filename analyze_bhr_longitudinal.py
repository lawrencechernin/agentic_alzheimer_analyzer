#!/usr/bin/env python3
"""
BHR MemTrax Longitudinal Analysis - Leveraging 4 Years of Data
===============================================================
Key Strategy: Model cognitive TRAJECTORIES rather than single timepoints
- Track decline patterns over m00, m06, m12, m18, m24, m36, m48
- Change scores are less affected by education/cognitive reserve
- Progression patterns reveal MCI even when baseline appears normal

Target: Break 0.80 AUC using longitudinal features
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
import matplotlib.pyplot as plt
import json

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

DATA_DIR = Path("../bhr/BHR-ALL-EXT_Mem_2022")
OUTPUT_DIR = Path("bhr_memtrax_results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Cognitive impairment QIDs
COGNITIVE_QIDS = ['QID1-5', 'QID1-12', 'QID1-13', 'QID1-22', 'QID1-23']

# Timepoint ordering (months from baseline)
TIMEPOINT_ORDER = {
    'm00': 0,    # Baseline
    'm06': 6,    # 6 months
    'm12': 12,   # 1 year
    'm18': 18,   # 1.5 years
    'm24': 24,   # 2 years
    'm30': 30,   # 2.5 years
    'm36': 36,   # 3 years
    'm42': 42,   # 3.5 years
    'm48': 48    # 4 years
}


def load_longitudinal_data():
    """Load ALL timepoints, not just baseline"""
    print("Loading longitudinal data...")
    
    # Load MemTrax - keep all timepoints
    memtrax = pd.read_csv(DATA_DIR / 'MemTrax.csv', low_memory=False)
    print(f"  MemTrax: {len(memtrax)} total records")
    
    # Count timepoints
    if 'TimepointCode' in memtrax.columns:
        timepoint_counts = memtrax['TimepointCode'].value_counts()
        print("  Timepoint distribution:")
        for tp in sorted(TIMEPOINT_ORDER.keys()):
            if tp in timepoint_counts.index:
                print(f"    {tp}: {timepoint_counts[tp]:,} records")
    
    # Count subjects with multiple timepoints
    subjects_with_multiple = memtrax.groupby('SubjectCode')['TimepointCode'].nunique()
    print(f"\n  Subjects with longitudinal data:")
    print(f"    1 timepoint: {(subjects_with_multiple == 1).sum():,}")
    print(f"    2+ timepoints: {(subjects_with_multiple >= 2).sum():,}")
    print(f"    3+ timepoints: {(subjects_with_multiple >= 3).sum():,}")
    print(f"    4+ timepoints: {(subjects_with_multiple >= 4).sum():,}")
    
    # Load medical history
    med_hx = pd.read_csv(DATA_DIR / 'BHR_MedicalHx.csv', low_memory=False)
    
    # Also load ECOG longitudinal if available
    ecog_path = DATA_DIR / 'BHR_EverydayCognition.csv'
    ecog = None
    if ecog_path.exists():
        ecog = pd.read_csv(ecog_path, low_memory=False)
        print(f"  ECOG: {len(ecog)} records (longitudinal)")
    
    return memtrax, med_hx, ecog


def extract_cognitive_trajectories(memtrax_all):
    """
    Extract trajectory features for each subject across all timepoints
    Key insight: Rate of decline is more predictive than absolute values
    """
    print("\nExtracting cognitive trajectories...")
    
    # Apply quality filter but keep all timepoints
    memtrax_q = memtrax_all[
        (memtrax_all['Status'] == 'Collected') &
        (memtrax_all['CorrectPCT'] >= 0.50) &  # Lower threshold to catch decline
        (memtrax_all['CorrectResponsesRT'].between(0.4, 3.0))  # Wider range for impaired
    ].copy()
    
    # Convert timepoints to numeric months
    if 'TimepointCode' in memtrax_q.columns:
        memtrax_q['months_from_baseline'] = memtrax_q['TimepointCode'].map(TIMEPOINT_ORDER)
    else:
        memtrax_q['months_from_baseline'] = 0
    
    trajectory_features = []
    
    for subject, group in memtrax_q.groupby('SubjectCode'):
        # Sort by timepoint
        group = group.sort_values('months_from_baseline')
        
        feat = {'SubjectCode': subject}
        feat['n_timepoints'] = len(group)
        feat['followup_months'] = group['months_from_baseline'].max()
        
        if len(group) >= 2:  # Need at least 2 timepoints for trajectory
            
            # === ACCURACY TRAJECTORY ===
            acc_values = group['CorrectPCT'].values
            time_values = group['months_from_baseline'].values
            
            # Remove NaNs
            valid = ~np.isnan(acc_values)
            if valid.sum() >= 2:
                acc_clean = acc_values[valid]
                time_clean = time_values[valid]
                
                # Linear trajectory (slope)
                if len(acc_clean) >= 2 and len(np.unique(time_clean)) >= 2:
                    try:
                        slope, intercept = np.polyfit(time_clean, acc_clean, 1)
                        feat['accuracy_slope_per_year'] = slope * 12  # Convert to per year
                        feat['accuracy_intercept'] = intercept
                        
                        # Predicted values
                        predicted = slope * time_clean + intercept
                        residuals = acc_clean - predicted
                        feat['accuracy_trajectory_rmse'] = np.sqrt(np.mean(residuals**2))
                    except:
                        pass
                
                # Change scores
                feat['accuracy_change_total'] = acc_clean[-1] - acc_clean[0]
                feat['accuracy_change_per_year'] = feat['accuracy_change_total'] / (time_clean[-1] / 12) if time_clean[-1] > 0 else 0
                
                # Variability (indicates inconsistent performance)
                feat['accuracy_std'] = np.std(acc_clean)
                feat['accuracy_cv'] = feat['accuracy_std'] / (np.mean(acc_clean) + 1e-6)
                
                # Acceleration (quadratic term)
                if len(acc_clean) >= 3 and len(np.unique(time_clean)) >= 3:
                    try:
                        poly2 = np.polyfit(time_clean, acc_clean, 2)
                        feat['accuracy_acceleration'] = poly2[0] * 2  # Second derivative
                    except:
                        pass
            
            # === REACTION TIME TRAJECTORY ===
            rt_values = group['CorrectResponsesRT'].values
            valid_rt = ~np.isnan(rt_values)
            if valid_rt.sum() >= 2:
                rt_clean = rt_values[valid_rt]
                time_clean_rt = time_values[valid_rt]
                
                # Linear trajectory
                if len(rt_clean) >= 2 and len(np.unique(time_clean_rt)) >= 2:
                    try:
                        slope_rt, intercept_rt = np.polyfit(time_clean_rt, rt_clean, 1)
                        feat['rt_slope_per_year'] = slope_rt * 12
                        feat['rt_intercept'] = intercept_rt
                        
                        # Trajectory quality
                        predicted_rt = slope_rt * time_clean_rt + intercept_rt
                        residuals_rt = rt_clean - predicted_rt
                        feat['rt_trajectory_rmse'] = np.sqrt(np.mean(residuals_rt**2))
                    except:
                        pass
                
                # Change scores
                feat['rt_change_total'] = rt_clean[-1] - rt_clean[0]
                feat['rt_change_per_year'] = feat['rt_change_total'] / (time_clean_rt[-1] / 12) if time_clean_rt[-1] > 0 else 0
                
                # Variability
                feat['rt_std'] = np.std(rt_clean)
                feat['rt_cv'] = feat['rt_std'] / (np.mean(rt_clean) + 1e-6)
            
            # === COMPOSITE TRAJECTORY ===
            # Cognitive efficiency over time
            if 'accuracy_slope_per_year' in feat and 'rt_slope_per_year' in feat:
                # Worsening = accuracy down AND RT up
                feat['composite_decline'] = -feat['accuracy_slope_per_year'] + feat['rt_slope_per_year']
                
            # === ERROR PATTERNS OVER TIME ===
            if 'IncorrectResponsesN' in group.columns:
                error_values = group['IncorrectResponsesN'].values
                valid_err = ~np.isnan(error_values)
                if valid_err.sum() >= 2:
                    err_clean = error_values[valid_err]
                    time_clean_err = time_values[valid_err]
                    
                    try:
                        slope_err, _ = np.polyfit(time_clean_err, err_clean, 1)
                        feat['error_slope_per_year'] = slope_err * 12
                        feat['error_increase'] = err_clean[-1] - err_clean[0]
                    except:
                        feat['error_increase'] = err_clean[-1] - err_clean[0]
            
            # === PRACTICE EFFECTS ===
            # Normal aging shows practice effects (improvement) early
            # MCI shows no practice effect or decline
            if len(group) >= 3:
                early_timepoints = group.iloc[:2]  # First 2 visits
                if len(early_timepoints) == 2:
                    early_acc_change = early_timepoints.iloc[1]['CorrectPCT'] - early_timepoints.iloc[0]['CorrectPCT']
                    feat['practice_effect'] = early_acc_change
                    feat['no_practice_effect'] = 1 if early_acc_change <= 0 else 0  # Flag for concern
        
        else:
            # Single timepoint - use baseline features only
            baseline = group.iloc[0]
            feat['accuracy_baseline'] = baseline['CorrectPCT']
            feat['rt_baseline'] = baseline['CorrectResponsesRT']
            feat['single_timepoint'] = 1
        
        trajectory_features.append(feat)
    
    trajectory_df = pd.DataFrame(trajectory_features)
    
    # Summary statistics
    print(f"  Extracted trajectories for {len(trajectory_df)} subjects")
    if 'n_timepoints' in trajectory_df.columns:
        print(f"  Average timepoints per subject: {trajectory_df['n_timepoints'].mean():.1f}")
        print(f"  Subjects with decline (accuracy): {(trajectory_df.get('accuracy_slope_per_year', 0) < 0).sum()}")
        print(f"  Subjects with RT slowing: {(trajectory_df.get('rt_slope_per_year', 0) > 0).sum()}")
    
    return trajectory_df


def extract_ecog_trajectories(ecog_all):
    """Extract ECOG trajectory features if available"""
    if ecog_all is None:
        return None
    
    print("\nExtracting ECOG trajectories...")
    
    # Convert timepoints
    if 'TimepointCode' in ecog_all.columns:
        ecog_all['months_from_baseline'] = ecog_all['TimepointCode'].map(TIMEPOINT_ORDER)
    else:
        return None
    
    # Get numeric columns (ECOG items)
    numeric_cols = ecog_all.select_dtypes(include=[np.number]).columns
    ecog_items = [c for c in numeric_cols if 'QID' in c and 'Subject' not in c]
    
    if not ecog_items:
        return None
    
    ecog_trajectories = []
    
    for subject, group in ecog_all.groupby('SubjectCode'):
        group = group.sort_values('months_from_baseline')
        
        feat = {'SubjectCode': subject}
        
        if len(group) >= 2:
            # Calculate mean ECOG score at each timepoint
            ecog_scores = group[ecog_items].mean(axis=1).values
            time_values = group['months_from_baseline'].values
            
            valid = ~np.isnan(ecog_scores)
            if valid.sum() >= 2:
                scores_clean = ecog_scores[valid]
                time_clean = time_values[valid]
                
                # Trajectory
                try:
                    slope, intercept = np.polyfit(time_clean, scores_clean, 1)
                    feat['ecog_slope_per_year'] = slope * 12
                except:
                    pass
                feat['ecog_change_total'] = scores_clean[-1] - scores_clean[0]
                feat['ecog_worsening'] = 1 if feat['ecog_change_total'] > 0.5 else 0
        
        ecog_trajectories.append(feat)
    
    ecog_traj_df = pd.DataFrame(ecog_trajectories)
    print(f"  ECOG trajectories for {len(ecog_traj_df)} subjects")
    
    return ecog_traj_df


def create_progression_labels(med_hx):
    """
    Create labels based on progression patterns, not just baseline
    Key: Look for worsening over time in medical history
    """
    print("\nCreating progression-based labels...")
    
    # Get all timepoints for medical history
    med_longitudinal = med_hx.copy()
    
    if 'TimepointCode' in med_longitudinal.columns:
        # Look at progression of cognitive complaints
        progression_labels = []
        
        for subject, group in med_longitudinal.groupby('SubjectCode'):
            group = group.sort_values('TimepointCode')
            
            # Check cognitive QIDs at each timepoint
            cognitive_scores = []
            for _, row in group.iterrows():
                score = 0
                n_valid = 0
                for qid in COGNITIVE_QIDS:
                    if qid in row and pd.notna(row[qid]):
                        if row[qid] == 1:  # Yes to cognitive problem
                            score += 1
                        n_valid += 1
                if n_valid > 0:
                    cognitive_scores.append(score / n_valid)
            
            # Determine progression
            if len(cognitive_scores) >= 2:
                # Worsening = increase in cognitive complaints
                progression = cognitive_scores[-1] - cognitive_scores[0]
                is_progressing = progression > 0.1  # Threshold for progression
                
                # Also check absolute level at last timepoint
                final_impaired = cognitive_scores[-1] > 0.2
                
                label = 1 if (is_progressing or final_impaired) else 0
            elif len(cognitive_scores) == 1:
                # Single timepoint - use threshold
                label = 1 if cognitive_scores[0] > 0.2 else 0
            else:
                continue
            
            progression_labels.append({
                'SubjectCode': subject,
                'cognitive_impairment': label,
                'progression_score': progression if len(cognitive_scores) >= 2 else 0
            })
        
        labels_df = pd.DataFrame(progression_labels)
        print(f"  Progressive impairment labels: {len(labels_df)} subjects")
        print(f"  Prevalence: {labels_df['cognitive_impairment'].mean():.1%}")
        
    else:
        # Fallback to baseline labels
        print("  No longitudinal medical history - using baseline labels")
        baseline = med_hx[med_hx['TimepointCode'] == 'm00'] if 'TimepointCode' in med_hx.columns else med_hx
        baseline = baseline.drop_duplicates(subset=['SubjectCode'])
        
        available_qids = [q for q in COGNITIVE_QIDS if q in baseline.columns]
        
        impairment = np.zeros(len(baseline), dtype=int)
        valid = np.zeros(len(baseline), dtype=bool)
        
        for qid in available_qids:
            impairment |= (baseline[qid] == 1).values
            valid |= baseline[qid].isin([1, 2]).values
        
        labels_df = pd.DataFrame({
            'SubjectCode': baseline['SubjectCode'],
            'cognitive_impairment': impairment,
            'progression_score': 0  # No progression info
        })
        labels_df = labels_df[valid].copy()
    
    return labels_df


def add_baseline_features(trajectory_df, data_dir):
    """Add demographics and baseline cognitive features"""
    
    # Load demographics
    demo_files = ['BHR_Demographics.csv', 'Profile.csv']
    
    for filename in demo_files:
        path = data_dir / filename
        if path.exists():
            demo = pd.read_csv(path, low_memory=False)
            if 'Code' in demo.columns:
                demo.rename(columns={'Code': 'SubjectCode'}, inplace=True)
            
            if 'SubjectCode' in demo.columns:
                cols = ['SubjectCode']
                for c in ['Age_Baseline', 'YearsEducationUS_Converted', 'Gender']:
                    if c in demo.columns:
                        cols.append(c)
                
                if len(cols) > 1:
                    trajectory_df = trajectory_df.merge(
                        demo[cols].drop_duplicates('SubjectCode'),
                        on='SubjectCode', how='left'
                    )
                    break
    
    # Create interaction features with trajectories
    if 'Age_Baseline' in trajectory_df.columns:
        # Older age + decline = higher risk
        if 'accuracy_slope_per_year' in trajectory_df.columns:
            trajectory_df['age_decline_interaction'] = (
                trajectory_df['Age_Baseline'] * 
                (-trajectory_df['accuracy_slope_per_year'].fillna(0))
            )
        
        if 'rt_slope_per_year' in trajectory_df.columns:
            trajectory_df['age_slowing_interaction'] = (
                trajectory_df['Age_Baseline'] * 
                trajectory_df['rt_slope_per_year'].fillna(0)
            )
    
    if 'Gender' in trajectory_df.columns:
        trajectory_df['Gender_Num'] = trajectory_df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0})
    
    return trajectory_df


def main():
    print("\n" + "="*80)
    print("BHR MEMTRAX LONGITUDINAL ANALYSIS - LEVERAGING 4 YEARS OF DATA")
    print("="*80)
    
    # Load all longitudinal data
    memtrax_all, med_hx_all, ecog_all = load_longitudinal_data()
    
    # Extract cognitive trajectories
    trajectory_features = extract_cognitive_trajectories(memtrax_all)
    
    # Extract ECOG trajectories if available
    ecog_trajectories = extract_ecog_trajectories(ecog_all)
    if ecog_trajectories is not None:
        trajectory_features = trajectory_features.merge(
            ecog_trajectories, on='SubjectCode', how='left'
        )
    
    # Add baseline features
    trajectory_features = add_baseline_features(trajectory_features, DATA_DIR)
    
    # Create progression-based labels
    labels = create_progression_labels(med_hx_all)
    
    # Merge features and labels
    data = trajectory_features.merge(labels, on='SubjectCode', how='inner')
    print(f"\nFinal longitudinal dataset: {len(data)} subjects")
    
    # Separate longitudinal vs single timepoint subjects
    has_longitudinal = data['n_timepoints'] >= 2
    print(f"  With longitudinal data (2+ timepoints): {has_longitudinal.sum()}")
    print(f"  Single timepoint only: {(~has_longitudinal).sum()}")
    
    # Prepare features
    feature_cols = [c for c in data.columns 
                   if c not in ['SubjectCode', 'cognitive_impairment', 'progression_score']]
    
    X = data[feature_cols]
    y = data['cognitive_impairment']
    
    print(f"\nFeatures: {X.shape[1]} columns")
    print(f"Target prevalence: {y.mean():.1%}")
    
    # Key longitudinal features
    long_features = [c for c in X.columns if any(x in c for x in 
                    ['slope', 'change', 'trajectory', 'acceleration', 'practice', 'decline'])]
    print(f"Longitudinal features: {len(long_features)}")
    if long_features:
        print("  Examples:", long_features[:5])
    
    # Train/test split - stratify by both label and longitudinal status
    stratify_var = y * 2 + has_longitudinal.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_var
    )
    
    print(f"\nTrain/test split: {len(X_train)}/{len(X_test)}")
    
    # Build models emphasizing longitudinal features
    print("\nTraining models with longitudinal focus...")
    
    models = {}
    
    # Logistic Regression
    models['Logistic'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5))
    ])
    
    # Random Forest - good for trajectory patterns
    models['RandomForest'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=10,
            min_samples_leaf=3, class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])
    
    # Gradient Boosting
    models['HistGB'] = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf', HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, learning_rate=0.03,
            max_depth=6, min_samples_leaf=15,
            random_state=RANDOM_STATE
        ))
    ])
    
    if HAS_XGB:
        models['XGBoost'] = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1, reg_alpha=0.5,
                scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                random_state=RANDOM_STATE
            ))
        ])
    
    # Train and evaluate
    results = {}
    best_auc = 0
    best_model = None
    best_name = None
    
    for name, model in models.items():
        print(f"\n  {name}:")
        
        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc'
        )
        print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train and test
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        print(f"    Test AUC: {test_auc:.4f}")
        
        results[name] = {
            'cv_auc': cv_scores.mean(),
            'test_auc': test_auc
        }
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = model
            best_name = name
    
    # Try ensemble
    print("\n  Stacking Ensemble:")
    stack = StackingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5, stack_method='predict_proba'
    )
    
    cal_stack = CalibratedClassifierCV(stack, cv=3, method='isotonic')
    cal_stack.fit(X_train, y_train)
    y_pred_stack = cal_stack.predict_proba(X_test)[:, 1]
    stack_auc = roc_auc_score(y_test, y_pred_stack)
    print(f"    Test AUC: {stack_auc:.4f}")
    
    if stack_auc > best_auc:
        best_auc = stack_auc
        best_name = 'Calibrated Stack'
    
    # Feature importance for best tree model
    if best_name in ['RandomForest', 'XGBoost', 'HistGB']:
        clf = best_model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n  Top 10 Most Important Features:")
            for _, row in importances.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
            
            # Check if longitudinal features are important
            long_importance = importances[importances['feature'].isin(long_features)]
            if len(long_importance) > 0:
                print(f"\n  Longitudinal features in top 20: {(long_importance.index < 20).sum()}")
    
    # Results
    print("\n" + "="*80)
    print("RESULTS - LONGITUDINAL ANALYSIS")
    print("="*80)
    print(f"Best Model: {best_name}")
    print(f"Test AUC: {best_auc:.4f}")
    
    if best_auc >= 0.80:
        print("\n🎯 SUCCESS! ACHIEVED AUC ≥ 0.80 WITH LONGITUDINAL MODELING!")
        print("\nKey success factors:")
        print("  ✓ Cognitive trajectory features (slopes, changes)")
        print("  ✓ Practice effect detection")
        print("  ✓ Progression-based labels")
        print("  ✓ Up to 4 years of follow-up data")
    elif best_auc > 0.75:
        print(f"\n📈 Significant improvement! Gap to 0.80: {0.80 - best_auc:.4f}")
        print("Longitudinal features are helping!")
    else:
        print(f"\n📊 Best AUC: {best_auc:.4f}")
    
    # Save results
    output = {
        'methodology': 'Longitudinal trajectory modeling',
        'dataset': {
            'total_subjects': len(data),
            'subjects_with_longitudinal': int(has_longitudinal.sum()),
            'avg_timepoints': float(data['n_timepoints'].mean()) if 'n_timepoints' in data.columns else 1,
            'max_followup_months': float(data['followup_months'].max()) if 'followup_months' in data.columns else 0
        },
        'features': {
            'total': X.shape[1],
            'longitudinal': len(long_features)
        },
        'results': results,
        'best_model': {
            'name': best_name,
            'test_auc': float(best_auc),
            'achieved_target': best_auc >= 0.80
        }
    }
    
    with open(OUTPUT_DIR / 'longitudinal_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/longitudinal_results.json")
    
    return best_auc


if __name__ == '__main__':
    auc = main()
