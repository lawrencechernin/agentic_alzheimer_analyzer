#!/usr/bin/env python3
"""
Test Benchmark Approach
========================
Try to replicate the exact benchmark methodology
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

def load_and_combine_oasis_data():
    """Load OASIS data exactly like the benchmark"""
    
    # Load both datasets
    cross_df = pd.read_csv("/Users/lawrencechernin/agentic_alzheimer_analyzer/training_data/oasis/oasis_cross-sectional.csv")
    long_df = pd.read_csv("/Users/lawrencechernin/agentic_alzheimer_analyzer/training_data/oasis/oasis_longitudinal.csv")
    
    print(f"📊 Cross-sectional: {cross_df.shape}")
    print(f"📊 Longitudinal: {long_df.shape}")
    
    # Harmonize column names
    cross_df = cross_df.rename(columns={
        'ID': 'Subject_ID',
        'M/F': 'Gender', 
        'Educ': 'EDUC'
    })
    
    long_df = long_df.rename(columns={
        'Subject ID': 'Subject_ID',
        'M/F': 'Gender'
    })
    
    # Get common columns
    common_cols = list(set(cross_df.columns) & set(long_df.columns))
    print(f"📋 Common columns: {common_cols}")
    
    # Select common columns and combine
    cross_common = cross_df[common_cols]
    long_common = long_df[common_cols]
    
    combined_df = pd.concat([cross_common, long_common], ignore_index=True)
    print(f"🔗 Combined dataset: {combined_df.shape}")
    
    # Drop rows missing CDR (target variable)
    before_cdr = len(combined_df)
    combined_df = combined_df.dropna(subset=['CDR'])
    after_cdr = len(combined_df)
    print(f"🎯 After dropping missing CDR: {after_cdr}/{before_cdr} subjects ({after_cdr/before_cdr*100:.1f}% retained)")
    
    # Check for CDR=2.0 (severe cases that benchmark might exclude)
    cdr_counts = combined_df['CDR'].value_counts().sort_index()
    print(f"📈 CDR distribution: {dict(cdr_counts)}")
    
    severe_cases = combined_df[combined_df['CDR'] == 2.0]
    print(f"🚨 Severe cases (CDR=2.0): {len(severe_cases)} subjects")
    
    # Try benchmark approach: exclude severe cases
    print(f"\n🧪 TESTING BENCHMARK APPROACH:")
    benchmark_df = combined_df[combined_df['CDR'] != 2.0].copy()
    print(f"   After excluding CDR=2.0: {len(benchmark_df)} subjects")
    
    return combined_df, benchmark_df

def prepare_features(df):
    """Prepare features exactly like benchmark"""
    
    # Select relevant columns (excluding CDR-related)
    feature_cols = ['Gender', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"🔧 Feature columns: {available_cols}")
    
    X = df[available_cols].copy()
    y = df['CDR'].copy()
    
    # Convert CDR to integer for classification
    y = y.astype(int)
    
    # Handle missing values
    # Mode for categorical (SES), median for numeric (MMSE)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numeric_imputer = SimpleImputer(strategy='median')
    
    if 'SES' in X.columns:
        X['SES'] = categorical_imputer.fit_transform(X[['SES']]).flatten()
    if 'MMSE' in X.columns:
        X['MMSE'] = numeric_imputer.fit_transform(X[['MMSE']]).flatten()
    
    # Encode gender
    if 'Gender' in X.columns:
        le = LabelEncoder()
        X['Gender'] = le.fit_transform(X['Gender'].fillna('Unknown'))
    
    # Drop any remaining NaN rows
    before_dropna = len(X)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    after_dropna = len(X)
    
    print(f"✅ After final cleanup: {after_dropna}/{before_dropna} subjects")
    
    return X, y

def run_benchmark_test():
    """Run the benchmark test"""
    
    print("🧠 BENCHMARK APPROACH TEST")
    print("=" * 40)
    
    # Load data
    combined_df, benchmark_df = load_and_combine_oasis_data()
    
    # Test both approaches
    for name, df in [("Combined (All CDR)", combined_df), ("Benchmark (No CDR=2.0)", benchmark_df)]:
        print(f"\n🧪 Testing {name}:")
        print(f"   Dataset size: {len(df)} subjects")
        
        X, y = prepare_features(df)
        print(f"   Final training set: {len(X)} subjects")
        
        if len(X) >= 50:  # Only run if we have enough data
            # Use simple GradientBoosting like benchmark
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring='accuracy')
            mean_cv = cv_scores.mean()
            std_cv = cv_scores.std()
            
            print(f"   📊 CV Accuracy: {mean_cv:.3f} ± {std_cv:.3f}")
            print(f"   📈 CV Range: {cv_scores.min():.3f} - {cv_scores.max():.3f}")
            
            # Check if this matches benchmark
            if abs(len(X) - 603) <= 5:  # Within 5 subjects of benchmark
                print(f"   🎯 POTENTIAL BENCHMARK MATCH! ({len(X)} ≈ 603 subjects)")
            
            if mean_cv >= 0.80:
                print(f"   🏆 BENCHMARK PERFORMANCE ACHIEVED! ({mean_cv:.1%} ≥ 80%)")

if __name__ == "__main__":
    run_benchmark_test()