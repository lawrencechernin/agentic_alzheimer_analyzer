#!/usr/bin/env python3
"""
Test script for CDR prediction enhancements
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cognitive_agent_enhancements import EnhancedCDRPredictor, integrate_enhancements
from agents.cognitive_analysis_agent import CognitiveAnalysisAgent
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_enhancements():
    """Test the CDR prediction enhancements"""
    
    print("\n" + "="*80)
    print("🧪 TESTING CDR PREDICTION ENHANCEMENTS")
    print("="*80)
    
    # Load OASIS data
    data_path = "./training_data/oasis/"
    
    print("\n📊 Loading OASIS data...")
    cross_df = pd.read_csv(f"{data_path}oasis_cross-sectional.csv")
    long_df = pd.read_csv(f"{data_path}oasis_longitudinal.csv")
    
    # Harmonize columns (matching benchmark approach)
    cross_df = cross_df.rename(columns={'ID': 'Subject_ID', 'M/F': 'Gender', 'Educ': 'EDUC'})
    long_df = long_df.rename(columns={'Subject ID': 'Subject_ID', 'M/F': 'Gender'})
    
    # Combine datasets
    common_cols = list(set(cross_df.columns) & set(long_df.columns))
    combined_df = pd.concat([cross_df[common_cols], long_df[common_cols]], ignore_index=True)
    
    print(f"   ✅ Loaded {len(combined_df)} records")
    
    # Initialize enhancer
    enhancer = EnhancedCDRPredictor(logger=logger)
    
    # Test 1: Brain Volume Enhancement
    print("\n🧠 Testing brain volume feature enhancement...")
    enhanced_df = enhancer.enhance_brain_volume_features(combined_df)
    
    new_features = [col for col in enhanced_df.columns if col not in combined_df.columns]
    print(f"   ✅ Added {len(new_features)} new features:")
    for feature in new_features[:5]:  # Show first 5
        print(f"      - {feature}")
    
    # Validate ASF-eTIV correlation
    if 'ASF' in enhanced_df.columns and 'eTIV' in enhanced_df.columns:
        corr = enhanced_df[['ASF', 'eTIV']].dropna().corr().iloc[0, 1]
        print(f"   📊 ASF-eTIV correlation: {corr:.3f} (expected: strongly negative)")
        if corr < -0.7:
            print("   ✅ Correlation validation PASSED")
        else:
            print("   ⚠️ Correlation weaker than expected")
    
    # Test 2: Atrophy Rate Calculation (if longitudinal data available)
    print("\n📈 Testing atrophy rate calculation...")
    if 'Visit' in long_df.columns:
        # Prepare longitudinal data with required columns
        long_test = long_df.copy()
        long_test = long_test.rename(columns={'Subject ID': 'Subject_ID'})
        
        atrophy_df = enhancer.calculate_brain_atrophy_rate(long_test)
        if not atrophy_df.empty:
            print(f"   ✅ Calculated atrophy for {len(atrophy_df)} subjects")
            
            # Show atrophy severity distribution
            if 'atrophy_severity' in atrophy_df.columns:
                severity_counts = atrophy_df['atrophy_severity'].value_counts()
                print("   📊 Atrophy severity distribution:")
                for severity, count in severity_counts.items():
                    print(f"      - {severity}: {count} subjects")
        else:
            print("   ⚠️ Could not calculate atrophy rates")
    
    # Test 3: Correlation-based Feature Selection
    print("\n🔍 Testing correlation-based feature selection...")
    
    # Prepare data for ML testing
    df_ml = enhanced_df.dropna(subset=['CDR'])
    
    # Prepare features
    feature_cols = [col for col in df_ml.columns if col not in ['CDR', 'Subject_ID', 'MRI_ID', 'Group']]
    numeric_cols = df_ml[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        X = df_ml[numeric_cols]
        y = df_ml['CDR']
        
        # Apply feature selection
        X_selected = enhancer.apply_correlation_based_feature_selection(X, y, threshold=0.05)
        
        print(f"   ✅ Selected {len(X_selected.columns)} features from {len(X.columns)} original")
        print("   📊 Top selected features:")
        for feature in X_selected.columns[:10]:  # Show top 10
            print(f"      - {feature}")
    
    # Test 4: Ensemble Model Performance
    print("\n🚀 Testing ensemble model performance...")
    
    if len(numeric_cols) > 0 and len(df_ml) > 100:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Prepare data
        X = df_ml[numeric_cols].fillna(0)  # Simple imputation for test
        y = df_ml['CDR']
        
        # Remove severe cases for benchmark comparison
        severe_mask = (y == 2.0)
        X = X[~severe_mask]
        y = y[~severe_mask]
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Test ensemble
        ensemble_results = enhancer.create_ensemble_model(X_train, y_train, X_test, y_test)
        
        print(f"   ✅ Ensemble CV accuracy: {ensemble_results['ensemble_cv_mean']:.3f} ± {ensemble_results['ensemble_cv_std']:.3f}")
        print(f"   ✅ Ensemble test accuracy: {ensemble_results.get('ensemble_test_accuracy', 0):.3f}")
        
        # Compare with individual models
        print("\n   📊 Model comparison:")
        for key, value in ensemble_results.items():
            if '_test_accuracy' in key and key != 'ensemble_test_accuracy':
                model_name = key.replace('_test_accuracy', '').upper()
                print(f"      - {model_name}: {value:.3f}")
        print(f"      - ENSEMBLE: {ensemble_results.get('ensemble_test_accuracy', 0):.3f}")
        
        # Check if we beat benchmark
        if ensemble_results.get('ensemble_test_accuracy', 0) > 0.807:
            print("\n   🎉 EXCEEDED BENCHMARK (80.7%)!")
        elif ensemble_results.get('ensemble_test_accuracy', 0) > 0.77:
            print("\n   ✅ Met colleague benchmark (77.2%)")
    
    print("\n" + "="*80)
    print("✅ ENHANCEMENT TESTING COMPLETE")
    print("="*80)
    
    return enhanced_df

if __name__ == "__main__":
    test_enhancements()