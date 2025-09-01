#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Alzheimer's Analysis Improvements
Tests integration and functionality of all enhancement modules
"""

import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# Import all improvement modules
from improvements.dynamic_model_framework import DynamicModelSelector, create_adaptive_pipeline
from improvements.modular_feature_engineering import FeatureEngineeringPipeline, create_feature_config
from improvements.auto_hyperparameter_optimization import AutoHyperparameterOptimizer, create_optimization_pipeline
from improvements.multi_target_support import MultiTargetAlzheimerPredictor, create_multi_target_pipeline

def create_sample_alzheimer_data(n_samples=500, add_noise=True, include_longitudinal=True):
    """
    Create realistic Alzheimer's dataset for testing
    """
    np.random.seed(42)
    
    # Base demographic and clinical features
    data = {
        'Subject_ID': [f'SUBJ_{i:04d}' for i in range(n_samples)],
        'Age': np.random.normal(70, 10, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples, p=[0.45, 0.55]),
        'EDUC': np.random.normal(14, 4, n_samples),
        'SES': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        
        # MRI brain volume measures
        'eTIV': np.random.normal(1500, 200, n_samples),
        'nWBV': np.random.normal(0.75, 0.08, n_samples),
        'ASF': np.random.normal(1.2, 0.15, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic relationships
    # ASF should be negatively correlated with eTIV (important validation)
    df['ASF'] = 2000 / df['eTIV'] + np.random.normal(0, 0.05, n_samples)
    
    # Age affects brain volume
    df['nWBV'] = df['nWBV'] - (df['Age'] - 70) * 0.002 + np.random.normal(0, 0.02, n_samples)
    
    # Education protective effect
    education_effect = (df['EDUC'] - 12) * 0.01
    df['nWBV'] = df['nWBV'] + education_effect
    
    # Create MMSE based on age, education, and brain volume
    df['MMSE'] = (30 
                  - (df['Age'] - 70) * 0.1 
                  + (df['EDUC'] - 12) * 0.3
                  + (df['nWBV'] - 0.75) * 20
                  + np.random.normal(0, 2, n_samples))
    df['MMSE'] = np.clip(df['MMSE'], 0, 30)
    
    # Create CDR based on MMSE and age - using discrete values only
    df['CDR'] = 0.0
    severe_mask = df['MMSE'] < 15
    moderate_mask = (df['MMSE'] >= 15) & (df['MMSE'] < 20)
    mild_mask = (df['MMSE'] >= 20) & (df['MMSE'] < 24)
    
    df.loc[severe_mask, 'CDR'] = np.random.choice([1.0, 2.0], severe_mask.sum(), p=[0.7, 0.3])
    df.loc[moderate_mask, 'CDR'] = np.random.choice([0.5, 1.0], moderate_mask.sum(), p=[0.6, 0.4])
    df.loc[mild_mask, 'CDR'] = np.random.choice([0.0, 0.5], mild_mask.sum(), p=[0.7, 0.3])
    
    # Ensure CDR is discrete by rounding
    df['CDR'] = df['CDR'].round(1)
    
    # Create diagnosis based on CDR
    df['diagnosis'] = 'Normal'
    df.loc[df['CDR'] > 0, 'diagnosis'] = 'Impaired'
    
    # Create binary high-risk indicator
    df['high_risk'] = (df['Age'] > 75) & (df['MMSE'] < 26)
    df['high_risk'] = df['high_risk'].astype(int)
    
    # Add longitudinal features if requested
    if include_longitudinal:
        df['Visit'] = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
        df['MR_Delay'] = np.random.normal(365, 60, n_samples)  # Days between visits
        df.loc[df['Visit'] == 1, 'MR_Delay'] = 0  # Baseline
    
    # Add some missing values to make it realistic
    if add_noise:
        missing_rate = 0.05
        for col in ['SES', 'MMSE', 'EDUC']:
            missing_mask = np.random.random(n_samples) < missing_rate
            df.loc[missing_mask, col] = np.nan
    
    # One-hot encode gender
    df = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
    
    return df

def test_dynamic_model_selection(df):
    """
    Test dynamic model selection framework
    """
    print("\n" + "="*60)
    print("🧪 TESTING DYNAMIC MODEL SELECTION")
    print("="*60)
    
    try:
        # Prepare data
        feature_cols = [col for col in df.columns if col not in ['CDR', 'Subject_ID', 'diagnosis', 'high_risk']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['CDR']
        
        # Remove missing values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Data: {len(X)} samples, {len(X.columns)} features")
        
        # Test model selection
        selector = DynamicModelSelector()
        models, pool_name = selector.select_model_pool(X, y, domain='alzheimer')
        
        print(f"✅ Selected model pool: {pool_name}")
        print(f"✅ Available models: {[name for name, _ in models]}")
        
        # Test optimization
        results = selector.optimize_models(X, y, domain='alzheimer')
        
        if results['best_model']:
            print(f"✅ Best model: {results['best_model']['name']} ({results['best_score']:.3f})")
            
            # Test recommendations
            recommendations = selector.get_model_recommendations(X, y)
            print(f"✅ Recommendations: {len(recommendations)} generated")
            
            return True
        else:
            print("❌ No models successfully trained")
            return False
            
    except Exception as e:
        print(f"❌ Dynamic model selection failed: {e}")
        return False

def test_feature_engineering(df):
    """
    Test modular feature engineering pipeline
    """
    print("\n" + "="*60)
    print("🧪 TESTING FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    try:
        original_features = len(df.columns)
        print(f"📊 Original features: {original_features}")
        
        # Test configuration creation
        config = create_feature_config('alzheimer')
        print(f"✅ Configuration created: {len(config['feature_engineering']['modules'])} modules")
        
        # Test pipeline
        pipeline = FeatureEngineeringPipeline(config)
        enhanced_df = pipeline.apply_pipeline(df, target_col='CDR')
        
        new_features = len(enhanced_df.columns)
        added_features = new_features - original_features
        
        print(f"✅ Enhanced features: {new_features} (+{added_features})")
        
        # Test auto-detection
        auto_modules = pipeline._auto_detect_modules(df)
        print(f"✅ Auto-detected modules: {len(auto_modules)}")
        
        return enhanced_df, added_features > 0
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return df, False

def test_hyperparameter_optimization(df):
    """
    Test automated hyperparameter optimization
    """
    print("\n" + "="*60)
    print("🧪 TESTING HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    try:
        # Prepare data
        feature_cols = [col for col in df.columns if col not in ['CDR', 'Subject_ID', 'diagnosis', 'high_risk']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['CDR']
        
        # Remove missing values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Data: {len(X)} samples, {len(X.columns)} features")
        
        # Test optimizer
        optimizer = AutoHyperparameterOptimizer(optimization_budget=20, scoring='f1_weighted')
        
        # Test recommendations
        recommendations = optimizer.get_optimization_recommendations(X, y)
        print(f"✅ Optimization recommendations: {len(recommendations)}")
        
        # Test model optimization
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42)
        
        results = optimizer.optimize_model('RandomForest', base_model, X, y, method='random')
        
        if results['model']:
            print(f"✅ Optimization successful: {results['best_score']:.3f}")
            print(f"✅ Best parameters found: {len(results['best_params'])} params")
            return True
        else:
            print("❌ Optimization failed")
            return False
            
    except Exception as e:
        print(f"❌ Hyperparameter optimization failed: {e}")
        return False

def test_multi_target_support(df):
    """
    Test multi-target prediction system
    """
    print("\n" + "="*60)
    print("🧪 TESTING MULTI-TARGET SUPPORT")
    print("="*60)
    
    try:
        # Test target detection
        predictor = MultiTargetAlzheimerPredictor()
        detected_targets = predictor.detect_targets(df)
        
        print(f"✅ Detected targets: {list(detected_targets.keys())}")
        
        # Test with specific targets
        target_columns = ['CDR', 'MMSE', 'diagnosis']
        available_targets = [col for col in target_columns if col in df.columns]
        
        print(f"📊 Testing with targets: {available_targets}")
        
        if not available_targets:
            print("❌ No suitable targets available")
            return False
        
        # Test target preparation
        y, target_info = predictor.prepare_targets(df, available_targets)
        
        print(f"✅ Prepared targets: {len(y.columns)} columns")
        for target, info in target_info.items():
            print(f"   {target}: {info['type']} ({info.get('n_classes', 'continuous')} classes)")
        
        # Test multi-target modeling
        feature_cols = [col for col in df.columns if col not in available_targets + ['Subject_ID']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # Remove missing values
        valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 50:
            print("⚠️ Insufficient data after preprocessing")
            return False
        
        results = predictor.fit_multi_target_models(X, y, target_info)
        
        individual_models = len(results.get('individual_models', {}))
        multi_output_models = len(results.get('multi_output_models', {}))
        
        print(f"✅ Individual models trained: {individual_models}")
        print(f"✅ Multi-output models trained: {multi_output_models}")
        
        return individual_models > 0
        
    except Exception as e:
        print(f"❌ Multi-target support failed: {e}")
        return False

def test_integrated_pipeline(df):
    """
    Test all improvements working together
    """
    print("\n" + "="*60)
    print("🧪 TESTING INTEGRATED PIPELINE")
    print("="*60)
    
    try:
        print("🔗 Running integrated analysis pipeline...")
        
        # Step 1: Feature Engineering
        config = create_feature_config('alzheimer')
        pipeline = FeatureEngineeringPipeline(config)
        enhanced_df = pipeline.apply_pipeline(df, target_col='CDR')
        
        original_features = len(df.columns)
        enhanced_features = len(enhanced_df.columns)
        print(f"   Features: {original_features} → {enhanced_features}")
        
        # Step 2: Prepare data for modeling
        feature_cols = [col for col in enhanced_df.columns 
                       if col not in ['CDR', 'Subject_ID', 'diagnosis', 'high_risk']]
        X = enhanced_df[feature_cols].select_dtypes(include=[np.number])
        y = enhanced_df['CDR']
        
        # Remove missing values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"   Clean data: {len(X)} samples, {len(X.columns)} features")
        
        # Step 3: Dynamic Model Selection + Optimization
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        optimization_results = create_optimization_pipeline(
            models, X, y, budget=20, method='random'
        )
        
        best_model = optimization_results.get('best_model', {})
        if best_model:
            print(f"   Best model: {best_model['name']} ({best_model['score']:.3f})")
        
        # Step 4: Multi-target analysis (if applicable)
        target_columns = ['CDR', 'MMSE']
        available_targets = [col for col in target_columns if col in enhanced_df.columns]
        
        if len(available_targets) > 1:
            multi_results = create_multi_target_pipeline(enhanced_df, available_targets)
            if 'performance_summary' in multi_results:
                individual_count = len(multi_results['performance_summary'].get('best_individual_models', {}))
                print(f"   Multi-target: {individual_count} individual models")
        
        print("✅ Integrated pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integrated pipeline failed: {e}")
        return False

def run_comprehensive_test():
    """
    Run comprehensive test suite for all improvements
    """
    print("🧪 COMPREHENSIVE TEST SUITE FOR ALZHEIMER'S ANALYSIS IMPROVEMENTS")
    print("="*80)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Create test data
    print("📊 Creating realistic Alzheimer's test dataset...")
    df = create_sample_alzheimer_data(n_samples=300, add_noise=True, include_longitudinal=True)
    
    print(f"   Dataset: {len(df)} samples, {len(df.columns)} features")
    print(f"   Targets available: CDR, MMSE, diagnosis, high_risk")
    print(f"   Missing values: {df.isna().sum().sum()} total")
    
    # Test results tracking
    test_results = {}
    
    # Run individual tests
    test_results['dynamic_models'] = test_dynamic_model_selection(df)
    
    enhanced_df, feature_success = test_feature_engineering(df)
    test_results['feature_engineering'] = feature_success
    
    test_results['hyperparameter_opt'] = test_hyperparameter_optimization(enhanced_df)
    
    test_results['multi_target'] = test_multi_target_support(df)
    
    test_results['integrated_pipeline'] = test_integrated_pipeline(df)
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Improvements are ready for production!")
        print("\n💡 Next Steps:")
        print("   1. Review INTEGRATION_GUIDE.md for implementation details")
        print("   2. Configure improvements for your specific dataset")
        print("   3. Integrate selected improvements into main framework")
        print("   4. Run validation on your actual data")
    else:
        print("⚠️  Some tests failed - review error messages above")
        print("   Consider running individual module tests for detailed debugging")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)