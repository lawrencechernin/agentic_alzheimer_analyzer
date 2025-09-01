#!/usr/bin/env python3
"""
Test Comprehensive Clinical Evaluation with F1 Scores
Demonstrates improved evaluation system with clinical metrics
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import improvements
from improvements.clinical_evaluation_metrics import ClinicalEvaluator
from improvements.dynamic_model_framework import DynamicModelSelector
from improvements.modular_feature_engineering import FeatureEngineeringPipeline, create_feature_config

def create_alzheimer_test_data(n_samples=400):
    """
    Create realistic Alzheimer's test data with proper class distribution
    """
    np.random.seed(42)
    
    # Base features
    data = {
        'Age': np.random.normal(72, 8, n_samples),
        'EDUC': np.random.normal(14, 3, n_samples),
        'SES': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'eTIV': np.random.normal(1500, 200, n_samples),
        'nWBV': np.random.normal(0.75, 0.08, n_samples),
        'ASF': np.random.normal(1.2, 0.15, n_samples),
        'Gender_M': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    }
    
    df = pd.DataFrame(data)
    
    # Create MMSE based on realistic relationships
    df['MMSE'] = (30 
                  - (df['Age'] - 65) * 0.15  # Age effect
                  + (df['EDUC'] - 12) * 0.4  # Education protective
                  + df['nWBV'] * 15          # Brain volume effect
                  + np.random.normal(0, 2.5, n_samples))
    df['MMSE'] = np.clip(df['MMSE'], 0, 30)
    
    # Create CDR based on MMSE and age - discrete values only
    df['CDR'] = 0.0
    
    # Define clear CDR categories
    severe_mask = df['MMSE'] <= 15
    moderate_mask = (df['MMSE'] > 15) & (df['MMSE'] <= 20)
    mild_mask = (df['MMSE'] > 20) & (df['MMSE'] <= 24)
    normal_mask = df['MMSE'] > 24
    
    # Assign CDR values deterministically for cleaner classification
    df.loc[severe_mask, 'CDR'] = 2.0
    df.loc[moderate_mask, 'CDR'] = 1.0
    df.loc[mild_mask, 'CDR'] = 0.5
    df.loc[normal_mask, 'CDR'] = 0.0
    
    # Add some noise to make it realistic but keep discrete
    noise_mask = np.random.random(n_samples) < 0.1
    df.loc[noise_mask & (df['CDR'] == 0.0), 'CDR'] = 0.5
    df.loc[noise_mask & (df['CDR'] == 2.0), 'CDR'] = 1.0
    
    # Ensure CDR is treated as integers for classification
    df['CDR'] = df['CDR'].astype(int)
    
    return df

def test_clinical_evaluation_system():
    """
    Test the comprehensive clinical evaluation system
    """
    print("🧪 COMPREHENSIVE CLINICAL EVALUATION TEST")
    print("=" * 60)
    
    # Create test data
    df = create_alzheimer_test_data(n_samples=300)
    print(f"📊 Created dataset: {len(df)} samples")
    print(f"   CDR distribution: {df['CDR'].value_counts().to_dict()}")
    
    # Apply feature engineering
    config = create_feature_config('alzheimer')
    pipeline = FeatureEngineeringPipeline(config)
    enhanced_df = pipeline.apply_pipeline(df, target_col='CDR')
    
    # Prepare data
    feature_cols = [col for col in enhanced_df.columns if col not in ['CDR']]
    X = enhanced_df[feature_cols].select_dtypes(include=[np.number])
    y = enhanced_df['CDR']
    
    # Remove missing values
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"📊 Clean data: {len(X)} samples, {len(X.columns)} features")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # Initialize clinical evaluator
    evaluator = ClinicalEvaluator()
    
    # Test models with comprehensive evaluation
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    model_results = []
    
    print("\n🔍 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n📊 Evaluating {name}...")
        
        try:
            # Comprehensive clinical evaluation
            metrics = evaluator.evaluate_model_comprehensive(model, X, y, cv=5)
            
            model_result = {
                'name': name,
                'metrics': metrics
            }
            model_results.append(model_result)
            
            # Print detailed results
            evaluator.print_clinical_summary(metrics, name)
            
        except Exception as e:
            print(f"❌ {name} evaluation failed: {e}")
    
    # Compare models
    if model_results:
        print(f"\n🏆 MODEL COMPARISON")
        print("=" * 60)
        
        comparison = evaluator.compare_models(model_results)
        
        print(f"🥇 Best F1 Score: {comparison['best_f1_model']['name']} "
              f"({comparison['best_f1_model']['f1_score']:.3f})")
        print(f"🥇 Best Precision: {comparison['best_precision_model']['name']} "
              f"({comparison['best_precision_model']['precision']:.3f})")
        print(f"🥇 Best Recall: {comparison['best_recall_model']['name']} "
              f"({comparison['best_recall_model']['recall']:.3f})")
        print(f"🥇 Best Clinical Score: {comparison['best_clinical_model']['name']} "
              f"({comparison['best_clinical_model']['clinical_score']:.3f})")
        
        acceptable_models = comparison['clinically_acceptable_models']
        print(f"🏥 Clinically Acceptable Models: {acceptable_models if acceptable_models else 'None'}")
        
        # Create summary table
        print(f"\n📋 PERFORMANCE SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Model':<20} {'F1 (W)':<10} {'F1 (M)':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10} {'Clinical':<10}")
        print("-" * 80)
        
        for result in model_results:
            m = result['metrics']
            f1_w = f"{m['f1_weighted_mean']:.3f}±{m['f1_weighted_std']:.2f}"
            f1_m = f"{m['f1_macro_mean']:.3f}±{m['f1_macro_std']:.2f}"
            prec = f"{m['precision_weighted_mean']:.3f}±{m['precision_weighted_std']:.2f}"
            rec = f"{m['recall_weighted_mean']:.3f}±{m['recall_weighted_std']:.2f}"
            acc = f"{m['accuracy_mean']:.3f}±{m['accuracy_std']:.2f}"
            clin = f"{m['clinical_quality_score']:.3f}"
            
            print(f"{result['name']:<20} {f1_w:<10} {f1_m:<10} {prec:<12} {rec:<10} {acc:<10} {clin:<10}")
    
    print(f"\n✅ Comprehensive evaluation complete!")
    return model_results

def demonstrate_f1_focus():
    """
    Demonstrate F1-focused evaluation for clinical applications
    """
    print("\n" + "=" * 60)
    print("🎯 F1-FOCUSED CLINICAL EVALUATION")
    print("=" * 60)
    
    print("Why F1 Score is Critical for Alzheimer's Prediction:")
    print("• Balances Precision & Recall - crucial for medical diagnosis")
    print("• Handles class imbalance - common in clinical datasets")
    print("• Weighted F1 accounts for different class sizes")
    print("• Macro F1 ensures all classes are predicted well")
    print("• More clinically relevant than accuracy alone")
    
    print(f"\n📊 Clinical Decision Matrix:")
    print("• High Precision = Few false alarms (misdiagnoses)")
    print("• High Recall = Few missed cases (early detection)")
    print("• High F1 = Optimal balance for clinical decisions")
    print("• Clinical Quality Score = Weighted combination for medical use")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Run comprehensive test
    results = test_clinical_evaluation_system()
    
    # Demonstrate F1 focus
    demonstrate_f1_focus()