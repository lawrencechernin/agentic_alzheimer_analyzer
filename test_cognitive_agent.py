#!/usr/bin/env python3
"""
Test Cognitive Analysis Agent
=============================
Quick test of just the ML portion
"""

import sys
import os
sys.path.append('/Users/lawrencechernin/agentic_alzheimer_analyzer')

import pandas as pd
import numpy as np
from agents.cognitive_analysis_agent import CognitiveAnalysisAgent
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def load_combined_oasis_data():
    """Load and combine OASIS data"""
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
    
    return combined_df

def test_cognitive_agent():
    """Test the cognitive analysis agent"""
    
    print("🧠 TESTING COGNITIVE ANALYSIS AGENT")
    print("=" * 40)
    
    # Load config
    config_path = "/Users/lawrencechernin/agentic_alzheimer_analyzer/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    combined_data = load_combined_oasis_data()
    print(f"✅ Loaded combined data: {combined_data.shape}")
    
    # Create agent
    agent = CognitiveAnalysisAgent(config)
    agent.combined_data = combined_data  # Set the data directly
    
    # Run the advanced CDR prediction (this is the key method)
    print("\n🔬 Running CDR prediction analysis...")
    results = agent._advanced_cdr_prediction()
    
    # Check results
    if 'best_model' in results:
        best_model = results['best_model']
        print(f"\n🏆 RESULTS:")
        print(f"   Best Model: {best_model.get('name', 'Unknown')}")
        print(f"   CV Accuracy: {best_model.get('cv_accuracy', 0):.3f}")
        print(f"   Test Accuracy: {best_model.get('test_accuracy', 0):.3f}")
        
        # Check if we achieved benchmark performance
        cv_acc = best_model.get('cv_accuracy', 0)
        if cv_acc >= 0.80:
            print(f"   🎯 BENCHMARK ACHIEVED! ({cv_acc:.1%} ≥ 80%)")
        else:
            print(f"   📊 Below benchmark: {cv_acc:.1%} < 80%")
    
    # Show all tested models
    if 'models_tested' in results:
        print(f"\n📋 All models tested:")
        for model in results['models_tested']:
            name = model.get('name', 'Unknown')
            cv_mean = model.get('cv_mean', 0)
            print(f"   {name}: {cv_mean:.3f}")

if __name__ == "__main__":
    test_cognitive_agent()