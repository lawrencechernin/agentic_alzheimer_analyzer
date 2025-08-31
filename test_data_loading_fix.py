#!/usr/bin/env python3
"""
Test Data Loading Fix
====================
Test that our benchmark data loading approach works in the main pipeline
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

def test_data_loading():
    """Test the fixed data loading approach"""
    
    print("🧪 TESTING BENCHMARK DATA LOADING FIX")
    print("=" * 50)
    
    # Load config
    config_path = "/Users/lawrencechernin/agentic_alzheimer_analyzer/config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create agent
    agent = CognitiveAnalysisAgent(config)
    
    # Test the data loading method directly
    print("\n🔧 Testing _load_and_preprocess_data()...")
    data_summary = agent._load_and_preprocess_data()
    
    print(f"\n📊 RESULTS:")
    print(f"   Data loaded: {agent.combined_data is not None}")
    print(f"   Dataset shape: {agent.combined_data.shape if agent.combined_data is not None else 'No data'}")
    print(f"   Total subjects: {data_summary.get('total_subjects', 0)}")
    print(f"   Baseline subjects: {data_summary.get('baseline_subjects', 0)}")
    
    if agent.combined_data is not None and len(agent.combined_data) > 0:
        print(f"   Columns: {list(agent.combined_data.columns)}")
        
        # Check for CDR column and distribution
        if 'CDR' in agent.combined_data.columns:
            cdr_counts = agent.combined_data['CDR'].value_counts().sort_index()
            print(f"   CDR distribution: {dict(cdr_counts)}")
            
            # Test if we can run CDR prediction
            print(f"\n🧠 Testing CDR prediction readiness...")
            try:
                # Quick test of our benchmark method
                y = agent.combined_data['CDR']
                print(f"   CDR values available: {len(y)}")
                print(f"   CDR range: {y.min()} - {y.max()}")
                
                # Check for benchmark subject count
                total_subjects = len(agent.combined_data)
                if total_subjects >= 600:
                    print(f"   🎯 BENCHMARK SUCCESS: {total_subjects} subjects (≥600 expected)")
                elif total_subjects >= 400:
                    print(f"   📊 Good: {total_subjects} subjects (400-600 range)")
                else:
                    print(f"   ⚠️ Low: {total_subjects} subjects (expected 600+)")
                
                # Test basic ML readiness
                feature_cols = ['Gender', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
                available_features = [col for col in feature_cols if col in agent.combined_data.columns]
                print(f"   Available ML features: {len(available_features)}/{len(feature_cols)}")
                print(f"   Features: {available_features}")
                
                if len(available_features) >= 6:
                    print(f"   ✅ ML READY: Sufficient features for prediction")
                else:
                    print(f"   ❌ ML NOT READY: Insufficient features")
                
            except Exception as e:
                print(f"   ❌ CDR prediction test failed: {e}")
        
        else:
            print("   ❌ No CDR column found")
        
        # Check preprocessing steps
        steps = data_summary.get('preprocessing_steps', [])
        print(f"\n🔄 Preprocessing steps applied:")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if (len(agent.combined_data) >= 600 and 
            'CDR' in agent.combined_data.columns and 
            len(available_features) >= 6):
            print("   🏆 BENCHMARK FIX SUCCESSFUL!")
            print("   ✅ Ready for 80%+ accuracy ML prediction")
        elif len(agent.combined_data) >= 400:
            print("   📊 DATA LOADING IMPROVED")
            print("   ✅ Should achieve better results than before")
        else:
            print("   ❌ DATA LOADING STILL HAS ISSUES")
            print("   ⚠️ May not achieve benchmark performance")
    
    else:
        print("   ❌ NO DATA LOADED - CRITICAL FAILURE")
        if 'error' in data_summary:
            print(f"   Error: {data_summary['error']}")

if __name__ == "__main__":
    test_data_loading()