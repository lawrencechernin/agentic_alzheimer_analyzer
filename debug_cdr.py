#!/usr/bin/env python3
"""
Debug CDR Missing Values
========================
Quick script to understand why we're losing 201 CDR values
"""

import pandas as pd
import numpy as np
import os
from typing import List

def load_and_combine_data(data_path: str) -> pd.DataFrame:
    """Load and combine OASIS data files"""
    all_files = []
    
    # Find all CSV files
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    print(f"📁 Found {len(all_files)} CSV files:")
    for f in all_files:
        print(f"   • {f}")
    
    # Load and combine
    combined_data = []
    for file_path in all_files:
        print(f"\n📈 Loading: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)
        print(f"   📊 Shape: {df.shape}")
        print(f"   📝 Columns: {list(df.columns)}")
        
        # Check CDR column
        if 'CDR' in df.columns:
            cdr_counts = df['CDR'].value_counts(dropna=False)
            print(f"   🎯 CDR distribution: {dict(cdr_counts)}")
            nan_count = df['CDR'].isna().sum()
            print(f"   ❌ CDR missing values: {nan_count}/{len(df)} ({nan_count/len(df)*100:.1f}%)")
        else:
            print(f"   ⚠️ No CDR column found")
        
        combined_data.append(df)
    
    # Combine all dataframes
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"\n🔗 Combined dataset shape: {combined_df.shape}")
        
        # Final CDR analysis
        if 'CDR' in combined_df.columns:
            cdr_counts = combined_df['CDR'].value_counts(dropna=False)
            print(f"🎯 Final CDR distribution: {dict(cdr_counts)}")
            nan_count = combined_df['CDR'].isna().sum()
            print(f"❌ Final CDR missing values: {nan_count}/{len(combined_df)} ({nan_count/len(combined_df)*100:.1f}%)")
            
            # Show some examples of missing CDR rows
            missing_cdr = combined_df[combined_df['CDR'].isna()]
            print(f"\n🔍 Sample rows with missing CDR:")
            print(missing_cdr[['Subject'] if 'Subject' in missing_cdr.columns else missing_cdr.columns[:5]].head())
        
        return combined_df
    else:
        print("❌ No data files found!")
        return pd.DataFrame()

if __name__ == "__main__":
    data_path = "/Users/lawrencechernin/agentic_alzheimer_analyzer/training_data/oasis"
    
    print("🧠 DEBUGGING CDR MISSING VALUES")
    print("=" * 40)
    
    combined_df = load_and_combine_data(data_path)
    
    print(f"\n📊 FINAL ANALYSIS:")
    print(f"   Total subjects: {len(combined_df)}")
    if 'CDR' in combined_df.columns:
        valid_cdr = combined_df.dropna(subset=['CDR'])
        print(f"   Subjects with CDR: {len(valid_cdr)}")
        print(f"   Subjects without CDR: {len(combined_df) - len(valid_cdr)}")
        print(f"   Data retention: {len(valid_cdr)/len(combined_df)*100:.1f}%")