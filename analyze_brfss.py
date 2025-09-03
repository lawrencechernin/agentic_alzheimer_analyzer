#!/usr/bin/env python3
"""
BRFSS-specific analysis script for surveillance data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Load the BRFSS data
print("📊 Loading BRFSS surveillance data...")
df = pd.read_csv('training_data/brfss/Alzheimers_Disease_And_Healthy_Aging.csv', low_memory=False)
print(f"✅ Loaded {len(df):,} records")

# Create output directory
output_dir = Path('outputs/brfss_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. TEMPORAL TRENDS ANALYSIS
print("\n📈 Analyzing temporal trends...")
temporal_trends = df.groupby(['YearStart', 'Topic'])['Data_Value'].agg(['mean', 'std', 'count'])
temporal_trends = temporal_trends.round(2)

# Key topics related to Alzheimer's
alzheimer_topics = df[df['Topic'].str.contains('Alzheimer|Dementia|Cognitive', case=False, na=False)]
if not alzheimer_topics.empty:
    print(f"   Found {len(alzheimer_topics):,} Alzheimer's-related records")
    
    # Analyze by year
    yearly_stats = alzheimer_topics.groupby('YearStart')['Data_Value'].agg(['mean', 'median', 'std', 'count'])
    print("\n   Yearly Statistics for Alzheimer's metrics:")
    print(yearly_stats)

# 2. GEOGRAPHIC ANALYSIS
print("\n🗺️ Analyzing geographic patterns...")
state_stats = df.groupby('LocationAbbr')['Data_Value'].agg(['mean', 'median', 'std', 'count'])
state_stats = state_stats.sort_values('mean', ascending=False)

print(f"   Top 5 states by average metric value:")
print(state_stats.head())
print(f"\n   Bottom 5 states by average metric value:")
print(state_stats.tail())

# 3. HEALTH TOPIC ANALYSIS
print("\n🏥 Analyzing health topics...")
topic_counts = df['Topic'].value_counts()
print(f"   Found {len(topic_counts)} unique health topics")
print("\n   Top 10 most common topics:")
for topic, count in topic_counts.head(10).items():
    print(f"   - {topic}: {count:,} records")

# 4. CLASS ANALYSIS
print("\n📋 Analyzing health classes...")
class_stats = df.groupby('Class')['Data_Value'].agg(['mean', 'median', 'count'])
class_stats = class_stats.sort_values('count', ascending=False)
print(class_stats)

# 5. STRATIFICATION ANALYSIS
print("\n👥 Analyzing demographic stratifications...")
if pd.notna(df['Stratification1']).any():
    strat1_counts = df['Stratification1'].value_counts()
    print(f"   Found {len(strat1_counts)} stratification categories")
    print("\n   Top stratifications:")
    for strat, count in strat1_counts.head(10).items():
        if pd.notna(strat):
            print(f"   - {strat}: {count:,} records")

# 6. CONFIDENCE INTERVAL ANALYSIS
print("\n📊 Analyzing data reliability (confidence intervals)...")
# Convert to numeric, handling any non-numeric values
df['High_Confidence_Limit'] = pd.to_numeric(df['High_Confidence_Limit'], errors='coerce')
df['Low_Confidence_Limit'] = pd.to_numeric(df['Low_Confidence_Limit'], errors='coerce')
df['CI_Width'] = df['High_Confidence_Limit'] - df['Low_Confidence_Limit']
reliable_data = df[df['CI_Width'] < df['CI_Width'].quantile(0.25)]
print(f"   Most reliable estimates (narrow CI): {len(reliable_data):,} records")
print(f"   Average CI width: {df['CI_Width'].mean():.2f}")

# 7. KEY INSIGHTS
insights = {
    'total_records': len(df),
    'unique_states': df['LocationAbbr'].nunique(),
    'year_range': f"{df['YearStart'].min()}-{df['YearEnd'].max()}",
    'unique_topics': len(topic_counts),
    'alzheimer_records': len(alzheimer_topics) if not alzheimer_topics.empty else 0,
    'average_value': df['Data_Value'].mean(),
    'median_value': df['Data_Value'].median()
}

# Create visualizations
print("\n🎨 Creating visualizations...")

# Plot 1: Temporal trends
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Yearly trends
yearly_means = df.groupby('YearStart')['Data_Value'].mean()
axes[0, 0].plot(yearly_means.index, yearly_means.values, marker='o')
axes[0, 0].set_title('Average Health Metrics Over Time')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Average Data Value')
axes[0, 0].grid(True)

# State distribution
top_states = state_stats.head(15)
axes[0, 1].barh(range(len(top_states)), top_states['mean'].values)
axes[0, 1].set_yticks(range(len(top_states)))
axes[0, 1].set_yticklabels(top_states.index)
axes[0, 1].set_title('Top 15 States by Average Metric')
axes[0, 1].set_xlabel('Average Data Value')

# Topic distribution
top_topics = topic_counts.head(10)
axes[1, 0].barh(range(len(top_topics)), top_topics.values)
axes[1, 0].set_yticks(range(len(top_topics)))
axes[1, 0].set_yticklabels([t[:40] + '...' if len(t) > 40 else t for t in top_topics.index])
axes[1, 0].set_title('Top 10 Health Topics')
axes[1, 0].set_xlabel('Number of Records')

# Data value distribution
axes[1, 1].hist(df['Data_Value'].dropna(), bins=50, edgecolor='black')
axes[1, 1].set_title('Distribution of Health Metric Values')
axes[1, 1].set_xlabel('Data Value')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(output_dir / 'brfss_analysis_summary.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved visualization to {output_dir / 'brfss_analysis_summary.png'}")

# Save insights
with open(output_dir / 'brfss_insights.json', 'w') as f:
    json.dump(insights, f, indent=2)
print(f"   ✅ Saved insights to {output_dir / 'brfss_insights.json'}")

# Generate summary report
print("\n" + "="*80)
print("📊 BRFSS SURVEILLANCE DATA ANALYSIS SUMMARY")
print("="*80)
print(f"\n🔍 Dataset Overview:")
print(f"   • Total records: {insights['total_records']:,}")
print(f"   • States covered: {insights['unique_states']}")
print(f"   • Time period: {insights['year_range']}")
print(f"   • Health topics: {insights['unique_topics']}")
print(f"   • Alzheimer's-specific records: {insights['alzheimer_records']:,}")

print(f"\n📈 Key Metrics:")
print(f"   • Average health metric value: {insights['average_value']:.2f}")
print(f"   • Median health metric value: {insights['median_value']:.2f}")

if not alzheimer_topics.empty:
    alz_states = alzheimer_topics.groupby('LocationAbbr')['Data_Value'].mean().sort_values(ascending=False)
    print(f"\n🧠 Alzheimer's Disease Insights:")
    print(f"   • States with highest Alzheimer's burden:")
    for state, value in alz_states.head(5).items():
        print(f"     - {state}: {value:.1f}")

print("\n✅ Analysis complete! Check outputs/brfss_analysis/ for detailed results.")
print("="*80)