#!/usr/bin/env python3
"""
Demographics Enrichment Utility
- Merges Age_Baseline, YearsEducationUS_Converted, Gender from multiple sources
- Accepts 'Code' as subject key and normalizes to 'SubjectCode'
- Adds derived features: squared terms, interactions, and cognitive reserve proxy
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np


def _read_and_normalize(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    if 'SubjectCode' not in df.columns and 'Code' in df.columns:
        df = df.rename(columns={'Code': 'SubjectCode'})
    if 'SubjectCode' not in df.columns:
        return None
    return df


def enrich_demographics(data_dir: Path, base: pd.DataFrame, subject_col: str = 'SubjectCode') -> pd.DataFrame:
    """Merge demographics (age, education, gender) and add derived interactions.

    Parameters
    - data_dir: path containing BHR CSV files
    - base: dataframe with at least SubjectCode column
    - subject_col: subject identifier column name

    Returns
    - Enriched dataframe (copy) with added demographic columns where available
    """
    df = base.copy()
    if subject_col != 'SubjectCode' and subject_col in df.columns:
        df = df.rename(columns={subject_col: 'SubjectCode'})

    sources: List[tuple[str, List[str]]] = [
        ('BHR_Demographics.csv', ['SubjectCode', 'Age_Baseline', 'YearsEducationUS_Converted', 'Gender']),
        ('Profile.csv', ['SubjectCode', 'YearsEducationUS_Converted', 'Age', 'Gender']),
        ('Participants.csv', ['SubjectCode', 'Age_Baseline', 'YearsEducationUS_Converted', 'Gender']),
        ('Subjects.csv', ['SubjectCode', 'Age_Baseline'])
    ]

    for filename, desired_cols in sources:
        src = _read_and_normalize(data_dir / filename)
        if src is None:
            continue
        keep_cols = [c for c in desired_cols if c in src.columns]
        if len(keep_cols) <= 1:
            continue
        src_small = src[keep_cols].drop_duplicates(subset=['SubjectCode'], keep='first').copy()
        before_cols = set(df.columns)
        df = df.merge(src_small, on='SubjectCode', how='left')
        added = [c for c in df.columns if c not in before_cols]
        if added:
            pass  # no print in library utility

    # Normalize core fields
    if 'Age' in df.columns and 'Age_Baseline' not in df.columns:
        df['Age_Baseline'] = pd.to_numeric(df['Age'], errors='coerce')
    if 'YearsEducationUS_Converted' in df.columns:
        df['YearsEducationUS_Converted'] = pd.to_numeric(df['YearsEducationUS_Converted'], errors='coerce')
    if 'Age_Baseline' in df.columns:
        df['Age_Baseline'] = pd.to_numeric(df['Age_Baseline'], errors='coerce')

    # Derived features
    if 'Age_Baseline' in df.columns:
        df['Age_Baseline_Squared'] = df['Age_Baseline'] ** 2
        df['Age_Per_Decade'] = df['Age_Baseline'] / 10.0
    if 'YearsEducationUS_Converted' in df.columns:
        df['Education_Years'] = df['YearsEducationUS_Converted']
        df['Education_Squared'] = df['YearsEducationUS_Converted'] ** 2
    if 'Age_Baseline' in df.columns and 'YearsEducationUS_Converted' in df.columns:
        df['Age_Education_Interaction'] = df['Age_Baseline'] * df['YearsEducationUS_Converted']
        df['CognitiveReserveProxy'] = df['YearsEducationUS_Converted'] / (df['Age_Baseline'] / 50.0 + 1e-6)

    # Gender numeric
    if 'Gender' in df.columns and 'Gender_Numeric' not in df.columns:
        gender_map = {'Male': 1, 'M': 1, 'Female': 0, 'F': 0}
        df['Gender_Numeric'] = df['Gender'].map(gender_map)

    if subject_col != 'SubjectCode':
        df = df.rename(columns={'SubjectCode': subject_col})

    return df 