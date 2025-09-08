#!/usr/bin/env python3
"""
BHR Target Curation
- Selects cognitive QIDs at baseline (m00) and excludes non-cognitive conditions
- Returns a clean binary label series for a requested target QID
"""
from typing import Tuple
import pandas as pd

COGNITIVE_QIDS = {
    'QID1-5': 'Dementia',
    'QID1-12': "Alzheimer's Disease",
    'QID1-13': 'Mild Cognitive Impairment',
    'QID1-22': 'Frontotemporal Dementia',
    'QID1-23': 'Lewy Body Disease',
}

EXCLUDE_QIDS = {
    'QID1-7',   # Parkinson's
    'QID1-19',  # Movement disorder
    'QID1-10',  # Stroke
}


def curate_cognitive_target(medical_df: pd.DataFrame,
                            target_qid: str = 'QID1-13',
                            subject_col: str = 'SubjectCode',
                            timepoint_col: str = 'TimepointCode') -> Tuple[pd.DataFrame, pd.Series]:
    """Filter baseline medical history and produce a clean binary target for a cognitive QID.

    Returns
    - filtered medical baseline dataframe (one row per subject)
    - label series y (1 for Yes, 0 for No)
    """
    if timepoint_col in medical_df.columns:
        med = medical_df[medical_df[timepoint_col] == 'm00'].copy()
    else:
        med = medical_df.copy()

    med = med.drop_duplicates(subset=[subject_col], keep='first')

    # Remove non-cognitive targets if present (no-op if absent)
    med = med.drop(columns=[c for c in EXCLUDE_QIDS if c in med.columns], errors='ignore')

    if target_qid not in med.columns:
        raise ValueError(f"Target QID '{target_qid}' not found in medical data")

    valid_mask = med[target_qid].isin([1.0, 2.0, 1, 2])
    med = med[valid_mask].copy()
    y = (med[target_qid].astype(float) == 1.0).astype(int)
    return med, y 