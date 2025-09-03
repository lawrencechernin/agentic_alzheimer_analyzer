#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path for package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def df():
    """Synthetic Alzheimer's-like dataset for tests expecting a 'df' fixture."""
    rng = np.random.default_rng(42)
    n_samples = 400

    data = {
        'Subject_ID': [f'SUBJ_{i:04d}' for i in range(n_samples)],
        'Age': rng.normal(70, 10, n_samples),
        'Gender': rng.choice(['M', 'F'], n_samples, p=[0.45, 0.55]),
        'EDUC': rng.normal(14, 4, n_samples),
        'SES': rng.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'eTIV': rng.normal(1500, 200, n_samples),
        'nWBV': rng.normal(0.75, 0.08, n_samples),
        'ASF': rng.normal(1.2, 0.15, n_samples),
    }
    frame = pd.DataFrame(data)

    # Physiological relationships
    frame['ASF'] = 2000 / frame['eTIV'] + rng.normal(0, 0.05, n_samples)
    frame['nWBV'] = frame['nWBV'] - (frame['Age'] - 70) * 0.002 + rng.normal(0, 0.02, n_samples)
    frame['nWBV'] = frame['nWBV'] + (frame['EDUC'] - 12) * 0.01

    frame['MMSE'] = (
        30
        - (frame['Age'] - 70) * 0.1
        + (frame['EDUC'] - 12) * 0.3
        + (frame['nWBV'] - 0.75) * 20
        + rng.normal(0, 2, n_samples)
    ).clip(0, 30)

    # Discrete CDR from MMSE
    frame['CDR'] = 0.0
    severe_mask = frame['MMSE'] < 15
    moderate_mask = (frame['MMSE'] >= 15) & (frame['MMSE'] < 20)
    mild_mask = (frame['MMSE'] >= 20) & (frame['MMSE'] < 24)
    frame.loc[severe_mask, 'CDR'] = rng.choice([1.0, 2.0], severe_mask.sum(), p=[0.7, 0.3])
    frame.loc[moderate_mask, 'CDR'] = rng.choice([0.5, 1.0], moderate_mask.sum(), p=[0.6, 0.4])
    frame.loc[mild_mask, 'CDR'] = rng.choice([0.0, 0.5], mild_mask.sum(), p=[0.7, 0.3])
    frame['CDR'] = frame['CDR'].round(1)

    frame['diagnosis'] = 'Normal'
    frame.loc[frame['CDR'] > 0, 'diagnosis'] = 'Impaired'
    frame['high_risk'] = ((frame['Age'] > 75) & (frame['MMSE'] < 26)).astype(int)

    # Longitudinal fields
    frame['Visit'] = rng.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
    frame['MR_Delay'] = rng.normal(365, 60, n_samples)
    frame.loc[frame['Visit'] == 1, 'MR_Delay'] = 0

    # Missingness
    for col in ['SES', 'MMSE', 'EDUC']:
        missing_mask = rng.random(n_samples) < 0.05
        frame.loc[missing_mask, col] = np.nan

    frame = pd.get_dummies(frame, columns=['Gender'], prefix='Gender')
    return frame 