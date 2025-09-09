#!/usr/bin/env python3
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_informant_residuals(data_dir: Path, base: pd.DataFrame, subject_col: str = 'SubjectCode') -> pd.DataFrame:
    df = base.copy()
    if subject_col != 'SubjectCode' and subject_col in df.columns:
        df = df.rename(columns={subject_col: 'SubjectCode'})

    targets: List[str] = []
    for name, csv in [('ECOG', 'BHR_EverydayCognition.csv'), ('SP_ECOG', 'BHR_SP_ECog.csv'), ('SP_ADL', 'BHR_SP_ADL.csv')]:
        p = data_dir / csv
        if not p.exists():
            continue
        eco = pd.read_csv(p, low_memory=False)
        if 'SubjectCode' not in eco.columns:
            continue
        if 'TimepointCode' in eco.columns:
            eco = eco[eco['TimepointCode'] == 'm00'].copy()
        num_cols = eco.select_dtypes(include=[np.number]).columns.tolist()
        keep = ['SubjectCode'] + num_cols
        eco_small = eco[keep].drop_duplicates(subset=['SubjectCode'], keep='first')
        if num_cols:
            eco_small[f'{name}_GlobalMean'] = eco_small[num_cols].mean(axis=1)
        feature_cols = ['SubjectCode'] + [c for c in eco_small.columns if c.startswith(f'{name}_')]
        df = df.merge(eco_small[feature_cols], on='SubjectCode', how='left')
        for c in feature_cols:
            if c != 'SubjectCode' and c not in targets:
                targets.append(c)

    if 'Age_Baseline' in df.columns and 'YearsEducationUS_Converted' in df.columns:
        X_demo = df[['Age_Baseline', 'YearsEducationUS_Converted']].apply(pd.to_numeric, errors='coerce')
        for t in targets:
            if t in df.columns:
                y = pd.to_numeric(df[t], errors='coerce')
                mask = X_demo.notna().all(axis=1) & y.notna()
                if mask.sum() >= 100:
                    lr = LinearRegression()
                    lr.fit(X_demo.loc[mask], y.loc[mask])
                    pred = lr.predict(X_demo.loc[mask])
                    df.loc[mask, f'{t}_Residual'] = y.loc[mask] - pred

    if subject_col != 'SubjectCode':
        df = df.rename(columns={'SubjectCode': subject_col})
    return df 