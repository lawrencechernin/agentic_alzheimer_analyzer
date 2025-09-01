#!/usr/bin/env python3
"""
ADNI Data Merger
Merge MemTrax, MRI, PET, and blood biomarker data for comprehensive analysis

This script helps merge all the downloaded ADNI data tables based on subject IDs
and visit codes to create a unified dataset for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ADNIDataMerger:
    def __init__(self, data_dir: str = "./adni_downloads"):
        """
        Initialize ADNI Data Merger
        
        Args:
            data_dir: Directory containing downloaded ADNI CSV files
        """
        self.data_dir = Path(data_dir)
        self.merged_data = None
        self.data_dict = {}
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        logger.info(f"Initialized data merger with directory: {data_dir}")
    
    def load_all_tables(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the data directory"""
        logger.info("Loading all CSV tables...")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data directory")
        
        for csv_file in csv_files:
            try:
                table_name = csv_file.stem
                df = pd.read_csv(csv_file)
                self.data_dict[table_name] = df
                logger.info(f"Loaded {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                logger.warning(f"Could not load {csv_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(self.data_dict)} tables")
        return self.data_dict
    
    def identify_key_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify key columns for merging (subject IDs and visit codes)
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (subject_id_columns, visit_columns)
        """
        subject_cols = []
        visit_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['rid', 'ptid', 'subject']):
                subject_cols.append(col)
            elif any(x in col_lower for x in ['viscode', 'visit']):
                visit_cols.append(col)
        
        return subject_cols, visit_cols
    
    def standardize_merge_keys(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        Standardize merge key columns across tables
        
        Args:
            df: DataFrame to standardize
            table_name: Name of the table
            
        Returns:
            DataFrame with standardized columns
        """
        df_copy = df.copy()
        
        # Common column mappings
        column_mappings = {
            'RID': 'RID',
            'PTID': 'PTID', 
            'VISCODE': 'VISCODE',
            'VISCODE2': 'VISCODE2',
            'EXAMDATE': 'EXAMDATE'
        }
        
        # Rename columns to standard names if they exist
        for old_col, new_col in column_mappings.items():
            if old_col in df_copy.columns and old_col != new_col:
                df_copy = df_copy.rename(columns={old_col: new_col})
        
        # Ensure RID is integer if present
        if 'RID' in df_copy.columns:
            df_copy['RID'] = pd.to_numeric(df_copy['RID'], errors='coerce').astype('Int64')
        
        # Convert dates to datetime if present
        date_columns = ['EXAMDATE', 'USERDATE', 'USERDATE2']
        for col in date_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        logger.info(f"Standardized {table_name}: shape {df_copy.shape}")
        return df_copy
    
    def create_memtrax_cohort(self) -> pd.DataFrame:
        """Create the base cohort from MemTrax participants"""
        logger.info("Creating MemTrax participant cohort...")
        
        # Find MemTrax table
        memtrax_tables = [k for k in self.data_dict.keys() if 'MEMTRAX' in k.upper()]
        
        if not memtrax_tables:
            raise ValueError("No MemTrax tables found in loaded data")
        
        # Use the main MemTrax table
        memtrax_table = memtrax_tables[0]
        memtrax_df = self.standardize_merge_keys(
            self.data_dict[memtrax_table], 
            memtrax_table
        )
        
        # Filter to completed tests only
        if 'DONE' in memtrax_df.columns:
            memtrax_df = memtrax_df[memtrax_df['DONE'] == 1]
            logger.info(f"Filtered to {memtrax_df.shape[0]} completed MemTrax assessments")
        
        logger.info(f"MemTrax cohort: {memtrax_df['RID'].nunique()} unique participants")
        return memtrax_df
    
    def merge_clinical_data(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Merge clinical and demographic data"""
        logger.info("Merging clinical data...")
        
        clinical_tables = ['PTDEMOG', 'ADNIMERGE', 'CDR', 'ECOG', 'MMSE', 'ADAS', 'MOCA']
        merged_df = base_df.copy()
        
        for table_name in clinical_tables:
            if table_name in self.data_dict:
                try:
                    clinical_df = self.standardize_merge_keys(
                        self.data_dict[table_name], 
                        table_name
                    )
                    
                    # Merge on RID and VISCODE if available
                    merge_cols = ['RID']
                    if 'VISCODE' in clinical_df.columns and 'VISCODE' in merged_df.columns:
                        merge_cols.append('VISCODE')
                    
                    merged_df = merged_df.merge(
                        clinical_df, 
                        on=merge_cols, 
                        how='left', 
                        suffixes=('', f'_{table_name}')
                    )
                    
                    logger.info(f"Merged {table_name}: {merged_df.shape[1]} total columns")
                    
                except Exception as e:
                    logger.warning(f"Could not merge {table_name}: {e}")
                    continue
        
        return merged_df
    
    def merge_neuroimaging_data(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Merge MRI and PET data"""
        logger.info("Merging neuroimaging data...")
        
        # MRI tables
        mri_tables = ['UCSFFSL', 'UCSFVOL', 'DTIROI']
        # PET tables  
        pet_tables = ['SUMMARYSUVR', 'AV45', 'FDG', 'PIB', 'AV1451']
        
        neuroimaging_tables = mri_tables + pet_tables
        merged_df = base_df.copy()
        
        for table_name in neuroimaging_tables:
            # Check for exact match or partial match
            matching_tables = [k for k in self.data_dict.keys() if table_name in k]
            
            if matching_tables:
                actual_table_name = matching_tables[0]  # Use first match
                try:
                    neuro_df = self.standardize_merge_keys(
                        self.data_dict[actual_table_name], 
                        actual_table_name
                    )
                    
                    # Merge on RID (neuroimaging might not have exact VISCODE match)
                    merged_df = merged_df.merge(
                        neuro_df, 
                        on='RID', 
                        how='left', 
                        suffixes=('', f'_{actual_table_name}')
                    )
                    
                    logger.info(f"Merged {actual_table_name}: {merged_df.shape[1]} total columns")
                    
                except Exception as e:
                    logger.warning(f"Could not merge {actual_table_name}: {e}")
                    continue
        
        return merged_df
    
    def merge_biomarker_data(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Merge blood/plasma and CSF biomarker data"""
        logger.info("Merging biomarker data...")
        
        biomarker_keywords = [
            'PLASMA', 'SIMOA', 'ELECSYS', 'LUMIPULSE', 
            'CSF', 'UPENNBIOMK', 'BIOMARK', 'APOERES'
        ]
        
        merged_df = base_df.copy()
        
        for keyword in biomarker_keywords:
            # Find tables containing this keyword
            matching_tables = [k for k in self.data_dict.keys() if keyword in k.upper()]
            
            for table_name in matching_tables:
                try:
                    biomarker_df = self.standardize_merge_keys(
                        self.data_dict[table_name], 
                        table_name
                    )
                    
                    # Merge on RID
                    merged_df = merged_df.merge(
                        biomarker_df, 
                        on='RID', 
                        how='left', 
                        suffixes=('', f'_{table_name}')
                    )
                    
                    logger.info(f"Merged {table_name}: {merged_df.shape[1]} total columns")
                    
                except Exception as e:
                    logger.warning(f"Could not merge {table_name}: {e}")
                    continue
        
        return merged_df
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """Create the unified multimodal dataset"""
        logger.info("Creating unified multimodal dataset...")
        
        # Load all tables
        self.load_all_tables()
        
        # Start with MemTrax cohort
        unified_df = self.create_memtrax_cohort()
        
        # Add clinical data
        unified_df = self.merge_clinical_data(unified_df)
        
        # Add neuroimaging data
        unified_df = self.merge_neuroimaging_data(unified_df)
        
        # Add biomarker data
        unified_df = self.merge_biomarker_data(unified_df)
        
        self.merged_data = unified_df
        
        logger.info(f"Final unified dataset: {unified_df.shape[0]} rows, {unified_df.shape[1]} columns")
        logger.info(f"Participants: {unified_df['RID'].nunique()}")
        
        return unified_df
    
    def generate_data_summary(self) -> Dict:
        """Generate summary statistics of the merged dataset"""
        if self.merged_data is None:
            raise ValueError("No merged data available. Run create_unified_dataset() first.")
        
        df = self.merged_data
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_participants': df['RID'].nunique(),
            'date_range': {
                'earliest': df['EXAMDATE'].min() if 'EXAMDATE' in df.columns else None,
                'latest': df['EXAMDATE'].max() if 'EXAMDATE' in df.columns else None
            },
            'missing_data': df.isnull().sum().sum(),
            'completion_rate': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        # Data type breakdown
        summary['column_types'] = df.dtypes.value_counts().to_dict()
        
        # Key variables availability
        key_vars = ['RID', 'PTID', 'VISCODE', 'EXAMDATE', 'AGE', 'GENDER', 'EDUCATION']
        summary['key_variables'] = {var: var in df.columns for var in key_vars}
        
        return summary
    
    def save_unified_dataset(self, output_path: str = "unified_memtrax_multimodal_dataset.csv") -> None:
        """Save the unified dataset to CSV"""
        if self.merged_data is None:
            raise ValueError("No merged data available. Run create_unified_dataset() first.")
        
        output_file = Path(output_path)
        self.merged_data.to_csv(output_file, index=False)
        
        logger.info(f"Unified dataset saved to: {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")


def main():
    """Main function to create unified dataset"""
    try:
        # Initialize merger
        merger = ADNIDataMerger("./adni_downloads")
        
        # Create unified dataset
        unified_data = merger.create_unified_dataset()
        
        # Generate summary
        summary = merger.generate_data_summary()
        
        print("\n" + "="*50)
        print("UNIFIED MEMTRAX MULTIMODAL DATASET SUMMARY")
        print("="*50)
        print(f"Total participants: {summary['unique_participants']:,}")
        print(f"Total observations: {summary['total_rows']:,}")
        print(f"Total variables: {summary['total_columns']:,}")
        print(f"Data completion rate: {summary['completion_rate']:.1f}%")
        
        if summary['date_range']['earliest']:
            print(f"Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
        
        print(f"\nKey variables available:")
        for var, available in summary['key_variables'].items():
            status = "✓" if available else "✗"
            print(f"  {status} {var}")
        
        # Save dataset
        merger.save_unified_dataset()
        
        print(f"\nUnified dataset saved as 'unified_memtrax_multimodal_dataset.csv'")
        print("This dataset includes:")
        print("• MemTrax cognitive assessments")
        print("• MRI structural and DTI data")
        print("• PET amyloid, tau, and FDG data") 
        print("• Blood/plasma biomarkers")
        print("• CSF biomarkers")
        print("• Clinical and demographic data")
        
    except Exception as e:
        logger.error(f"Error creating unified dataset: {e}")


if __name__ == "__main__":
    main()