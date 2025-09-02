# OASIS Dataset Setup Instructions

The Agentic Alzheimer's Analyzer is designed to work with the OASIS (Open Access Series of Imaging Studies) dataset for longitudinal dementia analysis. **This repository does not include the OASIS data files** - you need to download them yourself.

## 📥 How to Download OASIS Data

### Option 1: Kaggle (Recommended - Easiest)

1. **Visit the Kaggle dataset page**: 
   - Go to [https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)
   - Or search for "MRI and Alzheimers" on Kaggle

2. **Download the dataset**:
   - Click "Download" button (requires Kaggle account - free registration)
   - You'll get a ZIP file containing the CSV files

3. **Extract the files**:
   ```bash
   # Extract the downloaded archive
   unzip archive.zip
   
   # Copy the CSV files to this directory
   cp oasis_cross-sectional.csv /path/to/agentic_alzheimer_analyzer/training_data/oasis/
   cp oasis_longitudinal.csv /path/to/agentic_alzheimer_analyzer/training_data/oasis/
   ```

### Option 2: Direct from OASIS Website

1. **Visit OASIS Central**: [https://oasis-brains.org/](https://oasis-brains.org/)
2. **Create an account** (free registration required)
3. **Download OASIS-1 (Cross-sectional)** and **OASIS-2 (Longitudinal)** datasets
4. **Convert to CSV format** if needed and place in this directory

## 📁 Expected File Structure

After downloading, your `training_data/oasis/` directory should contain:

```
training_data/oasis/
├── README.md (this file)
├── oasis_cross-sectional.csv  # OASIS-1 cross-sectional data
└── oasis_longitudinal.csv     # OASIS-2 longitudinal data
```

## 📊 Dataset Information

### OASIS-1 (Cross-sectional)
- **Subjects**: ~400 individuals aged 18-96
- **Data**: Demographics, clinical assessments, brain volume measurements
- **Key Variables**: Age, Gender, Education, CDR, MMSE, brain volumes (eTIV, nWBV)

### OASIS-2 (Longitudinal) 
- **Subjects**: ~150 individuals with 2+ visits
- **Follow-up**: Up to 4 years
- **Data**: Same as OASIS-1 plus longitudinal progression tracking
- **Key Variables**: Timeline tracking, cognitive decline trajectories

## 🔍 What the Analyzer Will Do

Once you have the OASIS data in place, the Agentic Alzheimer's Analyzer will:

1. **Automatically discover** both CSV files
2. **Combine and harmonize** the cross-sectional and longitudinal datasets
3. **Apply advanced feature engineering** including brain volume normalization
4. **Run CDR prediction models** achieving 80%+ accuracy
5. **Generate comprehensive analysis** with visualizations and clinical insights

## ⚠️ Important Notes

- **File Privacy**: These CSV files contain research data and should not be committed to version control
- **Git Ignore**: The `training_data/` directory is already git-ignored to prevent accidental commits
- **Data Ethics**: Please respect the OASIS data usage terms and cite appropriately in any publications
- **File Names**: The analyzer expects exactly these filenames - don't rename them

## 🚀 Quick Verification

To verify your setup is correct:

```bash
# Check files are present
ls -la training_data/oasis/
# Should show: oasis_cross-sectional.csv, oasis_longitudinal.csv

# Test the analyzer can find them
python3 -c "
from core.datasets import get_adapter
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
adapter = get_adapter(config)
print(f'Adapter found: {type(adapter).__name__}')
print(f'Data available: {adapter.is_available() if adapter else False}')
"
```

Expected output:
```
Adapter found: OasisAdapter
Data available: True
```

## 🎯 Ready to Analyze

Once the files are in place, simply run:

```bash
python run_analysis.py
```

The system will automatically:
- Discover your OASIS datasets
- Run advanced CDR prediction analysis
- Generate research-quality reports and visualizations
- Provide clinical insights and recommendations

## 📚 Citation Requirements

If you use OASIS data in your research, please cite:

**OASIS-1**: Marcus, D.S., Wang, T.H., Parker, J., Csernansky, J.G., Morris, J.C., Buckner, R.L. Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults. Journal of Cognitive Neuroscience, 19, 1498-1507. doi: 10.1162/jocn.2007.19.9.1498

**OASIS-2**: Marcus, D.S., Fotenos, A.F., Csernansky, J.G., Morris, J.C., Buckner, R.L. Open access Series of Imaging Studies: Longitudinal MRI Data in Nondemented and Demented Older Adults. Journal of Cognitive Neuroscience, 22, 2677-2684. doi: 10.1162/jocn.2009.21407