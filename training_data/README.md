# Training Data Directory

This directory contains datasets used for training and evaluating the Agentic Alzheimer's Analyzer.

## 📁 Directory Structure

```
training_data/
├── README.md                    # This file
├── oasis/                      # OASIS MRI dataset
│   ├── README.md              # Dataset details and results
│   ├── oasis_cross-sectional.csv
│   └── oasis_longitudinal.csv
└── [future datasets]/         # Additional datasets can be added here
```

## 🎯 Current Performance Summary

| Dataset | Best Model | Accuracy | Weighted F1 | Key Features |
|---------|------------|----------|-------------|--------------|
| OASIS   | XGBoost    | 72.9%    | 0.715       | MMSE, eTIV, Gender, Age |

## 📋 Dataset Guidelines

When adding new datasets to this directory:

1. **Create a dedicated folder** for each dataset
2. **Include a README.md** with:
   - Data source and citations
   - Variable descriptions
   - Current best results from our framework
   - Any dataset-specific notes
3. **Use consistent naming** for data files
4. **Update config/config.yaml** to point to the new data path

## 🔧 Configuration

To use datasets in this directory, update your `config/config.yaml`:

```yaml
dataset:
  data_sources:
    - path: "./training_data/[dataset_name]/"
      type: "local_directory"
      description: "[Dataset description]"
```

## 📊 Adding New Datasets

For new Alzheimer's/dementia datasets, consider including:

- **Cognitive assessments**: MMSE, MoCA, CDR, etc.
- **Neuroimaging data**: MRI volumes, cortical thickness, etc.
- **Demographics**: Age, gender, education, SES
- **Biomarkers**: CSF, PET, genetic markers (if available)
- **Longitudinal tracking**: Multiple timepoints for progression analysis

---
*Maintained by the Agentic Alzheimer's Analyzer Project*