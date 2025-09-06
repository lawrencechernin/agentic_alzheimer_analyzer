# Using AD Workbench (ADDI) Exports

This guide explains how to run the analyzer on data exported from the Alzheimer's Disease Data Initiative (ADDI) / AD Workbench.

## 1) Export your data from Workbench
- Use Workbench tools to export relevant tables (clinical, cognitive assessments, demographics, etc.) as CSVs
- Keep file names descriptive (e.g., `cognitive_assessments.csv`, `demographics.csv`)
- Ensure subject identifiers are included consistently across files

## 2) Place files locally
Put your CSVs in a folder, for example:
```
training_data/addi_workbench_export/
  cognitive_assessments.csv
  demographics.csv
  clinical_history.csv
  visits.csv
```

## 3) Run the analyzer
Option A: Use CLI flags (no config edits required):
```
python run_analysis.py --offline --data-path training_data/addi_workbench_export/ --limit-rows 20000
```
- `--offline`: disables external AI calls; only deterministic summaries are used
- `--data-path`: points the system to your exported folder
- `--limit-rows`: enables sampling for large datasets

Option B: Configure `config/config.yaml`:
```yaml
dataset:
  name: "ADDI_Workbench"
  data_sources:
    - path: "./training_data/addi_workbench_export/"
      type: "local_directory"
  file_patterns:
    assessment_data:
      - "*.csv"
```
Then run:
```
python run_analysis.py
```

## 4) What the adapter does
- Discovers CSVs in the provided folder
- Detects a common subject identifier (e.g., `RID`, `PTID`, `Subject_ID`, `SubjectCode`)
- Selects baseline per subject (earliest date if available)
- Merges datasets conservatively to avoid Cartesian joins
- Produces a combined baseline table for analysis

## 5) Privacy and ethics
- Offline mode avoids sending any external requests for AI synthesis
- When online AI is enabled, only aggregated, anonymized summaries are used in prompts
- Avoid placing PHI in free-text columns; prefer coded fields

## 6) Troubleshooting
- If the system cannot find a subject ID column, rename your identifier to one of: `Subject_ID`, `SubjectCode`, `subject_id`, `participant_id`, `RID`, or `PTID`
- If merging leads to very small retained cohorts, verify subject ID consistency across files
- For very large exports, keep `--limit-rows` enabled and increase as needed 