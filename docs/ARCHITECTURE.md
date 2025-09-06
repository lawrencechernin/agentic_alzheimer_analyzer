# Architecture Overview

This document describes the high-level architecture, data flow, and extension points of the Agentic Alzheimer's Analyzer.

## Components

- Discovery Agent (`agents/discovery_agent.py`)
  - Enumerates files and schemas, maps variables via `config/data_dictionary.json`
  - Produces dataset characterization and analysis recommendations

- Cognitive Analysis Agent (`agents/cognitive_analysis_agent.py`)
  - Loads/cleans data (via adapters), runs correlation and modeling analyses
  - Generates figures and clinical/statistical summaries (with FDR correction)

- Literature Research Agent (`agents/literature_agent.py`)
  - Searches PubMed/Semantic Scholar, extracts findings, estimates novelty

- Orchestrator (`core/orchestrator.py`)
  - Coordinates agents, synthesizes insights, creates reports
  - Supports online AI providers or offline deterministic mode

- Token Manager (`core/token_manager.py`)
  - Tracks usage/cost across providers and enforces limits

## Dataset Adapters

Adapters are responsible for dataset-specific loading and minimal cleaning. They live under `core/datasets/` and implement:

- `BaseDatasetAdapter`
  - `is_available() -> bool`: indicates if data exists
  - `load_combined() -> DataFrame`: loads merged baseline frame
  - `data_summary() -> dict`: returns analysis-ready summary metadata

Current adapters:
- `OasisAdapter`: loads `training_data/oasis/oasis_cross-sectional.csv` and `oasis_longitudinal.csv`, harmonizes columns, and imputes SES/MMSE
- `BrfssAdapter`: discovers CSVs from configured paths/patterns and keeps numeric and key surveillance columns
- `GenericCSVAdapter`: loads one or more CSVs from a file/folder; supports sampling via `analysis.use_sampling`
- `ADDIWorkbenchAdapter`: ingests AD Workbench CSV exports, detects subject identifiers, selects baseline per subject, and merges safely

Selection:
- The agent uses `core/datasets.get_adapter(config)` to choose an adapter based on `config.dataset.name` keywords and availability. If no adapter is available, the agent falls back to the existing OASIS logic.

### Adapter selection matrix (name hints)
- Name contains `oasis` → `OasisAdapter`
- Name contains `brfss` or `healthy_aging` → `BrfssAdapter`
- Name contains `addi` or `workbench` → `ADDIWorkbenchAdapter`
- Name contains `generic`, `csv`, or `kaggle` → `GenericCSVAdapter`
- No match → try all in the order above and pick the first available

## Data Flow

1. Orchestrator loads config and initializes agents
2. Discovery Agent inspects files/variables and writes `outputs/dataset_discovery_results.json`
3. Cognitive Agent loads data (adapter or fallback), preprocesses, and computes:
   - Descriptives, cross-assessment correlations (with FDR), self/informant comparison
   - Predictive models (e.g., CDR) with F1-focused evaluation (if available)
   - Visualizations saved in `outputs/visualizations/`
4. Literature Agent researches and summarizes relevant studies
5. Orchestrator synthesizes insights (AI-based or offline rule-based) and emits reports

## AI Online vs Offline

- Offline: Set `ai_settings.offline_mode: true` to disable external calls and generate deterministic, rule-based insights and summaries (default in `config/config.yaml`)
- Online: Uses configured AI providers (`ai_providers`) for synthesis and summaries

## Extension Points

- New Dataset: Implement a `BaseDatasetAdapter` subclass and add hints to `core/datasets/__init__.py`
- New Analysis: Extend `CognitiveAnalysisAgent` with new methods and add results to saved artifacts
- New Literature Source: Add a client to `LiteratureResearchAgent` for additional APIs

## Outputs

- `outputs/complete_analysis_results.json`: end-to-end results snapshot
- `outputs/key_findings_summary.md`: executive/key findings summary
- `outputs/visualizations/`: figures (correlations, distributions, etc.)
- `outputs/literature_research_results.json`: literature findings and novelty

## Testing & CI

- Unit tests cover enhancements and agent behaviors
- A synthetic `df` fixture (`conftest.py`) supports deterministic tests
- Non-interactive tests skip prompting and external credentials 