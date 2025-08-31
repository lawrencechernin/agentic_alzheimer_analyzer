# Agentic Alzheimer's Analyzer

**Autonomous AI agents for accelerating Alzheimer's disease and related dementias (ADRD) research**

## 🎯 Overview

The Agentic Alzheimer's Analyzer is an open-source framework that uses autonomous AI agents to analyze Alzheimer's datasets, generate insights, and create grant-ready reports. The system is designed to be:

- **Autonomous**: Minimal human intervention required
- **Generalizable**: Works on any Alzheimer's dataset 
- **Comprehensive**: Discovery, analysis, literature research, and reporting
- **Cost-Controlled**: Token usage monitoring and caps
- **Research-Accelerating**: Reduces analysis time from months to hours

## 🧠 Current Experiment: ECOG-MemTrax Analysis

This implementation focuses on exploring relationships between:
- **ECOG**: Everyday Cognition questionnaire (self-report vs informant)
- **MemTrax**: Digital cognitive assessment (reaction time, accuracy)

## 🏗️ Architecture

### Multi-Agent System

1. **Discovery Agent** (`agents/discovery_agent.py`)
   - Automatically discovers and characterizes datasets
   - Maps variables to standardized ontologies
   - Assesses data quality and completeness

2. **Analysis Agent** (`agents/analysis_agent.py`)
   - Executes ECOG-MemTrax correlation analysis
   - Compares self-report vs informant ratings
   - Generates statistical summaries and visualizations

3. **Literature Agent** (`agents/literature_agent.py`)
   - Searches PubMed and Semantic Scholar
   - Extracts relevant findings and effect sizes
   - Identifies novel discoveries vs confirmatory findings

4. **Orchestrator** (`core/orchestrator.py`)
   - Coordinates all agents
   - Manages workflow and error handling
   - Generates final reports

### Support Systems

- **Token Manager** (`core/token_manager.py`)
  - Monitors API usage across providers (Claude, GPT, Gemini)
  - Enforces usage limits and cost controls
  - Suggests optimal provider selection

- **Configuration System** (`config/`)
  - `config.yaml`: Main settings and experiment parameters
  - `data_dictionary.json`: Variable mappings and ontologies
  - `usage_limits.json`: API usage limits and thresholds

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the framework
cd agentic_alzheimer_analyzer

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

**Set up API keys** (choose at least one AI provider):
```bash
# For Claude/Anthropic (recommended)
export ANTHROPIC_API_KEY="your_anthropic_key"

# For OpenAI GPT (alternative)
export OPENAI_API_KEY="your_openai_key" 

# For Google Gemini (alternative)
export GEMINI_API_KEY="your_gemini_key"
```

**Configure your dataset** in `config/config.yaml`:
```yaml
dataset:
  name: "BHR_2022_ECOG_MemTrax_Validation"  # Descriptive study name
  description: "Your study description"
  
  # Data sources - can be local paths, URLs, or cloud storage
  data_sources:
    - path: "../bhr.0/BHR-ALL-EXT_Mem_2022/"  # Your data path
      type: "local_directory"
      description: "Primary study data"
    # Add more sources if needed:
    # - url: "https://api.example.com/data"
    #   type: "api_endpoint"
    # - path: "s3://bucket/data/"
    #   type: "cloud_storage"

experiment:
  name: "BHR_2022_ECOG_MemTrax_Validation_Study"  # Specific experiment name
  dataset: "BHR_2022_ECOG_MemTrax_Validation"     # Links to dataset name
  
  primary_objectives:
    - "Compare ECOG self-report vs informant ratings"
    - "Correlate ECOG scores with MemTrax performance"
    - "Your additional research questions here"
```

### 3. Run Analysis

```bash
python run_analysis.py
```

The system will:
1. 🔍 Discover your dataset automatically
2. 🧠 Execute comprehensive analysis
3. 📚 Research relevant literature
4. 💡 Generate insights and recommendations
5. 📄 Create grant-ready reports

## 📁 Output Structure

After analysis, check the `outputs/` directory:

```
outputs/
├── complete_analysis_results.json      # Complete results
├── dataset_discovery_results.json      # Dataset characterization
├── ecog_memtrax_analysis_results.json  # Core analysis results
├── literature_research_results.json    # Literature findings
├── grant_application_section.md        # Grant preliminary data
├── executive_summary.md                # Executive summary
└── visualizations/                     # Generated plots
    ├── ecog_memtrax_correlations.png
    ├── self_informant_comparison.png
    └── cognitive_performance_distributions.png
```

## ⚙️ Configuration Guide

### Dataset Configuration

**Step 1: Set your data path**
Edit the `data_sources` section in `config/config.yaml`:
```yaml
data_sources:
  - path: "../your-data-directory/"     # Update this path
    type: "local_directory"
    description: "Your study description"
```

**Step 2: File discovery**
The system automatically discovers data files based on patterns:
```yaml
file_patterns:
  cognitive_data:
    - "*MemTrax*.csv"
    - "*cognitive*.csv"
  ecog_data:
    - "*ECOG*.csv" 
    - "*EverydayCognition*.csv"
  demographic_data:
    - "*Demographics*.csv"
```

**Step 3: Customize experiment**
Update the experiment section:
```yaml
experiment:
  name: "Your_Experiment_Name"
  primary_objectives:
    - "Your research question 1"
    - "Your research question 2"
```

### Variable Mapping

Variables are mapped using the data dictionary (`config/data_dictionary.json`):

```json
{
  "variable_mappings": {
    "cognitive_assessments": {
      "memtrax_reaction_time": {
        "possible_names": [
          "CorrectResponsesRT", "ResponseTime", "reaction_time"
        ],
        "description": "Reaction time for correct responses",
        "valid_range": [0.3, 3.0]
      }
    }
  }
}
```

### Token Management

Control AI API usage in `config/usage_limits.json`:

```json
{
  "claude": {
    "daily_token_limit": 1000000,
    "monthly_cost_limit": 1000.0,
    "enabled": true
  }
}
```

## 🔬 Research Applications

### Primary Use Cases

1. **ECOG-Digital Assessment Validation**
   - Correlate self-report questionnaires with objective measures
   - Identify discrepancies between self and informant reports
   - Validate digital cognitive assessments

2. **Biomarker Discovery**
   - Identify cognitive patterns predictive of decline
   - Stratify populations for biomarker testing
   - Generate hypotheses for validation studies

3. **Literature Synthesis**
   - Automatically contextualize findings within existing research
   - Identify research gaps and novel discoveries
   - Generate citation networks and meta-analyses

### Extensibility

The framework is designed to be easily extended:

- **New Datasets**: Simply update configuration files
- **New Analyses**: Add custom analysis modules
- **New AI Providers**: Extend the token manager
- **New Domains**: Modify data dictionary and ontologies

## 🤖 AI Integration

### Supported Providers

- **Claude (Anthropic)**: Primary analysis and reasoning
- **GPT (OpenAI)**: Alternative analysis and synthesis
- **Gemini (Google)**: Large-scale literature processing

### Token Usage Optimization

- Automatic provider selection based on usage
- Real-time cost monitoring and alerts
- Graceful fallbacks when limits are reached
- Usage analytics and optimization suggestions

## 📊 Analysis Capabilities

### Statistical Analyses

- Correlation analysis (Pearson, Spearman)
- Group comparisons (t-tests, Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Multiple comparison corrections

### Visualizations

- Correlation matrices and heatmaps
- Distribution plots and histograms
- Self-informant comparison plots
- Interactive dashboards (optional)

### Clinical Insights

- Risk stratification models
- Clinical cutoff recommendations
- Biomarker candidate identification
- Treatment target suggestions

## 🌐 Open Source Impact

### Research Democratization

- Makes advanced AI analysis accessible to all research groups
- Standardizes analysis approaches across studies
- Enables rapid replication and validation
- Reduces barriers to entry for computational research

### Community Contributions

- Plugin architecture for custom analyses
- Crowdsourced data dictionaries and ontologies
- Shared analysis templates and workflows
- Collaborative development of domain-specific modules

## 🔒 Privacy and Ethics

### Data Protection

- No data transmitted to AI services (analysis is local)
- Automatic anonymization of outputs
- Configurable privacy levels
- HIPAA-compatible deployment options

### Reproducible Research

- Complete audit trail of analysis steps
- Version control for all components
- Deterministic results with fixed random seeds
- Open-source transparency

## 🆘 Troubleshooting

### Common Issues

**"No data files found"**
- Check `data_sources` path in `config/config.yaml`
- Verify file naming patterns match `file_patterns`
- Ensure files are in CSV format
- Confirm the path exists and contains data files

**"Token limit exceeded"**
- Check usage with token manager
- Adjust limits in `config/usage_limits.json`
- Consider using different AI provider

**"Variable mapping failed"**
- Review variable names in your dataset
- Update `config/data_dictionary.json`
- Add custom variable mappings

### Support

For issues and contributions:
1. Check existing documentation
2. Search issue tracker (if available)
3. Create detailed issue report
4. Consider contributing improvements

## 🎯 Future Roadmap

### Near-term Enhancements

- [ ] Integration with imaging data (MRI, PET)
- [ ] Genetic data analysis capabilities
- [ ] Real-time collaboration features
- [ ] Enhanced visualization dashboard

### Long-term Vision

- [ ] Multi-site federated analysis
- [ ] Clinical decision support integration
- [ ] Real-world evidence generation
- [ ] Regulatory submission support

## 🏆 Citation

If you use this framework in your research, please cite:

```
Agentic Alzheimer's Analyzer: An autonomous AI framework for ADRD research acceleration.
[Year]. Available at: [URL]
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

We welcome contributions from the research community:

1. Fork the repository
2. Create feature branch
3. Add tests and documentation
4. Submit pull request
5. Engage in code review

Together, we can accelerate progress toward treatments and cures for Alzheimer's disease.

---

**🧠 Empowering researchers with autonomous AI to accelerate the fight against Alzheimer's disease.**