# Contributing to Agentic Alzheimer's Analyzer

Thank you for your interest in contributing to the Agentic Alzheimer's Analyzer! This project aims to democratize Alzheimer's research through autonomous AI agents, and we welcome contributions from researchers, developers, and domain experts worldwide.

## 🎯 Ways to Contribute

### 1. **Research Contributions**
- **New Dataset Adapters**: Add support for additional Alzheimer's datasets (ADNI, NACC, etc.)
- **Clinical Validation**: Validate findings against known benchmarks
- **Domain Expertise**: Improve cognitive assessment ontologies and data dictionaries
- **Literature Curation**: Enhance search strategies and validation frameworks

### 2. **Technical Contributions**
- **AI Agent Improvements**: Enhance analysis algorithms and AI reasoning
- **Visualization Enhancements**: Create better clinical research visualizations
- **Performance Optimization**: Improve analysis speed and memory efficiency
- **Bug Fixes**: Address issues and improve system reliability

### 3. **Documentation & Education**
- **Tutorial Creation**: Help new users get started
- **Research Examples**: Provide example analyses and interpretations
- **Best Practices**: Document analysis workflows and methodologies
- **Translation**: Make the framework accessible globally

## 🚀 Getting Started

### Prerequisites

1. **Python Environment** (3.8+)
2. **AI API Keys** (at least one of: Claude, OpenAI, Gemini)
3. **Research Background** (helpful for clinical contributions)
4. **Git/GitHub** familiarity

### Development Setup

```bash
# 1. Fork the repository on GitHub
# Click "Fork" at https://github.com/lawrencechernin/agentic_alzheimer_analyzer

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/agentic_alzheimer_analyzer.git
cd agentic_alzheimer_analyzer

# 3. Set up development environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 4. Set up pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# 5. Add upstream remote
git remote add upstream https://github.com/lawrencechernin/agentic_alzheimer_analyzer.git
```

### Testing Your Setup

```bash
# Verify the system works with sample data
python -c "
from core.datasets import get_adapter
print('✅ Core imports working')
"

# Run basic tests (when available)
python -m pytest tests/ -v
```

## 🔄 Development Workflow

### 1. **Create a Feature Branch**

```bash
# Always start from the latest main branch
git checkout main
git pull upstream main

# Create a descriptive feature branch
git checkout -b feature/add-adni-adapter
# or
git checkout -b fix/correlation-analysis-bug
# or 
git checkout -b docs/quickstart-tutorial
```

### 2. **Branch Naming Conventions**

- `feature/description` - New features or enhancements
- `fix/description` - Bug fixes
- `docs/description` - Documentation improvements
- `refactor/description` - Code restructuring
- `test/description` - Test additions or improvements

### 3. **Development Guidelines**

#### **Code Style**
```python
# Follow PEP 8 style guide
# Use descriptive variable names
memtrax_reaction_times = df['CorrectResponsesRT']  # Good
rt = df['CorrectResponsesRT']  # Less clear

# Add docstrings for functions
def analyze_cognitive_performance(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze cognitive performance metrics from assessment data.
    
    Args:
        data: DataFrame containing cognitive assessment results
        
    Returns:
        Dict containing analysis results and clinical insights
    """
    pass
```

#### **Commit Message Format**
```bash
# Use conventional commit format
git commit -m "feat: add ADNI dataset adapter with longitudinal support"
git commit -m "fix: resolve correlation analysis memory overflow"
git commit -m "docs: add quickstart tutorial for new users" 
git commit -m "refactor: simplify dataset discovery logic"
```

#### **Testing Requirements**
- Add tests for new features
- Ensure existing tests pass
- Test with multiple datasets if applicable
- Include edge case handling

```python
# Example test structure
def test_oasis_adapter_loading():
    """Test OASIS adapter correctly loads and combines datasets."""
    config = load_test_config()
    adapter = OasisAdapter(config)
    
    assert adapter.is_available()
    
    data = adapter.load_combined()
    assert len(data) > 0
    assert 'CDR' in data.columns
    assert 'Subject_ID' in data.columns
```

### 4. **Making Changes**

#### **For Dataset Adapters**
```python
# Create new adapter following the pattern
class YourDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Your initialization logic
        
    def is_available(self) -> bool:
        # Check if data files exist
        
    def load_combined(self) -> pd.DataFrame:
        # Load and combine your dataset
        
    def data_summary(self) -> Dict[str, Any]:
        # Provide dataset metadata
```

#### **For Analysis Enhancements**
- Update `agents/cognitive_analysis_agent.py` for new analysis methods
- Add to `config/data_dictionary.json` for new variable mappings
- Update visualizations in the analysis agent as needed

#### **For Documentation**
- Update README.md for major feature changes
- Add examples in `examples/` directory
- Update configuration guides as needed

### 5. **Testing Your Changes**

```bash
# Run comprehensive tests
python run_analysis.py  # Test full analysis pipeline

# Test specific components
python -c "
from agents.discovery_agent import DataDiscoveryAgent
agent = DataDiscoveryAgent()
result = agent.discover_datasets()
print('✅ Discovery agent working')
"

# Test with different configurations
# Modify config/config.yaml and test various scenarios
```

## 📝 Pull Request Process

### 1. **Before Submitting**

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up-to-date with main

```bash
# Update your branch with latest main
git checkout main
git pull upstream main
git checkout your-feature-branch
git rebase main
```

### 2. **Pull Request Template**

When creating a PR, please include:

```markdown
## 🎯 Description
Brief description of what this PR accomplishes.

## 🔬 Type of Change
- [ ] 🆕 New feature (dataset adapter, analysis method, etc.)
- [ ] 🐛 Bug fix
- [ ] 📚 Documentation update
- [ ] 🔄 Refactoring (no functional changes)
- [ ] ✅ Test additions/improvements

## 🧪 Testing
- [ ] Tested with OASIS dataset
- [ ] Tested with [other datasets if applicable]
- [ ] Added/updated tests
- [ ] All existing tests pass

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed the code
- [ ] Documented any complex logic
- [ ] No hardcoded values (uses config)
- [ ] Handles edge cases appropriately

## 🔍 Additional Context
Any additional information that reviewers should know.

## 📊 Results (if applicable)
Include analysis results, performance metrics, or screenshots.
```

### 3. **Pull Request Review Process**

1. **Automated Checks**: CI/CD will run basic tests
2. **Maintainer Review**: Core team will review for:
   - Code quality and style
   - Clinical accuracy and validity
   - Generalizability to other datasets
   - Performance implications
3. **Community Review**: Other contributors may provide feedback
4. **Approval & Merge**: After approval, maintainers will merge

### 4. **Review Timeline**
- **Simple fixes**: 1-3 days
- **New features**: 1-2 weeks
- **Major changes**: 2-4 weeks

We'll provide feedback promptly and work with you to get your contribution ready.

## 🎯 Contribution Areas

### **High-Priority Needs**

#### **Dataset Adapters** 🔥
- **ADNI (Alzheimer's Disease Neuroimaging Initiative)**
- **NACC (National Alzheimer's Coordinating Center)**  
- **UK Biobank cognitive assessments**
- **International datasets** (European, Asian cohorts)

#### **Analysis Enhancements** 🧠
- **Longitudinal trajectory modeling**
- **Multi-modal biomarker integration**
- **Advanced visualization techniques**
- **Clinical decision support algorithms**

#### **Research Validation** 📊
- **Cross-dataset validation studies**
- **Benchmark comparisons with published results**
- **Clinical significance threshold validation**
- **Statistical method improvements**

### **Documentation Needs** 📚
- **Video tutorials** for new users
- **Research methodology guides**
- **Configuration examples** for different study types
- **Clinical interpretation guides**

### **Community Building** 🌍
- **Translation** to other languages
- **Educational materials** for different audiences
- **Conference presentations** and workshops
- **Collaboration frameworks** for multi-site studies

## 🔬 Research Contribution Guidelines

### **Clinical Accuracy Standards**
- All analysis methods must be validated against established benchmarks
- Statistical approaches should follow clinical research best practices
- Effect size interpretations must align with Alzheimer's research standards
- Novel findings require literature validation

### **Generalizability Requirements**
- New features must work across different datasets
- Avoid hardcoded assumptions specific to one study
- Use configuration-driven approaches
- Test with multiple data structures

### **Reproducibility Standards**
- Document all analysis steps clearly
- Provide examples and test cases
- Include statistical parameters and thresholds
- Enable others to replicate results

## 🛡️ Code of Conduct

### **Our Standards**
- **Inclusive Environment**: Welcoming to all contributors regardless of background
- **Respectful Communication**: Professional and constructive feedback
- **Collaborative Spirit**: Focus on advancing Alzheimer's research together
- **Scientific Rigor**: Maintain high standards for research validity
- **Open Science**: Transparent methods and reproducible results

### **Expected Behavior**
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the research community
- Show empathy toward other contributors

### **Unacceptable Behavior**
- Harassment, discrimination, or exclusionary behavior
- Trolling, insulting comments, or personal attacks
- Publishing others' private information without permission
- Any conduct that would be inappropriate in a professional setting

## 🆘 Getting Help

### **Questions & Support**
- **GitHub Discussions**: For general questions and community discussions
- **Issues**: For bug reports and feature requests
- **Documentation**: Check README.md and docs/ directory first
- **Examples**: Look at examples/ directory for usage patterns

### **Maintainer Contact**
- Tag `@lawrencechernin` in issues/PRs for urgent matters
- Use GitHub Discussions for non-urgent questions
- Check existing issues before creating new ones

### **Community Resources**
- **Weekly Office Hours**: [Schedule TBD - community-driven]
- **Research Collaboration Channel**: [Platform TBD]
- **Conference Presentations**: [Update as available]

## 🎉 Recognition

### **Contributor Recognition**
- **Contributors List**: Added to README.md
- **Release Notes**: Major contributions highlighted
- **Academic Credit**: Co-authorship opportunities for significant research contributions
- **Conference Presentations**: Opportunities to present collaborative work

### **Types of Recognition**
- **Code Contributors**: Technical improvements and new features
- **Research Contributors**: Clinical validation and domain expertise
- **Documentation Contributors**: Tutorials, guides, and educational materials
- **Community Contributors**: Support, outreach, and collaboration building

## 🚀 Future Vision

Our goal is to build the **world's largest open-source framework for Alzheimer's research acceleration**. Your contributions help:

- **Democratize Research**: Make advanced analysis accessible globally
- **Accelerate Discovery**: Reduce research timelines from years to months
- **Improve Patient Outcomes**: Enable earlier detection and intervention
- **Foster Collaboration**: Connect researchers worldwide around shared tools
- **Advance Open Science**: Promote transparency and reproducibility

Every contribution, no matter how small, moves us closer to a world without Alzheimer's disease.

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

**🧠 Together, we can accelerate the fight against Alzheimer's disease through collaborative open-source research.**

Thank you for contributing to this important cause!