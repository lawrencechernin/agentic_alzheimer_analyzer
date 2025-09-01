# Contributing Guide

Thanks for your interest in contributing to the Agentic Alzheimer's Analyzer!

## Getting Started

- Fork the repository and clone your fork
- Python >= 3.9 recommended
- Create a virtual environment and install requirements:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```
pytest -q
```

Tests are designed to run non-interactively. A synthetic dataframe fixture (`conftest.py`) is provided to validate improvements without large datasets.

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes with clear, minimal edits and meaningful commit messages
3. Add/update tests and docs as appropriate
4. Run the full test suite and lint locally
5. Open a Pull Request (PR) to `main` with:
   - Summary of changes and motivation
   - Any breaking changes or migration notes
   - Screenshots of outputs if UI/plots changed

We use conventional commit prefixes where possible (feat, fix, docs, refactor, test, chore).

## Coding Standards

- Follow the existing code style (see editorconfig/formatting by example)
- Prefer descriptive names and early returns; avoid deep nesting
- Add concise docstrings for non-trivial functions and modules
- Keep imports explicit; avoid wildcard imports

## Adding a New Dataset Adapter

- Create a module under `core/datasets/` implementing `BaseDatasetAdapter`:
  - `is_available()` checks presence of necessary files
  - `load_combined()` returns a merged, analysis-ready `DataFrame`
  - `data_summary()` returns metadata used by the agent
- Register a selection hint (e.g., by name keyword) in `core/datasets/__init__.py`
- Add a short section in `README.md` if the dataset requires special setup

## Documentation

- Architecture: `ARCHITECTURE.md`
- Main README: usage overview, configuration tips, and outputs
- Include concise examples and commands in PRs when adding major features

## Community & Support

- Please search existing issues before filing a new one
- Use clear titles and include minimal repro steps when reporting bugs
- Be respectful and constructive in discussions and reviews

## License

By contributing, you agree that your contributions will be licensed under the MIT License. 