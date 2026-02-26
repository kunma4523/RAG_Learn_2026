# Contributions Directory
# ====================

This directory contains templates and guidelines for contributing to the project.

## Directory Structure

```
contributions/
├── README.md
├── templates/
│   ├── architecture_template.md    # Template for adding new architectures
│   ├── notebook_template.ipynb     # Template for notebooks
│   └── evaluation_template.py      # Template for evaluation scripts
└── examples/
    └── example_contribution.md     # Example contribution
```

## How to Contribute

### Adding a New Architecture

1. Create a new file in `docs/2-architectures/architectures/`
2. Use the template in `contributions/templates/architecture_template.md`
3. Add code example in `src/pipelines/`
4. Add notebook in `notebooks/`
5. Update the architecture table in `docs/2-architectures/README.md`

### Adding a New Evaluation Metric

1. Add metric in `src/evaluation/metrics.py`
2. Add tests in `tests/`
3. Update documentation

### Contributing Code

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a PR

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
