# Contributing to RawLLM

Thank you for your interest in contributing! This document covers how to set up a development environment, run the test suite, and submit changes.

## Development Environment

### Prerequisites

- Python 3.10 or newer
- `git`

### Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/dumb-orchestrator-poc.git
cd dumb-orchestrator-poc

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Install development/testing dependencies
pip install -r requirements-dev.txt

# 4. (Optional) install the package in editable mode for the rawllm CLI
pip install -e .

# 5. Copy the example environment file and fill in your API key(s)
cp .env.example .env
```

## Running the Tests

```bash
# Run the full test suite with coverage
pytest tests/ -v --cov=core --cov-fail-under=90

# Run a specific test file
pytest tests/test_plugin_manager.py -v

# Run with a specific test by name
pytest tests/ -k "test_rollback"
```

CI enforces a minimum coverage of 90% on the `core/` package. Please ensure new code is covered by tests.

## Code Style

- Line length: 120 characters (configured in CI `flake8` call)
- Type annotations are encouraged on all public functions
- Run `flake8 core/ plugins/ run.py --max-line-length=120 --ignore=E501,W503` before pushing

## Submitting a Pull Request

1. Create a feature branch from the relevant base branch.
2. Make your changes with clear, atomic commits.
3. Add or update tests to cover the new behaviour.
4. Verify that `pytest` passes and coverage stays above 90%.
5. Open a pull request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
6. Address any review feedback.

## Architecture Overview

See [`README.md`](README.md) for the project architecture and the `core/` package structure.

## Reporting Issues

Please use the [bug report](.github/ISSUE_TEMPLATE/bug_report.md) or [feature request](.github/ISSUE_TEMPLATE/feature_request.md) issue templates.
