# Contributing

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest
```

## Pull requests

- Keep preprocessing operations training-fold only.
- Add or update tests for behavioral changes.
- Run the complete test suite before submitting a change.
- Do not commit patient-level data, sample identifiers, model checkpoints, or third-party annotation files.
- Document changes to frozen study protocols separately from implementation fixes.
