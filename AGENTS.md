# Repository Guidelines

## Project Structure & Module Organization

`vllm_omni/` contains the main Python package, including entrypoints, engine code, workers, model executors, diffusion support, and model-specific implementations. Tests live under `tests/`, with markers for CPU/GPU, diffusion, omni, distributed, and benchmark coverage. Examples are split into `examples/offline_inference/` and `examples/online_serving/`. Benchmarks live in `benchmarks/`; recipes and deployment notes live in `recipes/` and `docs/`. UAD research notes and scripts are under `docs/uad/`.

## Build, Test, and Development Commands

Install for local development:

```bash
python -m pip install -e '.[dev]'
```

Run the test suite or a focused subset:

```bash
pytest
pytest tests/path/to/test_file.py -q
pytest -m "diffusion and not slow"
```

Run formatting and linting:

```bash
ruff check .
ruff format .
pre-commit run --all-files
```

Serve a model locally through the CLI:

```bash
vllm-omni serve <model> --host 0.0.0.0 --port 8000
```

## Coding Style & Naming Conventions

Python code uses 4-space indentation and a 120-character Ruff line length. Prefer typed public interfaces and keep mypy-friendly signatures for new modules. Use `snake_case` for functions and variables, `PascalCase` for classes, and module names that match existing package patterns. Do not introduce large cross-cutting abstractions unless they match existing engine, runner, scheduler, or model-executor boundaries.

## Testing Guidelines

Tests are discovered from `tests/` using `test_*.py` or `*_test.py`. Name test functions `test_*` and apply pytest markers from `pyproject.toml` when tests require GPU, distributed execution, diffusion models, or slow resources. For model or scheduler changes, add focused unit tests first; add smoke tests for serving or multimodal paths when behavior crosses request, runner, and output layers.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, sometimes with scopes such as `[CI]`, `[Test]`, or `[Docs]`. Keep commits focused and avoid mixing generated artifacts with source changes. PRs should include the motivation, key implementation details, test commands run, and any known hardware or model requirements. Include logs or screenshots only when they clarify serving behavior, benchmarks, or UI-facing examples.

## Security & Configuration Tips

Do not commit model weights, Hugging Face tokens, private endpoints, or large benchmark artifacts. Keep local outputs in ignored artifact directories. Prefer environment variables for credentials, for example `HF_TOKEN`, and document required GPU topology or model cache assumptions near scripts that need them.

## Agent-Specific Instructions

For step-based implementation work, complete one step, run the agreed validation, commit the related code, and push it to `fork/uad/dev`. Then stop and wait for user review before starting the next step. Record completed modifications and validation results in `PROGRESS.md`.
