default: lint format check test

# Install dependencies
install:
    uv sync

# Run linting checks
lint:
    uv run ruff check src

# Format code
format:
	uv run ruff check --select I --fix src
	uv run ruff format src

# Run ty type checker
ty:
    uv run ty check src

# Run type checking
check:
    just lint
    just ty
    just test

# Run tests
test:
    uv run pytest tests

# Run tests with coverage report
coverage:
    uv run pytest tests --cov=src/msbit --cov-report=term-missing

# Build and publish to PyPI
publish:
    uv build
    uv publish

# --- paper: sagepy comparison experiment ---

DATA := "paper/data"

# Install [paper] extras and fetch FASTA + test mzML
paper-setup:
    uv sync --extra paper
    uv run python -m paper.download {{DATA}}

# Build msbit BitmapIndex from FASTA
paper-build-db:
    uv run python -m paper.build_db {{DATA}}/human.fasta {{DATA}}/db

# Run msbit presearch (set BACKEND=gpu to test the CUDA path)
paper-run-msbit backend="cpu":
    uv run python -m paper.run_msbit {{DATA}}/db {{DATA}}/test.mzML {{DATA}}/msbit_{{backend}}.parquet --backend {{backend}}

# Run sagepy presearch on the same inputs
paper-run-sagepy:
    uv run python -m paper.run_sagepy {{DATA}}/human.fasta {{DATA}}/test.mzML {{DATA}}/sagepy.parquet

# Compute recall@K + timing aggregate
paper-compare backend="cpu":
    uv run python -m paper.compare {{DATA}}/msbit_{{backend}}.parquet {{DATA}}/sagepy.parquet {{DATA}}/compare_{{backend}}.json

# Final manuscript figure
paper-figures backend="cpu":
    uv run python -m paper.figures {{DATA}}/compare_{{backend}}.json {{DATA}}/figure_{{backend}}.png

# End-to-end on CPU
paper-all: paper-setup paper-build-db paper-run-msbit paper-run-sagepy paper-compare paper-figures
