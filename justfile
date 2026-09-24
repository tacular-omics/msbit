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

DATA    := "paper/data"
FASTA   := "/home/patrick-garrett/Data/Natalie/Histones/fasta/human_and_contaminants_filtered_0_05_argc_irt.fasta"
MZML    := "/home/patrick-garrett/Data/Natalie/Histones2/mzml/dda/argc/20250806_ArgC_DDA_HCD-FT_01.mzML.gz"

# Install [paper] extras
paper-setup:
    uv sync --extra paper

# Run both msbit and sagepy on the same in-memory DB (no index mismatch)
paper-run backend="cpu":
    uv run python -m paper.run_comparison {{FASTA}} {{MZML}} {{DATA}} --backend {{backend}}

# Compute recall@K + timing aggregate
paper-compare backend="cpu":
    uv run python -m paper.compare {{DATA}}/msbit.parquet {{DATA}}/sagepy.parquet {{DATA}}/compare_{{backend}}.json

# Final manuscript figure
paper-figures backend="cpu":
    uv run python -m paper.figures {{DATA}}/compare_{{backend}}.json {{DATA}}/figure_{{backend}}.png

# End-to-end on CPU (uses real ArgC data)
paper-all: paper-setup paper-run paper-compare paper-figures
