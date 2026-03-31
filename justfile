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
