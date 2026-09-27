.PHONY: help install test clean notebooks lint format run-v3

help:
	@echo "Available commands:"
	@echo "  make install    - Install package and dependencies"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Clean generated files"
	@echo "  make notebooks  - Start Jupyter notebook server"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make run-v3     - Run full v3 pipeline on revised data"

install:
	pip install -e .
	pip install -r requirements.txt

test:
	pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

notebooks:
	jupyter notebook notebooks/

lint:
	@echo "Running flake8..."
	-flake8 src/ --max-line-length=100 --ignore=E501,W503
	@echo "Running pylint..."
	-pylint src/ --max-line-length=100

format:
	@echo "Formatting with black..."
	black src/ --line-length=100
	@echo "Sorting imports..."
	isort src/

run-v3:
	@echo "Running improved v3 pipeline (data-revised -> data/processed/v3)..."
	python scripts/improved_pipeline.py
	@echo "Running updated v3 analytics (data/processed/v3 -> past-runs/v3/outputs)..."
	python scripts/run_updated_analysis.py
