#!/usr/bin/env -S just --justfile

# Virtual environment path alias
venv := "./venv/bin"

# List available commands
default:
    @just --list

# Install production dependencies only
install:
    [ -d venv ] || python3 -m venv venv
    {{venv}}/pip install -r requirements.txt

# Install both production and development dependencies
install-dev:
    {{venv}}/pip install -r requirements-dev.txt

# Run the Flask development server
run:
    {{venv}}/python app.py

# Format code using Ruff
fmt:
    {{venv}}/ruff format .
    {{venv}}/ruff check . --fix

# Check code without fixing
check:
    {{venv}}/ruff check .

# Run tests
test:
    {{venv}}/pytest

# Clean up Python cache files and temporary files
clean:
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -r {} +
    find . -type d -name "*.egg" -exec rm -r {} +
    find . -type d -name ".pytest_cache" -exec rm -r {} +
    find . -type d -name ".ruff_cache" -exec rm -r {} +

# Update requirements.txt with current production dependencies
freeze:
    {{venv}}/pip freeze | grep -v -f <({{venv}}/pip freeze -r requirements-dev.txt) > requirements.txt

# Update requirements-dev.txt with current development dependencies
freeze-dev:
    {{venv}}/pip freeze > requirements-dev.txt
