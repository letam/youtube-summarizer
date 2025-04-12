# List available commands
default:
    @just --list

# Install production dependencies only
install:
    pip install -r requirements.txt

# Install both production and development dependencies
install-dev:
    pip install -r requirements-dev.txt

# Run the Flask development server
run:
    python app.py

# Format code using Ruff
fmt:
    ruff format .
    ruff check . --fix

# Check code without fixing
check:
    ruff check .

# Run tests
test:
    pytest

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
    pip freeze | grep -v -f <(pip freeze -r requirements-dev.txt) > requirements.txt

# Update requirements-dev.txt with current development dependencies
freeze-dev:
    pip freeze > requirements-dev.txt