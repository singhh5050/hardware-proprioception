#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not in PATH."
  exit 1
fi

echo "Creating virtual environment in ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

if [[ -f "requirements.txt" ]]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
elif [[ -f "pyproject.toml" ]]; then
  echo "Installing project from pyproject.toml (including dev dependencies)..."
  pip install -e ".[dev]"
else
  echo "No requirements.txt or pyproject.toml found. Skipping dependency install."
fi

echo ""
echo "Setup complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
