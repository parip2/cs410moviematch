#!/bin/bash
set -e

echo " Movie Match Setup"

# Detect python command
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: Python is not installed or not on PATH."
    exit 1
fi

echo "[1/2] Creating virtual environment..."
$PY -m venv venv
echo ""

# Activate venv on Mac or Linux
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
# Activate venv on Windows Git Bash
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "ERROR: Could not activate virtual environment."
    exit 1
fi

echo "Virtual environment activated."
echo ""

echo "[2/2] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed."
echo ""

echo "[3/3] Generating cleaned CSV..."
$PY dataset_preprocessing_script.py
echo "CSV generation done."
echo ""

echo "Starting application..."
$PY app.py
