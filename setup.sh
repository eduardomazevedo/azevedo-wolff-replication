#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create output directory if it doesn't exist
if [ ! -d "output" ]; then
    echo "Creating output directory..."
    mkdir output
fi

echo "Setup complete! Virtual environment is ready to use."
echo "To activate the virtual environment, run: source .venv/bin/activate" 