#!/bin/bash

# Setup script for ChartsLLM project

echo "Setting up ChartsLLM project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data/chat
mkdir -p data/logs
mkdir -p data/metadata

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Prepare your training data in the data/ directories"
echo "2. Edit configs/base_model_config.yaml to configure training"
echo "3. Run: python scripts/train_base_chat_model.py"
echo "4. Then run: python scripts/train_domain_model.py"
echo ""
echo "To activate the virtual environment later, run:"
echo "  source venv/bin/activate"

