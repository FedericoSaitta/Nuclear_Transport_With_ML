#!/bin/bash
# setup.sh - One-command setup for nuclear-ml

set -e  # Exit on error

echo "🔧 Setting up nuclear-ml environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create environment
echo "📦 Creating conda environment..."
conda env create -f environment.yml

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "    conda activate nuclear-ml"