#!/bin/bash
# Run once on the login node to create the virtualenv.
# Usage: bash jobs/setup.sh

set -e
cd "$(dirname "$0")/.."

module purge
module load python/3.13.11

if [ ! -d "cs153-env" ]; then
    echo "Creating virtual environment..."
    python -m venv cs153-env
fi

source cs153-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install scikit-learn numpy matplotlib

echo "Setup complete. Activate with: source cs153-env/bin/activate"
