#!/bin/bash
# Run once on the login node to create the virtualenv.
# Usage: bash jobs/setup.sh

set -e
cd "$(dirname "$0")/.."

python3 -m venv cs153-env
source cs153-env/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install scikit-learn numpy matplotlib

echo "Setup complete. Activate with: source cs153-env/bin/activate"
