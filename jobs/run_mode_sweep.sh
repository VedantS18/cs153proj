#!/bin/bash
#SBATCH --job-name=cs153_modes
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=cs153_modes_%j.out
#SBATCH --exclude=oat-02,oat-04

# Run comprehensive erasure mode sweep.
# Submit 3 in parallel (one per category) after run_retrain_extended.sh completes:
#   sbatch --export=ALL,CONCEPTS="capital_cities element_symbols inventor_invention country_language historical_dates",SUFFIX=factual jobs/run_mode_sweep.sh
#   sbatch --export=ALL,CONCEPTS="gender_profession gender_emotion age_competence race_crime nationality_stereotype",SUFFIX=bias jobs/run_mode_sweep.sh
#   sbatch --export=ALL,CONCEPTS="hemingway shakespeare legal_text scientific_writing news_wire",SUFFIX=stylistic jobs/run_mode_sweep.sh

set -e
cd "$SLURM_SUBMIT_DIR"

source .env
export HF_TOKEN

source /etc/profile.d/z00_lmod.sh
module purge
module load python/3.13.11

SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"

source cs153-env/bin/activate
pip install -q datasets

echo "==== GPU ===="
nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

SUFFIX=${SUFFIX:-all}

echo ""
echo "==== Mode sweep: ${CONCEPTS:-all concepts} ===="
python scripts/sweep_erasure_modes.py \
    --model meta-llama/Llama-3.2-3B \
    --weights_path results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --rank_k 4 \
    ${CONCEPTS:+--concepts $CONCEPTS}

mv results/mode_sweep.json "results/mode_sweep_${SUFFIX}.json" 2>/dev/null || true
echo "Output: results/mode_sweep_${SUFFIX}.json"

echo ""
echo "==== Generating completion examples ===="
python scripts/generate_erasure_examples.py \
    --model meta-llama/Llama-3.2-3B \
    --weights_path results/probe_weights.json \
    --out_dir results \
    --max_new_tokens 35 \
    ${CONCEPTS:+--concepts $CONCEPTS}

mv results/erasure_examples.json "results/erasure_examples_${SUFFIX}.json" 2>/dev/null || true
mv results/erasure_examples.txt  "results/erasure_examples_${SUFFIX}.txt"  2>/dev/null || true
echo "Output: results/erasure_examples_${SUFFIX}.txt"

echo ""
echo "Done. $(date)"
