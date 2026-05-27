#!/bin/bash
#SBATCH --job-name=cs153_sweep
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_sweep_%j.out
#SBATCH --exclude=oat-02,oat-04

# Sweep erasure depth for a subset of concepts.
# Submit 3 jobs in parallel (one per category):
#   sbatch --export=ALL,CONCEPTS="capital_cities element_symbols inventor_invention country_language historical_dates",SUFFIX=factual jobs/run_depth_sweep.sh
#   sbatch --export=ALL,CONCEPTS="gender_profession gender_emotion age_competence race_crime nationality_stereotype",SUFFIX=bias jobs/run_depth_sweep.sh
#   sbatch --export=ALL,CONCEPTS="hemingway shakespeare legal_text scientific_writing news_wire",SUFFIX=stylistic jobs/run_depth_sweep.sh

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

echo "==== GPU ===="
nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

SUFFIX=${SUFFIX:-all}
OUT_FILE="results/depth_sweep_${SUFFIX}.json"

echo ""
echo "==== Erasure depth sweep: $CONCEPTS ===="
python scripts/sweep_erasure_depth.py \
    --model meta-llama/Llama-3.2-3B \
    --weights_path results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    ${CONCEPTS:+--concepts $CONCEPTS}

# Rename output to avoid collision between parallel jobs
mv results/depth_sweep.json "$OUT_FILE" 2>/dev/null || true
echo "Output: $OUT_FILE"

echo ""
echo "Done. $(date)"
