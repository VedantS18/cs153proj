#!/bin/bash
#SBATCH --job-name=cs153_causal
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_causal_%j.out
#SBATCH --exclude=oat-02,oat-04

# Single-layer causal erasure sweep.
# Submit 3 in parallel:
#   sbatch --export=ALL,CONCEPTS="capital_cities element_symbols inventor_invention country_language historical_dates",SUFFIX=factual jobs/run_causal_sweep.sh
#   sbatch --export=ALL,CONCEPTS="gender_profession gender_emotion age_competence race_crime nationality_stereotype",SUFFIX=bias jobs/run_causal_sweep.sh
#   sbatch --export=ALL,CONCEPTS="hemingway shakespeare legal_text scientific_writing news_wire",SUFFIX=stylistic jobs/run_causal_sweep.sh
# Also run with alpha=15 to find erasure threshold:
#   sbatch --export=ALL,CONCEPTS="capital_cities gender_profession hemingway",SUFFIX=repr_alpha15,ALPHA=15.0 jobs/run_causal_sweep.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

SUFFIX=${SUFFIX:-all}
ALPHA=${ALPHA:-1.0}

echo ""; echo "==== Causal layer sweep (alpha=$ALPHA): ${CONCEPTS:-all} ===="
python scripts/sweep_causal_layer.py \
    --model meta-llama/Llama-3.2-3B \
    --weights_path results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --alpha "$ALPHA" \
    ${CONCEPTS:+--concepts $CONCEPTS}

mv results/causal_layer_sweep*.json "results/causal_layer_sweep_${SUFFIX}.json" 2>/dev/null || true
echo "Output: results/causal_layer_sweep_${SUFFIX}.json"
echo ""; echo "Done. $(date)"
