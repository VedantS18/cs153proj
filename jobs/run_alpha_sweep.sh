#!/bin/bash
#SBATCH --job-name=cs153_alpha
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=cs153_alpha_%j.out
#SBATCH --exclude=oat-02,oat-04

# Alpha (steering strength) sweep at probe peak + candidate causal layers.
# One representative concept per category + all bias concepts.
# Submit with: sbatch jobs/run_alpha_sweep.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5

echo ""; echo "==== Alpha sweep ===="
python scripts/sweep_alpha.py \
    --model meta-llama/Llama-3.2-3B \
    --weights_path results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --concepts capital_cities element_symbols historical_dates \
               gender_profession gender_emotion race_crime \
               hemingway legal_text scientific_writing \
    --test_layers 1 20 24 27 \
    --alphas 1.0 2.0 5.0 10.0 15.0 20.0

echo ""; echo "Done. $(date)"
