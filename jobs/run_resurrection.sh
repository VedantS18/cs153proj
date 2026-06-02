#!/bin/bash
#SBATCH --job-name=cs153_resurrect
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=cs153_resurrect_%j.out
#SBATCH --exclude=oat-02,oat-04

# Concept resurrection curve — runs all 15 concepts in one job.
# Submit with: sbatch jobs/run_resurrection.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5

echo ""; echo "==== Resurrection curve ===="
python scripts/sweep_resurrection.py \
    --model meta-llama/Llama-3.2-3B \
    --weights_path results/probe_weights.json \
    --data_dir data/concepts \
    --out_dir results \
    --n_examples 80

echo ""; echo "Done. $(date)"
