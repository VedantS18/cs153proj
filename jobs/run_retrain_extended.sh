#!/bin/bash
#SBATCH --job-name=cs153_retrain
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cs153_retrain_%j.out
#SBATCH --exclude=oat-02,oat-04

# Retrain probes saving per-layer directions and top-K subspace directions.
# Must complete before run_mode_sweep.sh jobs.

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

echo "==== Extended probe retraining ===="
python scripts/train_probes.py \
    --act_dir "${SCRATCH}/cs153/activations" \
    --out_dir results \
    --n_folds 5 \
    --max_iter 1000

echo ""
echo "Done. $(date)"
