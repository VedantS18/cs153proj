#!/bin/bash
#SBATCH --job-name=cs153_erase
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_erase_%j.out
#SBATCH --exclude=oat-02,oat-04

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

echo ""
echo "==== Erasure evaluation ===="
python scripts/evaluate_erasure.py \
    --model meta-llama/Llama-3.2-3B \
    --act_dir "${SCRATCH}/cs153/activations" \
    --weights_path results/probe_weights.json \
    --out_dir results \
    --concepts ${CONCEPTS:-}

echo ""
echo "Done. $(date)"
