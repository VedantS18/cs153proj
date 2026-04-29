#!/bin/bash
#SBATCH --job-name=cs153_probes
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_probes_%j.out
#SBATCH --exclude=oat-02,oat-04

set -e
cd "$SLURM_SUBMIT_DIR"

source .env
export HF_TOKEN

module purge
module load python/3.13.11

SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"
mkdir -p "${SCRATCH}/cs153/activations"

source cs153-env/bin/activate

echo "==== GPU ===="
nvidia-smi 2>&1 | head -20
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ""
echo "==== Step 1: build dataset ===="
python scripts/build_dataset.py

echo ""
echo "==== Step 2: extract activations ===="
python scripts/extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --data_dir data/concepts \
    --out_dir "${SCRATCH}/cs153/activations" \
    --batch_size 16 \
    --max_length 64

# symlink so train_probes can find them locally
mkdir -p activations
ln -sfn "${SCRATCH}/cs153/activations" activations/scratch

echo ""
echo "==== Step 3: train probes ===="
python scripts/train_probes.py \
    --act_dir "${SCRATCH}/cs153/activations" \
    --out_dir results \
    --n_folds 5

echo ""
echo "==== Step 4: plot results ===="
python scripts/plot_probes.py

echo ""
echo "Done. $(date)"
