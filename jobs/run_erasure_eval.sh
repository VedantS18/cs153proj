#!/bin/bash
#SBATCH --job-name=cs153_erasure
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=cs153_erasure_%j.out
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
mkdir -p "$HF_HOME"
mkdir -p "${SCRATCH}/cs153/activations"

source cs153-env/bin/activate

echo "==== GPU ===="
nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ""
echo "==== Step 1: rebuild dataset (harder negatives) ===="
python scripts/build_dataset.py

echo ""
echo "==== Step 2: re-extract activations ===="
python scripts/extract_activations.py \
    --model meta-llama/Llama-3.2-3B \
    --data_dir data/concepts \
    --out_dir "${SCRATCH}/cs153/activations" \
    --batch_size 16 \
    --max_length 64

echo ""
echo "==== Step 3: retrain probes + save weights ===="
python scripts/train_probes.py \
    --act_dir "${SCRATCH}/cs153/activations" \
    --out_dir results \
    --n_folds 5

echo ""
echo "==== Step 4: plot probe results ===="
python scripts/plot_probes.py

echo ""
echo "==== Step 5: erasure evaluation ===="
python scripts/evaluate_erasure.py \
    --model meta-llama/Llama-3.2-3B \
    --act_dir "${SCRATCH}/cs153/activations" \
    --weights_path results/probe_weights.json \
    --out_dir results

echo ""
echo "Done. $(date)"
