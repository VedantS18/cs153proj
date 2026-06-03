#!/bin/bash
#SBATCH --job-name=cs153_new_styles
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=cs153_new_styles_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

echo "=== Step 1: Build dataset with new styles ==="
python scripts/build_dataset.py

echo "=== Step 2: Extract activations for new concepts ==="
for concept in jk_rowling cormac_mccarthy; do
    echo "--- Extracting activations: $concept ---"
    python scripts/extract_activations.py \
        --model meta-llama/Llama-3.2-3B \
        --concept $concept \
        --data_dir data/concepts \
        --out_dir results/activations
done

echo "=== Step 3: Train probes for new concepts ==="
for concept in jk_rowling cormac_mccarthy; do
    echo "--- Training probe: $concept ---"
    python scripts/train_probes.py \
        --concept $concept \
        --act_dir results/activations \
        --out_dir results/probes
done

echo "=== Step 4: Compute contrast directions for new concepts ==="
python scripts/compute_contrast_directions.py \
    --model meta-llama/Llama-3.2-3B \
    --concepts jk_rowling cormac_mccarthy \
    --probe_dir results/probes \
    --out results/contrast_directions.json

echo "=== Step 5: Style injection sweep for new concepts ==="
python scripts/sweep_repe.py \
    --model meta-llama/Llama-3.2-3B \
    --contrast_path results/contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir results \
    --inject \
    --out_prefix new_style_ \
    --concepts jk_rowling cormac_mccarthy \
    --alphas 0.0 5.0 10.0 20.0 40.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 80 \
    --n_seeds 3

echo "=== Done ==="
echo "Results in results/new_style_repe_sweep.json"
echo "Check the completions for alpha=5 and alpha=10 for demo quality"
