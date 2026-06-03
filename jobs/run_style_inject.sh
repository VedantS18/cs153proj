#!/bin/bash
#SBATCH --job-name=cs153_style
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cs153_style_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

echo "=== Style injection experiment ==="
echo "Adding contrast direction (subtract=False) to inject writing style into neutral prompts."
echo "Neutral prompts have no stylistic signal — any change is from steering alone."
echo ""

python scripts/sweep_repe.py \
    --model meta-llama/Llama-3.2-3B \
    --contrast_path results/contrast_directions.json \
    --probe_weights results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --inject \
    --out_prefix style_inject_ \
    --concepts hemingway shakespeare legal_text scientific_writing news_wire \
    --alphas 0.0 5.0 10.0 20.0 40.0 60.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 60 \
    --max_crows 0 \
    --n_seeds 3
