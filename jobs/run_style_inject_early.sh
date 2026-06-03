#!/bin/bash
#SBATCH --job-name=cs153_styE
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cs153_styE_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

# Style is encoded at layer 1 (cos~1.0). Inject early so the model
# has 22+ layers to express the style coherently. Low alpha — these
# vectors are well-calibrated and don't need high strength.
python scripts/sweep_repe.py \
    --model meta-llama/Llama-3.2-3B \
    --contrast_path results/contrast_directions.json \
    --probe_weights results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --inject \
    --out_prefix style_inject_early_ \
    --concepts hemingway shakespeare legal_text scientific_writing news_wire \
    --alphas 0.0 2.0 5.0 8.0 12.0 20.0 \
    --layer_frac 0.05 \
    --n_steer_layers 2 \
    --max_new_tokens 60 \
    --max_crows 0 \
    --n_seeds 3
