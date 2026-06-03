#!/bin/bash
#SBATCH --job-name=cs153_probe
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=cs153_probe_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

echo "=== Probe-direction steering control experiment ==="
echo "Using probe classifier weights as steering direction instead of contrast direction."
echo "Expected: CrowS-Pairs barely moves for bias concepts (cos~0.16),"
echo "          stylistic concept_test also barely moves (probe=contrast for these)."
echo ""

python scripts/sweep_repe.py \
    --model meta-llama/Llama-3.2-3B \
    --contrast_path results/contrast_directions.json \
    --probe_weights results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --steer_mode probe \
    --out_prefix probe_steer_ \
    --concepts gender_profession gender_emotion age_competence race_crime \
               nationality_stereotype hemingway shakespeare legal_text \
    --alphas 0.0 10.0 20.0 40.0 60.0 100.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 40 \
    --max_crows 300 \
    --n_seeds 2
