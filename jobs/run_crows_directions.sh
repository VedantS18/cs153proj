#!/bin/bash
#SBATCH --job-name=cs153_cwd
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=cs153_cwd_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

echo "=== Computing CrowS-Pairs contrast directions ==="
echo "Extracts activations from actual CrowS-Pairs sentences (sent_more vs sent_less)"
echo "so steering directions are exactly aligned with what the benchmark measures."

python scripts/compute_crows_directions.py \
    --model meta-llama/Llama-3.2-3B \
    --out results/crows_directions.json \
    --peak_layers results/peak_layers.json \
    --max_pairs 300
