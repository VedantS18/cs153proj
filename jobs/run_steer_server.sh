#!/bin/bash
#SBATCH --job-name=cs153_server
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=cs153_server_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

echo "Starting inference server on port 8787..."
echo "Node: $(hostname)"

python3 scripts/serve_steer.py \
    --model meta-llama/Llama-3.2-3B \
    --contrast results/contrast_directions.json \
    --port 8787 \
    --host 0.0.0.0
