#!/bin/bash
#SBATCH --job-name=cs153_judge
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=cs153_judge_%j.out
#SBATCH --exclude=oat-02,oat-04

# LLM-as-judge bias erasure evaluation.
# Pass 1: Llama 3.2 3B generates completions before/after erasure.
# Pass 2: Llama 3.2 1B judges each completion as Stereotyped/Neutral/Counter-stereotyped.
#
# Override target model for cross-model comparison:
#   sbatch --export=ALL,TARGET=meta-llama/Llama-3.2-3B,TAG=llama3b jobs/run_judge_bias.sh
#   sbatch --export=ALL,TARGET=mistralai/Mistral-7B-v0.1,TAG=mistral7b jobs/run_judge_bias.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate

TARGET=${TARGET:-"meta-llama/Llama-3.2-3B"}
JUDGE=${JUDGE:-"meta-llama/Llama-3.2-3B"}
TAG=${TAG:-"llama3b"}
WEIGHTS=${WEIGHTS:-"results/probe_weights.json"}

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ""
echo "==== LLM-as-judge bias evaluation ===="
echo "Target: $TARGET  |  Judge: $JUDGE  |  Tag: $TAG"
echo ""

python scripts/judge_bias_erasure.py \
    --target_model "$TARGET" \
    --judge_model "$JUDGE" \
    --weights_path "$WEIGHTS" \
    --out_dir results \
    --tag "$TAG" \
    --alphas 1.0 5.0 10.0 15.0 20.0 \
    --erase_layer_frac 0.9 \
    --max_new_tokens 30 \
    --n_seeds 3

echo ""
echo "Done. $(date)"
