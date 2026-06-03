#!/bin/bash
#SBATCH --job-name=cs153_repe
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_repe_%j.out
#SBATCH --exclude=oat-02,oat-04

# RepE-style constant steering sweep.
# Step 1: compute contrast directions (no GPU needed, fast)
# Step 2: run steering sweep with CrowS-Pairs + demo completions
#
# Submit: sbatch jobs/run_repe.sh
# Or for specific model: sbatch --export=ALL,MODEL="meta-llama/Llama-3.2-3B",TAG=llama3b jobs/run_repe.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate

MODEL=${MODEL:-"meta-llama/Llama-3.2-3B"}
TAG=${TAG:-"llama3b"}
ACT_DIR="${SCRATCH}/cs153/activations"

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5

echo ""
echo "==== [1/2] Compute contrast directions ===="
python scripts/compute_contrast_directions.py \
    --act_dir "$ACT_DIR" \
    --weights_path results/probe_weights.json \
    --out_dir results

echo ""
echo "==== [2/2] RepE steering sweep ===="
python scripts/sweep_repe.py \
    --model "$MODEL" \
    --contrast_path results/contrast_directions.json \
    --probe_weights results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    --alphas 0.0 10.0 20.0 40.0 60.0 100.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 40 \
    --max_crows 300 \
    --n_seeds 3

echo ""
echo "Done. $(date)"
