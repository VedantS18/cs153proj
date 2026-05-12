#!/bin/bash
#SBATCH --job-name=cs153_inspect
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=cs153_inspect_%j.out
#SBATCH --exclude=oat-02,oat-04

# Run with: sbatch --export=ALL,CONCEPT=capital_cities jobs/run_inspect.sh
# Or for multiple concepts at once, submit 4 jobs in parallel.

set -e
cd "$SLURM_SUBMIT_DIR"

source .env
export HF_TOKEN

source /etc/profile.d/z00_lmod.sh
module purge
module load python/3.13.11

SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"

source cs153-env/bin/activate

CONCEPT=${CONCEPT:-capital_cities}
echo "Inspecting concept: $CONCEPT"
echo ""

python scripts/inspect_erasure.py \
    --model meta-llama/Llama-3.2-3B \
    --concept "$CONCEPT" \
    --weights_path results/probe_weights.json \
    --act_dir "${SCRATCH}/cs153/activations" \
    --max_new_tokens 40 \
    --n_examples 6

echo ""
echo "Done. $(date)"
