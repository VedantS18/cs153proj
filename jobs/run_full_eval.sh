#!/bin/bash
#SBATCH --job-name=cs153_eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_eval_%j.out
#SBATCH --exclude=oat-02,oat-04

# Full evaluation: probe accuracy + concept-specific tests + MMLU collateral damage
# Run with: sbatch --export=ALL,CONCEPTS="capital_cities element_symbols" jobs/run_full_eval.sh
# Or for all concepts: sbatch jobs/run_full_eval.sh

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

echo "==== GPU ===="
nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ""
echo "==== Building concept test files ===="
python scripts/build_concept_test.py

echo ""
echo "==== Full erasure evaluation ===="
python scripts/evaluate_erasure.py \
    --model meta-llama/Llama-3.2-3B \
    --act_dir "${SCRATCH}/cs153/activations" \
    --weights_path results/probe_weights.json \
    --test_dir data/concept_test \
    --out_dir results \
    ${CONCEPTS:+--concepts $CONCEPTS}

echo ""
echo "Done. $(date)"
