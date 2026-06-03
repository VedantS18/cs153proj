#!/bin/bash
#SBATCH --job-name=cs153_alpha
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=cs153_alpha_%j.out
#SBATCH --exclude=oat-02,oat-04

# Rotation-corrected alpha sweep with CrowS-Pairs external validation.
# Uses per-layer probe directions (corrects for concept direction rotation).
# Tests layers scaled to model depth.
#
# Override via env vars:
#   MODEL, WEIGHTS_PATH, TAG (for output naming)
#   sbatch --export=ALL,MODEL="meta-llama/Llama-3.2-3B",WEIGHTS_PATH="results/probe_weights.json",TAG=llama3b jobs/run_alpha_sweep.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate
pip install -q datasets

MODEL=${MODEL:-"meta-llama/Llama-3.2-3B"}
WEIGHTS_PATH=${WEIGHTS_PATH:-"results/probe_weights.json"}
TAG=${TAG:-"llama3b"}

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5

# Scale test layers to model depth
N_LAYERS=$(python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('${MODEL}')
print(getattr(c, 'num_hidden_layers', 28))
")
L_MID=$(python -c "print(int(${N_LAYERS} * 0.7))")
L_LATE=$(python -c "print(int(${N_LAYERS} * 0.85))")
L_LAST=$(python -c "print(${N_LAYERS} - 1)")
echo "Model: $MODEL  |  layers: $N_LAYERS  |  testing: 1, $L_MID, $L_LATE, $L_LAST"

echo ""; echo "==== Rotation-corrected alpha sweep (CrowS-Pairs + per-layer dirs) ===="
python scripts/sweep_alpha.py \
    --model "$MODEL" \
    --weights_path "$WEIGHTS_PATH" \
    --test_dir data/concept_test \
    --out_dir results \
    --concepts capital_cities element_symbols historical_dates \
               gender_profession gender_emotion race_crime age_competence nationality_stereotype \
               hemingway legal_text scientific_writing \
    --test_layers 1 "$L_MID" "$L_LATE" "$L_LAST" \
    --alphas 1.0 2.0 5.0 10.0 15.0 20.0 \
    --per_layer_dirs \
    --max_crows 300

mv results/alpha_sweep.json "results/alpha_sweep_${TAG}.json" 2>/dev/null || true
echo "Output: results/alpha_sweep_${TAG}.json"
echo ""; echo "Done. $(date)"
