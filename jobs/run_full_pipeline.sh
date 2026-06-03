#!/bin/bash
#SBATCH --job-name=cs153_pipeline
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=cs153_pipeline_%j.out
#SBATCH --exclude=oat-02,oat-04

# Full pipeline for a new model: extract activations → train probes →
# direction rotation analysis → rotation-corrected alpha sweep (bias only)
#
# Submit one job per model:
#   sbatch --export=ALL,MODEL="meta-llama/Llama-3.2-1B",TAG=llama1b jobs/run_full_pipeline.sh
#   sbatch --export=ALL,MODEL="mistralai/Mistral-7B-v0.1",TAG=mistral7b jobs/run_full_pipeline.sh

set -e
cd "$SLURM_SUBMIT_DIR"
source .env; export HF_TOKEN
source /etc/profile.d/z00_lmod.sh; module purge; module load python/3.13.11
SCRATCH="/scratch/users/$USER"
export HF_HOME="${SCRATCH}/hf_cache"
source cs153-env/bin/activate

MODEL=${MODEL:-"meta-llama/Llama-3.2-1B"}
TAG=${TAG:-"llama1b"}

echo "==== GPU ===="; nvidia-smi 2>&1 | head -5
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

ACT_DIR="${SCRATCH}/cs153/activations_${TAG}"
mkdir -p "$ACT_DIR"

echo ""
echo "==== [1/4] Extract activations: $MODEL ===="
python scripts/extract_activations.py \
    --model "$MODEL" \
    --data_dir data/concepts \
    --out_dir "$ACT_DIR"

echo ""
echo "==== [2/4] Train probes ===="
python scripts/train_probes.py \
    --act_dir "$ACT_DIR" \
    --out_dir "results/${TAG}"

echo ""
echo "==== [3/4] Direction rotation analysis ===="
python scripts/analyze_direction_rotation.py \
    --weights_path "results/${TAG}/probe_weights.json" \
    --out_dir "results/${TAG}" \
    --model_tag "$TAG"

echo ""
echo "==== [4/4] Rotation-corrected alpha sweep (bias + CrowS-Pairs) ===="
python scripts/sweep_alpha.py \
    --model "$MODEL" \
    --weights_path "results/${TAG}/probe_weights.json" \
    --test_dir data/concept_test \
    --out_dir "results/${TAG}" \
    --concepts gender_profession gender_emotion age_competence race_crime nationality_stereotype \
               capital_cities historical_dates \
               hemingway legal_text \
    --test_layers 1 20 24 27 \
    --alphas 1.0 2.0 5.0 10.0 15.0 20.0

echo ""
echo "Done. $(date)"
