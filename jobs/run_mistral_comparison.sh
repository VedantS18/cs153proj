#!/bin/bash
#SBATCH --job-name=cs153_mistral
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_mistral_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

MODEL="mistralai/Mistral-7B-v0.1"
ACT_DIR="results/activations_mistral"
PROBE_DIR="results/probes_mistral"
OUT_DIR="results/mistral"

mkdir -p $ACT_DIR $PROBE_DIR $OUT_DIR

echo "=== Mistral 7B comparison run ==="
echo "Model: $MODEL"

echo ""
echo "=== Step 1: Extract activations ==="
python3 scripts/extract_activations.py \
    --model $MODEL \
    --data_dir data/concepts \
    --out_dir $ACT_DIR

echo ""
echo "=== Step 2: Train probes ==="
python3 scripts/train_probes.py \
    --act_dir $ACT_DIR \
    --out_dir $PROBE_DIR

echo ""
echo "=== Step 3: Compute contrast directions ==="
python3 scripts/compute_contrast_directions.py \
    --act_dir $ACT_DIR \
    --weights_path $PROBE_DIR/probe_weights.json \
    --out_dir $OUT_DIR

echo ""
echo "=== Step 4: Compute subspace overlap ==="
python3 scripts/analyze_subspace_overlap.py \
    --act_dir $ACT_DIR \
    --weights_path $PROBE_DIR/probe_weights.json \
    --out_dir $OUT_DIR

echo ""
echo "=== Step 5: RepE bias sweep (CrowS-Pairs) ==="
python3 scripts/sweep_repe.py \
    --model $MODEL \
    --contrast_path $OUT_DIR/contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir $OUT_DIR \
    --out_prefix mistral_ \
    --concepts age_competence gender_profession race_crime gender_emotion nationality_stereotype \
    --alphas 0.0 5.0 10.0 20.0 40.0 60.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 50 \
    --max_crows 300 \
    --n_seeds 2

echo ""
echo "=== Step 6: Style injection sweep ==="
python3 scripts/sweep_repe.py \
    --model $MODEL \
    --contrast_path $OUT_DIR/contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir $OUT_DIR \
    --inject \
    --out_prefix mistral_ \
    --concepts hemingway shakespeare legal_text scientific_writing news_wire \
    --alphas 0.0 5.0 10.0 20.0 40.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 80 \
    --n_seeds 3

echo ""
echo "=== Done ==="
echo "Outputs in $OUT_DIR/:"
ls $OUT_DIR/
