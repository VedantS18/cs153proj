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
PREFIX="mistral_"

echo "=== Mistral 7B comparison run ==="
echo "Model: $MODEL"
echo "All outputs prefixed with: $PREFIX"

echo ""
echo "=== Step 1: Extract activations ==="
for concept in hemingway shakespeare legal_text scientific_writing news_wire \
               age_competence gender_profession race_crime gender_emotion nationality_stereotype \
               capital_cities country_language element_symbols historical_dates inventor_invention; do
    echo "  $concept ..."
    python scripts/extract_activations.py \
        --model $MODEL \
        --concept $concept \
        --data_dir data/concepts \
        --out_dir results/activations_mistral
done

echo ""
echo "=== Step 2: Train probes ==="
for concept in hemingway shakespeare legal_text scientific_writing news_wire \
               age_competence gender_profession race_crime gender_emotion nationality_stereotype \
               capital_cities country_language element_symbols historical_dates inventor_invention; do
    echo "  $concept ..."
    python scripts/train_probes.py \
        --concept $concept \
        --act_dir results/activations_mistral \
        --out_dir results/probes_mistral
done

echo ""
echo "=== Step 3: Compute contrast directions ==="
python scripts/compute_contrast_directions.py \
    --model $MODEL \
    --all_concepts \
    --probe_dir results/probes_mistral \
    --out results/${PREFIX}contrast_directions.json

echo ""
echo "=== Step 4: Compute subspace overlap (geometry) ==="
python scripts/analyze_subspace_overlap.py \
    --probe_dir results/probes_mistral \
    --out results/${PREFIX}subspace_overlap.json

echo ""
echo "=== Step 5: RepE bias sweep (CrowS-Pairs) ==="
python scripts/sweep_repe.py \
    --model $MODEL \
    --contrast_path results/${PREFIX}contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir results \
    --out_prefix ${PREFIX} \
    --concepts age_competence gender_profession race_crime gender_emotion nationality_stereotype \
    --alphas 0.0 5.0 10.0 20.0 40.0 60.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 50 \
    --max_crows 300 \
    --n_seeds 2

echo ""
echo "=== Step 6: Style injection sweep ==="
python scripts/sweep_repe.py \
    --model $MODEL \
    --contrast_path results/${PREFIX}contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir results \
    --inject \
    --out_prefix ${PREFIX} \
    --concepts hemingway shakespeare legal_text scientific_writing news_wire \
    --alphas 0.0 5.0 10.0 20.0 40.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 80 \
    --n_seeds 3

echo ""
echo "=== Done ==="
echo "Key outputs:"
echo "  results/${PREFIX}repe_sweep.json       (bias, for CrowS comparison)"
echo "  results/${PREFIX}subspace_overlap.json  (geometry, for cross-model comparison)"
echo "  results/${PREFIX}style_inject_completions.json (style, for demo)"
