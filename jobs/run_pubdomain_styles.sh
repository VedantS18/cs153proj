#!/bin/bash
#SBATCH --job-name=cs153_pubdom
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=cs153_pubdom_%j.out
#SBATCH --exclude=oat-02,oat-04

source /etc/profile.d/z00_lmod.sh
module load python/3.13.1 2>/dev/null || module load python/3.12.1 2>/dev/null || true
source /home/users/vedants8/cs153proj/cs153-env/bin/activate

export HF_HOME=/scratch/users/vedants8/hf_cache
export TRANSFORMERS_CACHE=/scratch/users/vedants8/hf_cache

cd /home/users/vedants8/cs153proj

MODEL="meta-llama/Llama-3.2-3B"
ACT_DIR="results/activations"
PROBE_DIR="results/probes"

echo "=== Step 1: Fetch public domain texts ==="
python3 scripts/fetch_pubdomain_styles.py

echo ""
echo "=== Step 2: Build probe training datasets ==="
python3 scripts/build_pubdomain_dataset.py

echo ""
echo "=== Step 3: Extract activations (all concepts in data/concepts) ==="
python3 scripts/extract_activations.py \
    --model $MODEL \
    --data_dir data/concepts \
    --out_dir $ACT_DIR

echo ""
echo "=== Step 4: Train probes (all extracted concepts) ==="
python3 scripts/train_probes.py \
    --act_dir $ACT_DIR \
    --out_dir $PROBE_DIR

echo ""
echo "=== Step 5: Recompute contrast directions ==="
python3 scripts/compute_contrast_directions.py \
    --act_dir $ACT_DIR \
    --weights_path $PROBE_DIR/probe_weights.json \
    --out_dir results

echo ""
echo "=== Step 6: Recompute subspace overlap ==="
python3 scripts/analyze_subspace_overlap.py \
    --act_dir $ACT_DIR \
    --weights_path $PROBE_DIR/probe_weights.json \
    --out_dir results

echo ""
echo "=== Step 7: Style injection sweep — all new pubdomain authors ==="
python3 scripts/sweep_repe.py \
    --model $MODEL \
    --contrast_path results/contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir results \
    --inject \
    --out_prefix pubdom_ \
    --concepts hemingway fitzgerald austen dickens woolf \
    --alphas 0.0 5.0 10.0 20.0 40.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 80 \
    --n_seeds 3

echo ""
echo "=== Done ==="
echo "Sentences per concept:"
for concept in hemingway fitzgerald austen dickens woolf; do
    n=$(wc -l < data/concepts/${concept}.jsonl 2>/dev/null || echo 0)
    echo "  $concept: $n training pairs"
done
