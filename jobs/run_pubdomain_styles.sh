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

NEW_CONCEPTS="hemingway fitzgerald austen dickens woolf"
MODEL="meta-llama/Llama-3.2-3B"

echo "=== Step 1: Fetch public domain texts from Project Gutenberg ==="
python scripts/fetch_pubdomain_styles.py

echo ""
echo "=== Step 2: Build probe training datasets ==="
python scripts/build_pubdomain_dataset.py

echo ""
echo "=== Step 3: Extract residual stream activations ==="
for concept in $NEW_CONCEPTS; do
    echo "  --- $concept ---"
    python scripts/extract_activations.py \
        --model $MODEL \
        --concept $concept \
        --data_dir data/concepts \
        --out_dir results/activations
done

echo ""
echo "=== Step 4: Train linear probes ==="
for concept in $NEW_CONCEPTS; do
    echo "  --- $concept ---"
    python scripts/train_probes.py \
        --concept $concept \
        --act_dir results/activations \
        --out_dir results/probes
done

echo ""
echo "=== Step 5: Recompute contrast directions (all concepts) ==="
python scripts/compute_contrast_directions.py \
    --model $MODEL \
    --all_concepts \
    --probe_dir results/probes \
    --out results/contrast_directions.json

echo ""
echo "=== Step 6: Recompute subspace overlap (all 15+ concepts) ==="
python scripts/analyze_subspace_overlap.py \
    --probe_dir results/probes \
    --out results/subspace_overlap.json

echo ""
echo "=== Step 7: Style injection sweep — new real-author concepts ==="
python scripts/sweep_repe.py \
    --model $MODEL \
    --contrast_path results/contrast_directions.json \
    --test_dir data/concept_test \
    --out_dir results \
    --inject \
    --out_prefix pubdom_ \
    --concepts $NEW_CONCEPTS \
    --alphas 0.0 5.0 10.0 20.0 40.0 \
    --layer_frac 0.85 \
    --n_steer_layers 4 \
    --max_new_tokens 80 \
    --n_seeds 3

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  results/pubdom_repe_sweep.json           (style injection metrics)"
echo "  results/subspace_overlap.json            (updated geometry)"
echo "  results/contrast_directions.json         (updated directions)"
echo ""
echo "Sentences used per concept:"
for concept in $NEW_CONCEPTS; do
    n=$(wc -l < data/concepts/${concept}.jsonl 2>/dev/null || echo 0)
    echo "  $concept: $n training pairs"
done
