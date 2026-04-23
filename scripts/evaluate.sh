#!/bin/bash
#SBATCH --job-name=dla_eval
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#
# Generate samples from trained model and run evaluation.
#
# Usage:
#   sbatch scripts/evaluate.sh <checkpoint_path>
#   e.g.: sbatch scripts/evaluate.sh runs/run_20260324_120000/checkpoints/ckpt_epoch_0299.pt

set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint_path>"}
N_SAMPLES=64
RESULTS_DIR="results"

mkdir -p logs ${RESULTS_DIR}

echo "=== DLA Evaluation ==="
echo "Checkpoint: ${CHECKPOINT}"

# Activate env
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# 1. Generate samples
echo ""
echo "--- Generating samples ---"
python model/generate_samples.py \
    --checkpoint ${CHECKPOINT} \
    --n_samples ${N_SAMPLES} \
    --output_dir ${RESULTS_DIR}/generated

# 2. Fractal dimension of ground truth
echo ""
echo "--- Analyzing ground truth ---"
python eval/fractal_dim.py \
    --image_dir data/output/images \
    --label "Ground Truth (Random Walk)" \
    --output ${RESULTS_DIR}/fractal_dim_ground_truth.json

# 3. Fractal dimension of generated
echo ""
echo "--- Analyzing generated samples ---"
python eval/fractal_dim.py \
    --image_dir ${RESULTS_DIR}/generated/individual \
    --label "Generated (Custom DDPM)" \
    --output ${RESULTS_DIR}/fractal_dim_generated.json

# 4. Full comparison
echo ""
echo "--- Running comparison ---"
python eval/compare.py \
    --ground_truth data/output/images \
    --generated ${RESULTS_DIR}/generated/individual \
    --output_dir ${RESULTS_DIR}/comparison

echo ""
echo "Evaluation complete. Results in ${RESULTS_DIR}/"
