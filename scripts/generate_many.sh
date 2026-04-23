#!/bin/bash
#SBATCH --job-name=dla_gen_many
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --output=logs/gen_many_%j.out
#SBATCH --error=logs/gen_many_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint="vram40|vram48|vram80"
#SBATCH --time=01:00:00

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail

CHECKPOINT=${1:?"Usage: sbatch generate_many.sh <checkpoint_path> [n_samples]"}
N_SAMPLES=${2:-100}
OUTPUT_DIR=${3:-"results/many_samples"}

mkdir -p logs

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python model/generate_many.py \
    --checkpoint ${CHECKPOINT} \
    --n_samples ${N_SAMPLES} \
    --batch_size 16 \
    --output_dir ${OUTPUT_DIR}
