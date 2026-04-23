#!/bin/bash
#SBATCH --job-name=dla_ddpm
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#
# Train DDPM on DLA images.
# Submit: cd ~/Research/Machta/dla_project && sbatch scripts/train.sh

set -euo pipefail

# ---- Configuration ----
# 128x128 for fast training; DLA structure preserved at this resolution.
# Can scale to 256x256 later with more compute.
DATA_SRC="data/output/images"
IMAGE_SIZE=128
MODEL_DIM=64
DIM_MULTS="1 2 4 8"
EPOCHS=200
BATCH_SIZE=32
LR=1e-4
TIMESTEPS=1000
SAMPLING_STEPS=250

# ---- Copy data to node-local /tmp ----
LOCAL_DATA="/tmp/dla_train_${SLURM_JOB_ID}"
mkdir -p ${LOCAL_DATA}
echo "Copying training images to node-local /tmp ..."
t0=$SECONDS
cp ${DATA_SRC}/*.png ${LOCAL_DATA}/
elapsed=$((SECONDS - t0))
N_IMAGES=$(ls ${LOCAL_DATA}/*.png | wc -l)
echo "Copied ${N_IMAGES} images in ${elapsed}s"
trap "rm -rf ${LOCAL_DATA}" EXIT

# ---- Info ----
mkdir -p logs
echo ""
echo "=== DLA DDPM Training ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Data:       ${N_IMAGES} images (local copy)"
echo "Config:     ${IMAGE_SIZE}x${IMAGE_SIZE}, dim=${MODEL_DIM}, batch=${BATCH_SIZE}, epochs=${EPOCHS}"
echo ""

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ---- Train ----
python model/train.py \
    --data_dir ${LOCAL_DATA} \
    --image_size ${IMAGE_SIZE} \
    --model_dim ${MODEL_DIM} \
    --dim_mults ${DIM_MULTS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --timesteps ${TIMESTEPS} \
    --sampling_timesteps ${SAMPLING_STEPS} \
    --output_dir runs \
    --save_every 25 \
    --sample_every 25 \
    --n_samples 16 \
    --num_workers 0

echo "Training complete."
