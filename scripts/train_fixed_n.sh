#!/bin/bash
#SBATCH --job-name=dla_fixN
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=logs/train_fixN_%j.out
#SBATCH --error=logs/train_fixN_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=03:00:00

set -euo pipefail

# Fixed N=1000, 1px per particle, 256x256 images
DATA_SRC="data/fixed_n1000/images"
IMAGE_SIZE=256
MODEL_DIM=64
DIM_MULTS="1 2 4 8"
EPOCHS=150
BATCH_SIZE=16
LR=1e-4
TIMESTEPS=1000
SAMPLING_STEPS=250

# Copy to local /tmp
LOCAL_DATA="/tmp/dla_fixN_${SLURM_JOB_ID}"
mkdir -p ${LOCAL_DATA}
echo "Copying images to /tmp..."
cp ${DATA_SRC}/*.png ${LOCAL_DATA}/
N_IMAGES=$(ls ${LOCAL_DATA}/*.png | wc -l)
echo "Copied ${N_IMAGES} images"
trap "rm -rf ${LOCAL_DATA}" EXIT

mkdir -p logs
echo ""
echo "=== Fixed N=1000 DDPM Training ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Data:       ${N_IMAGES} images, fixed N=1000, 1px/particle"
echo "Config:     ${IMAGE_SIZE}x${IMAGE_SIZE}, dim=${MODEL_DIM}, batch=${BATCH_SIZE}, epochs=${EPOCHS}"
echo ""

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

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
    --output_dir runs_fixN \
    --save_every 25 \
    --sample_every 10 \
    --n_samples 16 \
    --num_workers 0

echo "Training complete."
