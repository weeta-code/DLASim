#!/bin/bash
#SBATCH --job-name=dla_clean
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --output=logs/train_clean_%j.out
#SBATCH --error=logs/train_clean_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="vram40|vram48|vram80"
#SBATCH --time=08:00:00
#SBATCH --requeue

# Avoid CUDA memory fragmentation (A100 OOM fix)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail

# Training v3 "clean":
#   - 10,000 clusters at N=1000 (fixed)
#   - Rendering: scale=2, disc_radius=1, 512x512 (clear tree structure, no blobs)
#   - Cosine schedule + v-prediction + min-SNR
#   - A100 GPU for speed

DATA_SRC="data/fixed_n1000_clean/images"
IMAGE_SIZE=512
MODEL_DIM=64
DIM_MULTS="1 2 4 8"
EPOCHS=100
BATCH_SIZE=4
GRAD_ACCUM=4           # effective batch = 16
LR=1e-4
TIMESTEPS=1000
SAMPLING_STEPS=500
RUN_NAME="clean_main"  # fixed run name so we can resume after preemption

# Copy to /tmp
LOCAL_DATA="/tmp/dla_clean_${SLURM_JOB_ID}"
mkdir -p ${LOCAL_DATA}
echo "Copying images to /tmp..."
t0=$SECONDS
cp ${DATA_SRC}/*.png ${LOCAL_DATA}/
elapsed=$((SECONDS - t0))
N_IMAGES=$(ls ${LOCAL_DATA}/*.png | wc -l)
echo "Copied ${N_IMAGES} images in ${elapsed}s"
trap "rm -rf ${LOCAL_DATA}" EXIT

mkdir -p logs
echo ""
echo "=== DLA DDPM v3 (CLEAN) Training ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Data:       ${N_IMAGES} images, fixed N=1000, scale=2 disc_radius=1"
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
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --timesteps ${TIMESTEPS} \
    --sampling_timesteps ${SAMPLING_STEPS} \
    --beta_schedule cosine \
    --objective pred_v \
    --min_snr \
    --output_dir runs_clean \
    --run_name ${RUN_NAME} \
    --save_every 5 \
    --sample_every 5 \
    --n_samples 16 \
    --num_workers 0

echo "Training complete."
