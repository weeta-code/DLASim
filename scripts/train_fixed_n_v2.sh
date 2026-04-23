#!/bin/bash
#SBATCH --job-name=dla_fN_v2
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=logs/train_fN_v2_%j.out
#SBATCH --error=logs/train_fN_v2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#
# Training run v2:
#   - 10,000 clusters (vs 2,400)
#   - 2px disc rendering (vs 1px)
#   - Cosine noise schedule (vs linear)
#   - V-prediction objective (vs epsilon)
#   - Min-SNR loss weighting
#   - 300 epochs (vs 150)

set -euo pipefail

# ---- Configuration ----
DATA_SRC="data/fixed_n1000_disc2/images"
IMAGE_SIZE=256
MODEL_DIM=64
DIM_MULTS="1 2 4 8"
EPOCHS=300
BATCH_SIZE=16
LR=1e-4
TIMESTEPS=1000
SAMPLING_STEPS=500          # More resampling levels for better quality

# Copy to local /tmp for fast I/O
LOCAL_DATA="/tmp/dla_fN_v2_${SLURM_JOB_ID}"
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
echo "=== Fixed-N DDPM Training v2 ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Data:       ${N_IMAGES} images, fixed N=1000, 2px disc rendering"
echo "Config:     ${IMAGE_SIZE}x${IMAGE_SIZE}, dim=${MODEL_DIM}, batch=${BATCH_SIZE}, epochs=${EPOCHS}"
echo "Objective:  v-prediction with cosine schedule + min-SNR loss"
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
    --beta_schedule cosine \
    --objective pred_v \
    --min_snr \
    --output_dir runs_v2 \
    --save_every 25 \
    --sample_every 10 \
    --n_samples 16 \
    --num_workers 0

echo "Training complete."
