#!/bin/bash
#SBATCH --job-name=dla_fixN_gen
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --output=logs/fixN_gen_%A_%a.out
#SBATCH --error=logs/fixN_gen_%A_%a.err
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#
# Generate 10,000 fixed-N DLA clusters in parallel.
# 100 tasks x 100 clusters each = 10,000 total.
#
# Submit from dla_project/ directory:
#   sbatch scripts/generate_fixed_n.sh

set -euo pipefail

# ---- Configuration ----
PARTICLES=1000                  # fixed N per cluster
IMG_SIZE=256                    # image resolution (used for raw cluster only)
CLUSTERS_PER_TASK=100
OUTPUT_DIR="data/fixed_n1000_raw_v2"
SIMULATOR="simulator/dla_sim"

BASE_SEED=100000
TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID * CLUSTERS_PER_TASK))

mkdir -p logs
echo "=== Fixed-N DLA Generation ==="
echo "Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Particles:  ${PARTICLES} (fixed)"
echo "Clusters:   ${CLUSTERS_PER_TASK}"
echo "Seed:       ${TASK_SEED}"

if [ ! -f "${SIMULATOR}" ]; then
    echo "Building simulator..."
    module load gcc/9.4.0
    make -C simulator
fi

./${SIMULATOR} \
    --particles ${PARTICLES} \
    --size ${IMG_SIZE} \
    --count ${CLUSTERS_PER_TASK} \
    --outdir ${OUTPUT_DIR} \
    --seed ${TASK_SEED} \
    --prefix "n1k_t${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
